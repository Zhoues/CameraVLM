from __future__ import annotations

import argparse
import os
import shutil
import stat
import subprocess
from datetime import UTC, datetime
from pathlib import Path


DEFAULT_TARGET_DIR = Path(
    "/share/project/zhouenshen/sfs/dataset/ActivePerception/Pano/vlm_camera_dataset/latent_data"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clear a directory efficiently on shared storage."
    )
    parser.add_argument(
        "--target-dir",
        type=Path,
        default=DEFAULT_TARGET_DIR,
        help="Directory to clear.",
    )
    parser.add_argument(
        "--strategy",
        choices=("move-and-sweep", "direct"),
        default="move-and-sweep",
        help=(
            "move-and-sweep: atomically move the whole directory aside, recreate it, "
            "then clean the moved tree. direct: delete the current contents in place."
        ),
    )
    parser.add_argument(
        "--cleanup-mode",
        choices=("detached", "blocking"),
        default="detached",
        help=(
            "detached returns after the directory is emptied and spawns background cleanup. "
            "blocking waits for the slow physical deletion to finish."
        ),
    )
    parser.add_argument(
        "--trash-root",
        type=Path,
        default=None,
        help=(
            "Sibling directory used to stage moved trees before cleanup. "
            "Defaults to <target parent>/.clean_trash."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would happen without modifying the filesystem.",
    )

    # Compatibility flags from older revisions. They are accepted so existing
    # commands do not break, but the move-and-sweep strategy does not use them.
    parser.add_argument("--num-workers", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--chunksize", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--task-depth", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--file-batch-size", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument(
        "--remove-empty-dirs",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def timestamp_tag() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def count_top_level_entries(root: Path) -> int:
    with os.scandir(root) as entries:
        return sum(1 for _ in entries)


def default_trash_root(root: Path) -> Path:
    return root.parent / ".clean_trash"


def validate_root(root: Path) -> None:
    if not root.exists():
        raise SystemExit(f"Target directory does not exist: {root}")
    if not root.is_dir():
        raise SystemExit(f"Target path is not a directory: {root}")


def ensure_safe_trash_root(root: Path, trash_root: Path) -> Path:
    trash_root = trash_root.expanduser().resolve()
    if trash_root == root or root in trash_root.parents:
        raise SystemExit(
            f"Trash root must be outside the target directory: {trash_root}"
        )
    trash_root.mkdir(parents=True, exist_ok=True)

    root_dev = root.parent.stat().st_dev
    trash_dev = trash_root.stat().st_dev
    if root_dev != trash_dev:
        raise SystemExit(
            "Trash root must be on the same filesystem as the target directory "
            "for atomic rename. Use a sibling path under the same parent."
        )
    return trash_root


def build_staging_dir(root: Path, trash_root: Path) -> Path:
    return trash_root / f"{root.name}.{timestamp_tag()}.{os.getpid()}"


def preserve_directory_metadata(src: Path, dst: Path) -> None:
    try:
        shutil.copystat(src, dst, follow_symlinks=False)
    except OSError:
        pass

    try:
        src_stat = src.stat()
        os.chown(dst, src_stat.st_uid, src_stat.st_gid)
    except (AttributeError, PermissionError, OSError):
        pass

    try:
        os.chmod(dst, stat.S_IMODE(src.stat().st_mode))
    except OSError:
        pass


def move_root_aside(root: Path, trash_root: Path) -> Path:
    staged_root = build_staging_dir(root, trash_root)
    os.replace(root, staged_root)

    try:
        root.mkdir()
        preserve_directory_metadata(staged_root, root)
    except Exception:  # noqa: BLE001
        os.replace(staged_root, root)
        raise

    return staged_root


def iter_top_level_paths(root: Path) -> list[str]:
    with os.scandir(root) as entries:
        return [entry.path for entry in entries]


def rm_command_available() -> bool:
    return shutil.which("rm") is not None


def run_rm(paths: list[str], *, detached: bool) -> int | None:
    command = ["rm", "-rf", "--one-file-system", "--", *paths]
    if detached:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        return process.pid

    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        raise SystemExit(f"rm failed with exit code {completed.returncode}")
    return None


def delete_path_python(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink(missing_ok=True)
        return
    shutil.rmtree(path)


def remove_dir_if_empty(path: Path) -> None:
    try:
        path.rmdir()
    except OSError:
        pass


def direct_delete(root: Path) -> None:
    paths = iter_top_level_paths(root)
    if not paths:
        print("Nothing to delete.")
        return

    print(f"Deleting {len(paths)} top-level entries in place.")
    if rm_command_available():
        run_rm(paths, detached=False)
        return

    for path_text in paths:
        delete_path_python(Path(path_text))


def move_and_sweep(
    root: Path,
    trash_root: Path,
    *,
    cleanup_mode: str,
) -> None:
    top_level_entries = count_top_level_entries(root)
    if top_level_entries == 0:
        print("Nothing to delete.")
        return

    print(
        f"Moving {top_level_entries} top-level entries out of {root} "
        "with an atomic directory rename."
    )
    staged_root = move_root_aside(root, trash_root)
    print(f"Target directory is now empty and recreated at: {root}")
    print(f"Staged old contents at: {staged_root}")

    if cleanup_mode == "detached":
        if rm_command_available():
            pid = run_rm([str(staged_root)], detached=True)
            print(f"Background cleanup started with PID {pid}.")
            return

        print(
            "System rm is unavailable, falling back to blocking Python cleanup."
        )

    print("Running blocking cleanup of the staged directory.")
    if rm_command_available():
        run_rm([str(staged_root)], detached=False)
        remove_dir_if_empty(trash_root)
        return

    shutil.rmtree(staged_root)
    remove_dir_if_empty(trash_root)


def main() -> None:
    args = parse_args()
    root = args.target_dir.expanduser().resolve()
    validate_root(root)

    trash_root = args.trash_root or default_trash_root(root)

    if args.dry_run:
        top_level_entries = count_top_level_entries(root)
        print(f"Strategy: {args.strategy}")
        print(f"Target directory: {root}")
        print(f"Top-level entries: {top_level_entries}")
        if args.strategy == "move-and-sweep":
            safe_trash_root = ensure_safe_trash_root(root, trash_root)
            staged_root = build_staging_dir(root, safe_trash_root)
            print(f"Trash root: {safe_trash_root}")
            print(f"Would stage old directory at: {staged_root}")
            print(f"Cleanup mode: {args.cleanup_mode}")
        else:
            print("Would delete current contents in place.")
        return

    if args.strategy == "move-and-sweep":
        safe_trash_root = ensure_safe_trash_root(root, trash_root)
        move_and_sweep(
            root,
            safe_trash_root,
            cleanup_mode=args.cleanup_mode,
        )
        return

    direct_delete(root)


if __name__ == "__main__":
    main()
