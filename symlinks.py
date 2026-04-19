import os
import json
from pathlib import Path
import argparse

def record_symlinks(root: Path, out_file: Path, ft: str):
    """
    Recursively scan `root` and record all symlinks.
    Stores: link_path -> link_target (as stored in the symlink)
    """
    root = root.resolve()
    symlinks = {}

    for dirpath, dirnames, filenames in os.walk(root):
        # dirpath: The path to the current directory the loop is visiting
        # dirnames: A list of the names of all subdirectories inside the current dirpath
        # filenames: A list of the names of all non-directory files (images, .csv, .txt, etc.) inside the current dirpath
        # 1. Sort in-place to ensure deterministic traversal of subdirectories
        enc_files = [f for f in filenames if f.lower().endswith(f'.{ft}')]
        
        dirnames.sort()
        # 2. Sort filenames to ensure deterministic processing of files
        enc_files.sort()

        dirpath = Path(dirpath)

        # Check files (now in alphabetical order)
        for name in enc_files:
            p = dirpath / name

            if p.is_symlink():
                symlinks[str(p)] = os.readlink(p)
            else:
                print(f"file {p} is not a symlink")
                assert False

        # Check directories (now in alphabetical order)
        for name in dirnames:
            p = dirpath / name
            if p.is_symlink():
                symlinks[str(p)] = os.readlink(p)

    with open(out_file, "w") as f:
        # This sorts the dictionary alphabetically by the path string
        json.dump(symlinks, f, indent=2, sort_keys=True)

    print(f"Recorded {len(symlinks)} symlinks -> {out_file}")

def restore_symlinks(map_file: Path, stop_on_missing_target=False):
    """
    Recreate symlinks from a previously saved map.
    """
    if not map_file.exists():
        print("No symlink map found.")
        return

    with open(map_file) as f:
        symlinks = json.load(f)

    restored = 0
    missing = 0

    for link, target in symlinks.items():
        link = Path(link)

        if !Path(target).is_file():
            print(f"target {target} doesn't exist")
            missing += 1
            if not stop_on_missing_target:
                continue
            else:
                print("Stopping")
                break
        # Ensure parent dir exists
        link.parent.mkdir(parents=True, exist_ok=True)

        # Remove stale file/link
        if link.exists() or link.is_symlink():
            link.unlink()

        #os.symlink(src, dst)
        #src: The path of the original file or directory (the target).
        #dst: The path where the symbolic link will be created.
        os.symlink(target, link)
        restored += 1

    print(f"Restored {restored} symlinks. Missing {missing} symlinks.")

def main(args):
    if args.action == 'record':
        record_symlinks(
            root=Path(args.root),
            out_file=Path(args.map_file),
            ft=args.enc,
        )
    elif args.action == 'restore':
        restore_symlinks(Path(args.map_file), args.stop_on_missing_target)
    else:
        raise ValueError(f"Unknown action {args.action}")

if __name__ == "__main__":
    # create the top-level parser
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='action', help='record, restore')

    # create the parser for the "record" command
    parser_record = subparsers.add_parser('record', help='record')
    parser_record.add_argument('--root', type=str, required=True, help='root')
    parser_record.add_argument('--map_file', type=str, required=True, help='map file, json')
    parser_record.add_argument('--enc', type=str, default='jpg',  help='image files encoding')

    # create the parser for the "restore" command
    parser_restore = subparsers.add_parser('restore', help='restore')
    parser_restore.add_argument('--map_file', type=str, required=True, help='map file, json')
    parser_restore.add_argument('--stop_on_missing_target', action='store_true', help='stop when a target is missing')
       
    args = parser.parse_args()

    main(args)
