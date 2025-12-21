import os
import json
from pathlib import Path
import argparse

def record_symlinks(root: Path, out_file: Path):
    """
    Recursively scan `root` and record all symlinks.
    Stores: link_path -> link_target (as stored in the symlink)
    """
    root = root.resolve()
    symlinks = {}

    for dirpath, dirnames, filenames in os.walk(root):
        dirpath = Path(dirpath)

        # Check files
        for name in filenames:
            p = dirpath / name
            if p.is_symlink():
                symlinks[str(p)] = os.readlink(p)

        # Check directories (important!)
        for name in dirnames:
            p = dirpath / name
            if p.is_symlink():
                symlinks[str(p)] = os.readlink(p)

    with open(out_file, "w") as f:
        json.dump(symlinks, f, indent=2)

    print(f"Recorded {len(symlinks)} symlinks -> {out_file}")

def restore_symlinks(map_file: Path):
    """
    Recreate symlinks from a previously saved map.
    """
    if not map_file.exists():
        print("No symlink map found.")
        return

    with open(map_file) as f:
        symlinks = json.load(f)

    restored = 0

    for link, target in symlinks.items():
        link = Path(link)

        # Ensure parent dir exists
        link.parent.mkdir(parents=True, exist_ok=True)

        # Remove stale file/link
        if link.exists() or link.is_symlink():
            link.unlink()

        os.symlink(target, link)
        restored += 1

    print(f"Restored {restored} symlinks.")

def main(args):
    if args.action == 'record':
        record_symlinks(
            root=Path(args.root),
            out_file=Path(args.map_file)
        )

    elif args.action == 'restore':
        restore_symlinks(Path(args.map_file))

if __name__ == "__main__":
    # create the top-level parser
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='action', help='record, restore')

    # create the parser for the "record" command
    parser_record = subparsers.add_parser('record', help='record')
    parser_record.add_argument('--root', type=str, required=True, help='root')
    parser_record.add_argument('--map_file', type=str, required=True, help='map file, json')

    # create the parser for the "restore" command
    parser_restore = subparsers.add_parser('restore', help='restore')
    parser_restore.add_argument('--map_file', type=str, required=True, help='map file, json')
       
    args = parser.parse_args()

    main(args)
