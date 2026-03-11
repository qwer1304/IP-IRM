import os
import argparse

def normalize_path(path):
    """Normalize path separators to forward slash."""
    return path.replace('\\', '/')

def dump_filenames(directory, output_file):
    """Dump all filenames in directory tree to a file."""
    filenames = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            filepath = os.path.join(root, file)
            rel_path = os.path.relpath(filepath, directory)
            filenames.append(normalize_path(rel_path))
    
    filenames.sort()
    
    out_dir = os.path.dirname(output_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    
    with open(output_file, 'w', encoding='ascii', errors='replace', newline='\n') as f:
        for filename in filenames:
            f.write(f"{filename}\n")
    
    print(f"Dumped {len(filenames)} filenames to {output_file}")

def compare_files(file_a, file_b):
    """Compare two dumped filename files bidirectionally."""
    with open(file_a, 'r', encoding='ascii', errors='replace') as f:
        content_a = f.read()
    with open(file_b, 'r', encoding='ascii', errors='replace') as f:
        content_b = f.read()

    # Debug: show raw first line bytes
    print(f"First line of {file_a} raw: {repr(content_a.splitlines()[0])}")
    print(f"First line of {file_b} raw: {repr(content_b.splitlines()[0])}")

    # Use splitlines() - handles \n, \r\n, \r
    lines_a = [normalize_path(l.strip()) for l in content_a.splitlines()]
    lines_a_nonempty = [l for l in lines_a if l]
    files_a = set(lines_a_nonempty)

    lines_b = [normalize_path(l.strip()) for l in content_b.splitlines()]
    lines_b_nonempty = [l for l in lines_b if l]
    files_b = set(lines_b_nonempty)

    print(f"Raw lines in {file_a}: {len(lines_a)}")
    print(f"Non-empty lines in {file_a}: {len(lines_a_nonempty)}")
    print(f"Unique files in {file_a}: {len(files_a)}")
    print(f"Raw lines in {file_b}: {len(lines_b)}")
    print(f"Non-empty lines in {file_b}: {len(lines_b_nonempty)}")
    print(f"Unique files in {file_b}: {len(files_b)}")

    if len(lines_a_nonempty) != len(files_a):
        print(f"WARNING: {len(lines_a_nonempty) - len(files_a)} duplicate entries in {file_a}")
    if len(lines_b_nonempty) != len(files_b):
        print(f"WARNING: {len(lines_b_nonempty) - len(files_b)} duplicate entries in {file_b}")

    in_a_not_b = files_a - files_b
    in_b_not_a = files_b - files_a

    print(f"\nFiles in {file_a} but NOT in {file_b}: {len(in_a_not_b)}")
    print(f"Files in {file_b} but NOT in {file_a}: {len(in_b_not_a)}")

    if not in_a_not_b and not in_b_not_a:
        print("OK - perfect match - all files present in both")
    else:
        if in_a_not_b:
            print(f"\nFAIL - {len(in_a_not_b)} files in {file_a} but NOT in {file_b}:")
            for f in sorted(in_a_not_b)[:20]:
                print(f"  {f}")
            if len(in_a_not_b) > 20:
                print(f"  ... and {len(in_a_not_b) - 20} more")

        if in_b_not_a:
            print(f"\nFAIL - {len(in_b_not_a)} files in {file_b} but NOT in {file_a}:")
            for f in sorted(in_b_not_a)[:20]:
                print(f"  {f}")
            if len(in_b_not_a) > 20:
                print(f"  ... and {len(in_b_not_a) - 20} more")

def compare_filenames(directory, reference_file):
    """Compare filenames in directory against reference file bidirectionally."""
    with open(reference_file, 'r', encoding='ascii', errors='replace') as f:
        content_ref = f.read()

    lines_ref = [normalize_path(l.strip()) for l in content_ref.splitlines()]
    lines_ref_nonempty = [l for l in lines_ref if l]
    reference_files = set(lines_ref_nonempty)

    current_files = set()
    current_list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            filepath = os.path.join(root, file)
            rel_path = normalize_path(os.path.relpath(filepath, directory))
            current_files.add(rel_path)
            current_list.append(rel_path)

    print(f"Files in directory: {len(current_list)}")
    print(f"Unique files in directory: {len(current_files)}")
    print(f"Non-empty lines in reference: {len(lines_ref_nonempty)}")
    print(f"Unique files in reference: {len(reference_files)}")

    if len(current_list) != len(current_files):
        print(f"WARNING: {len(current_list) - len(current_files)} duplicate entries in directory")
    if len(lines_ref_nonempty) != len(reference_files):
        print(f"WARNING: {len(lines_ref_nonempty) - len(reference_files)} duplicate entries in reference file")

    in_dir_not_in_file = current_files - reference_files
    in_file_not_in_dir = reference_files - current_files

    print(f"\nFiles in directory but NOT in reference: {len(in_dir_not_in_file)}")
    print(f"Files in reference but NOT in directory: {len(in_file_not_in_dir)}")

    if not in_dir_not_in_file and not in_file_not_in_dir:
        print("OK - perfect match - all files present in both")
    else:
        if in_dir_not_in_file:
            print(f"\nFAIL - {len(in_dir_not_in_file)} files in directory but NOT in reference:")
            for f in sorted(in_dir_not_in_file)[:20]:
                print(f"  {f}")
            if len(in_dir_not_in_file) > 20:
                print(f"  ... and {len(in_dir_not_in_file) - 20} more")

        if in_file_not_in_dir:
            print(f"\nFAIL - {len(in_file_not_in_dir)} files in reference but NOT in directory:")
            for f in sorted(in_file_not_in_dir)[:20]:
                print(f"  {f}")
            if len(in_file_not_in_dir) > 20:
                print(f"  ... and {len(in_file_not_in_dir) - 20} more")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Dump and compare directory file listings')
    parser.add_argument('mode', choices=['dump', 'compare', 'compare_files'],
                        help='dump: save filenames to file | compare: check dir against file | compare_files: compare two dump files')
    parser.add_argument('arg1', help='Directory (dump/compare) or file A (compare_files)')
    parser.add_argument('arg2', help='Output file (dump/compare) or file B (compare_files)')
    args = parser.parse_args()

    if args.mode == 'dump':
        dump_filenames(args.arg1, args.arg2)
    elif args.mode == 'compare':
        compare_filenames(args.arg1, args.arg2)
    elif args.mode == 'compare_files':
        compare_files(args.arg1, args.arg2)