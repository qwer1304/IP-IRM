import os
import argparse

def dump_filenames(directory, output_file):
    """Dump all filenames in directory tree to a file."""
    filenames = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            filepath = os.path.join(root, file)
            rel_path = os.path.relpath(filepath, directory)
            filenames.append(rel_path)
    
    filenames.sort()
    with open(output_file, 'w', encoding='ascii', errors='replace') as f:
        for filename in filenames:
            f.write(f"{filename}\n")
    
    print(f"Dumped {len(filenames)} filenames to {output_file}")

def compare_files(file_a, file_b):
    """Compare two dumped filename files bidirectionally."""
    with open(file_a, 'r', encoding='ascii', errors='replace') as f:
        files_a = set(line.strip() for line in f.readlines())
    
    with open(file_b, 'r', encoding='ascii', errors='replace') as f:
        files_b = set(line.strip() for line in f.readlines())
    
    in_a_not_b = files_a - files_b
    in_b_not_a = files_b - files_a
    
    print(f"Files in {file_a}: {len(files_a)}")
    print(f"Files in {file_b}: {len(files_b)}")
    
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
        reference_files = set(line.strip() for line in f.readlines())
    
    current_files = set()
    for root, dirs, files in os.walk(directory):
        for file in files:
            filepath = os.path.join(root, file)
            rel_path = os.path.relpath(filepath, directory)
            current_files.add(rel_path)
    
    in_dir_not_in_file = current_files - reference_files
    in_file_not_in_dir = reference_files - current_files
    
    print(f"Files in directory: {len(current_files)}")
    print(f"Files in reference: {len(reference_files)}")
    
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
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['dump', 'compare', 'compare_files'])
    parser.add_argument('arg1', help='Directory (dump/compare) or file A (compare_files)')
    parser.add_argument('arg2', help='Output file (dump/compare) or file B (compare_files)')
    args = parser.parse_args()
    
    if args.mode == 'dump':
        dump_filenames(args.arg1, args.arg2)
    elif args.mode == 'compare':
        compare_filenames(args.arg1, args.arg2)
    elif args.mode == 'compare_files':
        compare_files(args.arg1, args.arg2)