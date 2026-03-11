import os
import argparse

def dump_filenames(directory, output_file):
    """Dump all filenames in directory tree to a file."""
    filenames = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            filepath = os.path.join(root, file)
            # Store relative path for portability
            rel_path = os.path.relpath(filepath, directory)
            filenames.append(rel_path)
    
    filenames.sort()
    with open(output_file, 'w') as f:
        for filename in filenames:
            f.write(f"{filename}\n")
    
    print(f"Dumped {len(filenames)} filenames to {output_file}")

def compare_filenames(directory, reference_file):
    """Compare filenames in directory against reference file bidirectionally."""
    # Load reference filenames
    with open(reference_file, 'r') as f:
        reference_files = set(line.strip() for line in f.readlines())
    
    # Load current directory filenames
    current_files = set()
    for root, dirs, files in os.walk(directory):
        for file in files:
            filepath = os.path.join(root, file)
            rel_path = os.path.relpath(filepath, directory)
            current_files.add(rel_path)
    
    # Bidirectional set difference
    in_dir_not_in_file = current_files - reference_files
    in_file_not_in_dir = reference_files - current_files
    
    print(f"Files in directory: {len(current_files)}")
    print(f"Files in reference: {len(reference_files)}")
    
    if not in_dir_not_in_file and not in_file_not_in_dir:
        print(" Perfect match - all files present in both")
    else:
        if in_dir_not_in_file:
            print(f"\n {len(in_dir_not_in_file)} files in directory but NOT in reference:")
            for f in sorted(in_dir_not_in_file)[:20]:  # Show first 20
                print(f"  {f}")
            if len(in_dir_not_in_file) > 20:
                print(f"  ... and {len(in_dir_not_in_file) - 20} more")
        
        if in_file_not_in_dir:
            print(f"\n {len(in_file_not_in_dir)} files in reference but NOT in directory:")
            for f in sorted(in_file_not_in_dir)[:20]:  # Show first 20
                print(f"  {f}")
            if len(in_file_not_in_dir) > 20:
                print(f"  ... and {len(in_file_not_in_dir) - 20} more")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['dump', 'compare'])
    parser.add_argument('directory', help='Directory to scan')
    parser.add_argument('file', help='Reference file to dump to or compare against')
    args = parser.parse_args()
    
    if args.mode == 'dump':
        dump_filenames(args.directory, args.file)
    else:
        compare_filenames(args.directory, args.file)