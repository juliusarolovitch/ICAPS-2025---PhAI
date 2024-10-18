import os
import sys

def get_file_size(file_path):
    """
    Get the size of a file in bytes.
    """
    try:
        return os.path.getsize(file_path)
    except FileNotFoundError:
        return None

def convert_size(size_bytes):
    """
    Convert size in bytes to a human-readable format.
    """
    if size_bytes is None:
        return "N/A"
    
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script_name.py <file_path>")
        sys.exit(1)

    file_path = sys.argv[1]
    size_bytes = get_file_size(file_path)

    if size_bytes is None:
        print(f"File not found: {file_path}")
    else:
        print(f"File: {file_path}")
        print(f"Size: {convert_size(size_bytes)} ({size_bytes} bytes)")