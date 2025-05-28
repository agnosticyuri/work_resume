# -*- coding: utf-8 -*-
"""

@author: jske
"""


def check_ply_format(file_path):
    with open(file_path, "rb") as f:
        header = f.read(100).decode(errors="ignore")  # Read the first 100 bytes
        if "ascii" in header.lower():
            return "ASCII PLY FORMAT"
        elif "binary" in header.lower():
            return "Binary PLY FORMAT"
        else:
            return "Unknown format"

file_path = "your_file.ply"  # Replace with your PLY file path
print(check_ply_format(file_path))
