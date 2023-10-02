"""
Zoom all tomogram files in a directory
"""
import os

in_dir = ''
zoom_f = 0.5
out_sufix = '_bin2'


for f in os.listdir(in_dir):
    if os.path.splitext(f) is in valid_exts:
