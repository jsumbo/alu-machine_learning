#!/bin/bash

# Make all files in the current directory and its subdirectories executable
find . -type f -exec chmod +x {} \;

echo "All files have been made executable."