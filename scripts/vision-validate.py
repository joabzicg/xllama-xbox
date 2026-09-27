#!/usr/bin/env python3
"""Xbox vision matrix client. See --help; images and manual quality review required."""
from lan_validate import main

if __name__ == "__main__":
    main(["vision", *__import__("sys").argv[1:]])
