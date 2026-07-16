"""Application entry point."""
import os
import sys

SERVER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "server")
if SERVER_DIR not in sys.path:
    sys.path.insert(0, SERVER_DIR)

from app import main

if __name__ == "__main__":
    main()
