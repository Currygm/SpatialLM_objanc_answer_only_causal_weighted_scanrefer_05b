"""Portable preprocessing command."""
import sys
from data_preparation import main
if __name__ == "__main__": main(['normals'] + sys.argv[1:])
