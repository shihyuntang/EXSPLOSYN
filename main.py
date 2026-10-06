"""Launch EXSPLOSYN from a source checkout."""

from pathlib import Path
import sys

# Resolve the source directory independently of the working directory.
sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))

from exsplosyn.cli import main


if __name__ == '__main__':
    main()
