"""Start the local dubbing engine without installing the package first."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from dubtool.web_service import main  # noqa: E402


if __name__ == "__main__":
    main()
