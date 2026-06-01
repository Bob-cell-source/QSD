import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qsdrec.train import main


if __name__ == "__main__":
    if "--model-variant" not in sys.argv:
        sys.argv.extend(["--model-variant", "crsid_semhub"])
    main()
