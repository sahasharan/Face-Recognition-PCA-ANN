"""
predict.py
==========
Command-line inference script.

Usage
-----
    python predict.py path/to/image.jpg

Examples
--------
    python predict.py dataset/faces/Alia/face_1.jpg
    → Predicted Person: Alia
    → Confidence: 97.2%

    python predict.py unknown_person.jpg
    → Unknown Person
    → Confidence: 25.1%

Notes
-----
* Requires trained models in ``models/`` directory.
* Run ``python train.py`` first if models do not exist.
* The unknown-person threshold is set via ``config.UNKNOWN_THRESHOLD``.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.predictor import Predictor
from src.utils import get_logger

logger = get_logger(__name__)


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path>")
        sys.exit(1)

    image_path = Path(sys.argv[1])

    if not image_path.exists():
        print(f"Error: File not found — {image_path}")
        sys.exit(1)

    try:
        predictor = Predictor()
        result = predictor.predict(image_path)
        output = Predictor.format_result(result)
        print("\n" + "─" * 40)
        print(output)
        print("─" * 40 + "\n")
    except FileNotFoundError as e:
        print(
            f"\nModel files not found.\n"
            f"Please run  'python train.py'  before inference.\n"
            f"Details: {e}"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
