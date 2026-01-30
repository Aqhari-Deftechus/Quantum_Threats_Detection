from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.app.config import get_settings
from backend.app.vision.detector_factory import create_detector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SCRFD detection on a single image.")
    parser.add_argument("--image", required=True, help="Path to an image file.")
    parser.add_argument(
        "--fail-on-zero",
        action="store_true",
        help="Exit with non-zero status if no faces are detected.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Image not found: {image_path}")
        return 2
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Failed to read image: {image_path}")
        return 2

    settings = get_settings()
    detector = create_detector(settings)
    faces = detector.detect(image)
    print(f"Faces detected: {len(faces)}")
    if faces:
        face = max(faces, key=lambda f: f.score)
        print(f"Top face bbox: {[face.x1, face.y1, face.x2, face.y2]} score={face.score:.4f}")
    if args.fail_on_zero and not faces:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
