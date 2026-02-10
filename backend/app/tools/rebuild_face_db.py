from __future__ import annotations

from ..state import vision_service


def main() -> None:
    result = vision_service.rebuild_face_db()
    print("[INFO] Face DB rebuild status:")
    print(result)


if __name__ == "__main__":
    main()
