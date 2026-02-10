from __future__ import annotations

from pathlib import Path

from ..config import get_settings
from ..state import vision_service


def main() -> None:
    settings = get_settings()
    print("[INFO] Active Face Settings")
    print(f"[INFO] QTD_FACE_DATASET_DIR={settings.face_dataset_dir_raw}")
    print(f"[INFO] Resolved dataset dir={settings.face_dataset_dir_resolved}")
    print(f"[INFO] CWD={Path.cwd()}")
    print(f"[INFO] Env file={settings.active_face_env_file}")
    result = vision_service.rebuild_face_db()
    print("[INFO] Face DB rebuild status:")
    print(result)


if __name__ == "__main__":
    main()
