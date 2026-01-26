from __future__ import annotations

import sys

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.audit import verify_chain
from app.config import get_settings


def main() -> int:
    settings = get_settings()
    engine = create_engine(settings.db_url, echo=False, future=True)
    with Session(engine) as session:
        ok, bad_hash = verify_chain(session)
        if ok:
            print("Audit chain OK")
            return 0
        print(f"Audit chain broken at hash: {bad_hash}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
