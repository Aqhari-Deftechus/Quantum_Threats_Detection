from __future__ import annotations

import re


RTSP_CREDENTIALS_PATTERN = re.compile(r"(rtsp://)([^/@]+:[^/@]+)@")


def redact_rtsp(source: str) -> str:
    return RTSP_CREDENTIALS_PATTERN.sub(r"\1***:***@", source)
