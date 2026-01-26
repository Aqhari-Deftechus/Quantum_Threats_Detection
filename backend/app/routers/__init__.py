from .cameras import router as cameras_router
from .diagnostics import router as diagnostics_router
from .health import router as health_router
from .identities import router as identities_router
from .streaming import router as streaming_router

__all__ = ["cameras_router", "diagnostics_router", "health_router", "identities_router", "streaming_router"]
