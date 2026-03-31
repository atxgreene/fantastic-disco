"""Extended memory systems — scoped hierarchy and compression-aware SDI."""

from .scoped import ScopedMemoryHierarchy, MemoryScope
from .compressed_sdi import CompressedSDI

__all__ = ["ScopedMemoryHierarchy", "MemoryScope", "CompressedSDI"]
