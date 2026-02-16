"""
Fast Hashing Utilities for Cache Keys
======================================

Performance hierarchy (for cache key generation):
1. xxhash (10-20x faster than SHA256) - preferred
2. MD5 (3x faster than SHA256) - fallback
3. SHA256 - only for cryptographic security needs

NOTE: Cache keys do NOT require cryptographic security.
We only need fast, low-collision hashes for key generation.
"""

import hashlib

# Try to import xxhash for 10-20x faster hashing
try:
    import xxhash

    _HAS_XXHASH = True
except ImportError:
    _HAS_XXHASH = False

def fast_hash(content: str | bytes, length: int = 32) -> str:
    """
    Generate a fast hash for cache keys.

    Uses xxhash if available (10-20x faster than SHA256),
    falls back to MD5 (3x faster than SHA256).

    NOT for cryptographic purposes - use secure_hash() for that.

    Args:
        content: String or bytes to hash
        length: Length of output hash (default 32, max 32 for xxhash, 64 for MD5)

    Returns:
        Hexadecimal hash string truncated to specified length

    Performance:
        - xxhash: ~10GB/s on M4
        - MD5: ~500MB/s on M4
        - SHA256: ~200MB/s on M4
    """
    if isinstance(content, str):
        content = content.encode("utf-8")

    if _HAS_XXHASH:
        # xxhash64 is optimal for cache keys - fast and good distribution
        return xxhash.xxh64(content).hexdigest()[:length]
    else:
        # MD5 fallback - still 3x faster than SHA256 (not for security)
        return hashlib.md5(content, usedforsecurity=False).hexdigest()[:length]

def fast_hash_int(content: str | bytes) -> int:
    """
    Generate a fast integer hash for bloom filters/hash tables.

    Args:
        content: String or bytes to hash

    Returns:
        64-bit unsigned integer hash
    """
    if isinstance(content, str):
        content = content.encode("utf-8")

    if _HAS_XXHASH:
        return xxhash.xxh64_intdigest(content)
    else:
        # Use first 16 hex chars of MD5 as 64-bit int (not for security)
        return int(hashlib.md5(content, usedforsecurity=False).hexdigest()[:16], 16)

def secure_hash(content: str | bytes, length: int | None = None) -> str:
    """
    Generate a cryptographically secure hash.

    Use this for:
    - Password hashing (though prefer bcrypt/argon2 for that)
    - Content integrity verification
    - Security tokens/fingerprints

    Args:
        content: String or bytes to hash
        length: Optional length to truncate to (default: full 64-char hash)

    Returns:
        SHA256 hexadecimal hash string
    """
    if isinstance(content, str):
        content = content.encode("utf-8")

    digest = hashlib.sha256(content).hexdigest()
    return digest[:length] if length else digest

def cache_key(*args, length: int = 32) -> str:
    """
    Generate a cache key from multiple arguments.

    Example:
        key = cache_key("model_id", text, "v1")

    Args:
        *args: Arguments to include in cache key
        length: Length of output hash

    Returns:
        Fast hash of concatenated arguments
    """
    key_parts = []
    for arg in args:
        if arg is None:
            key_parts.append("None")
        elif isinstance(arg, (list, tuple)):
            key_parts.append(str(sorted(arg) if isinstance(arg, (list, set)) else arg))
        elif isinstance(arg, dict):
            key_parts.append(str(sorted(arg.items())))
        else:
            key_parts.append(str(arg))

    return fast_hash(":".join(key_parts), length)

# Module-level check for xxhash availability
def has_xxhash() -> bool:
    """Check if xxhash is available for optimal performance."""
    return _HAS_XXHASH
