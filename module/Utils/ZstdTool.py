try:
    import compression.zstd as _zstd

    def _compress(data: bytes, level: int) -> bytes:
        return _zstd.compress(data, level=level)

    def _decompress(data: bytes) -> bytes:
        return _zstd.decompress(data)
except ModuleNotFoundError:
    import zstandard as _zstd_compat  # type: ignore[import-untyped]

    def _compress(data: bytes, level: int) -> bytes:
        return _zstd_compat.ZstdCompressor(level=level).compress(data)

    def _decompress(data: bytes) -> bytes:
        return _zstd_compat.ZstdDecompressor().decompress(data)


class ZstdTool:
    """Zstd 压缩/解压工具类。

    优先使用 Python 3.14+ 标准库 compression.zstd，
    回退到第三方库 zstandard（兼容 Python 3.12/3.13）。
    """

    # 压缩级别（1-22，默认 3 是速度与压缩率的平衡点）
    COMPRESSION_LEVEL = 3

    @classmethod
    def compress(cls, data: bytes) -> bytes:
        """压缩数据"""
        return _compress(data, cls.COMPRESSION_LEVEL)

    @classmethod
    def decompress(cls, data: bytes) -> bytes:
        """解压数据"""
        return _decompress(data)
