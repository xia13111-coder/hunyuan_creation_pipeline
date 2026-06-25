class AssetRefinerError(RuntimeError):
    """Base class for asset-refiner errors."""


class BackendExecutionError(AssetRefinerError):
    """Raised when the geometry backend fails."""


class HunyuanApiError(AssetRefinerError):
    """Raised when the Tencent Hunyuan3D API backend fails."""
