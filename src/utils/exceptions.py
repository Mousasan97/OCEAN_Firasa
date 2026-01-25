"""
Custom exceptions for the application
"""


class OCEANException(Exception):
    """Base exception for OCEAN application"""

    def __init__(self, message: str, status_code: int = 500, details: dict = None):
        self.message = message
        self.status_code = status_code
        self.details = details or {}
        super().__init__(self.message)


class ModelNotLoadedException(OCEANException):
    """Raised when model is not loaded"""

    def __init__(self, message: str = "Model not loaded"):
        super().__init__(message, status_code=503)


class ModelLoadError(OCEANException):
    """Raised when model loading fails"""

    def __init__(self, message: str, details: dict = None):
        super().__init__(message, status_code=500, details=details)


class InvalidImageError(OCEANException):
    """Raised when image is invalid or corrupted"""

    def __init__(self, message: str = "Invalid or corrupted image"):
        super().__init__(message, status_code=400)


class InvalidVideoError(OCEANException):
    """Raised when video is invalid or corrupted"""

    def __init__(self, message: str = "Invalid or corrupted video"):
        super().__init__(message, status_code=400)


class FaceDetectionError(OCEANException):
    """Raised when face detection fails"""

    def __init__(self, message: str = "No face detected in image"):
        super().__init__(message, status_code=422)


class FileUploadError(OCEANException):
    """Raised when file upload fails"""

    def __init__(self, message: str, details: dict = None):
        super().__init__(message, status_code=400, details=details)


class FileSizeExceededError(OCEANException):
    """Raised when uploaded file exceeds size limit"""

    def __init__(self, max_size: int):
        message = f"File size exceeds maximum allowed size of {max_size / (1024*1024):.1f}MB"
        super().__init__(message, status_code=413)


class UnsupportedFileTypeError(OCEANException):
    """Raised when file type is not supported"""

    def __init__(self, extension: str, allowed: list, message: str = None):
        if message is None:
            message = f"File type '{extension}' not supported. Allowed: {', '.join(allowed)}"
        super().__init__(message, status_code=415, details={"allowed": allowed})


class PreprocessingError(OCEANException):
    """Raised when preprocessing fails"""

    def __init__(self, message: str, details: dict = None):
        super().__init__(message, status_code=422, details=details)


class PredictionError(OCEANException):
    """Raised when prediction fails"""

    def __init__(self, message: str, details: dict = None):
        super().__init__(message, status_code=500, details=details)


class CacheError(OCEANException):
    """Raised when cache operation fails"""

    def __init__(self, message: str):
        super().__init__(message, status_code=500)


class StorageError(OCEANException):
    """Raised when storage operation fails"""

    def __init__(self, message: str, details: dict = None):
        super().__init__(message, status_code=500, details=details)


class RateLimitExceededError(OCEANException):
    """Raised when rate limit is exceeded"""

    def __init__(self, retry_after: int = None):
        message = "Rate limit exceeded"
        if retry_after:
            message += f". Retry after {retry_after} seconds"
        super().__init__(message, status_code=429, details={"retry_after": retry_after})


class AuthenticationError(OCEANException):
    """Raised when authentication fails"""

    def __init__(self, message: str = "Authentication failed"):
        super().__init__(message, status_code=401)


class AuthorizationError(OCEANException):
    """Raised when authorization fails"""

    def __init__(self, message: str = "Insufficient permissions"):
        super().__init__(message, status_code=403)
