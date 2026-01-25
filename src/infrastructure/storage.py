"""
Storage layer for file uploads and results
Supports local filesystem and S3
"""
import shutil
from pathlib import Path
from typing import BinaryIO, Optional
from abc import ABC, abstractmethod
from datetime import datetime

from src.utils.config import settings
from src.utils.logger import get_logger
from src.utils.exceptions import StorageError

logger = get_logger(__name__)


class StorageBackend(ABC):
    """Abstract storage backend"""

    @abstractmethod
    def save(self, file_path: str, content: bytes) -> str:
        """Save file and return storage path"""
        pass

    @abstractmethod
    def load(self, file_path: str) -> bytes:
        """Load file content"""
        pass

    @abstractmethod
    def delete(self, file_path: str) -> None:
        """Delete file"""
        pass

    @abstractmethod
    def exists(self, file_path: str) -> bool:
        """Check if file exists"""
        pass


class LocalStorage(StorageBackend):
    """Local filesystem storage"""

    def __init__(self, base_dir: Optional[str] = None):
        """
        Initialize local storage

        Args:
            base_dir: Base directory for storage (uses config if None)
        """
        self.base_dir = Path(base_dir or settings.UPLOAD_DIR)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"LocalStorage initialized: {self.base_dir}")

    def _get_full_path(self, file_path: str) -> Path:
        """Get full path from relative path"""
        full_path = self.base_dir / file_path
        # Ensure path is within base_dir (security)
        try:
            full_path.resolve().relative_to(self.base_dir.resolve())
        except ValueError:
            raise StorageError(f"Invalid file path: {file_path}")
        return full_path

    def save(self, file_path: str, content: bytes) -> str:
        """Save file to local storage"""
        try:
            full_path = self._get_full_path(file_path)
            full_path.parent.mkdir(parents=True, exist_ok=True)

            with open(full_path, 'wb') as f:
                f.write(content)

            logger.info(f"Saved file: {full_path}")
            return str(file_path)

        except Exception as e:
            raise StorageError(f"Failed to save file: {str(e)}")

    def load(self, file_path: str) -> bytes:
        """Load file from local storage"""
        try:
            full_path = self._get_full_path(file_path)

            if not full_path.exists():
                raise StorageError(f"File not found: {file_path}")

            with open(full_path, 'rb') as f:
                content = f.read()

            logger.info(f"Loaded file: {full_path}")
            return content

        except StorageError:
            raise
        except Exception as e:
            raise StorageError(f"Failed to load file: {str(e)}")

    def delete(self, file_path: str) -> None:
        """Delete file from local storage"""
        try:
            full_path = self._get_full_path(file_path)

            if full_path.exists():
                full_path.unlink()
                logger.info(f"Deleted file: {full_path}")

        except Exception as e:
            logger.error(f"Failed to delete file: {e}")

    def exists(self, file_path: str) -> bool:
        """Check if file exists in local storage"""
        try:
            full_path = self._get_full_path(file_path)
            return full_path.exists()
        except:
            return False


class S3Storage(StorageBackend):
    """AWS S3 storage backend"""

    def __init__(
        self,
        bucket: Optional[str] = None,
        region: Optional[str] = None,
        access_key: Optional[str] = None,
        secret_key: Optional[str] = None
    ):
        """
        Initialize S3 storage

        Args:
            bucket: S3 bucket name (uses config if None)
            region: AWS region (uses config if None)
            access_key: AWS access key (uses config if None)
            secret_key: AWS secret key (uses config if None)
        """
        try:
            import boto3
        except ImportError:
            raise ImportError("boto3 required for S3Storage. Install with: pip install boto3")

        self.bucket = bucket or settings.S3_BUCKET
        self.region = region or settings.S3_REGION
        access_key = access_key or settings.S3_ACCESS_KEY
        secret_key = secret_key or settings.S3_SECRET_KEY

        if not self.bucket:
            raise StorageError("S3 bucket name required")

        # Initialize S3 client
        try:
            self.client = boto3.client(
                's3',
                region_name=self.region,
                aws_access_key_id=access_key,
                aws_secret_access_key=secret_key
            )
            # Test connection
            self.client.head_bucket(Bucket=self.bucket)
            logger.info(f"S3Storage initialized: {self.bucket}")
        except Exception as e:
            raise StorageError(f"Failed to initialize S3: {str(e)}")

    def save(self, file_path: str, content: bytes) -> str:
        """Save file to S3"""
        try:
            self.client.put_object(
                Bucket=self.bucket,
                Key=file_path,
                Body=content
            )
            logger.info(f"Saved to S3: s3://{self.bucket}/{file_path}")
            return file_path
        except Exception as e:
            raise StorageError(f"Failed to save to S3: {str(e)}")

    def load(self, file_path: str) -> bytes:
        """Load file from S3"""
        try:
            response = self.client.get_object(
                Bucket=self.bucket,
                Key=file_path
            )
            content = response['Body'].read()
            logger.info(f"Loaded from S3: s3://{self.bucket}/{file_path}")
            return content
        except Exception as e:
            raise StorageError(f"Failed to load from S3: {str(e)}")

    def delete(self, file_path: str) -> None:
        """Delete file from S3"""
        try:
            self.client.delete_object(
                Bucket=self.bucket,
                Key=file_path
            )
            logger.info(f"Deleted from S3: s3://{self.bucket}/{file_path}")
        except Exception as e:
            logger.error(f"Failed to delete from S3: {e}")

    def exists(self, file_path: str) -> bool:
        """Check if file exists in S3"""
        try:
            self.client.head_object(Bucket=self.bucket, Key=file_path)
            return True
        except:
            return False


class StorageService:
    """High-level storage service"""

    def __init__(self, backend: Optional[StorageBackend] = None):
        """
        Initialize storage service

        Args:
            backend: Storage backend (auto-selects based on config if None)
        """
        if backend is None:
            if settings.STORAGE_BACKEND == "s3":
                self.backend = S3Storage()
            else:
                self.backend = LocalStorage()
        else:
            self.backend = backend

        logger.info(f"StorageService initialized with backend: {type(self.backend).__name__}")

    def save_upload(
        self,
        filename: str,
        content: bytes,
        subfolder: Optional[str] = None
    ) -> str:
        """
        Save uploaded file with timestamp

        Args:
            filename: Original filename
            content: File content
            subfolder: Optional subfolder

        Returns:
            Storage path
        """
        # Generate unique filename with timestamp
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        name = Path(filename).stem
        ext = Path(filename).suffix

        unique_filename = f"{timestamp}_{name}{ext}"

        if subfolder:
            file_path = f"{subfolder}/{unique_filename}"
        else:
            file_path = unique_filename

        return self.backend.save(file_path, content)

    def save_result(
        self,
        filename: str,
        content: bytes,
        subfolder: str = "results"
    ) -> str:
        """Save prediction result"""
        return self.save_upload(filename, content, subfolder)

    def load_file(self, file_path: str) -> bytes:
        """Load file from storage"""
        return self.backend.load(file_path)

    def delete_file(self, file_path: str) -> None:
        """Delete file from storage"""
        self.backend.delete(file_path)

    def file_exists(self, file_path: str) -> bool:
        """Check if file exists"""
        return self.backend.exists(file_path)


# Global storage service instance
_storage_service: Optional[StorageService] = None


def get_storage_service() -> StorageService:
    """Get storage service instance (singleton)"""
    global _storage_service
    if _storage_service is None:
        _storage_service = StorageService()
    return _storage_service
