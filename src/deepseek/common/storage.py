"""
Cloud Storage Abstraction Layer
================================

Provides unified interface for cloud storage backends (S3, GCS, local).
Used for checkpoint storage, data loading, and model export.

Usage:
    # S3
    storage = get_storage("s3://my-bucket/checkpoints")
    storage.upload("./model.pt", "model.pt")
    
    # GCS
    storage = get_storage("gs://my-bucket/checkpoints")
    storage.download("model.pt", "./model.pt")
    
    # Local (fallback)
    storage = get_storage("file:///path/to/checkpoints")
    storage.list_files("models/")
"""

from __future__ import annotations

import logging
import os
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import BinaryIO
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


class StorageBackend(ABC):
    """Abstract base class for storage backends."""
    
    @abstractmethod
    def upload(self, local_path: str | Path, remote_path: str) -> None:
        """Upload a file to remote storage."""
        ...
    
    @abstractmethod
    def download(self, remote_path: str, local_path: str | Path) -> None:
        """Download a file from remote storage."""
        ...
    
    @abstractmethod
    def exists(self, remote_path: str) -> bool:
        """Check if a file exists in remote storage."""
        ...
    
    @abstractmethod
    def list_files(self, prefix: str = "") -> list[str]:
        """List files with given prefix."""
        ...
    
    @abstractmethod
    def delete(self, remote_path: str) -> None:
        """Delete a file from remote storage."""
        ...
    
    def upload_stream(self, stream: BinaryIO, remote_path: str) -> None:
        """Upload from a file-like object."""
        raise NotImplementedError("Stream upload not supported by this backend")
    
    def download_stream(self, remote_path: str) -> BinaryIO:
        """Download to a file-like object."""
        raise NotImplementedError("Stream download not supported by this backend")


class S3Storage(StorageBackend):
    """
    Amazon S3 storage backend.
    
    Requires boto3: pip install boto3
    
    Environment variables:
        AWS_ACCESS_KEY_ID
        AWS_SECRET_ACCESS_KEY
        AWS_DEFAULT_REGION
    """
    
    def __init__(self, bucket: str, prefix: str = ""):
        try:
            import boto3
            from botocore.exceptions import ClientError
            self.ClientError = ClientError
        except ImportError as e:
            raise ImportError(
                "boto3 is required for S3 storage. "
                "Install with: pip install boto3"
            ) from e
        
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self.client = boto3.client("s3")
        self.resource = boto3.resource("s3")
        logger.info(f"S3Storage initialized: bucket={bucket}, prefix={prefix}")
    
    def _full_path(self, path: str) -> str:
        """Get full S3 key with prefix."""
        path = path.strip("/")
        if self.prefix:
            return f"{self.prefix}/{path}"
        return path
    
    def upload(self, local_path: str | Path, remote_path: str) -> None:
        """Upload a file to S3."""
        local_path = Path(local_path)
        key = self._full_path(remote_path)
        
        self.client.upload_file(str(local_path), self.bucket, key)
        logger.info(f"Uploaded {local_path} to s3://{self.bucket}/{key}")
    
    def download(self, remote_path: str, local_path: str | Path) -> None:
        """Download a file from S3."""
        local_path = Path(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        key = self._full_path(remote_path)
        
        self.client.download_file(self.bucket, key, str(local_path))
        logger.info(f"Downloaded s3://{self.bucket}/{key} to {local_path}")
    
    def exists(self, remote_path: str) -> bool:
        """Check if a file exists in S3."""
        key = self._full_path(remote_path)
        try:
            self.client.head_object(Bucket=self.bucket, Key=key)
            return True
        except self.ClientError:
            return False
    
    def list_files(self, prefix: str = "") -> list[str]:
        """List files with given prefix."""
        full_prefix = self._full_path(prefix)
        
        paginator = self.client.get_paginator("list_objects_v2")
        files = []
        
        for page in paginator.paginate(Bucket=self.bucket, Prefix=full_prefix):
            for obj in page.get("Contents", []):
                # Remove the base prefix to get relative path
                key = obj["Key"]
                if self.prefix and key.startswith(self.prefix + "/"):
                    key = key[len(self.prefix) + 1:]
                files.append(key)
        
        return files
    
    def delete(self, remote_path: str) -> None:
        """Delete a file from S3."""
        key = self._full_path(remote_path)
        self.client.delete_object(Bucket=self.bucket, Key=key)
        logger.info(f"Deleted s3://{self.bucket}/{key}")
    
    def upload_stream(self, stream: BinaryIO, remote_path: str) -> None:
        """Upload from a file-like object."""
        key = self._full_path(remote_path)
        self.client.upload_fileobj(stream, self.bucket, key)
    
    def download_stream(self, remote_path: str) -> BinaryIO:
        """Download to a file-like object."""
        import io
        key = self._full_path(remote_path)
        buffer = io.BytesIO()
        self.client.download_fileobj(self.bucket, key, buffer)
        buffer.seek(0)
        return buffer


class GCSStorage(StorageBackend):
    """
    Google Cloud Storage backend.
    
    Requires google-cloud-storage: pip install google-cloud-storage
    
    Environment variables:
        GOOGLE_APPLICATION_CREDENTIALS (path to service account JSON)
    """
    
    def __init__(self, bucket: str, prefix: str = ""):
        try:
            from google.cloud import storage
            from google.cloud.exceptions import NotFound
            self.NotFound = NotFound
        except ImportError as e:
            raise ImportError(
                "google-cloud-storage is required for GCS storage. "
                "Install with: pip install google-cloud-storage"
            ) from e
        
        self.bucket_name = bucket
        self.prefix = prefix.strip("/")
        self.client = storage.Client()
        self.bucket = self.client.bucket(bucket)
        logger.info(f"GCSStorage initialized: bucket={bucket}, prefix={prefix}")
    
    def _full_path(self, path: str) -> str:
        """Get full GCS blob name with prefix."""
        path = path.strip("/")
        if self.prefix:
            return f"{self.prefix}/{path}"
        return path
    
    def upload(self, local_path: str | Path, remote_path: str) -> None:
        """Upload a file to GCS."""
        local_path = Path(local_path)
        blob_name = self._full_path(remote_path)
        
        blob = self.bucket.blob(blob_name)
        blob.upload_from_filename(str(local_path))
        logger.info(f"Uploaded {local_path} to gs://{self.bucket_name}/{blob_name}")
    
    def download(self, remote_path: str, local_path: str | Path) -> None:
        """Download a file from GCS."""
        local_path = Path(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        blob_name = self._full_path(remote_path)
        
        blob = self.bucket.blob(blob_name)
        blob.download_to_filename(str(local_path))
        logger.info(f"Downloaded gs://{self.bucket_name}/{blob_name} to {local_path}")
    
    def exists(self, remote_path: str) -> bool:
        """Check if a file exists in GCS."""
        blob_name = self._full_path(remote_path)
        blob = self.bucket.blob(blob_name)
        return blob.exists()
    
    def list_files(self, prefix: str = "") -> list[str]:
        """List files with given prefix."""
        full_prefix = self._full_path(prefix)
        
        blobs = self.client.list_blobs(self.bucket_name, prefix=full_prefix)
        files = []
        
        for blob in blobs:
            # Remove the base prefix to get relative path
            name = blob.name
            if self.prefix and name.startswith(self.prefix + "/"):
                name = name[len(self.prefix) + 1:]
            files.append(name)
        
        return files
    
    def delete(self, remote_path: str) -> None:
        """Delete a file from GCS."""
        blob_name = self._full_path(remote_path)
        blob = self.bucket.blob(blob_name)
        blob.delete()
        logger.info(f"Deleted gs://{self.bucket_name}/{blob_name}")
    
    def upload_stream(self, stream: BinaryIO, remote_path: str) -> None:
        """Upload from a file-like object."""
        blob_name = self._full_path(remote_path)
        blob = self.bucket.blob(blob_name)
        blob.upload_from_file(stream)
    
    def download_stream(self, remote_path: str) -> BinaryIO:
        """Download to a file-like object."""
        import io
        blob_name = self._full_path(remote_path)
        blob = self.bucket.blob(blob_name)
        buffer = io.BytesIO()
        blob.download_to_file(buffer)
        buffer.seek(0)
        return buffer


class LocalStorage(StorageBackend):
    """
    Local filesystem storage backend.
    
    Useful for development and testing.
    """
    
    def __init__(self, base_path: str | Path):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"LocalStorage initialized: path={self.base_path}")
    
    def _full_path(self, path: str) -> Path:
        """Get full local path."""
        return self.base_path / path.strip("/")
    
    def upload(self, local_path: str | Path, remote_path: str) -> None:
        """Copy a file to storage location."""
        local_path = Path(local_path)
        dest_path = self._full_path(remote_path)
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        
        shutil.copy2(local_path, dest_path)
        logger.info(f"Copied {local_path} to {dest_path}")
    
    def download(self, remote_path: str, local_path: str | Path) -> None:
        """Copy a file from storage location."""
        local_path = Path(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        src_path = self._full_path(remote_path)
        
        shutil.copy2(src_path, local_path)
        logger.info(f"Copied {src_path} to {local_path}")
    
    def exists(self, remote_path: str) -> bool:
        """Check if a file exists."""
        return self._full_path(remote_path).exists()
    
    def list_files(self, prefix: str = "") -> list[str]:
        """List files with given prefix."""
        search_path = self._full_path(prefix)
        
        if search_path.is_file():
            return [prefix]
        
        if not search_path.exists():
            return []
        
        files = []
        for path in search_path.rglob("*"):
            if path.is_file():
                rel_path = path.relative_to(self.base_path)
                files.append(str(rel_path))
        
        return files
    
    def delete(self, remote_path: str) -> None:
        """Delete a file."""
        path = self._full_path(remote_path)
        if path.is_file():
            path.unlink()
            logger.info(f"Deleted {path}")
        elif path.is_dir():
            shutil.rmtree(path)
            logger.info(f"Deleted directory {path}")


def get_storage(uri: str) -> StorageBackend:
    """
    Factory function to create appropriate storage backend from URI.
    
    Supported URI formats:
        - s3://bucket-name/path/prefix
        - gs://bucket-name/path/prefix
        - file:///absolute/path
        - /absolute/path (treated as local)
        - ./relative/path (treated as local)
    
    Args:
        uri: Storage URI
        
    Returns:
        StorageBackend instance
        
    Raises:
        ValueError: If URI scheme is not supported
    """
    parsed = urlparse(uri)
    scheme = parsed.scheme.lower()
    
    if scheme == "s3":
        bucket = parsed.netloc
        prefix = parsed.path.strip("/")
        return S3Storage(bucket, prefix)
    
    elif scheme == "gs":
        bucket = parsed.netloc
        prefix = parsed.path.strip("/")
        return GCSStorage(bucket, prefix)
    
    elif scheme == "file" or scheme == "":
        # Local filesystem
        if scheme == "file":
            path = parsed.path
        else:
            path = uri
        return LocalStorage(path)
    
    else:
        raise ValueError(
            f"Unsupported storage scheme: {scheme}. "
            f"Supported: s3://, gs://, file://, or local path"
        )


class CheckpointManager:
    """
    High-level checkpoint management with cloud storage.
    
    Provides atomic checkpoint saves with metadata tracking.
    """
    
    def __init__(self, storage: StorageBackend, checkpoint_prefix: str = "checkpoints"):
        self.storage = storage
        self.checkpoint_prefix = checkpoint_prefix
    
    def save_checkpoint(
        self,
        local_path: str | Path,
        step: int,
        metadata: dict | None = None,
    ) -> str:
        """
        Save a checkpoint with step number.
        
        Args:
            local_path: Path to local checkpoint file/directory
            step: Training step number
            metadata: Optional metadata to save alongside checkpoint
            
        Returns:
            Remote path where checkpoint was saved
        """
        import json
        
        local_path = Path(local_path)
        remote_dir = f"{self.checkpoint_prefix}/step_{step}"
        
        if local_path.is_file():
            # Single file checkpoint
            remote_path = f"{remote_dir}/{local_path.name}"
            self.storage.upload(local_path, remote_path)
        else:
            # Directory checkpoint
            for file_path in local_path.rglob("*"):
                if file_path.is_file():
                    rel_path = file_path.relative_to(local_path)
                    remote_path = f"{remote_dir}/{rel_path}"
                    self.storage.upload(file_path, remote_path)
        
        # Save metadata
        if metadata:
            import tempfile
            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
                json.dump(metadata, f)
                temp_path = f.name
            
            self.storage.upload(temp_path, f"{remote_dir}/metadata.json")
            os.unlink(temp_path)
        
        logger.info(f"Saved checkpoint to {remote_dir}")
        return remote_dir
    
    def load_checkpoint(
        self,
        step: int,
        local_dir: str | Path,
    ) -> dict | None:
        """
        Load a checkpoint by step number.
        
        Args:
            step: Training step number
            local_dir: Local directory to download checkpoint to
            
        Returns:
            Metadata dict if available, else None
        """
        import json
        
        local_dir = Path(local_dir)
        local_dir.mkdir(parents=True, exist_ok=True)
        remote_dir = f"{self.checkpoint_prefix}/step_{step}"
        
        # List all files in checkpoint
        files = self.storage.list_files(remote_dir)
        
        if not files:
            raise FileNotFoundError(f"Checkpoint not found: step_{step}")
        
        # Download all files
        metadata = None
        for file_path in files:
            if not file_path.startswith(remote_dir):
                continue
            
            rel_path = file_path[len(remote_dir):].strip("/")
            local_path = local_dir / rel_path
            self.storage.download(file_path, local_path)
            
            # Load metadata if present
            if rel_path == "metadata.json":
                with open(local_path) as f:
                    metadata = json.load(f)
        
        logger.info(f"Loaded checkpoint from {remote_dir}")
        return metadata
    
    def list_checkpoints(self) -> list[int]:
        """List all available checkpoint steps."""
        files = self.storage.list_files(self.checkpoint_prefix)
        
        steps = set()
        for f in files:
            parts = f.split("/")
            for part in parts:
                if part.startswith("step_"):
                    try:
                        step = int(part[5:])
                        steps.add(step)
                    except ValueError:
                        pass
        
        return sorted(steps)
    
    def get_latest_checkpoint(self) -> int | None:
        """Get the latest checkpoint step."""
        steps = self.list_checkpoints()
        return steps[-1] if steps else None
