"""
Data Caching Utilities with Parquet Support
Provides high-performance caching for data and preprocessing results

Author: Mohamed salem eddah
"""

import hashlib
import json
import pickle
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Optional, Union

import joblib
import pandas as pd

from .logging import get_logger

logger = get_logger(__name__)


class DataCache:
    """
    Intelligent data caching system with multiple backend support.

    Supports:
    - Parquet files (10x faster than CSV for large datasets)
    - Pickle files (for arbitrary Python objects)
    - Joblib compression (for numpy/pandas objects)
    """

    def __init__(self, cache_dir: Union[str, Path], enabled: bool = True):
        """
        Initialize data cache.

        Args:
            cache_dir: Directory to store cache files
            enabled: Whether caching is enabled
        """
        self.cache_dir = Path(cache_dir)
        self.enabled = enabled

        if self.enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Cache initialized at {self.cache_dir}")

    def _get_cache_key(self, key: str, params: dict = None) -> str:
        """
        Generate cache key from identifier and parameters.

        Args:
            key: Base cache key
            params: Additional parameters to include in key

        Returns:
            Hashed cache key
        """
        if params:
            # Create deterministic hash from parameters
            param_str = json.dumps(params, sort_keys=True)
            hash_suffix = hashlib.md5(param_str.encode()).hexdigest()[:8]
            return f"{key}_{hash_suffix}"
        return key

    def get_dataframe(
        self, key: str, params: dict = None, format: str = "parquet"
    ) -> Optional[pd.DataFrame]:
        """
        Get cached DataFrame.

        Args:
            key: Cache key
            params: Parameters to include in cache key
            format: File format ('parquet' or 'pickle')

        Returns:
            Cached DataFrame or None if not found
        """
        if not self.enabled:
            return None

        cache_key = self._get_cache_key(key, params)

        if format == "parquet":
            cache_file = self.cache_dir / f"{cache_key}.parquet"
            if cache_file.exists():
                logger.debug(f"Cache hit: {cache_key}")
                return pd.read_parquet(cache_file)
        elif format == "pickle":
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            if cache_file.exists():
                logger.debug(f"Cache hit: {cache_key}")
                return pd.read_pickle(cache_file)

        logger.debug(f"Cache miss: {cache_key}")
        return None

    def save_dataframe(
        self, df: pd.DataFrame, key: str, params: dict = None, format: str = "parquet"
    ):
        """
        Save DataFrame to cache.

        Args:
            df: DataFrame to cache
            key: Cache key
            params: Parameters to include in cache key
            format: File format ('parquet' or 'pickle')
        """
        if not self.enabled:
            return

        cache_key = self._get_cache_key(key, params)

        if format == "parquet":
            cache_file = self.cache_dir / f"{cache_key}.parquet"
            df.to_parquet(cache_file, compression="snappy", index=False)
            logger.debug(f"Saved to cache (parquet): {cache_key}")
        elif format == "pickle":
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            df.to_pickle(cache_file, compression="gzip")
            logger.debug(f"Saved to cache (pickle): {cache_key}")

    def get_object(self, key: str, params: dict = None) -> Optional[Any]:
        """
        Get cached Python object.

        Args:
            key: Cache key
            params: Parameters to include in cache key

        Returns:
            Cached object or None if not found
        """
        if not self.enabled:
            return None

        cache_key = self._get_cache_key(key, params)
        cache_file = self.cache_dir / f"{cache_key}.joblib"

        if cache_file.exists():
            logger.debug(f"Cache hit: {cache_key}")
            return joblib.load(cache_file)

        logger.debug(f"Cache miss: {cache_key}")
        return None

    def save_object(self, obj: Any, key: str, params: dict = None, compress: int = 3):
        """
        Save Python object to cache.

        Args:
            obj: Object to cache
            key: Cache key
            params: Parameters to include in cache key
            compress: Compression level (0-9, 0=no compression)
        """
        if not self.enabled:
            return

        cache_key = self._get_cache_key(key, params)
        cache_file = self.cache_dir / f"{cache_key}.joblib"

        joblib.dump(obj, cache_file, compress=compress)
        logger.debug(f"Saved to cache (joblib): {cache_key}")

    def clear(self, pattern: str = "*"):
        """
        Clear cache files matching pattern.

        Args:
            pattern: Glob pattern for files to delete
        """
        if not self.enabled:
            return

        deleted_count = 0
        for cache_file in self.cache_dir.glob(pattern):
            cache_file.unlink()
            deleted_count += 1

        logger.info(f"Cleared {deleted_count} cache files")

    def get_cache_info(self) -> dict:
        """
        Get information about cache usage.

        Returns:
            Dictionary with cache statistics
        """
        if not self.enabled:
            return {"enabled": False}

        total_size = sum(f.stat().st_size for f in self.cache_dir.rglob("*") if f.is_file())
        file_count = len(list(self.cache_dir.rglob("*")))

        return {
            "enabled": True,
            "cache_dir": str(self.cache_dir),
            "total_size_mb": total_size / (1024 * 1024),
            "file_count": file_count,
        }


def cached(
    cache_dir: Union[str, Path],
    key: str,
    params_from_args: Optional[list] = None,
    format: str = "parquet",
):
    """
    Decorator for caching function results.

    Args:
        cache_dir: Directory to store cache files
        key: Cache key (can include {arg_name} placeholders)
        params_from_args: List of argument names to include in cache key
        format: Cache format ('parquet', 'pickle', or 'object')

    Example:
        @cached(cache_dir='data/cache', key='processed_{dataset_name}',
                params_from_args=['dataset_name'])
        def process_data(dataset_name):
            # Expensive processing...
            return df
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            cache = DataCache(cache_dir)

            # Build cache parameters from function arguments
            params = {}
            if params_from_args:
                import inspect

                sig = inspect.signature(func)
                bound_args = sig.bind(*args, **kwargs)
                bound_args.apply_defaults()

                for param_name in params_from_args:
                    if param_name in bound_args.arguments:
                        params[param_name] = bound_args.arguments[param_name]

            # Try to load from cache
            if format in ["parquet", "pickle"]:
                cached_result = cache.get_dataframe(key, params, format=format)
            else:
                cached_result = cache.get_object(key, params)

            if cached_result is not None:
                logger.info(f"Loaded {key} from cache")
                return cached_result

            # Execute function
            result = func(*args, **kwargs)

            # Save to cache
            if format in ["parquet", "pickle"]:
                cache.save_dataframe(result, key, params, format=format)
            else:
                cache.save_object(result, key, params)

            return result

        return wrapper

    return decorator


def convert_csv_to_parquet(
    csv_path: Path,
    parquet_path: Optional[Path] = None,
    compression: str = "snappy",
    **read_csv_kwargs,
) -> Path:
    """
    Convert CSV file to Parquet format for faster loading.

    Args:
        csv_path: Path to CSV file
        parquet_path: Output path for Parquet file (default: same dir, .parquet extension)
        compression: Compression algorithm ('snappy', 'gzip', 'brotli')
        **read_csv_kwargs: Additional arguments for pd.read_csv

    Returns:
        Path to created Parquet file
    """
    if parquet_path is None:
        parquet_path = csv_path.with_suffix(".parquet")

    logger.info(f"Converting {csv_path} to Parquet format...")

    # Read CSV
    df = pd.read_csv(csv_path, **read_csv_kwargs)

    # Save as Parquet
    df.to_parquet(parquet_path, compression=compression, index=False)

    # Compare sizes
    csv_size = csv_path.stat().st_size / (1024 * 1024)
    parquet_size = parquet_path.stat().st_size / (1024 * 1024)

    logger.info(
        f"Conversion complete: {csv_size:.1f}MB (CSV) -> {parquet_size:.1f}MB (Parquet) "
        f"({100 * parquet_size / csv_size:.1f}% of original)"
    )

    return parquet_path
