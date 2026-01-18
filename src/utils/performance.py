"""
Performance Optimization Utilities
Provides vectorized operations and parallel processing helpers

Author: Mohamed salem eddah
"""

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from functools import partial
from typing import Any, Callable, Iterable, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from .logging import get_logger

logger = get_logger(__name__)


def vectorized_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply vectorized feature engineering operations.
    Much faster than row-wise operations using apply().

    Args:
        df: Input DataFrame

    Returns:
        DataFrame with engineered features
    """
    logger.info("Applying vectorized feature engineering...")

    # Ratio features (vectorized division)
    df["fwd_bwd_packet_ratio"] = np.where(
        df["Total Backward Packets"] > 0,
        df["Total Fwd Packets"] / df["Total Backward Packets"],
        df["Total Fwd Packets"],
    )

    df["fwd_bwd_length_ratio"] = np.where(
        df["Total Length of Bwd Packets"] > 0,
        df["Total Length of Fwd Packets"] / df["Total Length of Bwd Packets"],
        df["Total Length of Fwd Packets"],
    )

    # Log-transformed features (vectorized log)
    df["log_flow_duration"] = np.log1p(df["Flow Duration"].clip(lower=0))
    df["log_flow_bytes_s"] = np.log1p(df["Flow Bytes/s"].clip(lower=0))
    df["log_flow_packets_s"] = np.log1p(df["Flow Packets/s"].clip(lower=0))

    # Statistical features (vectorized operations)
    # Coefficient of variation
    df["fwd_packet_cv"] = np.where(
        df["Fwd Packet Length Mean"] > 0,
        df["Fwd Packet Length Std"] / df["Fwd Packet Length Mean"],
        0,
    )

    df["bwd_packet_cv"] = np.where(
        df["Bwd Packet Length Mean"] > 0,
        df["Bwd Packet Length Std"] / df["Bwd Packet Length Mean"],
        0,
    )

    # Packet size features (vectorized)
    df["total_packets"] = df["Total Fwd Packets"] + df["Total Backward Packets"]
    df["total_bytes"] = df["Total Length of Fwd Packets"] + df["Total Length of Bwd Packets"]
    df["avg_packet_size"] = np.where(
        df["total_packets"] > 0, df["total_bytes"] / df["total_packets"], 0
    )

    # Flow efficiency metrics
    df["bytes_per_packet_fwd"] = np.where(
        df["Total Fwd Packets"] > 0,
        df["Total Length of Fwd Packets"] / df["Total Fwd Packets"],
        0,
    )

    df["bytes_per_packet_bwd"] = np.where(
        df["Total Backward Packets"] > 0,
        df["Total Length of Bwd Packets"] / df["Total Backward Packets"],
        0,
    )

    logger.info("Vectorized feature engineering complete")
    return df


def parallel_apply(
    data: Iterable[Any],
    func: Callable,
    n_workers: int = None,
    use_threads: bool = False,
    description: str = "Processing",
    **func_kwargs,
) -> List[Any]:
    """
    Apply function to iterable in parallel.

    Args:
        data: Iterable to process
        func: Function to apply to each element
        n_workers: Number of parallel workers (default: CPU count)
        use_threads: Use threads instead of processes (for I/O-bound tasks)
        description: Description for progress bar
        **func_kwargs: Additional keyword arguments for func

    Returns:
        List of results
    """
    if n_workers is None:
        n_workers = mp.cpu_count()

    # Create partial function with kwargs
    if func_kwargs:
        func = partial(func, **func_kwargs)

    data_list = list(data)
    results = []

    executor_class = ThreadPoolExecutor if use_threads else ProcessPoolExecutor

    with executor_class(max_workers=n_workers) as executor:
        futures = [executor.submit(func, item) for item in data_list]

        for future in tqdm(
            as_completed(futures), total=len(futures), desc=description, leave=False
        ):
            try:
                results.append(future.result())
            except Exception as e:
                logger.error(f"Error in parallel processing: {e}")
                results.append(None)

    return results


def parallel_dataframe_processing(
    df: pd.DataFrame, func: Callable, n_workers: int = None, chunk_size: int = None
) -> pd.DataFrame:
    """
    Process DataFrame in parallel by splitting into chunks.

    Args:
        df: DataFrame to process
        func: Function that takes a DataFrame chunk and returns processed chunk
        n_workers: Number of parallel workers
        chunk_size: Size of each chunk (default: len(df) / n_workers)

    Returns:
        Processed DataFrame
    """
    if n_workers is None:
        n_workers = mp.cpu_count()

    if chunk_size is None:
        chunk_size = max(1, len(df) // n_workers)

    # Split dataframe into chunks
    chunks = [df.iloc[i : i + chunk_size] for i in range(0, len(df), chunk_size)]

    logger.info(f"Processing {len(chunks)} chunks in parallel with {n_workers} workers")

    # Process chunks in parallel
    processed_chunks = parallel_apply(
        chunks, func, n_workers=n_workers, use_threads=False, description="Processing chunks"
    )

    # Combine results
    result_df = pd.concat([chunk for chunk in processed_chunks if chunk is not None], axis=0)

    return result_df


def optimize_dataframe_memory(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Optimize DataFrame memory usage by downcasting numeric types.

    Args:
        df: DataFrame to optimize
        verbose: Print memory savings

    Returns:
        Optimized DataFrame
    """
    start_mem = df.memory_usage(deep=True).sum() / 1024**2

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()

            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if (
                    c_min > np.finfo(np.float16).min
                    and c_max < np.finfo(np.float16).max
                    and (c_max - c_min) < 65504
                ):
                    df[col] = df[col].astype(np.float32)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage(deep=True).sum() / 1024**2

    if verbose:
        logger.info(
            f"Memory usage optimized: {start_mem:.2f}MB -> {end_mem:.2f}MB "
            f"({100 * (start_mem - end_mem) / start_mem:.1f}% reduction)"
        )

    return df


def batch_generator(iterable: Iterable[Any], batch_size: int):
    """
    Generate batches from an iterable.

    Args:
        iterable: Input iterable
        batch_size: Size of each batch

    Yields:
        Batches of items
    """
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


class PerformanceMonitor:
    """Context manager for monitoring performance of code blocks."""

    def __init__(self, name: str = "Operation", log_memory: bool = True):
        """
        Initialize performance monitor.

        Args:
            name: Name of the operation being monitored
            log_memory: Whether to log memory usage
        """
        self.name = name
        self.log_memory = log_memory
        self.start_time = None
        self.start_memory = None

    def __enter__(self):
        """Start monitoring."""
        import time

        self.start_time = time.time()

        if self.log_memory:
            import psutil

            process = psutil.Process()
            self.start_memory = process.memory_info().rss / 1024**2

        logger.info(f"Starting {self.name}...")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop monitoring and log results."""
        import time

        elapsed_time = time.time() - self.start_time

        log_msg = f"{self.name} completed in {elapsed_time:.2f}s"

        if self.log_memory:
            import psutil

            process = psutil.Process()
            end_memory = process.memory_info().rss / 1024**2
            memory_delta = end_memory - self.start_memory
            log_msg += f" (Memory: {self.start_memory:.1f}MB -> {end_memory:.1f}MB, Δ{memory_delta:+.1f}MB)"

        logger.info(log_msg)


# Example usage:
# with PerformanceMonitor("Data loading"):
#     df = pd.read_csv("large_file.csv")
