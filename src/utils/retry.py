"""
Retry Logic with Exponential Backoff
Provides robust retry mechanisms for transient failures
"""

import asyncio
import time
from functools import wraps
from typing import Any, Callable, Optional, Tuple, Type

from .logging import get_logger

logger = get_logger(__name__)


def retry_with_backoff(
    max_retries: int = 4,
    initial_delay: float = 2.0,
    backoff_factor: float = 2.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    on_retry: Optional[Callable[[Exception, int], None]] = None,
):
    """
    Decorator for retrying functions with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds
        backoff_factor: Multiplier for delay on each retry
        exceptions: Tuple of exceptions to catch and retry
        on_retry: Optional callback called on each retry with (exception, attempt_number)

    Example:
        @retry_with_backoff(max_retries=3, initial_delay=1.0)
        def unreliable_function():
            # Function that might fail
            pass
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            delay = initial_delay
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e

                    if attempt == max_retries:
                        logger.error(
                            f"Function {func.__name__} failed after {max_retries} retries",
                            exc_info=True,
                        )
                        raise

                    logger.warning(
                        f"Attempt {attempt + 1}/{max_retries} for {func.__name__} failed: {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )

                    if on_retry:
                        on_retry(e, attempt + 1)

                    time.sleep(delay)
                    delay *= backoff_factor

            # Should never reach here, but just in case
            if last_exception:
                raise last_exception

        return wrapper

    return decorator


def async_retry_with_backoff(
    max_retries: int = 4,
    initial_delay: float = 2.0,
    backoff_factor: float = 2.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    on_retry: Optional[Callable[[Exception, int], None]] = None,
):
    """
    Decorator for retrying async functions with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds
        backoff_factor: Multiplier for delay on each retry
        exceptions: Tuple of exceptions to catch and retry
        on_retry: Optional callback called on each retry with (exception, attempt_number)

    Example:
        @async_retry_with_backoff(max_retries=3, initial_delay=1.0)
        async def unreliable_async_function():
            # Async function that might fail
            pass
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            delay = initial_delay
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e

                    if attempt == max_retries:
                        logger.error(
                            f"Async function {func.__name__} failed after {max_retries} retries",
                            exc_info=True,
                        )
                        raise

                    logger.warning(
                        f"Attempt {attempt + 1}/{max_retries} for {func.__name__} failed: {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )

                    if on_retry:
                        on_retry(e, attempt + 1)

                    await asyncio.sleep(delay)
                    delay *= backoff_factor

            # Should never reach here, but just in case
            if last_exception:
                raise last_exception

        return wrapper

    return decorator


class CircuitBreaker:
    """
    Circuit breaker pattern implementation to prevent cascading failures.

    States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Too many failures, requests fail immediately
    - HALF_OPEN: Testing if service recovered
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: Type[Exception] = Exception,
    ):
        """
        Initialize circuit breaker.

        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Seconds to wait before attempting recovery
            expected_exception: Exception type to catch
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception

        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function through circuit breaker."""
        if self.state == "OPEN":
            if time.time() - self.last_failure_time >= self.recovery_timeout:
                self.state = "HALF_OPEN"
                logger.info(f"Circuit breaker entering HALF_OPEN state")
            else:
                raise Exception(f"Circuit breaker is OPEN. Service unavailable.")

        try:
            result = func(*args, **kwargs)

            # Success - reset if in HALF_OPEN
            if self.state == "HALF_OPEN":
                self.state = "CLOSED"
                self.failure_count = 0
                logger.info(f"Circuit breaker entering CLOSED state (recovered)")

            return result

        except self.expected_exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()

            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"
                logger.error(
                    f"Circuit breaker OPEN after {self.failure_count} failures"
                )

            raise
