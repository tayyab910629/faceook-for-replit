"""
Retry decorator and utilities for robust error handling.
"""
import time
import logging
from functools import wraps
from typing import Callable, TypeVar, ParamSpec, Any

logger = logging.getLogger(__name__)

P = ParamSpec('P')
T = TypeVar('T')


def retry_with_backoff(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: tuple[type[Exception], ...] = (Exception,)
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """
    Retry decorator with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds
        backoff_factor: Multiplier for delay after each retry
        exceptions: Tuple of exceptions to catch and retry on

    Returns:
        Decorated function with retry logic
    """
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            delay = initial_delay
            last_exception: Exception | None = None

            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        logger.warning(
                            f"{func.__name__} failed (attempt {attempt + 1}/{max_retries}): {e}. "
                            f"Retrying in {delay:.2f}s..."
                        )
                        time.sleep(delay)
                        delay *= backoff_factor
                    else:
                        logger.error(f"{func.__name__} failed after {max_retries} attempts: {e}")

            if last_exception:
                raise last_exception
            raise RuntimeError(f"{func.__name__} failed after {max_retries} attempts")

        return wrapper
    return decorator

