import time
import functools
from typing import Type, Tuple, Optional, Callable
from deepseek.utils.logging import get_logger

logger = get_logger(__name__)

class DeepSeekError(Exception):
    """Base exception for DeepSeek errors."""
    pass

class ConfigurationError(DeepSeekError):
    """Configuration related errors."""
    pass

class ModelError(DeepSeekError):
    """Model related errors."""
    pass

class TrainingError(DeepSeekError):
    """Training related errors."""
    pass

class OOMError(TrainingError):
    """Out of memory error."""
    pass

def retry(
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    tries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    logger: Optional[Callable] = None,
):
    """
    Retry decorator with exponential backoff.
    
    Args:
        exceptions: Tuple of exceptions to catch
        tries: Number of attempts
        delay: Initial delay between retries
        backoff: Multiplier for delay
        logger: Logger to use
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            mtries, mdelay = tries, delay
            while mtries > 1:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    msg = f"{e}, Retrying in {mdelay} seconds..."
                    if logger:
                        logger.warning(msg)
                    else:
                        print(msg)
                    time.sleep(mdelay)
                    mtries -= 1
                    mdelay *= backoff
            return func(*args, **kwargs)
        return wrapper
    return decorator
