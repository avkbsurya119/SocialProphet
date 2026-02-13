"""
Helper utilities for SocialProphet.

Common functions used across the project.
"""

import os
import json
import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Union
from datetime import datetime
import pandas as pd
import numpy as np


def save_json(data: Dict, filepath: Union[str, Path]) -> None:
    """
    Save dictionary to JSON file.

    Args:
        data: Dictionary to save
        filepath: Output file path
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


def load_json(filepath: Union[str, Path]) -> Dict:
    """
    Load dictionary from JSON file.

    Args:
        filepath: Input file path

    Returns:
        Loaded dictionary
    """
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def save_pickle(obj: Any, filepath: Union[str, Path]) -> None:
    """
    Save object to pickle file.

    Args:
        obj: Object to save
        filepath: Output file path
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(filepath: Union[str, Path]) -> Any:
    """
    Load object from pickle file.

    Args:
        filepath: Input file path

    Returns:
        Loaded object
    """
    with open(filepath, "rb") as f:
        return pickle.load(f)


def get_timestamp_str() -> str:
    """
    Get current timestamp as string.

    Returns:
        Timestamp string (YYYYMMDD_HHMMSS)
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def format_number(num: float, precision: int = 2) -> str:
    """
    Format number for display.

    Args:
        num: Number to format
        precision: Decimal precision

    Returns:
        Formatted string
    """
    if num >= 1_000_000:
        return f"{num/1_000_000:.{precision}f}M"
    elif num >= 1_000:
        return f"{num/1_000:.{precision}f}K"
    else:
        return f"{num:.{precision}f}"


def calculate_percentage_change(old_value: float, new_value: float) -> float:
    """
    Calculate percentage change between two values.

    Args:
        old_value: Original value
        new_value: New value

    Returns:
        Percentage change
    """
    if old_value == 0:
        return 0 if new_value == 0 else float("inf")
    return ((new_value - old_value) / old_value) * 100


def ensure_list(value: Any) -> list:
    """
    Ensure value is a list.

    Args:
        value: Value to convert

    Returns:
        List
    """
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def safe_divide(a: float, b: float, default: float = 0) -> float:
    """
    Safely divide two numbers.

    Args:
        a: Numerator
        b: Denominator
        default: Default value if division fails

    Returns:
        Result or default
    """
    if b == 0:
        return default
    return a / b


def print_dataframe_info(df: pd.DataFrame, name: str = "DataFrame") -> None:
    """
    Print summary information about a DataFrame.

    Args:
        df: DataFrame to summarize
        name: Name to display
    """
    print(f"\n{'='*50}")
    print(f"{name} Summary")
    print(f"{'='*50}")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Data types:\n{df.dtypes}")
    print(f"\nMissing values:\n{df.isnull().sum()}")
    print(f"\nNumeric summary:\n{df.describe()}")
    print(f"{'='*50}\n")


def get_project_root() -> Path:
    """
    Get the project root directory.

    Returns:
        Path to project root
    """
    current = Path(__file__).resolve()
    # Go up from src/utils/helpers.py to project root
    return current.parent.parent.parent


def setup_logging(log_level: str = "INFO") -> None:
    """
    Set up basic logging configuration.

    Args:
        log_level: Logging level
    """
    import logging

    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def timer(func):
    """
    Decorator to time function execution.

    Args:
        func: Function to time

    Returns:
        Wrapped function
    """
    import time
    from functools import wraps

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        print(f"{func.__name__} completed in {elapsed:.2f}s")
        return result

    return wrapper


def chunk_list(lst: list, chunk_size: int) -> list:
    """
    Split a list into chunks.

    Args:
        lst: List to split
        chunk_size: Size of each chunk

    Returns:
        List of chunks
    """
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def flatten_dict(d: Dict, parent_key: str = "", sep: str = "_") -> Dict:
    """
    Flatten a nested dictionary.

    Args:
        d: Dictionary to flatten
        parent_key: Parent key prefix
        sep: Separator for keys

    Returns:
        Flattened dictionary
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def validate_date_format(date_str: str, format: str = "%Y-%m-%d") -> bool:
    """
    Validate if a string is in the expected date format.

    Args:
        date_str: Date string to validate
        format: Expected format

    Returns:
        True if valid, False otherwise
    """
    try:
        datetime.strptime(date_str, format)
        return True
    except ValueError:
        return False


def create_date_range(start: str, end: str, freq: str = "D") -> pd.DatetimeIndex:
    """
    Create a date range.

    Args:
        start: Start date
        end: End date
        freq: Frequency ("D"=daily, "H"=hourly, etc.)

    Returns:
        DatetimeIndex
    """
    return pd.date_range(start=start, end=end, freq=freq)
