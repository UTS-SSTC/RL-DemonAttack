"""
CSV logging utility for training metrics.

Provides a simple wrapper around Python's csv module for logging
structured training data to CSV files.
"""

import csv
import os


class CSVLogger:
    """
    Simple CSV logger for training metrics.

    Automatically creates the output directory if it doesn't exist,
    writes headers, and flushes after each log entry for real-time monitoring.

    Args:
        path: Path to the output CSV file.
        fieldnames: List of column names for the CSV file.
    """

    def __init__(self, path, fieldnames):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.f = open(path, "w", newline="", encoding="utf-8")
        self.w = csv.DictWriter(self.f, fieldnames=fieldnames)
        self.w.writeheader()
        self.f.flush()

    def log(self, **kwargs):
        """
        Write a row to the CSV file.

        Args:
            **kwargs: Key-value pairs matching the fieldnames specified in __init__.
        """
        self.w.writerow(kwargs)
        self.f.flush()

    def close(self):
        """Close the underlying file handle."""
        self.f.close()
