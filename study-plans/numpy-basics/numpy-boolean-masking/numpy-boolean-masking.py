import numpy as np

def row_summary(data: list, threshold: float) -> np.ndarray:
    """
    Returns a float64 array of shape (3, m, n): mask, any-row, all-row.
    """
    a = np.array(data, dtype=np.float64)
    return np.stack(
        [
            (a > threshold).astype(np.float64), 
            a * np.any(a > threshold, axis=1)[:, np.newaxis], 
            a * np.all(a > threshold, axis=1)[:, np.newaxis]
        ]
    ).astype(np.float64)