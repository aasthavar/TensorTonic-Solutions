import numpy as np

def filter_and_extract(data: list, row_start: int, row_stop: int, threshold: float) -> np.ndarray:
    """
    Returns matching values in row-major order as a 1D float64 array.
    """
    a = np.array(data, dtype=np.float64)
    block = a[row_start: row_stop]
    return block[block > threshold]
