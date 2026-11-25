import numpy as np

def reshape_array(data: list, operation: str) -> np.ndarray:
    """
    Returns a float64 array with the shape selected by operation.
    """
    a = np.array(data, dtype=np.float64)
    if operation == "flatten":
        return a.flatten()
    elif operation == "transpose":
        return a.T
    else:
        return np.expand_dims(a, axis=0)
