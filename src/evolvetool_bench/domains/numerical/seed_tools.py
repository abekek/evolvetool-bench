"""Shared seed tools for all Numerical Computation sessions.

These three tools are provided to the agent at the start of every session.
"""

SEED_TOOLS = [
    {
        "name": "compute_stats",
        "description": "Compute basic statistics (mean, median, std) for a list of numbers.",
        "implementation": '''
def compute_stats(numbers: list[float]) -> dict:
    """Compute mean, median, and standard deviation for a list of numbers."""
    import statistics
    if not numbers:
        return {"mean": None, "median": None, "std": None}
    return {
        "mean": statistics.mean(numbers),
        "median": statistics.median(numbers),
        "std": statistics.pstdev(numbers),
    }
''',
    },
    {
        "name": "solve_linear",
        "description": "Solve a 2x2 linear system Ax=b. Takes A as [[a,b],[c,d]] and b as [e,f].",
        "implementation": '''
def solve_linear(A: list[list[float]], b: list[float]) -> list[float]:
    """Solve a 2x2 linear system Ax=b using Cramer\'s rule."""
    a00, a01 = A[0]
    a10, a11 = A[1]
    b0, b1 = b
    det = a00 * a11 - a01 * a10
    if abs(det) < 1e-12:
        raise ValueError("Matrix is singular or near-singular.")
    x0 = (b0 * a11 - b1 * a01) / det
    x1 = (a00 * b1 - a10 * b0) / det
    return [x0, x1]
''',
    },
    {
        "name": "interpolate_1d",
        "description": "Linearly interpolate a value between two (x, y) points.",
        "implementation": '''
def interpolate_1d(x0: float, y0: float, x1: float, y1: float, x: float) -> float:
    """Return y by linear interpolation between (x0,y0) and (x1,y1) at position x."""
    if abs(x1 - x0) < 1e-15:
        raise ValueError("x0 and x1 must be distinct.")
    t = (x - x0) / (x1 - x0)
    return y0 + t * (y1 - y0)
''',
    },
]
