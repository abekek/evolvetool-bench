def solve_2x2_linear_system(a11, a12, b1, a21, a22, b2):
    """
    Solve a 2x2 linear system of equations using Cramer's rule.
    
    Solves the system:
    a11*x + a12*y = b1
    a21*x + a22*y = b2
    
    Utility: Solves linear systems of two equations with two unknowns using determinants
    Args:
        a11 (float): coefficient of x in first equation
        a12 (float): coefficient of y in first equation  
        b1 (float): constant term in first equation
        a21 (float): coefficient of x in second equation
        a22 (float): coefficient of y in second equation
        b2 (float): constant term in second equation
    Returns:
        dict: {'x': float, 'y': float} containing the solution, or error message if no unique solution
    """
    
    # Calculate main determinant
    det = a11 * a22 - a12 * a21
    
    # Check if system has unique solution
    if abs(det) < 1e-10:
        return {"error": "System has no unique solution (determinant is zero)"}
    
    # Calculate determinants for x and y using Cramer's rule
    det_x = b1 * a22 - a12 * b2
    det_y = a11 * b2 - b1 * a21
    
    # Solve for x and y
    x = det_x / det
    y = det_y / det
    
    return {"x": x, "y": y}