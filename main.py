import numpy as np

A = np.array([
    [3, -5, 47, 20],
    [11, 16, 17, 10],
    [56, 22, 11, -18],
    [17, 66, -12, 7]
], dtype=float)

B = np.array([18, 26, 34, 82], dtype=float)

def cramer(A, B):
    det_A = np.linalg.det(A)
    if det_A == 0:
        return None
    solutions = []
    for i in range(len(B)):
        Ai = A.copy()
        Ai[:, i] = B
        solutions.append(np.linalg.det(Ai) / det_A)
    return np.array(solutions)

def gauss_elimination(A, B):
    n = len(B)
    M = np.hstack((A, B.reshape(-1, 1)))
    for i in range(n):
        M[i] = M[i] / M[i, i]
        for j in range(i+1, n):
            M[j] -= M[j, i] * M[i]
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = M[i, -1] - np.dot(M[i, i+1:n], x[i+1:n])
    return x

def jacobi(A, B, tol=1e-6, max_iterations=100):
    n = len(B)
    x = np.zeros(n)
    for _ in range(max_iterations):
        x_new = np.zeros_like(x)
        for i in range(n):
            s = sum(A[i, j] * x[j] for j in range(n) if i != j)
            x_new[i] = (B[i] - s) / A[i, i]
        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            return x_new
        x = x_new
    return x

def gauss_seidel(A, B, tol=1e-6, max_iterations=100):
    n = len(B)
    x = np.zeros(n)
    for _ in range(max_iterations):
        x_new = x.copy()
        for i in range(n):
            s1 = sum(A[i, j] * x_new[j] for j in range(i))
            s2 = sum(A[i, j] * x[j] for j in range(i+1, n))
            x_new[i] = (B[i] - s1 - s2) / A[i, i]
        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            return x_new
        x = x_new
    return x

solution_cramer = cramer(A, B)
solution_gauss = gauss_elimination(A, B)
solution_jacobi = jacobi(A, B)
solution_gauss_seidel = gauss_seidel(A, B)

print("Cramer's Rule Solution:", solution_cramer)
print("Gauss Elimination Solution:", solution_gauss)
print("Jacobi Method Solution:", solution_jacobi)
print("Gauss-Seidel Method Solution:", solution_gauss_seidel)
