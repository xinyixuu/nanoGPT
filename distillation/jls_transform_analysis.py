import torch
import numpy as np
import matplotlib.pyplot as plt
import time

# Function to compute JL approximate matrix multiplication.
def jl_approx_matmul(A, B, k):
    # A: m x d, B: d x n
    d = A.shape[1]
    # Create a random Gaussian matrix and scale by 1/sqrt(k)
    R = torch.randn(k, d) / np.sqrt(k)
    # Project A and B: (A @ R.T) is m x k and (R @ B) is k x n.
    A_proj = A @ R.t()   
    B_proj = R @ B       
    return A_proj @ B_proj

# Define the matrix dimensions to be compared.
matrix_sizes = [100, 200, 500, 1000]

# Dictionaries to hold k-scan results for each matrix dimension.
error_dict = {}
time_dict = {}

# For each matrix size, scan various projection dimensions k.
for d in matrix_sizes:
    m = d
    n = d
    A = torch.randn(m, d)
    B = torch.randn(d, n)
    C_exact = A @ B
    # Define a range of k values from 10 up to d (20 points).
    k_values = np.linspace(10, d, 20, dtype=int)
    errors_k = []
    times_k = []
    for k in k_values:
        start = time.time()
        C_approx = jl_approx_matmul(A, B, k)
        t = time.time() - start
        err = torch.norm(C_exact - C_approx, p='fro') / torch.norm(C_exact, p='fro')
        errors_k.append(err.item())
        times_k.append(t)
    error_dict[d] = (k_values, errors_k)
    time_dict[d] = (k_values, times_k)
    print(f"Matrix {d}x{d}: k from {k_values[0]} to {k_values[-1]}, final relative error: {errors_k[-1]:.4f}")

# Plot 1: Relative error vs. projection dimension k (one curve per matrix size).
plt.figure(figsize=(8, 5))
for d in matrix_sizes:
    k_vals, err_vals = error_dict[d]
    plt.plot(k_vals, err_vals, marker='o', linestyle='-', label=f"{d}x{d}")
plt.xlabel('Projection Dimension k')
plt.ylabel('Relative Frobenius Norm Error')
plt.title('Approximation Error vs. Projection Dimension k')
plt.legend(title="Matrix Size")
plt.grid(True)
plt.savefig('jl_error_vs_k_all.png')
plt.show()

# Plot 2: Computation time vs. projection dimension k (one curve per matrix size).
plt.figure(figsize=(8, 5))
for d in matrix_sizes:
    k_vals, time_vals = time_dict[d]
    plt.plot(k_vals, time_vals, marker='o', linestyle='-', label=f"{d}x{d}")
plt.xlabel('Projection Dimension k')
plt.ylabel('Computation Time (seconds)')
plt.title('Computation Time vs. Projection Dimension k')
plt.legend(title="Matrix Size")
plt.grid(True)
plt.savefig('jl_time_vs_k_all.png')
plt.show()

