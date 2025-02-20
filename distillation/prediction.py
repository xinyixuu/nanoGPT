import numpy as np
import matplotlib.pyplot as plt

# Empirical data you provided
dims = np.array([32, 64, 128, 256, 512, 1024])
a_data = np.array([-2.9118010862688806, -2.074352092006171, -1.5565702664088292,
                   -1.0585414573927032, -0.7630612492394266, -0.5346467759931256])
b_data = np.array([77.73760361898873, 81.37124107773464, 84.52595124136722,
                   85.80028454159637, 87.150710164054, 87.93991441427792])

# Discovered regression formulas
def a_pred(D):
    # a(D) = -16 * D^(-0.489)
    return -16.0 * (D**-0.489)

def b_pred(D):
    # b(D) = 90 - 72.5 * D^(-0.517)
    return 90.0 - 72.5 * (D**-0.517)

# Generate predictions
pred_a = a_pred(dims)
pred_b = b_pred(dims)

# Compute a simple Mean Squared Error (MSE) as a measure of fit
mse_a = np.mean((a_data - pred_a)**2)
mse_b = np.mean((b_data - pred_b)**2)
print(f"MSE for a(D): {mse_a:.6f}")
print(f"MSE for b(D): {mse_b:.6f}")

# Set up the figure
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# --- Subplot for a(D) ---
axes[0].scatter(dims, a_data, color="red", label="Empirical a(D)", zorder=3)
axes[0].plot(dims, pred_a, color="blue", label="Fitted a(D)", zorder=2)
axes[0].set_xscale("log")  # optional, since dims are powers of 2
axes[0].set_xlabel("Dimension (D)")
axes[0].set_ylabel("a(D)")
axes[0].set_title("Empirical vs. Fitted a(D)")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# --- Subplot for b(D) ---
axes[1].scatter(dims, b_data, color="red", label="Empirical b(D)", zorder=3)
axes[1].plot(dims, pred_b, color="blue", label="Fitted b(D)", zorder=2)
axes[1].set_xscale("log")  # optional, since dims are powers of 2
axes[1].set_xlabel("Dimension (D)")
axes[1].set_ylabel("b(D)")
axes[1].set_title("Empirical vs. Fitted b(D)")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

