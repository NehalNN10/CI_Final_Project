import matplotlib.pyplot as plt
import numpy as np

# Sample data (replace with actual values)
iterations = np.arange(0, 10001, 500)
win_rates = [
    30 + 42 * (1 - np.exp(-i / 3000)) for i in iterations
]  # Simulated learning curve

plt.figure(figsize=(8, 4))
plt.plot(iterations, win_rates, "b-", linewidth=2, label="Our Agent")
plt.axhline(y=50, color="r", linestyle="--", label="MCTS Baseline")
plt.fill_between(
    iterations, [w - 5 for w in win_rates], [w + 5 for w in win_rates], alpha=0.2
)
plt.xlabel("Training Iterations", fontname="Times New Roman")
plt.ylabel("Win Rate (%)", fontname="Times New Roman")
plt.grid(linestyle=":")
plt.legend()
plt.show()