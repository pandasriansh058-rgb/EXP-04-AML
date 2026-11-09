import numpy as np
import matplotlib.pyplot as plt


def predict(x, w, b):
    return w * x + b


def compute_cost(y, y_pred):
    return np.sum((y - y_pred) ** 2) / len(y)


def gradient(x, y, y_pred):
    n = len(x)

    dw = (np.sum((y - y_pred) * x * -2)) / n
    db = (np.sum((y - y_pred) * -2)) / n

    return dw, db


def parameter_update(w, b, dw, db, alpha):
    w = w - alpha * dw
    b = b - alpha * db

    return w, b


# Dataset
x = np.array([2, 3, 4], dtype=float)
y = np.array([2, 3, 4], dtype=float)

n = len(x)

print("Input x:", x)
print("Actual y:", y)
print("Number of samples:", n)


# Initial parameters
w = 0.0
b = 0.0

alpha = 0.1
iterations = 3

print("Initial weight (w):", w)
print("Initial bias (b):", b)
print("Learning rate:", alpha)


# Initial prediction and cost
y_pred = predict(x, w, b)

print("Prediction:", y_pred)

cost = compute_cost(y, y_pred)

print("Initial Prediction:", y_pred)
print(f"Initial cost: {cost:.4f}")


# History lists
costs = []
w_history = []
b_history = []


# Gradient Descent
for i in range(iterations):

    y_pred_current_params = predict(x, w, b)

    dw, db = gradient(x, y, y_pred_current_params)

    w, b = parameter_update(w, b, dw, db, alpha)

    y_pred_after_update = predict(x, w, b)

    cost_after_update = compute_cost(y, y_pred_after_update)

    costs.append(cost_after_update)
    w_history.append(w)
    b_history.append(b)

    print("\n" + "-" * 30)
    print(f"Iteration {i + 1}:")
    print(
        f"The y-predict (after update) is {y_pred_after_update}"
    )
    print(
        f"The cost (after update) is {cost_after_update:.4f}"
    )
    print(
        f"Gradients: dw = {dw:.4f}, db = {db:.4f}"
    )
    print(
        f"The updated weight (w) is {w:.4f}"
    )
    print(
        f"The updated bias (b) is {b:.4f}"
    )


# Final values
final_w = w_history[-1]
final_b = b_history[-1]

print("\nFinal w:", final_w)
print("Final b:", final_b)

print("\nThe final equation is:")
print(f"y = {final_w:.4f}x + {final_b:.4f}")

print("The final cost is:", costs[-1])


# Plot Cost vs Iterations
plt.plot(range(1, iterations + 1), costs)

plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Cost Function vs Iterations")

plt.show()


# Final prediction for plotting
if iterations > 0:
    final_y_pred_for_plot = predict(x, final_w, final_b)
else:
    final_y_pred_for_plot = predict(x, w, b)


# Plot Linear Regression
plt.scatter(
    x,
    y,
    label="Actual Data"
)

plt.plot(
    x,
    final_y_pred_for_plot,
    label="Regression Line"
)

plt.xlabel("X")
plt.ylabel("Y")
plt.title("Linear Regression Fit")

plt.legend()
plt.show()