import numpy as np
import matplotlib.pyplot as plt


def generate_linear_data_with_noise(A, B, num_points, x_range=(0, 1), noise_std=0.5):
    """
    מייצר נקודות לינאריות עם רעש נורמלי
    """
    X = np.random.uniform(x_range[0], x_range[1], num_points)
    Y_perfect = A * X + B
    noise = np.random.normal(0, noise_std, num_points)
    Y = Y_perfect + noise
    return X, Y, Y_perfect


def calculate_least_squares_vectorized(X, Y):
    """
    חישוב A,B לפי נוסחת קו שגיאה מינימלי - בוקטורים
    a = (X·Y - n·x̄·ȳ) / (X·X - n·x̄²)
    b = ȳ - a·x̄
    """
    n = len(X)
    x_mean = np.mean(X)
    y_mean = np.mean(Y)

    xy_dot = np.dot(X, Y)  # Σxᵢyᵢ
    xx_dot = np.dot(X, X)  # Σxᵢ²

    numerator_a = xy_dot - n * x_mean * y_mean
    denominator_a = xx_dot - n * x_mean ** 2
    a_calculated = numerator_a / denominator_a
    b_calculated = y_mean - a_calculated * x_mean

    return a_calculated, b_calculated


# 1. הגדרת פרמטרים
A = 0.5  # שיפוע
B = 0.1  # חיתוך ציר Y
NUM_POINTS = 1000  # מספר נקודות

# 2. יצירת רעש בטווח 0-1
noise_level = np.random.uniform(0, 1)

# 3. יצירת הנקודות
X, Y, Y_perfect = generate_linear_data_with_noise(A, B, NUM_POINTS, noise_std=noise_level)

# 4. חישוב A,B מהנקודות
A_calc, B_calc = calculate_least_squares_vectorized(X, Y)

# 5. הצגת הנקודות והקווים
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# First plot - points with original line
ax1.scatter(X, Y, alpha=0.6, s=10, color='blue', label=f'Points ({NUM_POINTS})')
X_line = np.linspace(0, 1, 100)
Y_line = A * X_line + B
ax1.plot(X_line, Y_line, 'r-', linewidth=2, label=f'Original line: Y = {A}X + {B}')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_title('Points with Original Line')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Second plot - same points with calculated line
ax2.scatter(X, Y, alpha=0.6, s=10, color='blue', label=f'Points ({NUM_POINTS})')
Y_calc_line = A_calc * X_line + B_calc
ax2.plot(X_line, Y_calc_line, 'g-', linewidth=2, label=f'Calculated line: Y = {A_calc:.3f}X + {B_calc:.3f}')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_title('Points with Least Squares Line')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()