import numpy as np
import matplotlib.pyplot as plt

# פרמטרים
k = 5  # מספר משתנים מסבירים
n = 100  # מספר תצפיות
num_simulations = 20

# רשימות לשמירת תוצאות
r_squared_values = []
r_squared_adj_values = []
sigma_values = []

# יצירת טווח מבוקר של σ (מ-0.05 עד 2.0)
sigma_range = np.linspace(0.05, 2.0, num_simulations)

# ריצה של 20 סימולציות עם σ עולה
for sim in range(num_simulations):
    # שלב 1: הגרלת וקטור beta באורך k+5 (5 משתנים מקוריים + 5 נוספים)
    beta = np.random.uniform(0.5, 1.5, k + 5)

    # שלב 2: הגרלת מטריצת X בגודל n×k
    X_original = np.random.uniform(0, 1, (n, k))

    # הוספת 5 משתנים נוספים (אינטראקציות וריבועים)
    x1_x2 = X_original[:, 0] * X_original[:, 1]  # x₁ * x₂
    x1_squared = X_original[:, 0] ** 2  # x₁²
    x2_squared = X_original[:, 1] ** 2  # x₂²
    x3_x4 = X_original[:, 2] * X_original[:, 3]  # x₃ * x₄
    x1_x3 = X_original[:, 0] * X_original[:, 2]  # x₁ * x₃

    # צירוף כל המשתנים למטריצה אחת
    X = np.column_stack([X_original, x1_x2, x1_squared, x2_squared, x3_x4, x1_x3])

    # שלב 3: הגרלת וקטור שגיאה ε עם סטיית תקן מבוקרת
    sigma = sigma_range[sim]
    epsilon = np.random.normal(0, sigma, n)  # התפלגות נורמלית עם σ מבוקר

    # שלב 4: חישוב y = X·β + ε
    y = X @ beta + epsilon

    # שלב 5: חישוב R² (מקדם ההתאמה)
    y_pred = X @ beta  # ערכים חזויים (ללא שגיאה)
    y_mean = np.mean(y)

    # SST (Total Sum of Squares)
    SST = np.sum((y - y_mean) ** 2)

    # SSR (Residual Sum of Squares)
    SSR = np.sum((y - y_pred) ** 2)

    # R² = 1 - (SSR / SST)
    R_squared = 1 - (SSR / SST)

    # חישוב R² Adjusted
    # R²_adj = 1 - [(1 - R²) * (n - 1) / (n - p - 1)]
    # כאשר p = מספר המשתנים המסבירים
    p = X.shape[1]  # מספר העמודות = מספר המשתנים
    R_squared_adj = 1 - ((1 - R_squared) * (n - 1) / (n - p - 1))

    # שמירת התוצאות
    r_squared_values.append(R_squared)
    r_squared_adj_values.append(R_squared_adj)
    sigma_values.append(sigma)

# שלב 8: הצגת הגרף
plt.figure(figsize=(12, 7))

# R² רגיל
plt.scatter(sigma_values, r_squared_values, color='darkblue', alpha=0.7, s=120,
            edgecolors='black', linewidth=1.5, label='R²', zorder=3)
plt.plot(sigma_values, r_squared_values, 'b-', alpha=0.5, linewidth=2.5)

# R² Adjusted
plt.scatter(sigma_values, r_squared_adj_values, color='darkgreen', alpha=0.7, s=120,
            edgecolors='black', linewidth=1.5, label='R² Adjusted', zorder=3)
plt.plot(sigma_values, r_squared_adj_values, 'g-', alpha=0.5, linewidth=2.5)

plt.xlabel('σ (Standard Deviation of Error)', fontsize=14, fontweight='bold')
plt.ylabel('R² Values', fontsize=14, fontweight='bold')
plt.title(f'R² and R² Adjusted vs σ\n(k={k} original + 5 interaction variables = {k + 5} total, n={n} observations)',
          fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.4, linestyle='--')
plt.legend(fontsize=13, loc='upper right', framealpha=0.9)
plt.tight_layout()
plt.show()

# Print statistics
print("=" * 60)
print(f"Total variables: {k} original + 5 engineered = {k + 5}")
print(f"Engineered features: x₁*x₂, x₁², x₂², x₃*x₄, x₁*x₃")
print(f"σ range: [{min(sigma_values):.3f}, {max(sigma_values):.3f}]")
print(f"R² range: [{min(r_squared_values):.3f}, {max(r_squared_values):.3f}]")
print(f"R² Adjusted range: [{min(r_squared_adj_values):.3f}, {max(r_squared_adj_values):.3f}]")
print(f"Correlation (σ vs R²): {np.corrcoef(sigma_values, r_squared_values)[0, 1]:.4f}")
print(f"Correlation (σ vs R² Adj): {np.corrcoef(sigma_values, r_squared_adj_values)[0, 1]:.4f}")
print("=" * 60)
print("\nConclusion: As σ increases, both R² and R² Adjusted decrease!")
print("R² Adjusted is lower because it penalizes for the number of variables.")
print("Higher error = Worse fit")