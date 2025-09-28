import numpy as np
import matplotlib.pyplot as plt

# פרמטרים
k = 5  # מספר משתנים מסבירים
n = 100  # מספר תצפיות
num_simulations = 20

# רשימות לשמירת תוצאות
r_squared_values = []
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

    # שמירת התוצאות
    r_squared_values.append(R_squared)
    sigma_values.append(sigma)

# שלב 8: הצגת הגרף
plt.figure(figsize=(12, 7))
plt.scatter(sigma_values, r_squared_values, color='darkblue', alpha=0.7, s=120, edgecolors='black', linewidth=1.5)
plt.plot(sigma_values, r_squared_values, 'r-', alpha=0.6, linewidth=2.5, label='Trend')

plt.xlabel('σ (Standard Deviation of Error)', fontsize=14, fontweight='bold')
plt.ylabel('R² (Coefficient of Determination)', fontsize=14, fontweight='bold')
plt.title(f'Relationship between R² and σ\n(k={k} original + 5 interaction variables, n={n} observations)',
          fontsize=16, fontweight='bold')
plt.grid(True, alpha=0.4, linestyle='--')
plt.legend(fontsize=12, loc='upper right')
plt.tight_layout()
plt.show()

# Print statistics
print("=" * 50)
print(f"Total variables: {k} original + 5 engineered = {k + 5}")
print(f"Engineered features: x₁*x₂, x₁², x₂², x₃*x₄, x₁*x₃")
print(f"σ range: [{min(sigma_values):.3f}, {max(sigma_values):.3f}]")
print(f"R² range: [{min(r_squared_values):.3f}, {max(r_squared_values):.3f}]")
print(f"Correlation between σ and R²: {np.corrcoef(sigma_values, r_squared_values)[0, 1]:.4f}")
print("=" * 50)
print("\nConclusion: As σ increases, R² decreases!")
print("Higher error = Worse fit")