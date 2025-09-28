import numpy as np
import matplotlib.pyplot as plt


def linear_regression_analysis(k, n, num_datasets=50):
    """
    מבצע ניתוח רגרסיה לינארית על מספר סטים של נתונים

    Parameters:
    k (int): מספר המשתנים הבלתי תלויים
    n (int): מספר התצפיות
    num_datasets (int): מספר סטים של נתונים ליצירה

    Returns:
    r_squared_values (list): רשימת ערכי R²
    sigma_values (list): רשימת ערכי σ (סטיית תקן השגיאות)
    """

    r_squared_values = []
    sigma_values = []

    # יצירת רמות רעש משתנות כדי להדגים את הקשר בין σ ל-R²
    noise_levels = np.linspace(0.01, 0.5, num_datasets)

    for i in range(num_datasets):
        # 1. הגרלת וקטור beta באורך k בטווח 0-1
        beta = np.random.uniform(0, 1, k)

        # 2. הגרלת וקטור שגיאה ε עם רמת רעש משתנה
        noise_level = noise_levels[i]
        epsilon = np.random.normal(0, noise_level, n)

        # 3. הגרלת מטריצת X (n×k) בטווח 0-1
        X = np.random.uniform(0, 1, (n, k))

        # הדפסת המטריצות (רק עבור הסט הראשון)
        if i == 0:
            print(f"סט נתונים {i + 1}:")
            print(f"וקטור beta: {beta}")
            print(f"מטריצת X (5 שורות ראשונות):\n{X[:5]}")
            print(f"וקטור epsilon (5 ערכים ראשונים): {epsilon[:5]}")
            print(f"רמת רעש: {noise_level:.3f}")
            print("-" * 50)

        # חישוב y לפי המודל הלינארי: y = X @ beta + ε
        y = X @ beta + epsilon

        # חישוב החזיות
        y_pred = X @ beta

        # 5. חישוב R² (מקדם הקביעה)
        y_mean = np.mean(y)
        SST = np.sum((y - y_mean) ** 2)  # Total Sum of Squares
        SSE = np.sum((y - y_pred) ** 2)  # Error Sum of Squares

        r_squared = 1 - (SSE / SST)

        # 6. חישוב σ (סטיית התקן של השגיאות)
        residuals = y - y_pred
        sigma = np.std(residuals)

        r_squared_values.append(r_squared)
        sigma_values.append(sigma)

    return r_squared_values, sigma_values


def plot_r_vs_sigma(r_squared_values, sigma_values):
    """
    7. מציג גרף של R² כפונקציה של σ
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(sigma_values, r_squared_values)
    plt.xlabel('σ (סטיית תקן השגיאות)')
    plt.ylabel('R² (מקדם הקביעה)')
    plt.title('R² כפונקציה של σ\n(ככל ש-σ קטן יותר, R² גדול יותר)')
    plt.grid(True)
    plt.show()


# הרצת הקוד
if __name__ == "__main__":
    # 4. k,n כפרמטרים
    k = 50  # מספר משתנים בלתי תלויים
    n = 20  # מספר תצפיות

    # הרצת הניתוח
    r_squared_values, sigma_values = linear_regression_analysis(k, n)

    # הצגת הגרף
    plot_r_vs_sigma(r_squared_values, sigma_values)