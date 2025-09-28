import numpy as np
import matplotlib.pyplot as plt

# פרמטרים לדוגמה
delta = np.array([30, 40, 60])       # תשואה צפויה
sigma = np.array([12, 10, 8])     # volatility

# פונקציית רווח נטו מותאמת לסיכון
def net_profit(x):
    x = np.array(x)
    return np.sum(x * delta) - 0.5 * np.sum((x**2) * (sigma**2))

# גריד אחוזי השקעה
grid = np.linspace(0, 1, 20)

# חיפוש argminmax עם סכום ≤1
best_val = -np.inf
argminmax_coords = [0, 0, 0]

for x1 in grid:
    for x2 in grid:
        for x3 in grid:
            if x1 + x2 + x3 <= 1:
                val = net_profit([x1, x2, x3])
                if val > best_val:
                    best_val = val
                    argminmax_coords = [x1, x2, x3]

print("argminmax coordinates (x1, x2, x3):", argminmax_coords)
print("Net profit at argminmax:", net_profit(argminmax_coords))

# הכנת נתונים לגרף
X, Y, Z, C = [], [], [], []
for x1 in grid:
    for x2 in grid:
        for x3 in grid:
            if x1 + x2 + x3 <= 1:
                X.append(x1)
                Y.append(x2)
                Z.append(x3)                 # Z = השקעה במניה C
                C.append(net_profit([x1,x2,x3]))  # צבע = Net Profit

fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(X, Y, Z, c=C, cmap='viridis', s=50)
ax.set_xlabel('Investment in A')
ax.set_ylabel('Investment in B')
ax.set_zlabel('Investment in C')
plt.title('Investment Distribution with Net Profit as Color')

# מסמן את argminmax
ax.scatter(argminmax_coords[0], argminmax_coords[1], argminmax_coords[2],
           color='red', s=150, label='argminmax')

plt.legend()
plt.colorbar(sc, label='Net Profit')

# הוספת טקסט בתחתית עם ערכי argminmax
text_str = f"argminmax:\nx1={argminmax_coords[0]:.2f}, x2={argminmax_coords[1]:.2f}, x3={argminmax_coords[2]:.2f}\nNet Profit={net_profit(argminmax_coords):.2f}"
plt.gcf().text(0.15, 0.05, text_str, fontsize=12, bbox=dict(facecolor='white', alpha=0.7))

plt.show()
