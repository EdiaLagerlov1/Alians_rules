import numpy as np
import matplotlib.pyplot as plt


def find_best_line(num_points=1000, num_iterations=100, random_seed=None):
    """
    Function that finds the best line for randomly generated points

    Parameters:
    num_points (int): Number of points to generate
    num_iterations (int): Number of lines to generate and test
    random_seed (int): Random seed for reproducible results
    """

    if random_seed is not None:
        np.random.seed(random_seed)

    # Step 1-2: Generate random points in range 0-1
    print(f"Generating {num_points} random points...")
    points_x = np.random.uniform(0, 1, num_points)
    points_y = np.random.uniform(0, 1, num_points)

    # Step 3: Display the points
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.scatter(points_x, points_y, alpha=0.6, s=10)
    plt.title(f'{num_points} Random Points')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')

    # Variables to store the best line
    best_a = None
    best_b = None
    min_avg_distance = float('inf')
    all_results = []

    print(f"Searching for the best line from {num_iterations} attempts...")

    # Steps 4-8: Generate random lines and find the best one
    for i in range(num_iterations):
        # Step 4: Generate random a, b
        a = np.random.uniform(0, 1)  # slope
        b = np.random.uniform(0, 1)  # y-intercept

        # Step 5: Calculate average distances from the line
        # Line equation: y = ax + b
        # Distance from point (x_i, y_i) to line y = ax + b is:
        # |y_i - (ax_i + b)| / sqrt(1 + a^2)
        # For simplicity, we use vertical distance: |y_i - (ax_i + b)|

        predicted_y = a * points_x + b
        distances = np.abs(points_y - predicted_y)
        avg_distance = np.mean(distances)

        # Step 6: Store the results
        all_results.append((a, b, avg_distance))

        # Update the best line
        if avg_distance < min_avg_distance:
            min_avg_distance = avg_distance
            best_a = a
            best_b = b

        if (i + 1) % 20 == 0:
            print(f"Completed {i + 1} attempts...")

    # Step 9: The best line
    print(f"\nBest line found:")
    print(f"a (slope) = {best_a:.4f}")
    print(f"b (intercept) = {best_b:.4f}")
    print(f"Minimum average distance = {min_avg_distance:.4f}")
    print(f"Line equation: y = {best_a:.4f}x + {best_b:.4f}")

    # Step 10: Display the best line
    plt.subplot(1, 2, 2)
    plt.scatter(points_x, points_y, alpha=0.6, s=10, label='Random points')

    # Draw the best line
    x_line = np.linspace(0, 1, 100)
    y_line = best_a * x_line + best_b
    plt.plot(x_line, y_line, 'r-', linewidth=2,
             label=f'Best line: y = {best_a:.3f}x + {best_b:.3f}')

    plt.title(f'Best Line Found\n(Average distance: {min_avg_distance:.4f})')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')

    plt.tight_layout()
    plt.show()

    return best_a, best_b, min_avg_distance, all_results


def analyze_results(all_results):
    """
    Function to analyze the results
    """
    all_results = np.array(all_results)
    a_values = all_results[:, 0]
    b_values = all_results[:, 1]
    distances = all_results[:, 2]

    print(f"\nResults analysis:")
    print(f"Range of a values (slope): {np.min(a_values):.4f} - {np.max(a_values):.4f}")
    print(f"Range of b values (intercept): {np.min(b_values):.4f} - {np.max(b_values):.4f}")
    print(f"Range of average distances: {np.min(distances):.4f} - {np.max(distances):.4f}")
    print(f"Mean of average distances: {np.mean(distances):.4f}")

    # Display distribution of results
    plt.figure(figsize=(15, 4))

    plt.subplot(1, 3, 1)
    plt.hist(a_values, bins=20, alpha=0.7, edgecolor='black')
    plt.title('Distribution of a values (slope)')
    plt.xlabel('a')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 2)
    plt.hist(b_values, bins=20, alpha=0.7, edgecolor='black')
    plt.title('Distribution of b values (intercept)')
    plt.xlabel('b')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 3)
    plt.hist(distances, bins=20, alpha=0.7, edgecolor='black')
    plt.title('Distribution of average distances')
    plt.xlabel('Average distance')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# Run the main code
if __name__ == "__main__":
    # Run the function with required parameters
    best_a, best_b, min_distance, results = find_best_line(
        num_points=1000,
        num_iterations=100,
        #  random_seed=42  # for reproducible results
    )

    # Detailed analysis of results
    analyze_results(results)

    print(f"\n{'=' * 50}")
    print("Code completed successfully!")
    print(f"Best line found: y = {best_a:.4f}x + {best_b:.4f}")
    print(f"{'=' * 50}")