import numpy as np

# Configuration
ROWS = 5000
MIN_SUPPORT = 0.30  # 30%
MIN_CONFIDENCE = 0.70  # 70%


# Step 1: Generate the data
def generate_data():
    """Generate 5000x6 array with specified probabilities"""
    data = np.zeros((ROWS, 6), dtype=int)

    # Columns A, B (indices 0, 1): 70% probability of 1
    data[:, 0] = np.random.choice([0, 1], size=ROWS, p=[0.3, 0.7])
    data[:, 1] = np.random.choice([0, 1], size=ROWS, p=[0.3, 0.7])

    # Columns C, D (indices 2, 3): 50% probability of 1
    data[:, 2] = np.random.choice([0, 1], size=ROWS, p=[0.5, 0.5])
    data[:, 3] = np.random.choice([0, 1], size=ROWS, p=[0.5, 0.5])

    # Columns E, F (indices 4, 5): random (50/50)
    data[:, 4] = np.random.choice([0, 1], size=ROWS, p=[0.2, 0.8])
    data[:, 5] = np.random.choice([0, 1], size=ROWS, p=[0.2, 0.8])

    return data


# Helper function to generate combinations without itertools
def get_combinations(items, r):
    """Generate all combinations of r items from items tuple/list"""
    items = list(items)
    n = len(items)
    if r > n:
        return

    indices = list(range(r))
    yield tuple(items[i] for i in indices)

    while True:
        # Find the rightmost index that can be incremented
        i = r - 1
        while i >= 0 and indices[i] == i + n - r:
            i -= 1

        if i < 0:
            return

        # Increment this index and reset all indices to its right
        indices[i] += 1
        for j in range(i + 1, r):
            indices[j] = indices[j - 1] + 1

        yield tuple(items[i] for i in indices)


# Step 2: Find frequent itemsets using Apriori-like approach
def get_support(data, itemset):
    """Calculate support for an itemset"""
    mask = np.ones(len(data), dtype=bool)
    for col_idx in itemset:
        mask &= (data[:, col_idx] == 1)
    return np.sum(mask) / len(data)


def find_frequent_itemsets(data, min_support):
    """Find all itemsets that meet minimum support threshold"""
    columns = ['A', 'B', 'C', 'D', 'E', 'F']
    n_cols = data.shape[1]
    frequent_itemsets = {}

    # Level 1: Individual columns
    for col_idx in range(n_cols):
        itemset = (col_idx,)
        support = get_support(data, itemset)
        if support >= min_support:
            frequent_itemsets[itemset] = support

    # Level k: Combinations of size k
    current_level = list(frequent_itemsets.keys())
    k = 2

    while current_level:
        # Generate candidates of size k
        candidates = set()
        for i in range(len(current_level)):
            for j in range(i + 1, len(current_level)):
                # Join two (k-1)-itemsets
                union = tuple(sorted(set(current_level[i]) | set(current_level[j])))
                if len(union) == k:
                    candidates.add(union)

        # Test candidates
        next_level = []
        for itemset in candidates:
            support = get_support(data, itemset)
            if support >= min_support:
                frequent_itemsets[itemset] = support
                next_level.append(itemset)

        current_level = next_level
        k += 1

    return frequent_itemsets


# Step 3: Generate association rules
def generate_rules(data, frequent_itemsets, min_confidence):
    """Generate association rules from frequent itemsets"""
    columns = ['A', 'B', 'C', 'D', 'E', 'F']
    rules = []

    # For each frequent itemset with size >= 2
    for itemset, support in frequent_itemsets.items():
        if len(itemset) < 2:
            continue

        # Generate all non-empty subsets as antecedents
        for r in range(1, len(itemset)):
            for antecedent in get_combinations(itemset, r):
                consequent = tuple(col for col in itemset if col not in antecedent)

                # Calculate confidence: P(consequent | antecedent)
                antecedent_support = get_support(data, antecedent)
                if antecedent_support == 0:
                    continue

                confidence = support / antecedent_support

                if confidence >= min_confidence:
                    # Calculate lift
                    consequent_support = get_support(data, consequent)
                    lift = confidence / consequent_support if consequent_support > 0 else 0

                    rules.append({
                        'antecedent': [columns[i] for i in antecedent],
                        'consequent': [columns[i] for i in consequent],
                        'support': support,
                        'confidence': confidence,
                        'lift': lift
                    })

    # Sort by lift (descending), then by confidence (descending)
    rules.sort(key=lambda x: (x['lift'], x['confidence']), reverse=True)

    return rules


# Step 4: Display results
def display_rules(rules):
    """Display association rules in a readable format"""
    print(f"\n{'=' * 80}")
    print(f"ASSOCIATION RULES (Support >= {MIN_SUPPORT:.0%}, Confidence >= {MIN_CONFIDENCE:.0%})")
    print(f"Sorted by LIFT (best relationships first)")
    print(f"{'=' * 80}\n")
    print(f"Total rules found: {len(rules)}\n")

    if not rules:
        print("No rules found meeting the criteria.")
        return

    for i, rule in enumerate(rules, 1):
        antecedent_str = ', '.join([f"{col}=1" for col in rule['antecedent']])
        consequent_str = ', '.join([f"{col}=1" for col in rule['consequent']])

        print(f"Rule {i}: [Lift: {rule['lift']:.3f}]")
        print(f"  {antecedent_str} → {consequent_str}")
        print(f"  Support:    {rule['support']:.2%} ({int(rule['support'] * ROWS)} transactions)")
        print(f"  Confidence: {rule['confidence']:.2%}")
        print(f"  Lift:       {rule['lift']:.3f}")
        print()


# Main execution
def main():
    print("Generating data...")
    data = generate_data()

    print(f"Data shape: {data.shape}")
    print(f"\nColumn distributions:")
    columns = ['A', 'B', 'C', 'D', 'E', 'F']
    for i, col in enumerate(columns):
        pct = np.mean(data[:, i]) * 100
        print(f"  {col}: {pct:.1f}% ones")

    print(f"\nFinding frequent itemsets (support >= {MIN_SUPPORT:.0%})...")
    frequent_itemsets = find_frequent_itemsets(data, MIN_SUPPORT)
    print(f"Found {len(frequent_itemsets)} frequent itemsets")

    print(f"\nGenerating association rules (confidence >= {MIN_CONFIDENCE:.0%})...")
    rules = generate_rules(data, frequent_itemsets, MIN_CONFIDENCE)

    display_rules(rules)

    return data, rules


# Run the analysis
if __name__ == "__main__":
    data, rules = main()