# Association Rules Finder

A Python implementation of the Apriori algorithm for discovering association rules in binary data.

## What it does
Generates 5000 rows of synthetic binary data (6 columns: A-F) with specified probabilities, then finds meaningful patterns using association rule mining. Rules are filtered by minimum support (30%) and confidence (70%), sorted by lift strength.

## Usage
```bash
python find_rules.py
```

## Output
Displays all discovered rules in format: `Antecedent → Consequent` with support, confidence, and lift metrics. Higher lift values indicate stronger relationships between items.
