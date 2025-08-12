# Decision Tree from Scratch

A comprehensive, educational implementation of Decision Tree classifier built from scratch using only NumPy. This project demonstrates the core algorithms behind decision trees, including entropy calculation, information gain, and tree construction without relying on scikit-learn's implementation.

## 🌳 Overview

This implementation showcases:
- **Pure NumPy Implementation**: Built from ground up to understand tree construction algorithms
- **Information Gain**: Uses entropy-based splitting criteria for optimal decision boundaries
- **Configurable Parameters**: Customizable max depth, minimum samples split, and feature sampling
- **Complete Evaluation**: Includes accuracy metrics, classification report, and confusion matrix
- **Educational Focus**: Clear code structure with detailed algorithmic explanations

## ✨ Key Features

- **Custom Node Structure**: Efficient tree representation with leaf node detection
- **Entropy-Based Splitting**: Information gain calculation for optimal feature selection
- **Overfitting Prevention**: Built-in depth limiting and minimum sample constraints
- **Random Feature Selection**: Supports feature bagging for improved generalization
- **Comprehensive Testing**: Evaluated on breast cancer dataset with full metrics
- **Visualization Ready**: Confusion matrix plotting for model interpretation

## 🚀 Quick Start

### Installation

1. Clone the repository:
```bash
git clone https://github.com/hghaemi/decision_tree_from_scratch.git
cd decision_tree_from_scratch
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the notebook:
```bash
jupyter notebook decision_tree.ipynb
```

### Basic Usage

```python
from decision_tree import DecisionTree

# Initialize with custom parameters
clf = DecisionTree(
    max_depth=10,
    min_samples_split=5,
    n_features=None  # Use all features
)

# Train the model
clf.fit(X_train, y_train)

# Make predictions
predictions = clf.predict(X_test)
```

## 🧮 Algorithm Details

### Decision Tree Construction

The algorithm uses a **greedy, top-down approach**:

1. **Feature Selection**: Randomly sample features (if specified) or use all features
2. **Best Split Finding**: Evaluate all possible thresholds using information gain
3. **Tree Growing**: Recursively split data based on optimal feature/threshold pairs
4. **Stopping Criteria**: Stop when max depth reached, pure node found, or insufficient samples

### Information Gain Formula

```
Information Gain = H(parent) - [weighted average] H(children)

Where H(S) = -Σ(p_i * log2(p_i)) for each class i
```

### Key Components

#### Node Class
- **Internal Nodes**: Store feature index and threshold for splitting
- **Leaf Nodes**: Store the majority class prediction
- **Structure**: Binary tree with left/right child pointers

#### DecisionTree Class
- **fit()**: Constructs the tree using training data
- **predict()**: Traverses tree to make predictions
- **_grow_tree()**: Recursive tree building algorithm
- **_best_split()**: Finds optimal feature/threshold combination
- **_information_gain()**: Calculates entropy-based splitting criterion

## 📊 Performance

When tested on the breast cancer dataset:
- **High Accuracy**: Typically achieves 93-95% accuracy
- **Fast Training**: Efficient recursive construction
- **Good Interpretability**: Clear decision paths through tree structure
- **Robust Performance**: Handles both numerical and categorical features

## 🔍 Project Structure

```
decision_tree_from_scratch/
├── decision_tree.ipynb      # Main implementation and demo
├── requirements.txt         # Minimal dependencies
├── .gitignore              # Git ignore rules
└── README.md               # This documentation
```

### Core Implementation

```python
class Node:
    """Tree node with feature/threshold for internal nodes, value for leaves"""
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None)
    def is_leaf_node(self)

class DecisionTree:
    """Decision tree classifier with entropy-based splitting"""
    def __init__(self, min_samples_split=2, max_depth=100, n_features=None)
    def fit(X, y)
    def predict(X)
    def _grow_tree(X, y, depth=0)
    def _best_split(X, y, feat_idxs)
    def _information_gain(y, X_column, threshold)
```

## 🎓 Educational Value

Perfect for understanding:
- **Tree Construction**: How decisions trees are built step-by-step
- **Entropy & Information Theory**: Mathematical foundation of splitting criteria
- **Recursive Algorithms**: Tree building as a recursive process
- **Overfitting Prevention**: Role of hyperparameters in controlling complexity
- **Feature Selection**: Impact of random feature sampling on performance

## 🔧 Customization Options

### Hyperparameters

- **max_depth**: Maximum tree depth (default: 100)
- **min_samples_split**: Minimum samples required to split (default: 2)
- **n_features**: Number of features to consider per split (default: all)

### Advanced Usage

```python
# Prevent overfitting with conservative parameters
conservative_tree = DecisionTree(
    max_depth=5,
    min_samples_split=10,
    n_features=int(np.sqrt(n_features))  # Square root rule
)

# More aggressive tree for complex patterns
deep_tree = DecisionTree(
    max_depth=20,
    min_samples_split=2,
    n_features=None
)
```

## 📈 Comparison with Scikit-learn

| Feature | This Implementation | Scikit-learn |
|---------|-------------------|--------------|
| **Splitting Criterion** | Entropy (Information Gain) | Gini, Entropy, Log Loss |
| **Missing Values** | Not supported | Supported |
| **Pruning** | Pre-pruning only | Pre & Post-pruning |
| **Memory Usage** | Minimal | Optimized |
| **Speed** | Educational pace | Production optimized |
| **Interpretability** | Full transparency | Black box |

## 🐛 Known Limitations

- **No Missing Value Handling**: Requires clean data
- **No Post-Pruning**: Only pre-pruning via hyperparameters
- **Binary Classification Focus**: Designed primarily for binary problems
- **Memory Efficiency**: Not optimized for very large datasets

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- Add support for regression trees
- Implement post-pruning algorithms
- Add missing value handling
- Create tree visualization tools
- Add feature importance calculation

## 📚 Learning Resources

- **Information Theory**: Understanding entropy and information gain
- **Tree Algorithms**: CART, ID3, C4.5 algorithm comparisons
- **Ensemble Methods**: How decision trees form the basis for Random Forests
- **Overfitting**: Bias-variance tradeoff in tree-based models

## 📄 License

This project is open source and available under the MIT License.

---

**Educational Note**: This implementation prioritizes clarity and understanding over performance. For production use, consider scikit-learn's highly optimized DecisionTreeClassifier with additional features like pruning, missing value handling, and multi-output support.