# Soft-Margin SVM — Dual Form Implementation

A from-scratch implementation of the **Soft-Margin Support Vector Machine** solved via **Wolfe Dual optimization**, benchmarked against scikit-learn's `SVC` at increasing dataset sizes.

---

## Table of Contents

- [Background](#background)
- [The Primal Problem](#the-primal-problem)
- [The Wolfe Dual](#the-wolfe-dual)
- [The Kernel Trick](#the-kernel-trick)
- [Implementation Overview](#implementation-overview)
- [Hyperparameters](#hyperparameters)
- [Usage](#usage)
- [Benchmarking](#benchmarking)
- [Dependencies](#dependencies)

---

## Background

A **Support Vector Machine** finds the maximum-margin hyperplane separating two classes. For non-linearly separable data, the **soft-margin** formulation introduces slack variables $\xi_i \geq 0$ that allow individual points to violate the margin, controlled by a penalty parameter $C$.

The model is trained on 2D, non-separable data with labels $y_i \in \{-1, +1\}$.

---

## The Primal Problem

Given training data $\{(\mathbf{x}_i, y_i)\}_{i=1}^{n}$ with $\mathbf{x}_i \in \mathbb{R}^2$, we solve:

$$\min_{\mathbf{w},\, b,\, \boldsymbol{\xi}} \quad \frac{1}{2}\|\mathbf{w}\|^2 + C \sum_{i=1}^{n} \xi_i$$

$$\text{subject to} \quad y_i(\mathbf{w}^\top \mathbf{x}_i + b) \geq 1 - \xi_i, \quad \xi_i \geq 0 \quad \forall\, i$$

The slack variables $\xi_i$ measure the degree of margin violation:

| Value | Interpretation |
|---|---|
| $\xi_i = 0$ | Correctly classified, outside or on the margin |
| $0 < \xi_i \leq 1$ | Inside the margin, but correct side |
| $\xi_i > 1$ | Misclassified |

---

## The Wolfe Dual

Introducing Lagrange multipliers $\alpha_i \geq 0$ for the margin constraints and $\mu_i \geq 0$ for the non-negativity constraints on $\xi_i$, the Lagrangian is:

$$\mathcal{L} = \frac{1}{2}\|\mathbf{w}\|^2 + C\sum_i \xi_i - \sum_i \alpha_i \left[y_i(\mathbf{w}^\top \mathbf{x}_i + b) - 1 + \xi_i\right] - \sum_i \mu_i \xi_i$$

Setting partial derivatives to zero yields the stationarity conditions:

$$\frac{\partial \mathcal{L}}{\partial \mathbf{w}} = 0 \implies \mathbf{w} = \sum_i \alpha_i y_i \mathbf{x}_i$$

$$\frac{\partial \mathcal{L}}{\partial b} = 0 \implies \sum_i \alpha_i y_i = 0$$

$$\frac{\partial \mathcal{L}}{\partial \xi_i} = 0 \implies \alpha_i + \mu_i = C$$

Since $\mu_i \geq 0$, the third condition gives the **box constraint** $0 \leq \alpha_i \leq C$. Substituting back into $\mathcal{L}$ yields the **dual problem**:

$$\max_{\boldsymbol{\alpha}} \quad \sum_i \alpha_i - \frac{1}{2} \sum_{i,j} \alpha_i \alpha_j y_i y_j \, k(\mathbf{x}_i, \mathbf{x}_j)$$

$$\text{subject to} \quad 0 \leq \alpha_i \leq C, \quad \sum_i \alpha_i y_i = 0$$

This is a **convex quadratic program** in $\boldsymbol{\alpha}$, solved here via `cvxopt`.

### KKT Conditions & Support Vector Types

The KKT complementary slackness conditions produce three distinct cases:

| Condition | Type | Interpretation |
|---|---|---|
| $\alpha_i = 0$ | Non-support vector | Outside the margin |
| $0 < \alpha_i < C$ | Free support vector | On the margin; used to recover $b$ |
| $\alpha_i = C$ | Bound support vector | Inside or beyond the margin |

### Recovering the Bias

The bias $b$ is recovered by averaging over all **free support vectors** (those with $0 < \alpha_i < C$), which are guaranteed to satisfy the margin constraint with equality:

$$b = \frac{1}{|\mathcal{S}_{\text{free}}|} \sum_{i \in \mathcal{S}_{\text{free}}} \left( y_i - \sum_j \alpha_j y_j \, k(\mathbf{x}_j, \mathbf{x}_i) \right)$$

### Prediction

The decision function for a new point $\mathbf{x}$ is:

$$f(\mathbf{x}) = \text{sign}\!\left(\sum_i \alpha_i y_i \, k(\mathbf{x}_i, \mathbf{x}) + b\right)$$

---

## The Kernel Trick

The dual depends only on **dot products** $\mathbf{x}_i^\top \mathbf{x}_j$. Replacing these with a kernel function $k(\mathbf{x}_i, \mathbf{x}_j) = \phi(\mathbf{x}_i)^\top \phi(\mathbf{x}_j)$ implicitly maps data into a higher-dimensional feature space $\mathcal{F}$ without ever computing $\phi$ explicitly.

### Supported Kernels

**Linear:**

$$k(\mathbf{x}, \mathbf{z}) = \mathbf{x}^\top \mathbf{z}$$

**RBF (Gaussian)** — default for non-separable data:

$$k(\mathbf{x}, \mathbf{z}) = \exp\!\left(-\gamma \|\mathbf{x} - \mathbf{z}\|^2\right)$$

The RBF kernel corresponds to an **infinite-dimensional** feature map, yet evaluates in $O(n)$ time. The bandwidth parameter $\gamma = \frac{1}{2\sigma^2}$ controls locality: large $\gamma$ gives sharp local boundaries; small $\gamma$ gives smoother, more global boundaries.

A kernel $k$ is valid if and only if the Gram matrix $K_{ij} = k(\mathbf{x}_i, \mathbf{x}_j)$ is **positive semi-definite** for any finite set of points (Mercer's theorem).

---

## Implementation Overview

```
SoftMarginSVM
├── __init__(data_path)       # Load dataset from .xlsx
├── _load_data(data_path)     # Parse into DataFrame with columns x_1, x_2, class
├── print_df_info()           # Display head and describe()
│
├── _kernel(x, z)             # Scalar kernel evaluation: linear or RBF
├── _build_gram_matrix(X)     # Precompute n×n matrix K_ij = k(x_i, x_j)
│
├── fit(X, y)                 # Solve QP via cvxopt, extract alphas, compute b
├── _recover_bias(X, y)       # Average over free support vectors
│
├── predict(X)                # Returns array of ±1
└── score(X, y)               # Returns float accuracy
```

### QP Matrix Mapping

`cvxopt` solves $\min \frac{1}{2}\boldsymbol{\alpha}^\top P \boldsymbol{\alpha} + \mathbf{q}^\top \boldsymbol{\alpha}$ subject to $G\boldsymbol{\alpha} \leq \mathbf{h}$, $A\boldsymbol{\alpha} = \mathbf{b}$. The dual maps as:

| `cvxopt` variable | Value |
|---|---|
| $P$ | $y_i y_j K_{ij}$ |
| $\mathbf{q}$ | $-\mathbf{1} \in \mathbb{R}^n$ |
| $G$ | $[-I \;\; I]^\top \in \mathbb{R}^{2n \times n}$ |
| $\mathbf{h}$ | $[\mathbf{0} \;\; C\cdot\mathbf{1}]^\top \in \mathbb{R}^{2n}$ |
| $A$ | $\mathbf{y}^\top \in \mathbb{R}^{1 \times n}$ |
| $\mathbf{b}$ | $0$ |

A small jitter $\epsilon I$ (where $\epsilon = 10^{-8}$) is added to $P$ to ensure positive semi-definiteness under floating point.

---

## Hyperparameters

| Parameter | Description | Tested Values |
|---|---|---|
| $C > 0$ | Margin-violation tradeoff | `0.1`, `100` |
| `kernel` | Kernel function | `'linear'`, `'rbf'` |
| $\gamma > 0$ | RBF bandwidth ($\gamma = 1/2\sigma^2$) | tunable |

**Effect of $C$:**

- **Large $C$** (e.g. $C = 100$): heavily penalizes violations $\Rightarrow$ smaller margin, fewer misclassifications, more sensitive to noise
- **Small $C$** (e.g. $C = 0.1$): tolerates violations $\Rightarrow$ larger margin, more misclassifications, better generalization on noisy data

---

## Usage

```python
from pathlib import Path
from svm import SoftMarginSVM

# Initialize and inspect data
svm = SoftMarginSVM(data_path=Path("Proj2&3DataSet.xlsx"))
svm.print_df_info()

# Prepare inputs (labels must be ±1)
X = svm.inputs.to_numpy()
y = svm.targets.to_numpy()  # ensure {-1, +1}

# Fit with C = 0.1
svm.fit(X, y, C=0.1, kernel='rbf', gamma=0.5)
print(f"Accuracy (C=0.1): {svm.score(X, y):.4f}")

# Fit with C = 100
svm.fit(X, y, C=100, kernel='rbf', gamma=0.5)
print(f"Accuracy (C=100): {svm.score(X, y):.4f}")
```

---

## Benchmarking

Training time is measured against `sklearn.svm.SVC` (LIBSVM backend, SMO algorithm) at increasing dataset sizes $n$.

**Subsampling schedule:** $n \in \{50, 100, 200, 400, \ldots, N_{\text{full}}\}$

**Complexity comparison:**

| Implementation | Solver | Expected Complexity |
|---|---|---|
| This implementation | `cvxopt` dense QP | $O(n^3)$ |
| `sklearn.svm.SVC` | LIBSVM (SMO) | $O(n^2)$ in practice |

Results are plotted on a **log-log scale** of wall-clock training time vs. $n$. The slope of each line approximates the polynomial exponent, making the scaling difference visually clear.

---

## Dependencies

```
numpy
pandas
cvxopt
scikit-learn
openpyxl
matplotlib
```

Install via:

```bash
pip install numpy pandas cvxopt scikit-learn openpyxl matplotlib
```
