# Assignment 1 — My Notes and Reflections

## Overview

This assignment focuses on understanding and implementing **automatic differentiation (AD)** — a fundamental tool used in modern machine learning frameworks.  

In machine learning, there are several ways to compute gradients, such as:
- Numerical differentiation (finite difference)
- Symbolic differentiation
- Automatic differentiation via computational graphs (what we implement here)

The first part of the assignment involves completing the core infrastructure of an automatic differentiation engine by implementing missing operators and evaluators in `auto_diff.py`.  

The second part asks us to implement logistic regression, stochastic gradient descent (SGD), and softmax classification using the AD framework — and to extend the operator set if needed.

---

## Part 1: Implementing Automatic Differentiation

### 🔧 What I Found Difficult

1. **Understanding the existing framework**

   It took some time to understand how the AD engine is structured — especially how `Node`, `Op`, and `Evaluator` classes interact.

2. **Matrix transformations and gradient flow**

   Implementing gradients for matrix-based operations (e.g., `MatMul`, `Sum`, etc.) was challenging, particularly in terms of ensuring the correct shapes and broadcasting behavior. Properly propagating gradients through the computation graph required careful debugging and validation.

---

## Part 2: Logistic Regression and Softmax Implementation

### 🔧 What I Found Difficult

1. **Writing a custom `softmax` operator**

   Since softmax wasn't provided, it had to be manually implemented using existing operators. This required translating the softmax equation into a sequence of computational steps that fit within the AD framework.

2. **Numerical instability in softmax**

   Training with a naive `exp(z)` implementation led to overflow issues and unstable behavior.  
   Although the assignment mentioned that numerical stability wasn't required, this still caused practical issues. A more stable formulation, such as using `log(sum(exp(z)))`, helped mitigate overflow in some cases.  
   Reference: [https://github.com/Sunxiaochuan256/CMU-MLSYS-hw0](https://github.com/Sunxiaochuan256/CMU-MLSYS-hw0)

---

## Still Unsure About

- Exploding Gradient Problem Solution other than numerically stable version

---

## Resources I Found Helpful

- [mlsyscourse.org/materials](https://mlsyscourse.org/materials)
- [Sunxiaochuan256/CMU-MLSYS-hw0 GitHub repo](https://github.com/Sunxiaochuan256/CMU-MLSYS-hw0)

---

💬 I'm open to feedback or discussion — feel free to open an issue or PR if you have suggestions!
