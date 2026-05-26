<style>
/* CSS Styling for A4 Print Layout */
@media print {
  @page {
    size: A4;
    margin: 20mm 15mm 20mm 15mm;
  }
  body {
    font-family: system-ui, -apple-system, sans-serif;
    font-size: 11pt;
    line-height: 1.5;
    color: #000 !important;
  }
  h1, h2, h3, h4 {
    page-break-after: avoid;
    break-after: avoid;
    color: #000 !important;
  }
  pre, blockquote, table, img, svg, .mermaid {
    page-break-inside: avoid;
    break-inside: avoid;
    max-width: 100% !important;
  }
}
</style>

# 🧠 Neural Networks Study Guide: Lecture 1 - Part 2 (Perceptron Learning & Delta Rule)

This study guide details the classification boundaries, learning algorithms, trace tables, and mathematical update rules for Single-Layer Perceptrons (SLP) using both the classical **Perceptron Learning Algorithm** and the **Delta Rule**.

---

## 1. Single-Layer Perceptron (SLP) & Decision Boundaries

A Single-Layer Perceptron maps inputs directly to one or more output units to perform linear classification.

### Architecture with Multiple Outputs
When a classification task involves multiple output classes, inputs are mapped directly to multiple output units ($\Omega_1, \Omega_2, \Omega_3$).

<div style="width: 100%; text-align: center;">
<svg width="450" height="200" viewBox="0 0 450 200" xmlns="http://www.w3.org/2000/svg" style="display: inline-block; margin: 15px auto; max-width: 100%;">
<line x1="80" y1="30" x2="350" y2="40" stroke="#bbb" stroke-width="1" />
<line x1="80" y1="65" x2="350" y2="40" stroke="#bbb" stroke-width="1" />
<line x1="80" y1="100" x2="350" y2="40" stroke="#bbb" stroke-width="1" />
<line x1="80" y1="135" x2="350" y2="40" stroke="#bbb" stroke-width="1" />
<line x1="80" y1="170" x2="350" y2="40" stroke="#bbb" stroke-width="1" />
<line x1="80" y1="30" x2="350" y2="100" stroke="#888" stroke-width="1.2" />
<line x1="80" y1="65" x2="350" y2="100" stroke="#888" stroke-width="1.2" />
<line x1="80" y1="100" x2="350" y2="100" stroke="#888" stroke-width="1.2" />
<line x1="80" y1="135" x2="350" y2="100" stroke="#888" stroke-width="1.2" />
<line x1="80" y1="170" x2="350" y2="100" stroke="#888" stroke-width="1.2" />
<line x1="80" y1="30" x2="350" y2="160" stroke="#bbb" stroke-width="1" />
<line x1="80" y1="65" x2="350" y2="160" stroke="#bbb" stroke-width="1" />
<line x1="80" y1="100" x2="350" y2="160" stroke="#bbb" stroke-width="1" />
<line x1="80" y1="135" x2="350" y2="160" stroke="#bbb" stroke-width="1" />
<line x1="80" y1="170" x2="350" y2="160" stroke="#bbb" stroke-width="1" />
<circle cx="80" cy="30" r="14" fill="#e3f2fd" stroke="#1e88e5" stroke-width="2" />
<text x="80" y="34" font-family="system-ui, -apple-system, sans-serif" font-size="11" text-anchor="middle" font-weight="bold" fill="#1565c0">i₁</text>
<circle cx="80" cy="65" r="14" fill="#e3f2fd" stroke="#1e88e5" stroke-width="2" />
<text x="80" y="69" font-family="system-ui, -apple-system, sans-serif" font-size="11" text-anchor="middle" font-weight="bold" fill="#1565c0">i₂</text>
<circle cx="80" cy="100" r="14" fill="#e3f2fd" stroke="#1e88e5" stroke-width="2" />
<text x="80" y="104" font-family="system-ui, -apple-system, sans-serif" font-size="11" text-anchor="middle" font-weight="bold" fill="#1565c0">i₃</text>
<circle cx="80" cy="135" r="14" fill="#e3f2fd" stroke="#1e88e5" stroke-width="2" />
<text x="80" y="139" font-family="system-ui, -apple-system, sans-serif" font-size="11" text-anchor="middle" font-weight="bold" fill="#1565c0">i₄</text>
<circle cx="80" cy="170" r="14" fill="#e3f2fd" stroke="#1e88e5" stroke-width="2" />
<text x="80" y="174" font-family="system-ui, -apple-system, sans-serif" font-size="11" text-anchor="middle" font-weight="bold" fill="#1565c0">i₅</text>
<circle cx="350" cy="40" r="16" fill="#f3e5f5" stroke="#8e24aa" stroke-width="2" />
<text x="350" y="44" font-family="system-ui, -apple-system, sans-serif" font-size="11" text-anchor="middle" font-weight="bold" fill="#6a1b9a">Ω₁</text>
<circle cx="350" cy="100" r="16" fill="#f3e5f5" stroke="#8e24aa" stroke-width="2" />
<text x="350" y="104" font-family="system-ui, -apple-system, sans-serif" font-size="11" text-anchor="middle" font-weight="bold" fill="#6a1b9a">Ω₂</text>
<circle cx="350" cy="160" r="16" fill="#f3e5f5" stroke="#8e24aa" stroke-width="2" />
<text x="350" y="164" font-family="system-ui, -apple-system, sans-serif" font-size="11" text-anchor="middle" font-weight="bold" fill="#6a1b9a">Ω₃</text>
<text x="80" y="12" font-family="system-ui, -apple-system, sans-serif" font-size="10" text-anchor="middle" fill="#666">Inputs</text>
<text x="350" y="18" font-family="system-ui, -apple-system, sans-serif" font-size="10" text-anchor="middle" fill="#666">Outputs</text>
</svg>
</div>

### Decision Boundary Dimensionality
The geometric shape of the decision boundary is directly determined by the dimensionality of the input feature space:

1.  **1D Space (Single Feature $x$)**:
    *   The decision boundary is a **single point** along the number line.
    *   **Linearly Separable Case**:
        ```text
        Number Line:  ---[ Blue (-0.5) ]---[ Blue (1.1) ]---[ Blue (1.2) ]---[ Blue (2.6) ]---| Boundary Point |---[ Red (2.8) ]---[ Red (3.4) ]---[ Red (4.0) ]--->
        ```
        A single threshold point splits the classes.
    *   **Non-Separable Case (Need Multiple Boundaries)**:
        *   Pattern `Blue, Blue, Red, Blue`: Requires at least 2 separator boundaries; a single-layer perceptron (SLP) cannot represent this.
        *   Pattern `Blue, Blue, Blue, Red, Blue, Red, Red`: Requires at least 3 separator boundaries; cannot be solved by an SLP.

2.  **2D Space (Two Features $x_1, x_2$)**:
    *   The decision boundary is a **straight line**.
    *   **Linearly Separable Dataset Example**:
        *   *Target 0 (Blue)*: $(2, 3), (3, 4), (7, 2)$
        *   *Target 1 (Red)*: $(-3, 3), (-1, 2), (0, 1), (0, -2)$
        *   *Solution*: A single diagonal line separating the red cluster on the left from the blue cluster on the right.
    *   **Non-Linearly Separable Dataset Example** (Adding coordinates $(3, 8)$ and $(7, 5)$ as Target class 1):
        *   Adding these points forces the red and blue clusters to overlap, making linear separation impossible; an SLP cannot resolve this.

3.  **3D Space (Three Features)**:
    *   The decision boundary is a **2D flat plane**.

4.  **Higher Dimensional Space ($N > 3$ Features)**:
    *   The decision boundary is an **$(N-1)$-dimensional hyperplane**.

---

## 2. The Perceptron Learning Algorithm

The Perceptron Learning Algorithm is a supervised learning method for binary classifiers. It adjusts connection weights only when the network makes an incorrect prediction.

### Algorithm Pseudocode
```text
While there exists a training pattern p in P where the error is too large:
    Input pattern p into the network and compute output y
    For each output neuron Ω:
        If y_Ω == t_Ω:
            Output is correct; do nothing
        Else:
            If y_Ω == 0:
                For all input neurons i:
                    w_{i,Ω} := w_{i,Ω} + o_i   (Increase weight by input value)
            If y_Ω == 1:
                For all input neurons i:
                    w_{i,Ω} := w_{i,Ω} - o_i   (Decrease weight by input value)
```
Where:
*   $y_\Omega$: Calculated network output for unit $\Omega$.
*   $t_\Omega$: Target (ground-truth) output class ($0$ or $1$).
*   $o_i$: Input value from neuron $i$.

### Decision Rule
The prediction is determined by:
$$y = \begin{cases} 1 & \text{if } net \geq \theta \\ 0 & \text{if } net < \theta \end{cases}$$
Where $net = w_1 x_1 + w_2 x_2 + w_{\text{bias}} \cdot \text{bias}$, and $\theta$ is the threshold.

### Weight Update Trace Table (Step-by-Step)
Given initial weights $w_1 = 0.1, w_2 = 0.2, w_{\text{bias}} = -0.2$, inputs $x_1, x_2$, bias value $1$, threshold $\theta = 0.1$, and target output $t$:

| Epoch | $x_1$ | $x_2$ | bias | $w_1$ | $w_2$ | $w_{\text{bias}}$ | $net$ | Output $y$ | Target $t$ | Action & Weight Adjustment |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **1** | 0 | 0 | 1 | 0.1 | 0.2 | -0.2 | -0.2 | 0 | 0 | Correct ($y=t$). No change. |
| **1** | 0 | 1 | 1 | 0.1 | 0.2 | -0.2 | 0.0 | 0 | 1 | **Incorrect** ($y=0, t=1$). Update weights:<br>$w_1 := 0.1 + 0 = 0.1$<br>$w_2 := 0.2 + 1 = 1.2$<br>$w_{\text{bias}} := -0.2 + 1 = 0.8$ |
| **1** | 1 | 0 | 1 | 0.1 | 1.2 | 0.8 | 0.9 | 1 | 1 | Correct ($y=t$). No change. |
| **1** | 1 | 1 | 1 | 0.1 | 1.2 | 0.8 | 2.1 | 1 | 1 | Correct ($y=t$). No change. |

*Note: After one complete epoch, the new weights are $w_1 = 1.1, w_2 = 1.2, w_{\text{bias}} = -0.2$ (following final updates at the start of Epoch 2).*

---

## 3. Single-Layer Perceptron using the Delta Rule

The Delta Rule (also known as the Adaline rule or Widrow-Hoff rule) updates the weights continuously based on the difference between the target and calculated output. Unlike the binary perceptron rule, it incorporates a **learning rate ($\eta$)** to control the step size of learning.

### Mathematical Formulation
The general weight update formula for input neuron $i$ mapping to output unit $\Omega$ is:
$$w_{i,\Omega} := w_{i,\Omega} + \eta \cdot o_i \cdot (t_\Omega - y_\Omega)$$

For a standard 2-input network with a bias:
$$w_1 := w_1 + \eta \cdot x_1 \cdot (t - y)$$
$$w_2 := w_2 + \eta \cdot x_2 \cdot (t - y)$$
$$w_{\text{bias}} := w_{\text{bias}} + \eta \cdot \text{bias} \cdot (t - y)$$
Where:
*   $\eta$: Learning rate (e.g., $0.1$).
*   $(t - y)$: The prediction error.

### Weight Update Trace Table (Delta Rule)
Given learning rate $\eta = 0.1$, initial weights $w_1 = 0.1, w_2 = 0.2, w_{\text{bias}} = -0.2$, inputs $x_1, x_2$, bias value $1$, and target output $t$:

| Epoch | $x_1$ | $x_2$ | bias | $w_1$ | $w_2$ | $w_{\text{bias}}$ | $net$ | Output $y$ | Target $t$ | Action & Step-by-Step Update |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **1** | 0 | 0 | 1 | 0.1 | 0.2 | -0.2 | -0.2 | 0 | 0 | Correct ($t-y = 0$). No change. |
| **1** | 0 | 1 | 1 | 0.1 | 0.2 | -0.2 | 0.0 | 0 | 1 | **Incorrect** ($t-y = 1$). Update weights:<br>$w_1 := 0.1 + 0.1 \cdot 0 \cdot 1 = 0.1$<br>$w_2 := 0.2 + 0.1 \cdot 1 \cdot 1 = 0.3$<br>$w_{\text{bias}} := -0.2 + 0.1 \cdot 1 \cdot 1 = -0.1$ |
| **1** | 1 | 0 | 1 | 0.1 | 0.3 | -0.1 | 0.0 | 0 | 1 | **Incorrect** ($t-y = 1$). Update weights:<br>$w_1 := 0.1 + 0.1 \cdot 1 \cdot 1 = 0.2$<br>$w_2 := 0.3 + 0.1 \cdot 0 \cdot 1 = 0.3$<br>$w_{\text{bias}} := -0.1 + 0.1 \cdot 1 \cdot 1 = 0.0$ |
| **1** | 1 | 1 | 1 | 0.2 | 0.3 | 0.0 | 0.5 | 1 | 1 | Correct ($t-y = 0$). No change. |
