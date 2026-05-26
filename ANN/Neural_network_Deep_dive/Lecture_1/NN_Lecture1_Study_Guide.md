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

# 🧠 Neural Networks Study Guide: Lecture 1 (Introduction & Foundations)

This study guide summarizes the biological foundations, artificial neuron modeling, activation functions, structural architectures (Perceptrons, Multi-Layer Feed-Forward Networks, and Recurrent Networks), logical gate realizations, and learning algorithms based on the chronological sequence of the Lecture 1 slides.

---

## 📌 Lecture Agenda
1.  **Biological Background**: Overview of biological neurons.
2.  **Artificial Neuron**: Structural elements mapping biology to computation.
3.  **Classes of Neural Networks**:
    *   Perceptrons
    *   Multi-Layered Feed-Forward Networks
    *   Recurrent Networks
4.  **Modeling the Neuron**: Mathematical notations and weighted summation.
5.  **Activation Functions**: Step, Sign, Sigmoid, Identity, and ReLU.
6.  **Linear Separability & Logical Functions**: AND, OR, and XOR gate representations.
7.  **The Learning Cycle**: Error calculation and weight update rule.

---

## 1. Biological Background & ANN Ingredients

Artificial neural networks are inspired by the cell structures and communication methods of the human brain.

### The Biological Neuron Components
*   **Dendrites**: Branching input structures that receive incoming chemical/electrical signals from neighboring cells.
*   **Cell Body (Soma)**: The nucleus and center of the cell, which accumulates and processes incoming signals.
*   **Axon**: A single long cable-like structure wrapped in a insulating **Myelin sheath** that carries action potentials (electrical output) away from the cell body.
*   **Synapses**: Terminals at the end of axon branches that transmit the output signals to the dendrites of the next neurons.

### Mapping Biology to Artificial Neural Networks (ANN)

| Biological Component | Artificial Model Component | Functional Description |
| :--- | :--- | :--- |
| **Dendrites** | Input Features ($x_1, x_2, \dots, x_n$) | Receives features or output activations from preceding layers. |
| **Synapses** | Weights ($w_{i1}, w_{i2}, \dots, w_{in}$) | Scales the input signals, representing the connection strength. |
| **Soma (Cell Body)** | Summation Node ($\sum x_j w_{ij}$) | Aggregates all scaled inputs to calculate a total net input. |
| **Firing Threshold** | Activation Function ($g$) | Determines the firing output based on the input threshold. |
| **Axon / Terminal** | Output Activation ($y_i$) | Transmits the output activation value to the next layer. |

---

## 2. Modeling & Simulation of an Artificial Neuron

An artificial neuron acts as a mathematical processing unit that computes a weighted sum of its inputs, adds a bias (or threshold), and applies an activation function.

### Mathematical Definitions
*   $a_j$: The activation value of a preceding unit $j$ (input signal).
*   $w_{j,i}$: The connection weight from unit $j$ to the current unit $i$.
*   $in_i$: The total weighted sum of inputs entering unit $i$:
    $$in_i = \sum_j W_{j,i} a_j$$
*   $g$: The activation function applied to the net input.
*   $a_i$: The final activation value (output) of unit $i$:
    $$a_i = g(in_i)$$

### Artificial Neuron Flowchart

```mermaid
graph TD
    X1["Input a₁"] -->|Weight W_{1,i}| Sum["Summation Node (Σ) <br> in_i = Σ W_{j,i} aⱼ"]
    X2["Input a₂"] -->|Weight W_{2,i}| Sum
    Xn["Input aₙ"] -->|Weight W_{n,i}| Sum
    Sum --> Act["Activation Function g"]
    Act --> Out["Output (a_i)"]
    
    style Sum fill:#f5faff,stroke:#1565c0,stroke-width:2px
    style Act fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
    style Out fill:#fff3e0,stroke:#e65100,stroke-width:2px
```

---

## 3. Activation Functions

Activation functions inject non-linear capabilities into neural networks. Five primary activation functions are defined in this lecture:

### 1. Step Function
Outputs a binary value based on a fixed threshold $t$.
$$\text{Step}_t(x) = \begin{cases} 1 & \text{if } x \geq t \\ 0 & \text{if } x < t \end{cases}$$

### 2. Sign Function
Outputs a bipolar step output symmetric around zero.
$$\text{Sign}(x) = \begin{cases} +1 & \text{if } x \geq 0 \\ -1 & \text{if } x < 0 \end{cases}$$

### 3. Sigmoid Function
A smooth, continuous, S-shaped curve bounded strictly between $0$ and $1$.
$$\text{Sigmoid}(x) = \frac{1}{1 + e^{-x}}$$

### 4. Identity Function
Outputs the input value directly without changes (linear activation).
$$f(x) = x$$

### 5. ReLU (Rectified Linear Unit)
A piecewise linear function that outputs zero for negative inputs, and acts as an identity function for positive values.
$$\text{ReLU}(x) = \max(0, x)$$

---

## 4. Classes of Neural Network Architectures

Neural networks are grouped into three structural classes based on their layers and feed-forward or recurrent connections:

### Class I: Perceptrons
A Perceptron is a single-layer neural network model with inputs directly connected to the output node.
*   **Formula**: Labeled mathematically as:
    $$a = g(-W_0 + W_1 a_1 + W_2 a_2)$$
    Here, $-W_0$ is the bias (also termed negative threshold), and $g$ is a binary step activation function ($g(in) = 1$ if $in > 0$ else $0$).

### Class II: Multi-Layer Feed-Forward Networks
Consists of multiple layers of units where connections flow strictly forward from inputs, through hidden layers, to outputs.
*   **Layers**: Contains an **Input layer**, one or more **Hidden layers**, and an **Output layer**.
*   **Flow**: Output links of one layer connect only to input links of the next layer (no backwards loops or connections within the same layer).
*   **Representational Capacity**: Addition of hidden layers enables multi-layer networks to represent complex, non-linear functions.
*   **Example Architecture Layout (Slide 12)**:
    *   **Input**: $4$ features (represented as input size $[N, 4]$).
    *   **Hidden Layer 1**: $5$ units (weights $W_1$ of size $[4, 5]$, activation function $f_1$).
    *   **Hidden Layer 2**: $7$ units (weights $W_2$ of size $[5, 7]$, activation function $f_2$).
    *   **Output Layer**: $3$ units (weights $W_o$ of size $[7, 3]$, output size $[N, 3]$).

### Class III: Recurrent Networks
A network where connection links can form loops, allowing outputs to flow back into previous layers or back into the same neuron.
*   Allows the network to preserve internal states (memory), making it suitable for temporal or sequential data streams (such as speech or handwriting recognition).

---

## 5. Logical Gate Realization & Linear Separability

Single-layer perceptrons can model basic logical gates, but their capability is limited by the linear separability of the logical functions.

### Truth Tables of Logic Gates

| Input A | Input B | AND Output | OR Output | XOR Output |
| :---: | :---: | :---: | :---: | :---: |
| 0 | 0 | 0 | 0 | 0 |
| 0 | 1 | 0 | 1 | 1 |
| 1 | 0 | 0 | 1 | 1 |
| 1 | 1 | 1 | 1 | 0 |

### 1. AND Gate Implementation
*   **Configuration**: Weights $w_1 = 1, w_2 = 1$, and threshold $t = 2$.
*   **Function**: $Y = \text{Step}_2(x_1 \cdot 1 + x_2 \cdot 1)$.
    *   $x = (1, 1) \to 1(1) + 1(1) = 2 \geq 2 \to Y = 1$
    *   $x = (1, 0) \to 1(1) + 0(1) = 1 < 2 \to Y = 0$
    *   $x = (0, 1) \to 0(1) + 1(1) = 1 < 2 \to Y = 0$
    *   $x = (0, 0) \to 0(1) + 0(1) = 0 < 2 \to Y = 0$

### 2. OR Gate Implementation
*   **Configuration**: Weights $w_1 = 2, w_2 = 2$, and threshold $t = 2$.
*   **Function**: $Y = \text{Step}_2(x_1 \cdot 2 + x_2 \cdot 2)$.
    *   $x = (1, 1) \to 1(2) + 1(2) = 4 \geq 2 \to Y = 1$
    *   $x = (1, 0) \to 1(2) + 0(2) = 2 \geq 2 \to Y = 1$
    *   $x = (0, 1) \to 0(2) + 1(2) = 2 \geq 2 \to Y = 1$
    *   $x = (0, 0) \to 0(2) + 0(2) = 0 < 2 \to Y = 0$

### 3. Linear Separability & XOR
*   **Definition**: A dataset is **linearly separable** if a single straight line (or hyperplane in higher dimensions) can perfectly separate the positive output classes (1) from the negative output classes (0).
*   **Perceptron Limit**: A single-layer Perceptron can only represent linearly separable functions (like AND and OR).
*   **The XOR Problem**: The XOR gate outputs $1$ for opposite inputs and $0$ for identical inputs. Plotting these classes in a 2D grid puts the 1s and 0s diagonally opposite each other, making it impossible to separate them with a single straight line.

#### Linear Separability Visual Comparison

<div style="width: 100%; text-align: center;">
<svg width="600" height="240" viewBox="0 0 600 240" xmlns="http://www.w3.org/2000/svg" style="display: inline-block; margin: 20px auto; max-width: 100%;">
<defs>
<marker id="arrow" viewBox="0 0 10 10" refX="5" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
<path d="M 0 0 L 10 5 L 0 10 z" fill="#888" />
</marker>
</defs>
<g transform="translate(0, 0)">
<text x="100" y="20" font-family="system-ui, -apple-system, sans-serif" font-size="13" font-weight="bold" text-anchor="middle" fill="#333">AND Gate</text>
<text x="100" y="34" font-family="system-ui, -apple-system, sans-serif" font-size="10" fill="#2e7d32" text-anchor="middle" font-weight="bold">Linearly Separable</text>
<line x1="40" y1="180" x2="170" y2="180" stroke="#888" stroke-width="1.5" marker-end="url(#arrow)" />
<line x1="40" y1="180" x2="40" y2="50" stroke="#888" stroke-width="1.5" marker-end="url(#arrow)" />
<text x="175" y="184" font-family="system-ui, -apple-system, sans-serif" font-size="10" fill="#666">x₁</text>
<text x="36" y="45" font-family="system-ui, -apple-system, sans-serif" font-size="10" fill="#666">x₂</text>
<text x="30" y="184" font-family="system-ui, -apple-system, sans-serif" font-size="10" fill="#888">0</text>
<text x="140" y="195" font-family="system-ui, -apple-system, sans-serif" font-size="10" fill="#888">1</text>
<line x1="140" y1="180" x2="140" y2="184" stroke="#888" stroke-width="1" />
<text x="28" y="84" font-family="system-ui, -apple-system, sans-serif" font-size="10" fill="#888">1</text>
<line x1="36" y1="80" x2="40" y2="80" stroke="#888" stroke-width="1" />
<line x1="90" y1="50" x2="170" y2="130" stroke="#f57c00" stroke-width="2" stroke-dasharray="4 3" />
<text x="150" y="75" font-family="system-ui, -apple-system, sans-serif" font-size="9" fill="#f57c00" font-weight="bold">Boundary</text>
<circle cx="40" cy="180" r="7" fill="#e53935" stroke="#b71c1c" stroke-width="1.5" />
<text x="40" y="171" font-family="system-ui, -apple-system, sans-serif" font-size="9" text-anchor="middle" font-weight="bold" fill="#333">0</text>
<circle cx="140" cy="180" r="7" fill="#e53935" stroke="#b71c1c" stroke-width="1.5" />
<text x="140" y="171" font-family="system-ui, -apple-system, sans-serif" font-size="9" text-anchor="middle" font-weight="bold" fill="#333">0</text>
<circle cx="40" cy="80" r="7" fill="#e53935" stroke="#b71c1c" stroke-width="1.5" />
<text x="40" y="71" font-family="system-ui, -apple-system, sans-serif" font-size="9" text-anchor="middle" font-weight="bold" fill="#333">0</text>
<circle cx="140" cy="80" r="7" fill="#4caf50" stroke="#1b5e20" stroke-width="1.5" />
<text x="140" y="71" font-family="system-ui, -apple-system, sans-serif" font-size="9" text-anchor="middle" font-weight="bold" fill="#333">1</text>
</g>
<g transform="translate(200, 0)">
<text x="100" y="20" font-family="system-ui, -apple-system, sans-serif" font-size="13" font-weight="bold" text-anchor="middle" fill="#333">OR Gate</text>
<text x="100" y="34" font-family="system-ui, -apple-system, sans-serif" font-size="10" fill="#2e7d32" text-anchor="middle" font-weight="bold">Linearly Separable</text>
<line x1="40" y1="180" x2="170" y2="180" stroke="#888" stroke-width="1.5" marker-end="url(#arrow)" />
<line x1="40" y1="180" x2="40" y2="50" stroke="#888" stroke-width="1.5" marker-end="url(#arrow)" />
<text x="175" y="184" font-family="system-ui, -apple-system, sans-serif" font-size="10" fill="#666">x₁</text>
<text x="36" y="45" font-family="system-ui, -apple-system, sans-serif" font-size="10" fill="#666">x₂</text>
<text x="30" y="184" font-family="system-ui, -apple-system, sans-serif" font-size="10" fill="#888">0</text>
<text x="140" y="195" font-family="system-ui, -apple-system, sans-serif" font-size="10" fill="#888">1</text>
<line x1="140" y1="180" x2="140" y2="184" stroke="#888" stroke-width="1" />
<text x="28" y="84" font-family="system-ui, -apple-system, sans-serif" font-size="10" fill="#888">1</text>
<line x1="36" y1="80" x2="40" y2="80" stroke="#888" stroke-width="1" />
<line x1="20" y1="130" x2="90" y2="200" stroke="#f57c00" stroke-width="2" stroke-dasharray="4 3" />
<text x="25" y="150" font-family="system-ui, -apple-system, sans-serif" font-size="9" fill="#f57c00" font-weight="bold">Boundary</text>
<circle cx="40" cy="180" r="7" fill="#e53935" stroke="#b71c1c" stroke-width="1.5" />
<text x="40" y="171" font-family="system-ui, -apple-system, sans-serif" font-size="9" text-anchor="middle" font-weight="bold" fill="#333">0</text>
<circle cx="140" cy="180" r="7" fill="#4caf50" stroke="#1b5e20" stroke-width="1.5" />
<text x="140" y="171" font-family="system-ui, -apple-system, sans-serif" font-size="9" text-anchor="middle" font-weight="bold" fill="#333">1</text>
<circle cx="40" cy="80" r="7" fill="#4caf50" stroke="#1b5e20" stroke-width="1.5" />
<text x="40" y="71" font-family="system-ui, -apple-system, sans-serif" font-size="9" text-anchor="middle" font-weight="bold" fill="#333">1</text>
<circle cx="140" cy="80" r="7" fill="#4caf50" stroke="#1b5e20" stroke-width="1.5" />
<text x="140" y="71" font-family="system-ui, -apple-system, sans-serif" font-size="9" text-anchor="middle" font-weight="bold" fill="#333">1</text>
</g>
<g transform="translate(400, 0)">
<text x="100" y="20" font-family="system-ui, -apple-system, sans-serif" font-size="13" font-weight="bold" text-anchor="middle" fill="#333">XOR Gate</text>
<text x="100" y="34" font-family="system-ui, -apple-system, sans-serif" font-size="10" fill="#c62828" text-anchor="middle" font-weight="bold">Non-Linearly Separable</text>
<line x1="40" y1="180" x2="170" y2="180" stroke="#888" stroke-width="1.5" marker-end="url(#arrow)" />
<line x1="40" y1="180" x2="40" y2="50" stroke="#888" stroke-width="1.5" marker-end="url(#arrow)" />
<text x="175" y="184" font-family="system-ui, -apple-system, sans-serif" font-size="10" fill="#666">x₁</text>
<text x="36" y="45" font-family="system-ui, -apple-system, sans-serif" font-size="10" fill="#666">x₂</text>
<text x="30" y="184" font-family="system-ui, -apple-system, sans-serif" font-size="10" fill="#888">0</text>
<text x="140" y="195" font-family="system-ui, -apple-system, sans-serif" font-size="10" fill="#888">1</text>
<line x1="140" y1="180" x2="140" y2="184" stroke="#888" stroke-width="1" />
<text x="28" y="84" font-family="system-ui, -apple-system, sans-serif" font-size="10" fill="#888">1</text>
<line x1="36" y1="80" x2="40" y2="80" stroke="#888" stroke-width="1" />
<line x1="40" y1="180" x2="140" y2="80" stroke="#ddd" stroke-width="1" stroke-dasharray="2 2" />
<line x1="40" y1="80" x2="140" y2="180" stroke="#ddd" stroke-width="1" stroke-dasharray="2 2" />
<circle cx="40" cy="180" r="7" fill="#e53935" stroke="#b71c1c" stroke-width="1.5" />
<text x="40" y="171" font-family="system-ui, -apple-system, sans-serif" font-size="9" text-anchor="middle" font-weight="bold" fill="#333">0</text>
<circle cx="140" cy="180" r="7" fill="#4caf50" stroke="#1b5e20" stroke-width="1.5" />
<text x="140" y="171" font-family="system-ui, -apple-system, sans-serif" font-size="9" text-anchor="middle" font-weight="bold" fill="#333">1</text>
<circle cx="40" cy="80" r="7" fill="#4caf50" stroke="#1b5e20" stroke-width="1.5" />
<text x="40" y="71" font-family="system-ui, -apple-system, sans-serif" font-size="9" text-anchor="middle" font-weight="bold" fill="#333">1</text>
<circle cx="140" cy="80" r="7" fill="#e53935" stroke="#b71c1c" stroke-width="1.5" />
<text x="140" y="71" font-family="system-ui, -apple-system, sans-serif" font-size="9" text-anchor="middle" font-weight="bold" fill="#333">0</text>
</g>
<g transform="translate(180, 220)">
<circle cx="10" cy="10" r="5" fill="#4caf50" stroke="#1b5e20" stroke-width="1" />
<text x="20" y="13" font-family="system-ui, -apple-system, sans-serif" font-size="10" fill="#333">Output = 1 (True)</text>
<circle cx="140" cy="10" r="5" fill="#e53935" stroke="#b71c1c" stroke-width="1" />
<text x="150" y="13" font-family="system-ui, -apple-system, sans-serif" font-size="10" fill="#333">Output = 0 (False)</text>
</g>
</svg>
</div>

---

## 6. The Learning Process

Training a neural network involves an iterative feedback loop where weights are adjusted dynamically based on output errors.

### Pseudocode Algorithm (Slide 19)
```text
While epoch produces an error:
    Present network with next inputs from epoch
    Error = Target (T) - Output (O)
    If Error != 0 then:
        Wⱼ = Wⱼ + LR * Iⱼ * Error
    End If
End While
```

### Component Breakdown
*   **epoch**: A single complete pass over all training datasets through the neural network.
*   **Target (T)**: The ground-truth (actual) expected output value.
*   **Output (O)**: The current predicted output generated by the network.
*   **Error (Err)**: The difference between target and prediction ($T - O$).
*   **LR (Learning Rate)**: A small positive scalar coefficient controlling the adjustment step size.
*   **$I_j$**: The input value of feature $j$.
*   **$W_j$**: The connection weight associated with input feature $j$.
*   **Weight Update Formula**:
    $$W_{j,\text{new}} = W_{j,\text{old}} + \alpha \cdot \text{Input}_j \cdot \text{Error}$$
    where $\alpha$ is the learning rate.
