# Unit II — Feedforward Neural Networks

> **INT511 – Neural Networks** | M.Tech Level Notes
> **Coverage:** Multi-layer neural network · XOR problem · Backpropagation algorithm · Cost functions · Gradient descent · Overfitting · Regularization techniques

---

## Table of Contents

1. [Multi-Layer Neural Network — Architecture & Notation](#1-multi-layer-neural-network--architecture--notation)
2. [The XOR Problem](#2-the-xor-problem)
3. [Backpropagation — Complete Derivation](#3-backpropagation--complete-derivation)
4. [Backpropagation — Fully Worked Numerical](#4-backpropagation--fully-worked-numerical)
5. [Cost Functions](#5-cost-functions)
6. [Gradient Descent — Theory and Convergence](#6-gradient-descent--theory-and-convergence)
7. [Overfitting and the Bias–Variance Decomposition](#7-overfitting-and-the-biasvariance-decomposition)
8. [Regularization Techniques](#8-regularization-techniques)
9. [Weight Initialization (Xavier & He Derivations)](#9-weight-initialization-xavier--he-derivations)
10. [Practical Training Recipe & Debugging](#10-practical-training-recipe--debugging)
11. [Solved Numericals](#11-solved-numericals)
12. [Viva / Exam Pointers](#12-viva--exam-pointers)

---

## 1. Multi-Layer Neural Network — Architecture & Notation

### 1.1 Architecture

```
   INPUT          HIDDEN 1        HIDDEN 2         OUTPUT
   layer 0         layer 1         layer 2         layer L

   x₁ ●──────┐   ┌──► ● a₁⁽¹⁾ ──┐  ┌──► ● ──┐   ┌──► ● ŷ₁
             ├───┤              ├──┤        ├───┤
   x₂ ●──────┤   ├──► ● a₂⁽¹⁾ ──┤  ├──► ● ──┤   ├──► ● ŷ₂
             ├───┤              ├──┤        ├───┤
   x₃ ●──────┘   └──► ● a₃⁽¹⁾ ──┘  └──► ● ──┘   └──► ● ŷ₃
                     W⁽¹⁾,b⁽¹⁾      W⁽²⁾,b⁽²⁾     W⁽³⁾,b⁽³⁾
                     φ⁽¹⁾           φ⁽²⁾          φ⁽³⁾ (softmax/linear)

   ───────────────── FORWARD PASS  (activations) ─────────────────►
   ◄──────────────── BACKWARD PASS (deltas δ)  ────────────────────
```

### 1.2 Canonical notation (used throughout these notes)

| Symbol | Meaning | Shape |
|---|---|---|
| $L$ | number of layers (excluding input) | scalar |
| $n_\ell$ | width of layer $\ell$ | scalar |
| $\mathbf W^{(\ell)}$ | weights from layer $\ell{-}1$ to $\ell$ | $n_\ell \times n_{\ell-1}$ |
| $\mathbf b^{(\ell)}$ | biases of layer $\ell$ | $n_\ell \times 1$ |
| $\mathbf z^{(\ell)}$ | pre-activation (induced local field) | $n_\ell \times 1$ |
| $\mathbf a^{(\ell)}$ | activation, $=\varphi^{(\ell)}(\mathbf z^{(\ell)})$ | $n_\ell \times 1$ |
| $\boldsymbol\delta^{(\ell)}$ | error signal $\partial\mathcal L/\partial \mathbf z^{(\ell)}$ | $n_\ell \times 1$ |

**Forward recursion:**

$$
\mathbf a^{(0)} = \mathbf x, \qquad
\mathbf z^{(\ell)} = \mathbf W^{(\ell)}\mathbf a^{(\ell-1)} + \mathbf b^{(\ell)},\qquad
\mathbf a^{(\ell)} = \varphi^{(\ell)}\!\left(\mathbf z^{(\ell)}\right),\qquad
\hat{\mathbf y} = \mathbf a^{(L)} .
$$

Parameter count: $\;P = \sum_{\ell=1}^{L} \left(n_\ell n_{\ell-1} + n_\ell\right)$.

**Example:** a 784–256–128–10 MLP has
$784(256)+256 + 256(128)+128 + 128(10)+10 = 200\,960 + 32\,896 + 1\,290 = 235\,146$ parameters.

### 1.3 Batched form (what frameworks actually compute)

With a mini-batch $\mathbf X \in \mathbb{R}^{B\times n_0}$ (rows = samples):

$$
\mathbf Z^{(\ell)} = \mathbf A^{(\ell-1)}\mathbf W^{(\ell)\mathsf T} + \mathbf 1_B \mathbf b^{(\ell)\mathsf T},\qquad
\mathbf A^{(\ell)} = \varphi(\mathbf Z^{(\ell)}) .
$$

Forward FLOPs $\approx 2B\sum_\ell n_\ell n_{\ell-1}$; backward $\approx 2\times$ forward; so training $\approx 3\times$ inference cost per sample.

### 1.4 Why depth: the region-counting argument

A ReLU MLP is a **continuous piecewise-linear** function. Each hidden unit contributes a hyperplane that folds input space. The maximum number of linear regions for $L$ hidden layers of width $w$ on $n$ inputs (Montúfar et al. 2014):

$$
\mathcal R \;\ge\; \left(\prod_{\ell=1}^{L-1}\left\lfloor \frac{w}{n}\right\rfloor^{\,n}\right)\sum_{j=0}^{n}\binom{w}{j}
\;=\; \Omega\!\left(\left(\tfrac{w}{n}\right)^{n(L-1)} w^{n}\right).
$$

Regions grow **polynomially in width but exponentially in depth**. This is the precise sense in which "deep" beats "wide".

---

## 2. The XOR Problem

### 2.1 Statement and impossibility

XOR truth table with class labels:

```
      x₂
      │
    1 ●(0,1)→1        ○(1,1)→0
      │
      │
    0 ○(0,0)→0        ●(1,0)→1
      └──────────────────────── x₁
      0                1

   ● class 1   ○ class 0
   NO single straight line separates ● from ○
```

**Proof of non-separability** (see Unit I §N2 for the inequality version). Alternative *convex-hull* proof:
Class-0 points $\{(0,0),(1,1)\}$ and class-1 points $\{(0,1),(1,0)\}$ are the two **diagonals** of the unit square. Their convex hulls are the two diagonal line segments, which **intersect at $(\tfrac12,\tfrac12)$**. Two sets whose convex hulls intersect cannot be separated by a hyperplane (separating hyperplane theorem). ∎

Generalisation: XOR is the $n{=}2$ case of **parity**, $y = \bigoplus_i x_i$, which requires $\Omega(2^n)$ units at depth 2 but only $O(n)$ units at depth $O(\log n)$ (XOR-tree).

### 2.2 Solution 1 — threshold units (OR minus AND)

$$
h_1 = \Theta(x_1 + x_2 - 0.5) \;\;(\text{OR}),\qquad
h_2 = \Theta(x_1 + x_2 - 1.5) \;\;(\text{AND}),\qquad
y = \Theta(h_1 - h_2 - 0.5)
$$

```
              w=1
    x₁ ──────────────► ┌──────┐ h₁
        \      w=1     │θ=0.5 │───── +1 ──┐
         \  ┌──────────►└──────┘           │   ┌──────┐
          \/                               ├──►│θ=0.5 │──► y
          /\  w=1      ┌──────┐ h₂         │   └──────┘
         /  └──────────►│θ=1.5 │───── −1 ──┘
    x₂ ──────────────► └──────┘
              w=1
```

| $x_1$ | $x_2$ | $x_1{+}x_2$ | $h_1$ (OR) | $h_2$ (AND) | $h_1-h_2-0.5$ | $y$ | XOR |
|---|---|---|---|---|---|---|---|
| 0 | 0 | 0 | 0 | 0 | −0.5 | 0 | 0 ✓ |
| 0 | 1 | 1 | 1 | 0 | +0.5 | 1 | 1 ✓ |
| 1 | 0 | 1 | 1 | 0 | +0.5 | 1 | 1 ✓ |
| 1 | 1 | 2 | 1 | 1 | −0.5 | 0 | 0 ✓ |

**Geometric interpretation:** the hidden layer maps the four inputs to $\{(0,0),(1,0),(1,0),(1,1)\}$ — the two class-1 points **collapse onto the same hidden representation** $(1,0)$, and the problem becomes linearly separable in $h$-space. *Representation learning in its simplest form.*

### 2.3 Solution 2 — ReLU network (Goodfellow's construction)

$$
\mathbf h = \text{ReLU}\!\left(\mathbf W\mathbf x + \mathbf c\right),\quad
\mathbf W = \begin{pmatrix}1&1\\1&1\end{pmatrix},\;
\mathbf c=\begin{pmatrix}0\\-1\end{pmatrix};\qquad
y = \mathbf w^{\mathsf T}\mathbf h + b,\quad \mathbf w=\begin{pmatrix}1\\-2\end{pmatrix},\; b=0.
$$

| $\mathbf x$ | $\mathbf W\mathbf x$ | $+\mathbf c$ | $\mathbf h = \text{ReLU}$ | $y = h_1 - 2h_2$ |
|---|---|---|---|---|
| (0,0) | (0,0) | (0,−1) | (0,0) | **0** ✓ |
| (0,1) | (1,1) | (1,0) | (1,0) | **1** ✓ |
| (1,0) | (1,1) | (1,0) | (1,0) | **1** ✓ |
| (1,1) | (2,2) | (2,1) | (2,1) | 2 − 2 = **0** ✓ |

Exactly zero error, no training required — this proves *representational* sufficiency; learning it by gradient descent is a separate (and non-trivial: XOR has a saddle-rich landscape) matter.

### 2.4 Solution 3 — feature augmentation

Add the product feature $x_3 = x_1x_2$. Then
$$
y = \Theta(x_1 + x_2 - 2x_3 - 0.5)
$$
separates perfectly in $\mathbb R^3$. This is **Cover's theorem in action** (Unit I §7) and is precisely the idea behind kernel methods and RBF networks (Unit V).

---

## 3. Backpropagation — Complete Derivation

### 3.1 Setup

Objective for one sample: $\mathcal L\big(\mathbf a^{(L)}, \mathbf y\big)$. We need $\dfrac{\partial \mathcal L}{\partial W^{(\ell)}_{ij}}$ and $\dfrac{\partial\mathcal L}{\partial b^{(\ell)}_i}$ for **all** $\ell$.

Naïve finite differences cost $O(P)$ forward passes ⇒ $O(P^2)$ total. Backprop computes **all** $P$ derivatives in $O(P)$ — a *reverse-mode automatic differentiation* on the computational graph.

### 3.2 Define the local error signal (delta)

$$
\boxed{\;\delta^{(\ell)}_i \;\triangleq\; \frac{\partial \mathcal L}{\partial z^{(\ell)}_i}\;}
$$

Everything reduces to computing $\boldsymbol\delta^{(\ell)}$.

### 3.3 Step 1 — Output layer delta (BP1)

$$
\delta^{(L)}_i = \frac{\partial \mathcal L}{\partial z^{(L)}_i}
= \sum_k \frac{\partial \mathcal L}{\partial a^{(L)}_k}\frac{\partial a^{(L)}_k}{\partial z^{(L)}_i}
$$

For an **element-wise** output activation ($a_k = \varphi(z_k)$), the Jacobian is diagonal and

$$
\boxed{\;\boldsymbol\delta^{(L)} = \nabla_{\mathbf a^{(L)}}\mathcal L \;\odot\; \varphi'\!\left(\mathbf z^{(L)}\right)\;}\qquad (\odot=\text{Hadamard})
$$

For **softmax** output, use the full Jacobian $\mathbf J = \mathrm{diag}(\mathbf a)-\mathbf a\mathbf a^{\mathsf T}$, i.e. $\boldsymbol\delta^{(L)} = \mathbf J\,\nabla_{\mathbf a}\mathcal L$.

### 3.4 Step 2 — Backward recursion (BP2)

$z^{(\ell)}_i$ influences $\mathcal L$ **only** through the next layer's pre-activations. By the multivariate chain rule:

$$
\delta^{(\ell)}_i = \frac{\partial\mathcal L}{\partial z^{(\ell)}_i}
= \sum_{k=1}^{n_{\ell+1}} \frac{\partial\mathcal L}{\partial z^{(\ell+1)}_k}\cdot\frac{\partial z^{(\ell+1)}_k}{\partial z^{(\ell)}_i}
= \sum_k \delta^{(\ell+1)}_k \cdot \frac{\partial z^{(\ell+1)}_k}{\partial z^{(\ell)}_i}.
$$

Now, since $z^{(\ell+1)}_k = \sum_j W^{(\ell+1)}_{kj}\,\varphi(z^{(\ell)}_j) + b^{(\ell+1)}_k$:

$$
\frac{\partial z^{(\ell+1)}_k}{\partial z^{(\ell)}_i} = W^{(\ell+1)}_{ki}\,\varphi'\!\left(z^{(\ell)}_i\right).
$$

Substituting:

$$
\boxed{\;\delta^{(\ell)}_i = \varphi'\!\left(z^{(\ell)}_i\right)\sum_{k} W^{(\ell+1)}_{ki}\,\delta^{(\ell+1)}_k
\quad\Longleftrightarrow\quad
\boldsymbol\delta^{(\ell)} = \left(\mathbf W^{(\ell+1)\mathsf T}\boldsymbol\delta^{(\ell+1)}\right)\odot \varphi'\!\left(\mathbf z^{(\ell)}\right)\;}
$$

**Interpretation.** The error is propagated backwards through the **transpose** of the same weight matrix used in the forward pass, then gated by the local derivative. This is *reverse-mode AD*: the backward pass is a linear network with transposed weights.

### 3.5 Step 3 — Parameter gradients (BP3, BP4)

$$
\frac{\partial \mathcal L}{\partial W^{(\ell)}_{ij}}
= \frac{\partial\mathcal L}{\partial z^{(\ell)}_i}\cdot\frac{\partial z^{(\ell)}_i}{\partial W^{(\ell)}_{ij}}
= \delta^{(\ell)}_i \, a^{(\ell-1)}_j
\qquad\Longrightarrow\qquad
\boxed{\;\nabla_{\mathbf W^{(\ell)}}\mathcal L = \boldsymbol\delta^{(\ell)}\,\mathbf a^{(\ell-1)\mathsf T}\;}
$$

$$
\frac{\partial\mathcal L}{\partial b^{(\ell)}_i} = \delta^{(\ell)}_i
\qquad\Longrightarrow\qquad
\boxed{\;\nabla_{\mathbf b^{(\ell)}}\mathcal L = \boldsymbol\delta^{(\ell)}\;}
$$

> **The four backprop equations (memorise this block):**
> $$
> \begin{aligned}
> \text{(BP1)}\quad &\boldsymbol\delta^{(L)} = \nabla_{\mathbf a}\mathcal L \odot \varphi'(\mathbf z^{(L)})\\
> \text{(BP2)}\quad &\boldsymbol\delta^{(\ell)} = \left(\mathbf W^{(\ell+1)\mathsf T}\boldsymbol\delta^{(\ell+1)}\right)\odot\varphi'(\mathbf z^{(\ell)})\\
> \text{(BP3)}\quad &\partial\mathcal L/\partial \mathbf b^{(\ell)} = \boldsymbol\delta^{(\ell)}\\
> \text{(BP4)}\quad &\partial\mathcal L/\partial \mathbf W^{(\ell)} = \boldsymbol\delta^{(\ell)}\mathbf a^{(\ell-1)\mathsf T}
> \end{aligned}
> $$

### 3.6 The algorithm

> **Algorithm: Backpropagation (one mini-batch)**
> **Input:** batch $\{(\mathbf x_b,\mathbf y_b)\}_{b=1}^{B}$, params $\{\mathbf W^{(\ell)},\mathbf b^{(\ell)}\}$, lr $\eta$
> 1. **Forward:** $\mathbf a^{(0)}\!=\!\mathbf x$; for $\ell=1..L$: $\mathbf z^{(\ell)}\!=\!\mathbf W^{(\ell)}\mathbf a^{(\ell-1)}\!+\!\mathbf b^{(\ell)}$, $\mathbf a^{(\ell)}\!=\!\varphi(\mathbf z^{(\ell)})$. **Cache** $\mathbf z^{(\ell)},\mathbf a^{(\ell)}$.
> 2. **Output error:** $\boldsymbol\delta^{(L)}$ via BP1.
> 3. **Backward:** for $\ell = L{-}1$ down to $1$: $\boldsymbol\delta^{(\ell)}$ via BP2.
> 4. **Gradients:** $\nabla_{\mathbf W^{(\ell)}} = \frac1B\sum_b \boldsymbol\delta_b^{(\ell)}\mathbf a_b^{(\ell-1)\mathsf T}$, $\nabla_{\mathbf b^{(\ell)}} = \frac1B\sum_b \boldsymbol\delta_b^{(\ell)}$.
> 5. **Update:** $\mathbf W^{(\ell)} \leftarrow \mathbf W^{(\ell)} - \eta\nabla_{\mathbf W^{(\ell)}}$, likewise $\mathbf b^{(\ell)}$.
>
> **Complexity:** time $O(B\sum_\ell n_\ell n_{\ell-1})$ per pass; memory $O(B\sum_\ell n_\ell)$ for the cached activations (the real bottleneck in deep nets → *gradient checkpointing* trades compute for memory: $O(\sqrt L)$ memory at ~30 % extra compute).

### 3.7 Gradient checking (numerical verification)

$$
\frac{\partial\mathcal L}{\partial\theta_i} \approx \frac{\mathcal L(\theta + \epsilon e_i) - \mathcal L(\theta - \epsilon e_i)}{2\epsilon} + O(\epsilon^2)
$$

Use the central difference (error $O(\epsilon^2)$, not $O(\epsilon)$), $\epsilon \approx 10^{-5}$ in float64, and compare via the **relative error**

$$
\text{rel} = \frac{\|\nabla_{\text{analytic}} - \nabla_{\text{numeric}}\|_2}{\|\nabla_{\text{analytic}}\|_2 + \|\nabla_{\text{numeric}}\|_2} \;<\; 10^{-7}\;\;\text{(good)},\;\; >10^{-4}\;\;\text{(bug)}.
$$

Turn off dropout and kinks (ReLU at 0) while checking.

### 3.8 Vanishing / exploding gradients — the formal statement

Unrolling BP2 from layer $L$ back to layer $\ell$:

$$
\boldsymbol\delta^{(\ell)} = \left[\prod_{k=\ell+1}^{L} \mathbf D^{(k-1)}\mathbf W^{(k)\mathsf T}\right]\boldsymbol\delta^{(L)},
\qquad \mathbf D^{(k)} = \mathrm{diag}\!\left(\varphi'(\mathbf z^{(k)})\right).
$$

Taking norms,

$$
\|\boldsymbol\delta^{(\ell)}\| \le \left(\max_k\|\mathbf D^{(k)}\|\;\max_k\|\mathbf W^{(k)}\|\right)^{L-\ell}\|\boldsymbol\delta^{(L)}\|.
$$

Let $\rho = \|\mathbf D\|\|\mathbf W\|$. Then:
- $\rho < 1 \Rightarrow$ **exponential decay** (vanishing gradient) — early layers stop learning;
- $\rho > 1 \Rightarrow$ **exponential growth** (exploding gradient) — NaNs.

With sigmoid, $\|\mathbf D\|\le 0.25$, so $\rho<1$ unless $\|W\|>4$. **Remedies:** ReLU-family ($\varphi'\in\{0,1\}$), careful initialisation (§9), BatchNorm/LayerNorm, residual connections $\mathbf a^{(\ell)} = \mathbf a^{(\ell-1)} + F(\mathbf a^{(\ell-1)})$ giving $\partial\mathbf a^{(\ell)}/\partial\mathbf a^{(\ell-1)} = \mathbf I + \partial F/\partial\mathbf a$ (identity path ⇒ $\rho\approx1$), and gradient clipping $\mathbf g \leftarrow \mathbf g\cdot\min(1, c/\|\mathbf g\|)$.

---

## 4. Backpropagation — Fully Worked Numerical

**Network:** 2 – 2 – 1, sigmoid everywhere, loss $E = \tfrac12(t-\hat y)^2$, learning rate $\eta = 0.5$.

**Initial parameters**

| Param | Value | | Param | Value |
|---|---|---|---|---|
| $w_{11}$ (x₁→h₁) | 0.15 | | $v_1$ (h₁→o) | 0.40 |
| $w_{12}$ (x₂→h₁) | 0.20 | | $v_2$ (h₂→o) | 0.45 |
| $w_{21}$ (x₁→h₂) | 0.25 | | $b_3$ (bias o) | 0.60 |
| $w_{22}$ (x₂→h₂) | 0.30 | | | |
| $b_1$ (bias h₁) | 0.35 | | $b_2$ (bias h₂) | 0.35 |

**Input** $\mathbf x = (0.05,\,0.10)$, **target** $t = 0.01$.

### Step 1 — Forward pass

$$
z_{h_1} = 0.15(0.05) + 0.20(0.10) + 0.35 = 0.0075 + 0.0200 + 0.35 = \mathbf{0.3775}
$$
$$
a_{h_1} = \sigma(0.3775) = \frac{1}{1+e^{-0.3775}} = \frac{1}{1+0.685570} = \mathbf{0.593270}
$$
$$
z_{h_2} = 0.25(0.05) + 0.30(0.10) + 0.35 = 0.0125 + 0.0300 + 0.35 = \mathbf{0.3925}
$$
$$
a_{h_2} = \sigma(0.3925) = \frac{1}{1+0.675356} = \mathbf{0.596884}
$$
$$
z_o = 0.40(0.593270) + 0.45(0.596884) + 0.60 = 0.237308 + 0.268598 + 0.60 = \mathbf{1.105906}
$$
$$
\hat y = \sigma(1.105906) = \frac{1}{1+0.330910} = \mathbf{0.751365}
$$

$$
E = \tfrac12(0.01 - 0.751365)^2 = \tfrac12(0.741365)^2 = \tfrac12(0.549622) = \mathbf{0.274811}
$$

### Step 2 — Output delta (BP1)

$$
\frac{\partial E}{\partial \hat y} = -(t - \hat y) = \hat y - t = 0.751365 - 0.01 = 0.741365
$$
$$
\frac{\partial \hat y}{\partial z_o} = \hat y(1-\hat y) = 0.751365 \times 0.248635 = 0.186816
$$
$$
\boxed{\;\delta_o = 0.741365 \times 0.186816 = \mathbf{0.138498}\;}
$$

### Step 3 — Output-layer gradients (BP3, BP4)

$$
\frac{\partial E}{\partial v_1} = \delta_o\, a_{h_1} = 0.138498(0.593270) = \mathbf{0.082159}
$$
$$
\frac{\partial E}{\partial v_2} = \delta_o\, a_{h_2} = 0.138498(0.596884) = \mathbf{0.082660}
$$
$$
\frac{\partial E}{\partial b_3} = \delta_o = \mathbf{0.138498}
$$

### Step 4 — Hidden deltas (BP2)

$$
\delta_{h_1} = \delta_o\,v_1\,a_{h_1}(1-a_{h_1}) = 0.138498(0.40)(0.593270)(0.406730)
$$
$$
= 0.055399 \times 0.241300 = \mathbf{0.013368}
$$
$$
\delta_{h_2} = \delta_o\,v_2\,a_{h_2}(1-a_{h_2}) = 0.138498(0.45)(0.596884)(0.403116)
$$
$$
= 0.062324 \times 0.240613 = \mathbf{0.014996}
$$

### Step 5 — Hidden-layer gradients

| Gradient | Computation | Value |
|---|---|---|
| $\partial E/\partial w_{11}$ | $\delta_{h_1}x_1 = 0.013368(0.05)$ | 0.00066840 |
| $\partial E/\partial w_{12}$ | $\delta_{h_1}x_2 = 0.013368(0.10)$ | 0.00133680 |
| $\partial E/\partial w_{21}$ | $\delta_{h_2}x_1 = 0.014996(0.05)$ | 0.00074980 |
| $\partial E/\partial w_{22}$ | $\delta_{h_2}x_2 = 0.014996(0.10)$ | 0.00149960 |
| $\partial E/\partial b_1$ | $\delta_{h_1}$ | 0.01336800 |
| $\partial E/\partial b_2$ | $\delta_{h_2}$ | 0.01499600 |

> Note how the hidden gradients are **two orders of magnitude smaller** than the output gradients ($10^{-3}$ vs $10^{-1}$). This *is* the vanishing-gradient phenomenon appearing in a 2-layer net; imagine 20 layers.

### Step 6 — Parameter update ($\eta = 0.5$)

| Param | Old | $-\eta\,\nabla$ | New |
|---|---|---|---|
| $v_1$ | 0.40 | −0.041080 | **0.358920** |
| $v_2$ | 0.45 | −0.041330 | **0.408670** |
| $b_3$ | 0.60 | −0.069249 | **0.530751** |
| $w_{11}$ | 0.15 | −0.000334 | **0.149666** |
| $w_{12}$ | 0.20 | −0.000668 | **0.199332** |
| $w_{21}$ | 0.25 | −0.000375 | **0.249625** |
| $w_{22}$ | 0.30 | −0.000750 | **0.299250** |
| $b_1$ | 0.35 | −0.006684 | **0.343316** |
| $b_2$ | 0.35 | −0.007498 | **0.342502** |

### Step 7 — Verify the loss decreased

$$
z_{h_1}' = 0.149666(0.05)+0.199332(0.10)+0.343316 = 0.370732 \Rightarrow a_{h_1}' = 0.591639
$$
$$
z_{h_2}' = 0.249625(0.05)+0.299250(0.10)+0.342502 = 0.384908 \Rightarrow a_{h_2}' = 0.595055
$$
$$
z_o' = 0.358920(0.591639)+0.408670(0.595055)+0.530751 = 0.212349+0.243182+0.530751 = 0.986282
$$
$$
\hat y' = \sigma(0.986282) = 0.728353,\qquad
E' = \tfrac12(0.01-0.728353)^2 = \mathbf{0.258016}
$$

$$
\Delta E = 0.258016 - 0.274811 = \mathbf{-0.016795} \quad ✓\;\text{loss decreased}
$$

**First-order sanity check:** predicted decrease $\approx \eta\|\nabla E\|^2 = 0.5\big(0.082159^2+0.082660^2+0.138498^2 + \dots\big) \approx 0.5(0.0292) \approx 0.0146$. Observed $0.0168$; the small discrepancy is the second-order (curvature) term — here helping because the step moved into a steeper descent region.

---

## 5. Cost Functions

### 5.1 Mean Squared Error (MSE / SSE)

$$
\mathcal L_{\text{MSE}} = \frac{1}{2N}\sum_{i=1}^{N}\|\mathbf y_i - \hat{\mathbf y}_i\|_2^2,
\qquad \frac{\partial \mathcal L}{\partial \hat y} = \hat y - y .
$$

**MLE derivation.** Assume $y = f_\theta(x) + \epsilon$, $\epsilon\sim\mathcal N(0,\sigma^2)$. Then
$$
\log p(\mathcal D|\theta) = \sum_i \log \frac{1}{\sqrt{2\pi\sigma^2}}e^{-\frac{(y_i - f_\theta(x_i))^2}{2\sigma^2}}
= -\frac{1}{2\sigma^2}\sum_i (y_i-f_\theta(x_i))^2 - \frac N2\log(2\pi\sigma^2).
$$
Maximising the log-likelihood $\equiv$ minimising SSE. **So MSE assumes homoscedastic Gaussian noise** — inappropriate for classification.

### 5.2 Cross-Entropy

**Binary (Bernoulli MLE):**
$$
\mathcal L_{\text{BCE}} = -\frac1N\sum_i\Big[y_i\log\hat y_i + (1-y_i)\log(1-\hat y_i)\Big]
$$

**Categorical (Multinoulli MLE):**
$$
\mathcal L_{\text{CCE}} = -\frac1N\sum_{i=1}^{N}\sum_{c=1}^{C} t_{ic}\log \hat y_{ic}
$$

**Information-theoretic identity:**
$$
H(p, q) = H(p) + D_{\text{KL}}(p\|q).
$$
Since the empirical $p$ (one-hot) has $H(p)=0$, **minimising cross-entropy $\equiv$ minimising KL divergence** between the empirical and model distributions.

### 5.3 The learning-slowdown problem — why CE beats MSE for classification

**With sigmoid output + MSE:**
$$
\frac{\partial \mathcal L}{\partial z} = (\hat y - y)\,\underbrace{\sigma'(z)}_{\to\,0\text{ when saturated}} = (\hat y-y)\hat y(1-\hat y)
$$
If $y=1$ but $\hat y=0.001$ (confidently wrong), $\sigma' = 0.001(0.999)\approx 10^{-3}$ ⇒ **gradient $\approx 10^{-3}$: learning stalls exactly when the error is largest.**

**With sigmoid output + BCE:**
$$
\frac{\partial\mathcal L}{\partial \hat y} = -\frac{y}{\hat y} + \frac{1-y}{1-\hat y} = \frac{\hat y - y}{\hat y(1-\hat y)}
$$
$$
\frac{\partial\mathcal L}{\partial z} = \frac{\hat y-y}{\hat y(1-\hat y)}\cdot \hat y(1-\hat y) = \boxed{\;\hat y - y\;}
$$
The saturating factor **cancels exactly**. The same happens for softmax+CCE (Unit I §8.6). For the same wrong prediction, the gradient is $0.001-1 = -0.999$ — three orders of magnitude larger.

**Convexity note:** MSE with a sigmoid output is **non-convex even for a single neuron**; BCE with a sigmoid is **convex in $\mathbf w$**. This is a second, independent reason to prefer CE.

### 5.4 Catalogue of loss functions

| Loss | Formula | Output activation | Use |
|---|---|---|---|
| MSE / L2 | $\tfrac12(y-\hat y)^2$ | linear | regression, Gaussian noise |
| MAE / L1 | $\lvert y-\hat y\rvert$ | linear | robust regression (Laplace noise) |
| Huber | $\begin{cases}\tfrac12 e^2 & \lvert e\rvert\le\delta\\ \delta(\lvert e\rvert-\tfrac\delta2)& \text{else}\end{cases}$ | linear | robust + differentiable at 0 |
| BCE | $-[y\log\hat y + (1{-}y)\log(1{-}\hat y)]$ | sigmoid | binary / multi-label |
| CCE | $-\sum_c t_c\log \hat y_c$ | softmax | multi-class |
| KL | $\sum_c p_c \log(p_c/q_c)$ | softmax | distillation, VAE |
| Hinge | $\max(0, 1-y\hat y)$ | linear | SVM-style margin |
| Focal | $-\alpha(1-\hat y)^\gamma\log\hat y$ | sigmoid | class imbalance |
| Contrastive/InfoNCE | $-\log\frac{e^{s^+/\tau}}{\sum e^{s/\tau}}$ | — | self-supervised |

### 5.5 The general "canonical link" theorem

For any **exponential-family** output distribution with natural parameter $z$ and its canonical link activation $\varphi$, minimising the negative log-likelihood always gives

$$
\boxed{\;\frac{\partial\mathcal L}{\partial \mathbf z^{(L)}} = \hat{\mathbf y} - \mathbf y\;}
$$

| Distribution | Activation | Loss |
|---|---|---|
| Gaussian | identity | MSE |
| Bernoulli | sigmoid | BCE |
| Categorical | softmax | CCE |
| Poisson | exp | Poisson NLL |

*This is why frameworks fuse `softmax_cross_entropy` into one op — both for numerical stability (log-sum-exp) and because the fused gradient is trivially $\hat y - y$.*

---

## 6. Gradient Descent — Theory and Convergence

### 6.1 The three variants

| Variant | Update uses | Steps/epoch | Gradient noise | Memory |
|---|---|---|---|---|
| **Batch GD** | all $N$ samples | 1 | none | $O(N)$ |
| **Stochastic GD** | 1 sample | $N$ | very high | $O(1)$ |
| **Mini-batch GD** | $B$ samples ($32$–$512$) | $N/B$ | $\propto 1/\sqrt B$ | $O(B)$ |

$$
\theta_{k+1} = \theta_k - \eta\,\nabla_\theta \hat{\mathcal L}_{\mathcal B_k}(\theta_k),\qquad
\mathbb{E}\!\left[\nabla \hat{\mathcal L}_{\mathcal B}\right] = \nabla\mathcal L \;\;(\text{unbiased}),\qquad
\mathrm{Var} = \frac{\Sigma}{B}.
$$

### 6.2 Convergence of batch GD on a smooth objective

**Assumption ($L$-smoothness):** $\|\nabla f(x)-\nabla f(y)\|\le L\|x-y\|$, equivalently
$$
f(y) \le f(x) + \nabla f(x)^{\mathsf T}(y-x) + \tfrac{L}{2}\|y-x\|^2 .
$$

Put $y = x - \eta\nabla f(x)$:
$$
f(x^+) \le f(x) - \eta\|\nabla f\|^2 + \tfrac{L\eta^2}{2}\|\nabla f\|^2
= f(x) - \eta\left(1 - \tfrac{L\eta}{2}\right)\|\nabla f\|^2 .
$$

**Descent guarantee:** need $\eta < 2/L$. With the optimal $\eta = 1/L$:
$$
f(x^+) \le f(x) - \frac{1}{2L}\|\nabla f(x)\|^2 .
$$
Telescoping over $K$ steps gives the non-convex rate
$$
\min_{k\le K}\|\nabla f(x_k)\|^2 \le \frac{2L\big(f(x_0)-f^\star\big)}{K}
\qquad\Rightarrow\qquad O(1/\sqrt K)\ \text{for the gradient norm.}
$$

**Strongly convex case ($\mu$-strong convexity):** linear convergence
$$
f(x_K)-f^\star \le \left(1 - \frac{\mu}{L}\right)^{K}\big(f(x_0)-f^\star\big) = \left(1-\frac1\kappa\right)^K(\cdot),\quad \kappa = L/\mu .
$$

**Quadratic case, exact analysis.** For $f(\theta)=\tfrac12\theta^{\mathsf T}\mathbf H\theta$ with $\mathbf H = \mathbf Q\Lambda\mathbf Q^{\mathsf T}$, in the eigenbasis $\tilde\theta = \mathbf Q^{\mathsf T}\theta$:
$$
\tilde\theta_i^{(k)} = (1-\eta\lambda_i)^{k}\,\tilde\theta_i^{(0)} .
$$
- Stability requires $|1-\eta\lambda_i|<1 \;\forall i \Rightarrow \boxed{0<\eta<2/\lambda_{\max}}$.
- Optimal $\eta^\star = \dfrac{2}{\lambda_{\min}+\lambda_{\max}}$, giving contraction factor $\dfrac{\kappa-1}{\kappa+1}$.
- **Ill-conditioning ($\kappa\gg1$) is the fundamental problem:** the smallest direction converges $\kappa$ times slower ⇒ the classic **zig-zag in a narrow ravine**. Momentum (Unit VI) reduces the dependence from $\kappa$ to $\sqrt\kappa$.

### 6.3 SGD convergence (Robbins–Monro)

With learning-rate schedule $\eta_k$ satisfying
$$
\sum_{k=1}^{\infty}\eta_k = \infty \quad\text{(enough total travel)},\qquad
\sum_{k=1}^{\infty}\eta_k^2 < \infty \quad\text{(noise averages out)},
$$
SGD converges almost surely to a stationary point. E.g. $\eta_k = \eta_0/k$ ✓; $\eta_k = \eta_0/\sqrt k$ satisfies the first but not the second (still gives $O(1/\sqrt k)$ in convex settings).

**Rate (convex, bounded variance $\sigma^2$):** $\mathbb E[f(\bar x_K)] - f^\star = O\!\left(\dfrac{\sigma}{\sqrt K}\right)$ — *sub-linear regardless of $\kappa$*, because noise, not curvature, dominates.

### 6.4 Loss-surface geometry in high dimensions

- **Critical points:** at a random critical point of a high-dimensional random function, the Hessian eigenvalues follow a semicircle law; the probability that *all* $P$ eigenvalues are positive is $\approx e^{-cP}$. ⇒ **Saddle points vastly outnumber local minima.**
- **Index–loss relation:** critical points with high loss are overwhelmingly saddles; local minima cluster near the global value (Dauphin et al. 2014, Choromanska et al. 2015).
- **Practical implication:** the enemy is *plateaus around saddles*, not bad local minima. Escape mechanisms: SGD noise, momentum, adaptive methods.
- **Mode connectivity:** independently trained solutions are often connected by low-loss paths ⇒ the minima form a connected manifold, not isolated wells.
- **Flat vs sharp minima:** flat minima (small $\|\mathbf H\|$) generalise better (PAC-Bayes / MDL argument). Large batch → sharp minima → worse generalisation; SAM (sharpness-aware minimisation) explicitly optimises $\max_{\|\epsilon\|\le\rho} \mathcal L(\theta+\epsilon)$.

---

## 7. Overfitting and the Bias–Variance Decomposition

### 7.1 The phenomenon

```
 Error
   │  ╲                                        ╱ validation
   │   ╲                                     ╱
   │    ╲                                  ╱
   │     ╲__________________       _____╱
   │                        ╲____╱  ← best model (early stopping point)
   │      ╲
   │        ╲__________________________________  training
   └──────────────────────────────────────────────► capacity / epochs
       underfit   |    sweet spot    |   overfit
      (high bias) |                  | (high variance)
```

### 7.2 Full derivation of the bias–variance decomposition

Let $y = f(x) + \epsilon$, $\mathbb E[\epsilon]=0$, $\mathrm{Var}(\epsilon)=\sigma^2$. Let $\hat f_{\mathcal D}$ be the model trained on dataset $\mathcal D$. The expected squared error at a fixed $x$:

$$
\mathbb E_{\mathcal D,\epsilon}\!\left[(y - \hat f_{\mathcal D}(x))^2\right]
$$

Insert $\pm \bar f(x)$ where $\bar f(x)=\mathbb E_{\mathcal D}[\hat f_{\mathcal D}(x)]$, and $\pm f(x)$:

$$
= \mathbb E\!\left[\Big( \underbrace{(f + \epsilon) - \bar f}_{A} + \underbrace{\bar f - \hat f_{\mathcal D}}_{B} \Big)^2\right]
= \mathbb E[A^2] + 2\mathbb E[AB] + \mathbb E[B^2]
$$

Cross term: $\mathbb E[AB] = \mathbb E[(f+\epsilon-\bar f)]\,\mathbb E[(\bar f - \hat f_{\mathcal D})] = (\cdot)\times 0 = 0$ (independence of $\epsilon$ and $\mathcal D$, and $\mathbb E_{\mathcal D}[\bar f - \hat f_{\mathcal D}]=0$).

$$
\mathbb E[A^2] = \mathbb E[(f-\bar f)^2] + 2\mathbb E[\epsilon(f-\bar f)] + \mathbb E[\epsilon^2] = (f-\bar f)^2 + \sigma^2
$$
$$
\mathbb E[B^2] = \mathbb E_{\mathcal D}\!\left[(\hat f_{\mathcal D}-\bar f)^2\right] = \mathrm{Var}_{\mathcal D}(\hat f)
$$

$$
\boxed{\;\underbrace{\mathbb E\big[(y-\hat f)^2\big]}_{\text{expected test error}} = \underbrace{\big(f(x)-\bar f(x)\big)^2}_{\text{Bias}^2} + \underbrace{\mathrm{Var}_{\mathcal D}\big(\hat f(x)\big)}_{\text{Variance}} + \underbrace{\sigma^2}_{\text{irreducible noise}}\;}
$$

| Regime | Bias | Variance | Symptom |
|---|---|---|---|
| Underfitting | high | low | train error ≈ test error, both large |
| Overfitting | low | high | train error ≪ test error |
| Just right | balanced | balanced | small gap, small error |

### 7.3 Capacity measures

- **VC dimension.** For a network with $P$ weights and threshold units, $\text{VC} = O(P\log P)$; for piecewise-linear (ReLU) nets of depth $L$, $\text{VC} = O(PL\log P)$.
- **Generalisation bound.** With probability $1-\delta$:

  $$
  R(\theta) \le \hat R(\theta) + O\!\left(\sqrt{\frac{\text{VC}\log(N/\text{VC}) + \log(1/\delta)}{N}}\right).
  $$

- **Rademacher complexity** gives tighter, norm-based bounds: $\hat{\mathfrak R}_N(\mathcal F) \le \frac{\prod_\ell \|W^{(\ell)}\|_F}{\sqrt N}$ — this justifies *weight-norm* regularisation directly.

### 7.4 The deep-learning paradox and double descent

Modern nets have $P \gg N$ and can fit **random labels** perfectly (Zhang et al. 2017) — so classical VC bounds are vacuous. Empirically:

```
Test
error │╲
      │ ╲      ╱╲  ← interpolation threshold (P ≈ N)
      │  ╲___╱   ╲
      │            ╲______________  ← modern over-parameterised regime
      └──────────────────────────────► model capacity P
       classical U   |  double descent
```

**Explanation:** beyond the interpolation threshold, among the infinitely many zero-training-error solutions, SGD exhibits an *implicit bias* toward minimum-norm / maximum-margin solutions, which generalise well.

---

## 8. Regularization Techniques

### 8.1 $L_2$ regularization (weight decay / ridge / Tikhonov)

$$
\tilde{\mathcal L}(\theta) = \mathcal L(\theta) + \frac{\lambda}{2}\|\mathbf w\|_2^2
\qquad\Longrightarrow\qquad
\nabla\tilde{\mathcal L} = \nabla\mathcal L + \lambda \mathbf w
$$
$$
\mathbf w \leftarrow \mathbf w - \eta(\nabla\mathcal L + \lambda\mathbf w) = \underbrace{(1-\eta\lambda)}_{\text{shrinkage}}\mathbf w - \eta\nabla\mathcal L
$$

**Analysis in the Hessian eigenbasis (the key derivation).** Quadratic approximation around the unregularised optimum $\mathbf w^\star$:
$$
\hat{\mathcal L}(\mathbf w) = \mathcal L(\mathbf w^\star) + \tfrac12(\mathbf w-\mathbf w^\star)^{\mathsf T}\mathbf H(\mathbf w - \mathbf w^\star).
$$
Setting $\nabla\big[\hat{\mathcal L} + \tfrac\lambda2\|\mathbf w\|^2\big]=0$:
$$
\mathbf H(\tilde{\mathbf w}-\mathbf w^\star) + \lambda\tilde{\mathbf w} = 0 \Rightarrow \tilde{\mathbf w} = (\mathbf H+\lambda\mathbf I)^{-1}\mathbf H\,\mathbf w^\star .
$$
With $\mathbf H = \mathbf Q\Lambda\mathbf Q^{\mathsf T}$:
$$
\boxed{\;\mathbf Q^{\mathsf T}\tilde{\mathbf w} = \mathrm{diag}\!\left(\frac{\lambda_i}{\lambda_i + \lambda}\right)\mathbf Q^{\mathsf T}\mathbf w^\star\;}
$$

**Interpretation:** directions with **large curvature** ($\lambda_i \gg \lambda$) are essentially untouched; directions with **small curvature** ($\lambda_i\ll\lambda$) are shrunk toward zero. $L_2$ *removes the directions in which the data does not constrain the solution.*

**Bayesian view:** $L_2$ = MAP estimation with a Gaussian prior $\mathbf w \sim \mathcal N(0,\tau^2\mathbf I)$, $\lambda = \sigma^2/\tau^2$.

> **AdamW note:** $L_2$ penalty and weight decay are **not** the same under adaptive optimisers, because the penalty gets divided by $\sqrt{\hat v}$. AdamW decouples them: $\mathbf w \leftarrow \mathbf w - \eta\big(\hat m/(\sqrt{\hat v}+\epsilon) + \lambda \mathbf w\big)$. (See Unit VI.)

### 8.2 $L_1$ regularization (LASSO) — why it produces sparsity

$$
\tilde{\mathcal L} = \mathcal L + \lambda\|\mathbf w\|_1,\qquad
\nabla = \nabla\mathcal L + \lambda\,\mathrm{sgn}(\mathbf w)
$$

**Derivation of the soft-threshold solution.** Assume a diagonal Hessian $\mathbf H=\mathrm{diag}(h_i)$ (as in Goodfellow §7.1.2). Per coordinate:
$$
\min_{w_i}\; \tfrac{h_i}{2}(w_i - w_i^\star)^2 + \lambda|w_i| .
$$
Sub-differential condition $h_i(w_i-w_i^\star) + \lambda\,\partial|w_i| \ni 0$ gives
$$
\boxed{\;\tilde w_i = \mathrm{sgn}(w_i^\star)\max\!\left(|w_i^\star| - \frac{\lambda}{h_i},\;0\right)\;}
$$

**So if $|w_i^\star| \le \lambda/h_i$, the weight is set exactly to ZERO** — a hard, exact sparsification. Contrast with $L_2$, which only *multiplies* weights by $\lambda_i/(\lambda_i+\lambda) \ne 0$.

**Geometric intuition:** the $L_1$ ball $\{\|w\|_1\le t\}$ is a cross-polytope whose **vertices lie on the axes**; the elliptical loss contour typically first touches it at a vertex ⇒ zeros. The $L_2$ ball is smooth, so contact is generically off-axis.

**Bayesian view:** Laplace prior $p(w_i)\propto e^{-\lambda|w_i|}$.

| | $L_1$ | $L_2$ |
|---|---|---|
| Penalty | $\lambda\sum\lvert w_i\rvert $ | $\tfrac\lambda2\sum w_i^2$ |
| Gradient | $\lambda\mathrm{sgn}(w)$ (constant magnitude) | $\lambda w$ (proportional) |
| Solution | **sparse** (feature selection) | dense, shrunk |
| Prior | Laplace | Gaussian |
| Differentiable at 0 | No | Yes |
| Elastic net | $\alpha\lambda\Vert w\Vert _1 + \tfrac{(1-\alpha)\lambda}{2}\Vert w\Vert _2^2$ — combines both | |

### 8.3 Dropout

**Training:** for each unit independently, sample $r_j \sim \text{Bernoulli}(p)$ (keep-probability $p$) and set
$$
\tilde a_j = \frac{r_j}{p}\,a_j \quad\text{(inverted dropout — scale at train time)}.
$$
**Testing:** use all units unchanged (no scaling needed with inverted dropout).

**Why the $1/p$ scaling.** $\mathbb E[\tilde a_j] = \frac1p\,\mathbb E[r_j]\,a_j = \frac1p (p)a_j = a_j$ ⇒ the expected pre-activation of the next layer is identical at train and test time.

**Interpretation 1 — Exponential ensemble.** A network with $n$ droppable units defines $2^n$ sub-networks sharing weights. Dropout trains them jointly; test-time weight scaling is a cheap approximation to the **geometric mean** of their predictions. (For a single softmax layer it is *exactly* the normalised geometric mean.)

**Interpretation 2 — Adaptive $L_2$.** For linear regression with dropout on inputs, taking the expectation over the mask gives
$$
\mathbb E_{\mathbf r}\left[\|\mathbf y - (\mathbf r\odot \mathbf X)\mathbf w\|^2\right]
= \|\mathbf y - p\mathbf X\mathbf w\|^2 + p(1-p)\sum_j \|\mathbf X_{:,j}\|^2 w_j^2 ,
$$
i.e. dropout $\equiv$ **$L_2$ penalty scaled by the input feature norms** (a data-dependent ridge).

**Interpretation 3 — Co-adaptation breaking.** No unit can rely on a specific partner being present ⇒ redundant, robust features.

**Practice:** $p_{\text{keep}} = 0.5$ for fully-connected hidden layers, $0.8$–$0.9$ for inputs; rarely used in conv layers (use BatchNorm/spatial dropout instead); usually removed when BatchNorm is present (variance-shift conflict).

### 8.4 Early stopping — and its exact equivalence to $L_2$

Monitor validation loss; stop after $p$ epochs ("patience") without improvement; restore the best checkpoint.

**Theorem.** For a quadratic loss with Hessian $\mathbf H=\mathbf Q\Lambda\mathbf Q^{\mathsf T}$ and GD started at $\mathbf w^{(0)}=\mathbf 0$ with learning rate $\eta$, after $\tau$ steps:
$$
\mathbf Q^{\mathsf T}\mathbf w^{(\tau)} = \left[\mathbf I - (\mathbf I - \eta\Lambda)^{\tau}\right]\mathbf Q^{\mathsf T}\mathbf w^\star .
$$
Compare with the $L_2$ result $\mathbf Q^{\mathsf T}\tilde{\mathbf w} = \Lambda(\Lambda+\lambda\mathbf I)^{-1}\mathbf Q^{\mathsf T}\mathbf w^\star$, i.e. componentwise $\frac{\lambda_i}{\lambda_i+\lambda} = 1 - \frac{\lambda}{\lambda_i+\lambda}$.
Matching the two requires $(1-\eta\lambda_i)^\tau = \frac{\lambda}{\lambda_i+\lambda}$. For small $\eta\lambda_i$, $\log(1-\eta\lambda_i)\approx-\eta\lambda_i$, giving

$$
\boxed{\;\tau \approx \frac{1}{\eta\lambda} \quad\Longleftrightarrow\quad \lambda \approx \frac{1}{\tau\eta}\;}
$$

**Early stopping is $L_2$ regularization with an implicit $\lambda = 1/(\eta\tau)$** — training longer = weaker regularization. This is one of the most-asked derivations at M.Tech level.

### 8.5 Data augmentation

$$
\hat{\mathcal L} = \frac1N\sum_i \mathbb E_{T\sim\mathcal T}\big[\mathcal L(f_\theta(T(x_i)), y_i)\big]
$$
Encodes **invariances** as a prior. Images: flips, crops, colour jitter, Cutout, Mixup ($\tilde x = \lambda x_i + (1-\lambda)x_j$, $\tilde y=\lambda y_i+(1-\lambda)y_j$), CutMix, RandAugment. Text: back-translation, synonym replacement. Audio: SpecAugment.

**Noise injection to inputs** with variance $\sigma^2$ is provably equivalent (to $O(\sigma^2)$) to a **Tikhonov penalty on the Jacobian**: $\lambda\,\mathbb E\|\nabla_x f\|^2$.

### 8.6 Batch Normalization

$$
\mu_{\mathcal B} = \frac1B\sum_{b}z_b,\quad
\sigma^2_{\mathcal B}=\frac1B\sum_b (z_b-\mu_{\mathcal B})^2,\quad
\hat z_b = \frac{z_b-\mu_{\mathcal B}}{\sqrt{\sigma^2_{\mathcal B}+\epsilon}},\quad
y_b = \gamma\hat z_b + \beta
$$

Backprop through BN (needed for exams):
$$
\frac{\partial\mathcal L}{\partial \hat z_b} = \frac{\partial\mathcal L}{\partial y_b}\gamma,\quad
\frac{\partial\mathcal L}{\partial\sigma^2} = \sum_b \frac{\partial\mathcal L}{\partial\hat z_b}(z_b-\mu)\cdot\left(-\tfrac12\right)(\sigma^2+\epsilon)^{-3/2},
$$
$$
\frac{\partial\mathcal L}{\partial\mu} = \sum_b\frac{\partial\mathcal L}{\partial\hat z_b}\cdot\frac{-1}{\sqrt{\sigma^2+\epsilon}} + \frac{\partial\mathcal L}{\partial\sigma^2}\cdot\frac{-2\sum_b(z_b-\mu)}{B},
$$
$$
\frac{\partial\mathcal L}{\partial z_b} = \frac{\partial\mathcal L}{\partial\hat z_b}\frac{1}{\sqrt{\sigma^2+\epsilon}} + \frac{\partial\mathcal L}{\partial\sigma^2}\frac{2(z_b-\mu)}{B} + \frac{\partial\mathcal L}{\partial\mu}\frac1B .
$$

**Effects:** smoother loss landscape (better-behaved Lipschitz constant of the gradient — Santurkar et al.), allows larger $\eta$, mild regularization via batch noise. **Inference** uses running averages of $\mu,\sigma^2$. Variants: LayerNorm (over features — Transformers, RNNs), GroupNorm, InstanceNorm, RMSNorm.

### 8.7 Other techniques

| Technique | Mechanism |
|---|---|
| **Max-norm** | project $\Vert \mathbf w_j\Vert _2 \le c$ after each update; pairs well with dropout |
| **Label smoothing** | $t_c \leftarrow (1-\epsilon)t_c + \epsilon/C$; prevents over-confident logits, improves calibration |
| **Parameter sharing** | CNN weight tying; a *hard* prior of translation equivariance |
| **Bagging / ensembles** | averaging $M$ independently-trained nets reduces variance by $\approx 1/M$ when errors are uncorrelated |
| **Multi-task learning** | shared representation acts as a prior |
| **Stochastic depth** | randomly skip residual blocks |
| **Gradient noise** | add $\mathcal N(0,\sigma_t^2)$ to gradients, $\sigma_t^2=\eta/(1+t)^\gamma$ |
| **Sharpness-aware min. (SAM)** | minimise worst-case loss in an $\ell_2$ ball ⇒ flat minima |

---

## 9. Weight Initialization (Xavier & He Derivations)

**Why not zeros?** All units in a layer compute the same thing, receive the same gradient, and stay identical forever — the **symmetry-breaking problem**. Why not large random? Saturation / explosion.

### 9.1 Xavier / Glorot (for tanh, sigmoid — symmetric, $\varphi'(0)\approx1$)

**Forward requirement:** keep $\mathrm{Var}(z^{(\ell)}) = \mathrm{Var}(z^{(\ell-1)})$.

With $z_i = \sum_{j=1}^{n_{\text{in}}} w_{ij}a_j$, independent zero-mean $w$ and $a$:
$$
\mathrm{Var}(z) = n_{\text{in}}\mathrm{Var}(w)\mathrm{Var}(a)
\;\Rightarrow\; \mathrm{Var}(w) = \frac{1}{n_{\text{in}}} .
$$

**Backward requirement:** keep $\mathrm{Var}(\delta^{(\ell)}) = \mathrm{Var}(\delta^{(\ell+1)})$. By BP2 (with $\varphi'\approx1$):
$$
\mathrm{Var}(\delta^{(\ell)}) = n_{\text{out}}\mathrm{Var}(w)\mathrm{Var}(\delta^{(\ell+1)})
\;\Rightarrow\;\mathrm{Var}(w) = \frac{1}{n_{\text{out}}} .
$$

The two cannot both hold unless $n_{\text{in}}=n_{\text{out}}$; Glorot takes the **harmonic compromise**:

$$
\boxed{\;\mathrm{Var}(w) = \frac{2}{n_{\text{in}} + n_{\text{out}}}\;}
$$

Uniform form: $w \sim \mathcal U\!\left[-\sqrt{\dfrac{6}{n_{\text{in}}+n_{\text{out}}}},\; \sqrt{\dfrac{6}{n_{\text{in}}+n_{\text{out}}}}\right]$ (since $\mathrm{Var}(\mathcal U[-a,a]) = a^2/3$).

### 9.2 He / Kaiming (for ReLU)

ReLU zeroes half of the pre-activations. If $z$ is symmetric about 0,
$$
\mathbb E[\text{ReLU}(z)^2] = \tfrac12\mathbb E[z^2] \Rightarrow \mathrm{Var}(a) = \tfrac12\mathrm{Var}(z).
$$
Substituting into the forward variance condition:
$$
\mathrm{Var}(z^{(\ell)}) = n_{\text{in}}\mathrm{Var}(w)\cdot\tfrac12\mathrm{Var}(z^{(\ell-1)}) \;\overset{!}{=}\; \mathrm{Var}(z^{(\ell-1)})
$$
$$
\boxed{\;\mathrm{Var}(w) = \frac{2}{n_{\text{in}}}\;\;\;\Big(w\sim\mathcal N\!\left(0, \sqrt{2/n_{\text{in}}}^2\right)\Big)}
$$

Without the factor 2, activations shrink by $2^{-L/2}$; for $L=30$ that is $\approx 3\times10^{-5}$ — the network is effectively dead at initialisation. **This one factor of 2 made 30-layer networks trainable.**

**Other schemes:** LSUV (data-driven layer-wise rescaling), orthogonal init ($\mathbf W^{\mathsf T}\mathbf W=\mathbf I$, ideal for RNNs — preserves norms exactly), and zero-init of the last BN $\gamma$ in each residual block.

---

## 10. Practical Training Recipe & Debugging

```
1. Sanity check     : overfit 1 batch (loss → 0). If not, there's a bug.
2. Initial loss     : for C-class CE it must start near ln(C). C=10 → 2.303
3. Gradient check   : central differences, rel. error < 1e-7 (float64)
4. LR range test    : sweep η ×10 from 1e-6 → 1; pick just below divergence
5. Add regularisation: weight decay → dropout → augmentation, one at a time
6. Monitor          : train/val loss curves, gradient norms per layer,
                      activation histograms (watch for dead ReLUs / saturation)
7. Diagnose:
     train↑ val↑    → underfit  : bigger model, longer, higher LR
     train↓ val↑    → overfit   : more data/aug, more regularisation, smaller model
     loss = NaN     → explode   : lower LR, clip grads, check log(0) / div-by-0
     loss flat      → dead      : check init, dead ReLUs, LR too small
     spiky val loss → LR too high or batch too small
```

**Loss must start at $\ln C$:** at init, softmax outputs $\approx 1/C$, so $\mathcal L = -\ln(1/C) = \ln C$. If your MNIST run starts at 7.0 instead of 2.303, your initialisation or label encoding is broken.

---

## 11. Solved Numericals

### N1. Forward pass with ReLU + softmax, and the CE gradient

Network 3–2–3. $\mathbf x = (1, 0.5, -1)^{\mathsf T}$, true class $= 2$ (index from 1).

$$
\mathbf W^{(1)} = \begin{pmatrix}0.2 & -0.5 & 0.3\\ 0.7 & 0.1 & -0.4\end{pmatrix},\;
\mathbf b^{(1)} = \begin{pmatrix}0.1\\ -0.2\end{pmatrix},\qquad
\mathbf W^{(2)}=\begin{pmatrix}0.5 & -0.3\\ -0.2 & 0.8\\ 0.6 & 0.1\end{pmatrix},\;
\mathbf b^{(2)} = \begin{pmatrix}0\\0.1\\-0.1\end{pmatrix}
$$

**Hidden:**
$$
z^{(1)}_1 = 0.2(1) + (-0.5)(0.5) + 0.3(-1) + 0.1 = 0.2 - 0.25 - 0.3 + 0.1 = -0.25 \Rightarrow a_1 = 0
$$
$$
z^{(1)}_2 = 0.7(1) + 0.1(0.5) + (-0.4)(-1) - 0.2 = 0.7+0.05+0.4-0.2 = 0.95 \Rightarrow a_2 = 0.95
$$
So $\mathbf a^{(1)} = (0,\,0.95)^{\mathsf T}$ — **unit 1 is inactive**, and will receive zero gradient this step.

**Output logits:**
$$
z^{(2)} = \begin{pmatrix}0.5(0)+(-0.3)(0.95)+0\\ -0.2(0)+0.8(0.95)+0.1 \\ 0.6(0)+0.1(0.95)-0.1\end{pmatrix}
= \begin{pmatrix}-0.285\\ 0.860\\ -0.005\end{pmatrix}
$$

**Softmax** (shift by $0.860$): $(-1.145,\,0,\,-0.865)$ ⇒ $e^{\cdot} = (0.31819,\,1,\,0.42104)$, $S = 1.73923$.
$$
\hat{\mathbf y} = (0.18295,\; 0.57497,\; 0.24208),\qquad \textstyle\sum = 1 ✓
$$
$$
\mathcal L = -\ln(0.57497) = \mathbf{0.55351}
$$
$$
\boldsymbol\delta^{(2)} = \hat{\mathbf y} - \mathbf t = (0.18295,\; -0.42503,\; 0.24208)
$$

**Hidden delta:**
$$
\mathbf W^{(2)\mathsf T}\boldsymbol\delta^{(2)} =
\begin{pmatrix}0.5 & -0.2 & 0.6\\ -0.3 & 0.8 & 0.1\end{pmatrix}
\begin{pmatrix}0.18295\\-0.42503\\0.24208\end{pmatrix}
=\begin{pmatrix}0.09148+0.08501+0.14525\\ -0.05489-0.34002+0.02421\end{pmatrix}
=\begin{pmatrix}0.32174\\ -0.37070\end{pmatrix}
$$
ReLU derivative: $\varphi'(z^{(1)}) = (0, 1)$ (unit 1 inactive).
$$
\boldsymbol\delta^{(1)} = (0.32174\times 0,\;\; -0.37070\times 1) = (0,\; -0.37070)
$$

**Gradients:**
$$
\nabla_{\mathbf W^{(2)}} = \boldsymbol\delta^{(2)}\mathbf a^{(1)\mathsf T} =
\begin{pmatrix}0.18295\\-0.42503\\0.24208\end{pmatrix}(0\;\;0.95)
=\begin{pmatrix}0 & 0.17380\\ 0 & -0.40378\\ 0 & 0.22998\end{pmatrix}
$$
$$
\nabla_{\mathbf W^{(1)}} = \boldsymbol\delta^{(1)}\mathbf x^{\mathsf T} =
\begin{pmatrix}0\\-0.37070\end{pmatrix}(1\;\;0.5\;\;-1)
= \begin{pmatrix}0 & 0 & 0\\ -0.37070 & -0.18535 & 0.37070\end{pmatrix}
$$
**Observation:** the entire first row of $\nabla_{\mathbf W^{(1)}}$ is zero — hidden unit 1 learns nothing from this sample. If it stays inactive across the whole dataset, it is a **dead ReLU**.

---

### N2. MSE vs Cross-Entropy gradient magnitude

Sigmoid output, target $y=1$, prediction $\hat y = 0.01$ (confidently wrong).

| Loss | $\partial\mathcal L/\partial z$ | Value |
|---|---|---|
| MSE | $(\hat y - y)\hat y(1-\hat y) = (-0.99)(0.01)(0.99)$ | $-0.0098$ |
| BCE | $\hat y - y = 0.01 - 1$ | $-0.99$ |

**Ratio = 101×.** With $\hat y = 10^{-4}$: MSE gradient $\approx -10^{-4}$, BCE $\approx -1$ ⇒ ratio $\approx 10^{4}$. MSE would need ~10 000× more steps to correct the same mistake.

---

### N3. $L_2$ shrinkage factors

$\mathbf H$ has eigenvalues $\lambda = (100,\;10,\;1,\;0.1,\;0.01)$; take $\lambda_{\text{reg}} = 1$.

| $\lambda_i$ | Shrinkage $\dfrac{\lambda_i}{\lambda_i+1}$ | Effect |
|---|---|---|
| 100 | 0.990 | essentially unchanged |
| 10 | 0.909 | mildly shrunk |
| 1 | 0.500 | halved |
| 0.1 | 0.091 | strongly suppressed |
| 0.01 | 0.0099 | ≈ eliminated |

Directions the data barely constrains are removed — exactly the intended behaviour.

---

### N4. $L_1$ soft-thresholding

$w^\star = (2.0,\,0.3,\,-0.05,\,-1.2)$, diagonal Hessian $h_i = 1$ for all $i$, $\lambda = 0.4$.

$$
\tilde w_i = \mathrm{sgn}(w_i^\star)\max(|w_i^\star| - 0.4,\,0)
$$

| $w_i^\star$ | $\lvert w_i^\star\rvert -0.4$ | $\tilde w_i$ |
|---|---|---|
| 2.00 | 1.60 | **1.60** |
| 0.30 | −0.10 → 0 | **0** (pruned) |
| −0.05 | −0.35 → 0 | **0** (pruned) |
| −1.20 | 0.80 | **−0.80** |

**Sparsity = 50 %.** Under $L_2$ with $\lambda=0.4$: $\tilde w = w^\star/1.4 = (1.43,\,0.214,\,-0.036,\,-0.857)$ — nothing is exactly zero.

---

### N5. Early stopping ↔ $L_2$ equivalence, numerically

Training with $\eta = 0.01$, stopping at $\tau = 250$ iterations.
$$
\lambda_{\text{implicit}} \approx \frac{1}{\eta\tau} = \frac{1}{0.01\times250} = \mathbf{0.4}
$$
Verify on a direction with $\lambda_i = 5$:
- Early stopping factor: $1-(1-\eta\lambda_i)^\tau = 1-(1-0.05)^{250} = 1 - 0.95^{250}$.
  $\ln(0.95^{250}) = 250\ln0.95 = 250(-0.051293) = -12.823 \Rightarrow 0.95^{250}=2.7\times10^{-6}$. Factor $\approx 0.9999973$.
- $L_2$ factor: $\lambda_i/(\lambda_i+\lambda) = 5/5.4 = 0.9259$.

Now a low-curvature direction, $\lambda_i = 0.01$:
- Early stopping: $1-(1-0.0001)^{250} = 1-e^{-0.025} = 0.02469$.
- $L_2$: $0.01/0.41 = 0.02439$.

The two agree closely in the small-curvature regime where regularization actually matters ✓.

---

### N6. Dropout expectation and variance

A layer of $n=100$ units with activation $a=1$ each, keep probability $p = 0.5$, inverted dropout, next-layer weights $w_j = 0.1$.

Without dropout: $z = \sum_j w_j a_j = 100(0.1)(1) = 10$.
With dropout: $z = \sum_j w_j \frac{r_j}{p}a_j = 0.2\sum_j r_j$, $r_j\sim\text{Ber}(0.5)$.
$$
\mathbb E[z] = 0.2\times 100 \times 0.5 = \mathbf{10}\quad ✓\;(\text{unbiased})
$$
$$
\mathrm{Var}(z) = 0.2^2\times 100\times p(1-p) = 0.04\times100\times0.25 = 1 \Rightarrow \text{SD}=1
$$
So the injected multiplicative noise has ~10 % relative magnitude — a substantial but not destructive perturbation, which is exactly the regularizing signal.

---

### N7. He vs Xavier initialisation through 20 ReLU layers

Layer width $n=512$, ReLU.
- **Xavier** ($\mathrm{Var}(w) = 1/n$ forward form): per layer, $\mathrm{Var}(z^{(\ell)}) = n\cdot\frac1n\cdot\frac12\mathrm{Var}(z^{(\ell-1)}) = \frac12 \mathrm{Var}(z^{(\ell-1)})$.
  After 20 layers: factor $2^{-20} = 9.5\times 10^{-7}$ — activations vanish.
- **He** ($\mathrm{Var}(w)=2/n$): $\mathrm{Var}(z^{(\ell)}) = n\cdot\frac2n\cdot\frac12\mathrm{Var}(z^{(\ell-1)}) = \mathrm{Var}(z^{(\ell-1)})$.
  After 20 layers: factor $1.0$ — perfectly preserved ✓

---

### N8. Bias–variance numerically

Three models trained on 100 bootstrap samples, evaluated at one test point with $f(x)=2.0$, $\sigma^2=0.1$:

| Model | $\bar f(x)$ | $\mathrm{Var}(\hat f)$ | Bias² | Total = Bias²+Var+σ² |
|---|---|---|---|---|
| Linear (degree 1) | 1.20 | 0.02 | $(2.0-1.2)^2 = 0.64$ | 0.76 |
| Degree 4 | 1.95 | 0.15 | 0.0025 | **0.2525** |
| Degree 15 | 2.01 | 1.40 | 0.0001 | 1.5001 |

The degree-4 model wins — not because it has the lowest bias or lowest variance, but the best **sum**.

---

### N9. Number of linear regions

A ReLU MLP with $n=2$ inputs, $L=4$ hidden layers of width $w=10$:
$$
\mathcal R \ge \left\lfloor \tfrac{10}{2}\right\rfloor^{2(4-1)}\sum_{j=0}^{2}\binom{10}{j}
= 5^{6}\,(1 + 10 + 45) = 15\,625 \times 56 = \mathbf{875\,000}
$$
A single hidden layer with the same total $40$ units gives at most $\sum_{j=0}^{2}\binom{40}{j} = 1+40+780 = \mathbf{821}$.
**Depth buys ~1000× more regions with identical parameter budget.**

---

## 12. Viva / Exam Pointers

**Likely long questions**
1. Derive the four backpropagation equations from first principles; state the computational complexity.
2. Solve XOR with a 2-2-1 network (threshold and ReLU); explain why one layer fails.
3. Derive the bias–variance decomposition.
4. Show that $L_1$ produces exact zeros (soft-thresholding) while $L_2$ does not.
5. Prove that early stopping is equivalent to $L_2$ regularization with $\lambda \approx 1/(\eta\tau)$.
6. Derive the Xavier and He initialisation variances.
7. Show that sigmoid+BCE (and softmax+CCE) eliminate the learning-slowdown factor.
8. Complete one full forward–backward–update pass on a small network (numerical).

**Traps**
- $\delta$ is $\partial\mathcal L/\partial z$ (**pre**-activation), not $\partial\mathcal L/\partial a$.
- BP2 uses $\mathbf W^{(\ell+1)\mathsf T}$ — the weights of the layer **ahead**, not the current layer.
- With softmax output the delta uses the **full Jacobian**; only with CE does it simplify to $\hat y - t$.
- Dropout scaling: inverted dropout scales at **train** time by $1/p$; classical dropout scales at test time by $p$. Do not do both.
- BatchNorm at inference uses **running statistics**, not batch statistics.
- Bias terms are not usually $L_2$-regularized (they don't cause overfitting and regularizing them adds bias).

**One-line formula sheet**

$$
\boldsymbol\delta^{(L)} = \nabla_a\mathcal L\odot\varphi'(z^{(L)}) \;\;|\;\;
\boldsymbol\delta^{(\ell)} = (W^{(\ell+1)\mathsf T}\boldsymbol\delta^{(\ell+1)})\odot\varphi'(z^{(\ell)}) \;\;|\;\;
\nabla_{W^{(\ell)}} = \boldsymbol\delta^{(\ell)}a^{(\ell-1)\mathsf T}
$$
$$
\mathbb E[(y-\hat f)^2] = \text{Bias}^2 + \text{Var} + \sigma^2 \;\;|\;\;
\tilde w_i^{L_2} = \tfrac{\lambda_i}{\lambda_i+\lambda}w_i^\star \;\;|\;\;
\tilde w_i^{L_1} = \mathrm{sgn}(w^\star_i)(|w^\star_i|-\lambda/h_i)_+
$$
$$
\lambda_{\text{early-stop}}\approx \tfrac{1}{\eta\tau}\;\;|\;\;
\mathrm{Var}(w)_{\text{He}} = 2/n_{\text{in}}\;\;|\;\;
\mathrm{Var}(w)_{\text{Xavier}} = 2/(n_{\text{in}}+n_{\text{out}}) \;\;|\;\;
0<\eta<2/\lambda_{\max}
$$

---

*Previous: [Unit I](./Unit-1.md) · Next: [Unit III: Feedbackward Neural Networks](./Unit-3.md)*
