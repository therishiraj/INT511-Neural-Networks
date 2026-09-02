# Unit I — Introduction to Neural Networks

> **INT511 – Neural Networks** | M.Tech Level Notes
> **Coverage:** Overview of neural networks · Biological inspiration and artificial neurons · Types of learning · McCulloch–Pitts neural network · Perceptron · Activation functions (Threshold, Sigmoid, Tanh, ReLU, Softmax)

---

## Table of Contents

1. [Overview of Neural Networks](#1-overview-of-neural-networks)
2. [Biological Inspiration and Artificial Neurons](#2-biological-inspiration-and-artificial-neurons)
3. [Types of Learning](#3-types-of-learning)
4. [McCulloch–Pitts Neuron](#4-mcculloch–pitts-neuron)
5. [The Perceptron](#5-the-perceptron)
6. [Perceptron Convergence Theorem (Full Proof)](#6-perceptron-convergence-theorem-full-proof)
7. [Capacity of a Perceptron — Cover's Counting Theorem](#7-capacity-of-a-perceptron--covers-counting-theorem)
8. [Activation Functions](#8-activation-functions)
9. [Solved Numericals](#9-solved-numericals)
10. [Viva / Exam Pointers](#10-viva--exam-pointers)

---

## 1. Overview of Neural Networks

### 1.1 Formal definition

An **artificial neural network (ANN)** is a parameterised, directed, weighted computational graph
$G = (V, E)$ that realises a family of functions

$$
\mathcal{F}_\Theta = \{\, f_\theta : \mathcal{X} \subseteq \mathbb{R}^{n} \to \mathcal{Y} \subseteq \mathbb{R}^{m} \;\mid\; \theta \in \Theta \subseteq \mathbb{R}^{P} \,\}
$$

where each node applies an affine map followed by a (usually non-linear) scalar function, and
learning is the search for $\theta^\star$ minimising an empirical risk

$$
\theta^\star = \arg\min_{\theta \in \Theta} \; \underbrace{\frac{1}{N}\sum_{i=1}^{N} \mathcal{L}\!\left(f_\theta(x_i), y_i\right)}_{\hat{R}_{\text{emp}}(\theta)} \;+\; \lambda\,\Omega(\theta).
$$

**Statistical learning framing.** The true objective is the *expected risk*
$R(\theta) = \mathbb{E}_{(x,y)\sim \mathcal{D}}[\mathcal{L}(f_\theta(x), y)]$, which is inaccessible.
The excess risk decomposes as

$$
R(\hat\theta) - R^\star = \underbrace{\left(R(\hat\theta) - \inf_{\theta}R(\theta)\right)}_{\text{estimation error}} + \underbrace{\left(\inf_{\theta}R(\theta) - R^\star\right)}_{\text{approximation error}} .
$$

Neural networks are attractive because the **approximation error term can be driven to zero** for a wide class of targets (Universal Approximation), while depth controls the growth of the parameter count needed to do so.

### 1.2 The three defining properties

| Property | Meaning | Consequence |
|---|---|---|
| **Massive parallelism** | $O(10^2$–$10^{11})$ simple units acting simultaneously | Fault tolerance, graceful degradation |
| **Distributed representation** | A concept is a *pattern of activity*, not a single unit | Robustness to unit failure, generalisation |
| **Adaptivity** | Free parameters modified by data | Learning replaces explicit programming |

### 1.3 Universal Approximation Theorem (Cybenko 1989, Hornik 1991)

Let $\sigma:\mathbb{R}\to\mathbb{R}$ be continuous, bounded and non-constant (a *discriminatory* / non-polynomial function). Then for every $f \in C(I_n)$, $I_n=[0,1]^n$, and $\varepsilon>0$ there exist $M\in\mathbb{N}$, $\alpha_j, b_j \in \mathbb{R}$, $w_j \in \mathbb{R}^n$ such that

$$
F(x) = \sum_{j=1}^{M} \alpha_j\, \sigma\!\left(w_j^{\mathsf T} x + b_j\right), \qquad \sup_{x\in I_n} |F(x) - f(x)| < \varepsilon .
$$

**Leshno–Pinkus (1993) sharpening:** a shallow network is universal **iff** $\sigma$ is *not* a polynomial.

**Critical reading (M.Tech level).**
- The theorem is **existential**, not constructive: it says nothing about how to *find* $\alpha_j, w_j$.
- Width may be **exponential** in $n$: Barron's bound gives $\|f - F\|_{L^2}^2 \le C_f^2 / M$ where $C_f = \int \|\omega\|\,|\hat f(\omega)|\,d\omega$ is the *Barron constant*; for badly-behaved $f$, $C_f$ blows up.
- **Depth separation** (Telgarsky 2016): there exist functions computable by a ReLU net of depth $k^2$ with $O(k)$ units that require $\Omega(2^{k})$ units at depth $k$. This is the theoretical justification of *deep* learning.

### 1.4 Historical timeline (with the two "AI winters")

```
1943  McCulloch & Pitts     Threshold logic neuron; NN = universal Boolean machine
1949  Hebb                  "Cells that fire together wire together" -> Hebbian rule
1958  Rosenblatt            Perceptron + convergence theorem
1960  Widrow & Hoff         ADALINE, LMS (delta) rule  -- first gradient method
1969  Minsky & Papert       "Perceptrons": XOR limitation  ==> WINTER I
1974  Werbos                Backpropagation (PhD thesis, ignored)
1982  Hopfield              Energy function, associative memory (physics revival)
1982  Kohonen               Self-Organizing Map
1985  Ackley/Hinton/Sejn.   Boltzmann machine
1986  Rumelhart/Hinton/W.   Backprop popularised (PDP volumes)
1989  Cybenko / LeCun       UAT / CNN with backprop (LeNet)
1995  Vapnik                SVM outperforms NNs ==> WINTER II
1997  Hochreiter&Schmid.    LSTM
2006  Hinton                DBN + greedy layerwise pretraining ("deep learning")
2012  Krizhevsky            AlexNet on GPU -> ImageNet breakthrough
2014-17 GAN, ResNet, BatchNorm, Adam, Attention/Transformer
2018+ Self-supervised pretraining, scaling laws, foundation models
```

---

## 2. Biological Inspiration and Artificial Neurons

### 2.1 The biological neuron

```
                       Dendrites (input, ~10^4 synapses)
                        \  |  /
                         \ | /
          ┌───────────────●───────────────┐
          │           SOMA (cell body)    │   Spatio-temporal summation
          │   V_m evolves per cable eqn   │   of post-synaptic potentials
          └───────────────┬───────────────┘
                          │  Axon hillock  -- threshold ≈ -55 mV
                          │
        ══════════════════╪══════════════════   AXON (myelinated,
                          │                      saltatory conduction)
                    ┌─────┴─────┐
                Axon terminals / synaptic boutons
                    │  neurotransmitter release
                    ▼
                Next neuron's dendrite
```

**Integrate-and-fire dynamics** (the biophysical ancestor of the artificial neuron):

$$
\tau_m \frac{dV(t)}{dt} = -\big(V(t)-V_{\text{rest}}\big) + R_m I(t),
\qquad \text{fire and reset if } V(t) \ge V_{\text{th}}
$$

with $\tau_m = R_m C_m \approx 10\text{–}20\ \text{ms}$. Integrating the linear ODE over a synaptic input current gives an exponentially-weighted sum of inputs — i.e. **a weighted sum followed by a threshold**, which is exactly the McCulloch–Pitts abstraction.

**Synaptic plasticity** — the biological substrate of "weights":
- **LTP / LTD** (long-term potentiation/depression) modify synaptic efficacy $w_{ij}$.
- **STDP** (spike-timing-dependent plasticity):

  $$
  \Delta w =
  \begin{cases}
  A_{+}\exp(-\Delta t/\tau_{+}), & \Delta t = t_{\text{post}} - t_{\text{pre}} > 0 \quad (\text{potentiation})\\[4pt]
  -A_{-}\exp(\Delta t/\tau_{-}), & \Delta t < 0 \quad (\text{depression})
  \end{cases}
  $$

This is the *causal, temporally asymmetric* refinement of Hebb's rule.

### 2.2 The artificial neuron (Perceptron unit / node)

```
    x₁ ──w₁────┐
    x₂ ──w₂────┤        ┌──────────┐         ┌────────┐
     ⋮    ⋮     ├──►  Σ │ v = wᵀx+b│  ──►  φ │  φ(v)  │ ──► y
    xₙ ──wₙ────┤        └──────────┘         └────────┘
     1 ──b ────┘        (induced local        (activation
                          field / net input)     function)
```

$$
\boxed{\;v_k = \sum_{j=1}^{n} w_{kj}x_j + b_k = \mathbf{w}_k^{\mathsf T}\mathbf{x} + b_k,\qquad y_k = \varphi(v_k)\;}
$$

$v_k$ is called the **induced local field** (Haykin) or *net input / pre-activation*.

**Bias absorption trick.** Set $x_0 \equiv 1$, $w_{k0} \equiv b_k$, so $v_k = \tilde{\mathbf w}_k^{\mathsf T}\tilde{\mathbf x}$ with $\tilde{\mathbf x}\in\mathbb{R}^{n+1}$. This is used throughout the perceptron proof.

**Geometric meaning.** $v_k = 0$ defines a hyperplane $H$ with unit normal $\mathbf w/\|\mathbf w\|$ and signed distance from origin $-b/\|\mathbf w\|$. Distance of any point $\mathbf x_0$:
$$
d(\mathbf x_0, H) = \frac{|\mathbf w^{\mathsf T}\mathbf x_0 + b|}{\|\mathbf w\|_2}.
$$

### 2.3 Biological vs Artificial — quantitative comparison

| Aspect | Biological neuron | Artificial neuron |
|---|---|---|
| Signal | Spike train (all-or-none, ~1 ms) | Real-valued scalar |
| Coding | Rate / temporal / population code | Analogue magnitude |
| Speed | ~10⁻³ s per operation | ~10⁻⁹ s |
| Count | ~8.6×10¹⁰ neurons, ~10¹⁴–10¹⁵ synapses | 10⁶–10¹² parameters |
| Energy | ~20 W whole brain | ~10²–10⁶ W (GPU cluster) |
| Learning | STDP, neuromodulation, local | Global gradient (backprop) — **not biologically plausible** (weight transport problem) |
| Topology | Sparse, recurrent, small-world | Usually dense, layered, feedforward |
| Precision | Stochastic, noisy, ~1–2 bits | FP32/FP16/INT8 deterministic |

> **Interview point:** backprop requires the *transpose* of the forward weight matrix in the backward pass ("weight transport problem"). Biologically plausible surrogates: feedback alignment, target propagation, predictive coding, equilibrium propagation.

---

## 3. Types of Learning

### 3.1 Taxonomy by supervision signal

```
                          LEARNING
                             │
   ┌───────────────┬─────────┴─────────┬──────────────────┐
Supervised     Unsupervised       Reinforcement      Hybrid / other
(x, y)           (x only)          (s,a,r,s')      semi-, self-, active,
                                                    transfer, meta-
```

**(a) Supervised learning.** Data $\mathcal{D}=\{(x_i,y_i)\}_{i=1}^N \sim \mathcal{D}^N$. Learn $f:\mathcal X\to\mathcal Y$.
- *Regression:* $\mathcal Y = \mathbb{R}^m$, typically MSE loss.
- *Classification:* $\mathcal Y = \{1,\dots,C\}$, cross-entropy loss.
- Error-correction: $\Delta w_{kj} = \eta\, e_k\, x_j$ where $e_k = d_k - y_k$ (delta / LMS rule).

**(b) Unsupervised learning.** Only $\{x_i\}$. Objectives: density estimation $p(x)$, clustering, dimensionality reduction, generative modelling. Includes **Hebbian** and **competitive** learning (Units IV & V).

**(c) Reinforcement learning.** MDP $(\mathcal S,\mathcal A,P,r,\gamma)$; maximise
$$
J(\pi) = \mathbb{E}_\pi\!\left[\sum_{t=0}^{\infty}\gamma^{t} r_t\right],\qquad
Q^\pi(s,a) = \mathbb{E}\big[r + \gamma Q^\pi(s',a')\big].
$$
Feedback is a **scalar, delayed, evaluative** signal — not a target vector.

**(d) Semi-supervised / self-supervised.** $N_\ell \ll N_u$; exploit the cluster/manifold assumption, or invent a pretext task (masking, contrastive InfoNCE) so that supervision is generated from the data itself.

### 3.2 Taxonomy by learning *rule* (Haykin's five)

| # | Rule | Update | Character |
|---|---|---|---|
| 1 | **Error-correction** | $\Delta w_{kj}=\eta e_k x_j$ | Supervised, local, gradient of $\tfrac12 e^2$ |
| 2 | **Hebbian** | $\Delta w_{kj}=\eta y_k x_j$ | Unsupervised, correlational, **unstable** (unbounded growth) |
| 3 | **Competitive** | $\Delta w_{kj}=\eta(x_j-w_{kj})$ for winner only | Unsupervised, WTA, clustering |
| 4 | **Boltzmann** | $\Delta w_{kj}=\eta(\rho_{kj}^{+}-\rho_{kj}^{-})$ | Stochastic, correlation difference (Unit III) |
| 5 | **Memory-based** | store $\{(x_i,d_i)\}$, use local neighbourhood | Lazy learner (k-NN, RBF) |

**Oja's rule** — stabilising Hebb by normalisation. Starting from
$w(t+1) = \dfrac{w + \eta y x}{\|w + \eta y x\|}$ and expanding to first order in $\eta$:

$$
\boxed{\;\Delta w = \eta\, y\,(x - y\,w)\;}
$$

At convergence $\mathbb{E}[\Delta w]=0 \Rightarrow \mathbf{C}w = \lambda w$ with $\mathbf C=\mathbb E[xx^{\mathsf T}]$ and $\|w\|=1$: **a single Hebbian neuron with Oja's rule extracts the first principal component.** (Direct bridge to Unit IV.)

---

## 4. McCulloch–Pitts Neuron

### 4.1 Definition (1943)

A MP neuron has $n$ **excitatory** inputs $x_1..x_n \in\{0,1\}$, $m$ **inhibitory** inputs $z_1..z_m\in\{0,1\}$, and a threshold $\theta \in \mathbb{Z}$:

$$
y =
\begin{cases}
1, & \text{if } \displaystyle\sum_{i=1}^{n} x_i \ge \theta \;\text{ and }\; \sum_{j=1}^{m} z_j = 0\\[6pt]
0, & \text{otherwise}
\end{cases}
$$

**Absolute inhibition:** a *single* active inhibitory input vetoes firing regardless of excitation.

**Constraints (why it is not a perceptron):**
- weights are fixed at $+1$ (excitatory) / veto (inhibitory) — **no learning**;
- inputs and output are strictly binary;
- unit time delay per neuron ⇒ networks compute in discrete time steps.

### 4.2 Realising Boolean functions

Let $g(x)=\sum x_i$ and $y=f(g)= \mathbb{1}[g\ge\theta]$.

| Function | Realisation | $\theta$ | Check |
|---|---|---|---|
| AND($x_1,x_2$) | 2 excitatory | $\theta=2$ | $1{+}1\ge2$ ✓ |
| OR($x_1,x_2$) | 2 excitatory | $\theta=1$ | any one suffices |
| NOT($x$) | 1 inhibitory, 0 excitatory | $\theta=0$ | $x{=}0\Rightarrow y{=}1$; $x{=}1\Rightarrow$ veto |
| NOR | 2 inhibitory | $\theta=0$ | fires only if both 0 |
| $x_1 \wedge \overline{x_2}$ | $x_1$ excit., $x_2$ inhib. | $\theta=1$ | ✓ |
| NAND | $\overline{x_1}\vee\overline{x_2}$: two-layer | — | universal gate ⇒ **MP nets are Turing-complete for combinational logic** |

### 4.3 XOR with MP neurons (two layers required)

$$
x_1 \oplus x_2 = (x_1 \wedge \overline{x_2}) \;\vee\; (\overline{x_1} \wedge x_2)
$$

```
        x₁ ──(+)──►┌───────┐
                   │ N₁ θ=1│───(+)──►┌────────┐
        x₂ ──(–)──►└───────┘         │ N₃ θ=1 │──► y = x₁ ⊕ x₂
                                     │  (OR)  │
        x₁ ──(–)──►┌───────┐         └────────┘
                   │ N₂ θ=1│───(+)──►    ▲
        x₂ ──(+)──►└───────┘  ───────────┘
        (+) excitatory   (–) inhibitory
```

**Truth-table verification**

| $x_1$ | $x_2$ | $N_1$ (x₁ ∧ ¬x₂) | $N_2$ (¬x₁ ∧ x₂) | $y=N_1\vee N_2$ | XOR |
|---|---|---|---|---|---|
| 0 | 0 | 0 | 0 | 0 | 0 ✓ |
| 0 | 1 | 0 (veto) | 1 | 1 | 1 ✓ |
| 1 | 0 | 1 | 0 (veto) | 1 | 1 ✓ |
| 1 | 1 | 0 (veto) | 0 (veto) | 0 | 0 ✓ |

Latency: 2 time steps (one per layer).

### 4.4 Geometric view

An MP neuron implements $\mathbb{1}\!\left[\mathbf 1^{\mathsf T}x \ge \theta\right]$ — a hyperplane with **normal fixed along $\mathbf 1$**; only the offset $\theta$ is free. Hence it can only realise **symmetric threshold (majority-type) functions**. The perceptron generalises this by letting the normal rotate.

**Counting argument.** Number of Boolean functions of $n$ variables $=2^{2^n}$. Number of *linearly separable* ones, $LS(n)$:

| $n$ | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| $2^{2^n}$ | 4 | 16 | 256 | 65 536 | 4.29×10⁹ |
| $LS(n)$ | 4 | 14 | 104 | 1 882 | 94 572 |
| fraction | 1.0 | 0.875 | 0.406 | 0.0287 | 2.2×10⁻⁵ |

The fraction $\to 0$ **super-exponentially**: single-layer separability is a vanishingly rare property. This is the quantitative form of the Minsky–Papert critique.

---

## 5. The Perceptron

### 5.1 Model

$$
y = \mathrm{sgn}(v) = \mathrm{sgn}\!\big(\mathbf w^{\mathsf T}\mathbf x + b\big),\qquad
\mathrm{sgn}(v)=\begin{cases}+1 & v\ge 0\\ -1 & v<0\end{cases}
$$

Difference from MP: **real-valued adjustable weights**, real-valued inputs, a learning algorithm, and a bias.

### 5.2 Perceptron learning algorithm (error-driven, online)

> **Algorithm (Rosenblatt, 1958)**
> 1. Initialise $\mathbf w(0) = \mathbf 0$ (or small random), $t \leftarrow 0$, choose $\eta>0$.
> 2. For each epoch, for each sample $(\mathbf x_i, d_i)$, $d_i\in\{+1,-1\}$:
> &nbsp;&nbsp;&nbsp;**a.** $y_i(t) = \mathrm{sgn}(\mathbf w(t)^{\mathsf T}\mathbf x_i)$   *(bias absorbed)*
> &nbsp;&nbsp;&nbsp;**b.** If $y_i(t) \ne d_i$: $\; \mathbf w(t+1) = \mathbf w(t) + \eta\, d_i\, \mathbf x_i$; $\;t\leftarrow t+1$
> &nbsp;&nbsp;&nbsp;**c.** Else $\mathbf w(t+1) = \mathbf w(t)$ *(no update on correct classification)*
> 3. Repeat until an epoch passes with zero errors.

Equivalent compact form: $\;\Delta\mathbf w = \tfrac{\eta}{2}\,(d_i - y_i)\,\mathbf x_i$, since $d_i-y_i \in\{0,\pm2\}$.

**Why the update helps.** After an update on a misclassified $(\mathbf x_i,d_i)$:
$$
d_i\,\mathbf w(t+1)^{\mathsf T}\mathbf x_i = d_i\,\mathbf w(t)^{\mathsf T}\mathbf x_i + \eta\, d_i^2\|\mathbf x_i\|^2
= \underbrace{d_i\,\mathbf w(t)^{\mathsf T}\mathbf x_i}_{<0} + \eta\|\mathbf x_i\|^2 ,
$$
i.e. the margin on that sample **strictly increases** by $\eta\|\mathbf x_i\|^2$.

### 5.3 Perceptron criterion as a loss function

The perceptron rule is (sub-)gradient descent on

$$
J_p(\mathbf w) = \sum_{i \in \mathcal M} -\,d_i\, \mathbf w^{\mathsf T}\mathbf x_i
= \sum_{i=1}^{N}\max\!\big(0,\; -d_i \mathbf w^{\mathsf T}\mathbf x_i\big),
$$

$\mathcal M$ = misclassified set. Indeed $\nabla_{\mathbf w} J_p = -\sum_{i\in\mathcal M} d_i \mathbf x_i$, so $\mathbf w \leftarrow \mathbf w - \eta\nabla J_p$ reproduces step 2b.

**Comparison of margin-based losses** (with $z = d\,\mathbf w^{\mathsf T}\mathbf x$):

| Model | Loss $\ell(z)$ | Property |
|---|---|---|
| Perceptron | $\max(0,-z)$ | zero loss at $z=0^+$ → no margin guarantee |
| SVM (hinge) | $\max(0, 1-z)$ | enforces margin ⇒ unique maximum-margin solution |
| Logistic | $\log(1+e^{-z})$ | smooth, probabilistic |
| ADALINE (LMS) | $\tfrac12 (d - \mathbf w^{\mathsf T}\mathbf x)^2$ | uses **linear** output, not $\mathrm{sgn}$ |

### 5.4 Perceptron vs ADALINE (Widrow–Hoff)

| | Perceptron | ADALINE |
|---|---|---|
| Error computed on | $\mathrm{sgn}(v)$ (post-activation) | $v$ (pre-activation) |
| Update | $\eta(d-y)x$, $y=\pm1$ | $\eta(d-v)x$ |
| Loss | Perceptron criterion (piecewise linear) | MSE (quadratic, convex, smooth) |
| Non-separable data | oscillates forever | converges to LMS optimum |
| Stability bound | — | $0 < \eta < 2/\lambda_{\max}(\mathbf R)$, $\mathbf R=\mathbb E[xx^{\mathsf T}]$ |

---

## 6. Perceptron Convergence Theorem (Full Proof)

### 6.1 Statement (Novikoff, 1962)

Let $\mathcal D=\{(\mathbf x_i,d_i)\}_{i=1}^N$, $d_i\in\{\pm1\}$, satisfy:
1. **Boundedness:** $\|\mathbf x_i\| \le R$ for all $i$;
2. **Linear separability with margin $\gamma>0$:** $\exists\, \mathbf w^\star$ with $\|\mathbf w^\star\|=1$ such that $d_i\,\mathbf w^{\star\mathsf T}\mathbf x_i \ge \gamma \;\forall i$.

Then the perceptron algorithm started at $\mathbf w(0)=\mathbf 0$ with $\eta=1$ makes at most

$$
\boxed{\; k_{\max} \le \left(\frac{R}{\gamma}\right)^{2} \;}
$$

updates (mistakes), **independent of $N$ and of the dimension $n$**.

### 6.2 Proof

Let $\mathbf w_k$ be the weight vector *after* the $k$-th mistake, $\mathbf w_0=\mathbf 0$. Suppose the $k$-th mistake occurs on $(\mathbf x_{i_k}, d_{i_k})$, so $\mathbf w_k = \mathbf w_{k-1} + d_{i_k}\mathbf x_{i_k}$.

**Step 1 — Lower bound (the projection onto $\mathbf w^\star$ grows linearly).**

$$
\mathbf w^{\star\mathsf T}\mathbf w_k = \mathbf w^{\star\mathsf T}\mathbf w_{k-1} + d_{i_k}\,\mathbf w^{\star\mathsf T}\mathbf x_{i_k}
\;\ge\; \mathbf w^{\star\mathsf T}\mathbf w_{k-1} + \gamma .
$$

By induction from $\mathbf w^{\star\mathsf T}\mathbf w_0 = 0$:

$$
\mathbf w^{\star\mathsf T}\mathbf w_k \;\ge\; k\gamma. \quad (6.1)
$$

**Step 2 — Upper bound (the norm grows only as $\sqrt{k}$).**

$$
\|\mathbf w_k\|^2 = \|\mathbf w_{k-1}\|^2 + 2\,d_{i_k}\underbrace{\mathbf w_{k-1}^{\mathsf T}\mathbf x_{i_k}d_{i_k}/d_{i_k}}_{\text{see below}} + \|\mathbf x_{i_k}\|^2 .
$$

Precisely, $\|\mathbf w_k\|^2 = \|\mathbf w_{k-1}\|^2 + 2\,d_{i_k}\mathbf w_{k-1}^{\mathsf T}\mathbf x_{i_k} + d_{i_k}^2\|\mathbf x_{i_k}\|^2$. Because a **mistake** occurred, $d_{i_k}\mathbf w_{k-1}^{\mathsf T}\mathbf x_{i_k} \le 0$; and $d_{i_k}^2=1$, $\|\mathbf x_{i_k}\|^2\le R^2$. Hence

$$
\|\mathbf w_k\|^2 \le \|\mathbf w_{k-1}\|^2 + R^2 \;\;\Longrightarrow\;\; \|\mathbf w_k\|^2 \le kR^2 . \quad (6.2)
$$

**Step 3 — Combine via Cauchy–Schwarz.** Since $\|\mathbf w^\star\|=1$,

$$
k\gamma \;\overset{(6.1)}{\le}\; \mathbf w^{\star\mathsf T}\mathbf w_k \;\le\; \|\mathbf w^\star\|\,\|\mathbf w_k\| = \|\mathbf w_k\| \;\overset{(6.2)}{\le}\; \sqrt{k}\,R .
$$

Therefore $k\gamma \le \sqrt k R \Rightarrow \sqrt k \le R/\gamma \Rightarrow k \le R^2/\gamma^2$. $\blacksquare$

### 6.3 Remarks (what examiners probe)

- **Why $\eta$ does not matter (with $\mathbf w_0 = \mathbf 0$):** scaling $\eta$ scales every $\mathbf w_k$ by $\eta$; the sign of $\mathbf w^{\mathsf T}\mathbf x$ is unchanged, so the *sequence of mistakes is identical*.
- The bound depends on $\gamma$, the **geometric margin**; if data are barely separable ($\gamma\to0$), the bound explodes — connecting directly to the SVM idea of *maximising* $\gamma$.
- **Non-separable case:** the algorithm never terminates. Fixes: pocket algorithm (keep best-so-far weights), averaged perceptron, or the Freund–Schapire bound

  $$
  k \le \left(\frac{2(R + D)}{\gamma}\right)^{2}, \quad D = \sqrt{\textstyle\sum_i \xi_i^2},\;\; \xi_i = \max(0,\gamma - d_i\mathbf w^{\star\mathsf T}\mathbf x_i).
  $$

- The bound is **dimension-free** — a very early example of a margin-based generalisation guarantee.

### 6.4 Limitations of the single-layer perceptron

1. Cannot represent XOR / parity / connectedness (Minsky & Papert 1969).
2. Solution is **not unique** and generally **not max-margin** (depends on data order & init).
3. No probabilistic output.
4. Fraction of learnable Boolean functions $\to 0$ (Section 4.4).
5. Convergence time can be exponential in the input dimension for adversarial orderings even when separable (though *mistake count* is bounded).

---

## 7. Capacity of a Perceptron — Cover's Counting Theorem

**Theorem (Cover, 1965).** The number of *linearly separable dichotomies* of $N$ points in **general position** in $\mathbb R^{n}$ (through the origin) is

$$
C(N,n) = 2\sum_{k=0}^{n-1}\binom{N-1}{k}.
$$

**Consequences.**
- If $N \le n$: $C(N,n) = 2^{N}$ — *all* dichotomies are realisable (data are shatterable).
- The probability that a random dichotomy is separable, $P(N,n) = C(N,n)/2^N$, has a sharp transition at $N = 2n$: $P(2n, n) = 1/2$.
- **Perceptron capacity $= 2n$** patterns; **VC dimension of a hyperplane in $\mathbb R^n$ (with bias) $= n+1$.**

This theorem returns in **Unit V**, where it justifies why projecting into a *higher-dimensional* nonlinear feature space (RBF hidden layer) makes patterns linearly separable.

---

## 8. Activation Functions

### 8.1 Why nonlinearity is essential

For a network of $L$ purely linear layers,
$$
f(x) = W_L(W_{L-1}(\cdots W_1 x)) = \underbrace{\left(\textstyle\prod_{\ell} W_\ell\right)}_{=\;W_{\text{eff}}} x ,
$$
which is a **single linear map**. Depth adds *zero* representational power without $\varphi$. (It does change optimisation dynamics — "deep linear networks" are a real research object — but the function class is unchanged.)

### 8.2 Threshold / Heaviside / Step

$$
\varphi(v)=\begin{cases}1,& v\ge0\\ 0,& v<0\end{cases}
\qquad\text{(bipolar variant: } \mathrm{sgn}(v)\in\{-1,+1\})
$$

$$
\varphi'(v)=0 \;\;\forall v\ne0, \qquad \varphi'(0) \text{ undefined } (=\delta(v)\text{ distributionally}).
$$

**Verdict:** gradient is identically zero ⇒ **backpropagation impossible**. Historical only (MP, perceptron). Modern re-appearance: *straight-through estimator* in binarised networks, where the backward pass pretends $\varphi'(v)=\mathbb 1[|v|\le1]$.

### 8.3 Logistic Sigmoid

$$
\sigma(v)=\frac{1}{1+e^{-av}} \in (0,1), \qquad a = \text{slope parameter (usually }1)
$$

**Derivative (derive it, don't memorise it):**

$$
\frac{d\sigma}{dv} = \frac{d}{dv}\left(1+e^{-av}\right)^{-1}
= -\left(1+e^{-av}\right)^{-2}\cdot(-a e^{-av})
= a\,\frac{e^{-av}}{(1+e^{-av})^{2}}
$$
$$
= a\cdot \frac{1}{1+e^{-av}}\cdot\frac{e^{-av}}{1+e^{-av}}
= \boxed{\,a\,\sigma(v)\big(1-\sigma(v)\,)\,}
$$

**Key numbers.** $\sigma(0)=0.5$, $\sigma'(0)=a/4 = 0.25$ (**maximum**), $\sigma'(\pm4)\approx0.0177$, $\sigma'(\pm 6)\approx 0.0025$.

**Vanishing-gradient computation.** Through $L$ sigmoid layers the Jacobian magnitude is bounded by
$$
\left|\prod_{\ell=1}^{L}\sigma'(v_\ell)\,w_\ell\right| \le \left(\tfrac14 |w|_{\max}\right)^{L}.
$$
With $|w|\le1$, after $L=10$ layers the gradient is $\le 4^{-10} \approx 10^{-6}$. **This single line is the reason deep sigmoid nets failed pre-2006.**

**Other issues:**
- Output not zero-centred ⇒ all gradients w.r.t. a shared input have the same sign ⇒ **zig-zag** optimisation path.
- $\exp$ is comparatively expensive.
- **Still essential** as the output unit for binary classification (it is the inverse-logit, giving calibrated $P(y{=}1|x)$) and as gating in LSTM/GRU.

**Probabilistic identity.** For a two-class problem with equal-covariance Gaussians,
$$
P(C_1|x) = \sigma\!\left(\log\frac{p(x|C_1)P(C_1)}{p(x|C_2)P(C_2)}\right) = \sigma(w^{\mathsf T}x+b).
$$
So sigmoid is not arbitrary — it is the **canonical link of the Bernoulli exponential family**.

### 8.4 Hyperbolic Tangent

$$
\tanh(v)=\frac{e^{v}-e^{-v}}{e^{v}+e^{-v}} = \frac{1-e^{-2v}}{1+e^{-2v}} \in (-1,1)
$$

**Relation to sigmoid** (derive):
$$
\tanh(v) = 2\sigma(2v) - 1 \quad\Longleftrightarrow\quad \sigma(v) = \tfrac12\left(1+\tanh(v/2)\right).
$$

**Derivative:**
$$
\frac{d}{dv}\tanh(v) = \frac{(e^v+e^{-v})^2-(e^v-e^{-v})^2}{(e^v+e^{-v})^2} = 1-\tanh^2(v),\qquad \tanh'(0)=1 .
$$

**Advantage over sigmoid:** zero-centred output ⇒ zero-mean activations propagate ⇒ better-conditioned Hessian (LeCun's *Efficient BackProp*: recommended $\varphi(v)=1.7159\tanh(\tfrac23 v)$, which gives $\varphi(\pm1)=\pm1$ and $\varphi'$ maximal near the operating point). Still saturates.

### 8.5 ReLU and its family

$$
\text{ReLU}(v)=\max(0,v),\qquad
\text{ReLU}'(v)=\begin{cases}1,& v>0\\ 0,& v<0\\ \text{[0,1] (subgradient, take 0)},& v=0\end{cases}
$$

**Why it works:**
- Gradient is exactly **1** on the active half-line ⇒ no attenuation ⇒ trains 10–100× faster (AlexNet).
- Induces **sparse** codes (~50 % zeros at init) — biologically closer, computationally cheaper.
- Piecewise linear ⇒ the network partitions input space into convex polytopes; the number of linear regions of a ReLU net with $L$ layers of width $w$ on $n$ inputs grows as $\Omega\!\left((w/n)^{n(L-1)}w^{n}\right)$ — **exponential in depth**, the formal statement of depth's power.

**Problems:** not zero-centred; unbounded above; **dying ReLU** — if $v_k<0$ for *all* training inputs, $\partial \mathcal L/\partial w_k = 0$ permanently. Typically triggered by a large learning rate pushing $b$ very negative.

**Variants:**

| Name | Definition | Notes |
|---|---|---|
| Leaky ReLU | $\max(\alpha v, v)$, $\alpha{=}0.01$ | never dies |
| PReLU | same, $\alpha$ **learned** | +1 param/channel |
| ELU | $v$ if $v>0$ else $\alpha(e^{v}-1)$ | mean activation ≈0, smooth |
| SELU | $\lambda\,\text{ELU}_\alpha(v)$, $\lambda{=}1.0507,\alpha{=}1.6733$ | *self-normalising* fixed point $(\mu,\sigma^2)=(0,1)$ |
| Softplus | $\ln(1+e^{v})$ | smooth ReLU; $\frac{d}{dv}\text{softplus}=\sigma(v)$ |
| GELU | $v\,\Phi(v)\approx 0.5v\!\left(1+\tanh\!\big[\sqrt{2/\pi}(v+0.044715v^3)\big]\right)$ | stochastic-regulariser view; Transformers |
| Swish/SiLU | $v\,\sigma(\beta v)$ | found by NAS; smooth, non-monotone |
| Mish | $v\tanh(\text{softplus}(v))$ | smoother loss landscape |

**Weight initialisation is coupled to $\varphi$** (proved in Unit II):
$$
\text{Xavier/Glorot: } \mathrm{Var}(w)=\frac{2}{n_{\text{in}}+n_{\text{out}}} \;\;(\tanh);\qquad
\text{He/Kaiming: } \mathrm{Var}(w)=\frac{2}{n_{\text{in}}} \;\;(\text{ReLU}).
$$

### 8.6 Softmax (vector-valued)

$$
\boxed{\; y_i = \text{softmax}(\mathbf v)_i = \frac{e^{v_i}}{\sum_{j=1}^{C} e^{v_j}},\qquad y_i>0,\;\; \sum_{i=1}^{C} y_i = 1 \;}
$$

**Numerical stability (mandatory in practice).** Since softmax is invariant to a constant shift, $\text{softmax}(\mathbf v) = \text{softmax}(\mathbf v - c)$, always use $c = \max_j v_j$:
$$
y_i = \frac{e^{v_i - v_{\max}}}{\sum_j e^{v_j - v_{\max}}}
$$
otherwise $e^{v}$ overflows for $v \gtrsim 709$ (FP64) or $v\gtrsim 88$ (FP32).

**Jacobian derivation.** For $i = k$:
$$
\frac{\partial y_i}{\partial v_i}
= \frac{e^{v_i}S - e^{v_i}e^{v_i}}{S^2},\quad S=\textstyle\sum_j e^{v_j}
= y_i - y_i^2 = y_i(1-y_i).
$$
For $i \ne k$:
$$
\frac{\partial y_i}{\partial v_k} = \frac{0\cdot S - e^{v_i}e^{v_k}}{S^2} = -y_i y_k .
$$
Combined:
$$
\boxed{\;\frac{\partial y_i}{\partial v_k} = y_i(\delta_{ik} - y_k)
\quad\Longleftrightarrow\quad
\mathbf J = \mathrm{diag}(\mathbf y) - \mathbf y\mathbf y^{\mathsf T}\;}
$$
Note $\mathbf J$ is symmetric PSD, and $\mathbf J\mathbf 1 = \mathbf 0$ ⇒ **rank $C-1$** (consistent with the shift-invariance / one redundant degree of freedom).

**The key simplification with cross-entropy.** With $\mathcal L = -\sum_i t_i\ln y_i$ and one-hot $\mathbf t$:
$$
\frac{\partial \mathcal L}{\partial v_k}
= -\sum_i \frac{t_i}{y_i}\cdot y_i(\delta_{ik}-y_k)
= -\sum_i t_i\delta_{ik} + y_k\sum_i t_i
= \boxed{\,y_k - t_k\,}
$$
The messy Jacobian collapses to a plain error signal. (Same structure as sigmoid + binary cross-entropy — see Unit II §cost functions.)

**Temperature.** $y_i(T) = e^{v_i/T}/\sum_j e^{v_j/T}$. As $T\to0^+$, softmax $\to$ argmax (one-hot); as $T\to\infty$, $\to$ uniform $1/C$. Used in knowledge distillation, RL exploration, and sampling from language models. Softmax is also the **Gibbs/Boltzmann distribution** with $v_i = -E_i$, $T$ = temperature — the exact link to Unit III.

### 8.7 Comparison summary

| $\varphi$ | Range | $\varphi'$ range | Zero-centred | Saturates | Cost | Typical use |
|---|---|---|---|---|---|---|
| Threshold | $\{0,1\}$ | 0 a.e. | No | — | trivial | MP, perceptron |
| Sigmoid | $(0,1)$ | $(0,0.25]$ | No | both ends | high | binary output, gates |
| Tanh | $(-1,1)$ | $(0,1]$ | **Yes** | both ends | high | RNN hidden, shallow nets |
| ReLU | $[0,\infty)$ | $\{0,1\}$ | No | left only (dies) | **cheap** | default hidden |
| Leaky/PReLU | $\mathbb R$ | $\{\alpha,1\}$ | ~Yes | no | cheap | deep CNN |
| ELU/SELU | $(-\alpha,\infty)$ | $(0,1]$ | Yes | left soft | medium | self-normalising nets |
| GELU/Swish | $\approx(-0.28,\infty)$ | smooth | ~Yes | no | medium | Transformers |
| Softmax | $(0,1)^C$, sums 1 | Jacobian | — | yes | medium | multi-class output |

---

## 9. Solved Numericals

### N1. MP neuron — design a 3-input majority gate
**Ask:** Realise $y=1$ iff at least 2 of $x_1,x_2,x_3$ are 1.
**Solution:** All three excitatory, $\theta = 2$.
Check: $(1,1,0)\to g=2\ge2\Rightarrow1$ ✓; $(1,0,0)\to g=1<2\Rightarrow0$ ✓; $(1,1,1)\to3\ge2\Rightarrow1$ ✓.
As a perceptron: $w=(1,1,1)$, $b=-1.5$, $y=\mathbb 1[x_1+x_2+x_3-1.5\ge0]$.

---

### N2. Prove XOR is not linearly separable
Suppose $\exists (w_1,w_2,b)$ with $y=\mathbb 1[w_1x_1+w_2x_2+b\ge0]$ realising XOR. Then:

| Pattern | Requirement | Inequality |
|---|---|---|
| $(0,0)\to0$ | $b<0$ | (i) |
| $(0,1)\to1$ | $w_2+b\ge0$ | (ii) |
| $(1,0)\to1$ | $w_1+b\ge0$ | (iii) |
| $(1,1)\to0$ | $w_1+w_2+b<0$ | (iv) |

Add (ii)+(iii): $w_1+w_2+2b \ge 0 \Rightarrow w_1+w_2 \ge -2b$.
From (iv): $w_1+w_2 < -b$.
Hence $-2b \le w_1+w_2 < -b \Rightarrow -2b < -b \Rightarrow -b<0 \Rightarrow b>0$, contradicting (i). ∎

---

### N3. Perceptron training — full iteration table (AND gate, bipolar)

Data (bipolar, bias absorbed as $x_0=1$): $\eta=1$, $\mathbf w(0)=(0,0,0)$ with $\mathbf w=(b,w_1,w_2)$.

| $x_0$ | $x_1$ | $x_2$ | $d$ |
|---|---|---|---|
| 1 | −1 | −1 | −1 |
| 1 | −1 | +1 | −1 |
| 1 | +1 | −1 | −1 |
| 1 | +1 | +1 | +1 |

Rule: $y = \mathrm{sgn}(\mathbf w^{\mathsf T}\mathbf x)$ with $\mathrm{sgn}(0)=+1$; on error $\mathbf w \leftarrow \mathbf w + d\,\mathbf x$.

**Epoch 1**

| Step | $\mathbf x$ | $\mathbf w$ before | $v$ | $y$ | $d$ | Update? | $\mathbf w$ after |
|---|---|---|---|---|---|---|---|
| 1 | (1,−1,−1) | (0,0,0) | 0 | +1 | −1 | ✔ $-\mathbf x$ | (−1, 1, 1) |
| 2 | (1,−1,+1) | (−1,1,1) | −1−1+1=−1 | −1 | −1 | ✘ | (−1, 1, 1) |
| 3 | (1,+1,−1) | (−1,1,1) | −1+1−1=−1 | −1 | −1 | ✘ | (−1, 1, 1) |
| 4 | (1,+1,+1) | (−1,1,1) | −1+1+1=+1 | +1 | +1 | ✘ | (−1, 1, 1) |

**Epoch 2** — recheck pattern 1: $v = -1+(-1)+(-1) = -3 \Rightarrow y=-1=d$ ✓. Patterns 2–4 unchanged ✓.
**Converged in 1 update.** Final: $b=-1,\;w_1=w_2=1$; decision line $x_1+x_2 = 1$.

*Sanity check with the theorem:* $R = \max\|\mathbf x\| = \sqrt3$. A valid $\mathbf w^\star$ is $\tfrac{1}{\sqrt3}(-1,1,1)$, giving $\gamma = \min_i d_i\mathbf w^{\star\mathsf T}\mathbf x_i = 1/\sqrt3$. Bound: $k \le R^2/\gamma^2 = 3/(1/3) = 9$. Observed $k=1 \le 9$ ✓.

---

### N4. Perceptron with a non-trivial 2-D dataset

$\mathcal D$: $\;\mathbf x_1=(1,1)^{\mathsf T},d{=}{+}1$; $\mathbf x_2=(2,-2)^{\mathsf T},d{=}{-}1$; $\mathbf x_3=(-1,2)^{\mathsf T},d{=}{+}1$; $\mathbf x_4=(-2,1)^{\mathsf T},d{=}{-}1$.
Augmented $\tilde{\mathbf x}=(1,x_1,x_2)$, $\eta=1$, $\mathbf w_0=(0,0,0)$.

| It. | Sample | $\tilde{\mathbf x}$ | $\mathbf w$ | $v$ | $y$ | $d$ | New $\mathbf w$ |
|---|---|---|---|---|---|---|---|
| 1 | 1 | (1,1,1) | (0,0,0) | 0 | +1 | +1 | (0,0,0) |
| 2 | 2 | (1,2,−2) | (0,0,0) | 0 | +1 | −1 | (−1,−2,2) |
| 3 | 3 | (1,−1,2) | (−1,−2,2) | −1+2+4=5 | +1 | +1 | (−1,−2,2) |
| 4 | 4 | (1,−2,1) | (−1,−2,2) | −1+4+2=5 | +1 | −1 | (−2,−4,1) |
| 5 | 1 | (1,1,1) | (−2,−4,1) | −2−4+1=−5 | −1 | +1 | (−1,−3,2) |
| 6 | 2 | (1,2,−2) | (−1,−3,2) | −1−6−4=−11 | −1 | −1 | — |
| 7 | 3 | (1,−1,2) | (−1,−3,2) | −1+3+4=6 | +1 | +1 | — |
| 8 | 4 | (1,−2,1) | (−1,−3,2) | −1+6+2=7 | +1 | −1 | (−2,−5,1) |
| 9 | 1 | (1,1,1) | (−2,−5,1) | −2−5+1=−6 | −1 | +1 | (−1,−4,2) |
| 10 | 2 | (1,2,−2) | (−1,−4,2) | −1−8−4=−13 | −1 | −1 | — |
| 11 | 3 | (1,−1,2) | (−1,−4,2) | −1+4+4=7 | +1 | +1 | — |
| 12 | 4 | (1,−2,1) | (−1,−4,2) | −1+8+2=9 | +1 | −1 | (−2,−6,1) |
| 13 | 1 | (1,1,1) | (−2,−6,1) | −7 | −1 | +1 | (−1,−5,2) |
| 14 | 2 | — | (−1,−5,2) | −1−10−4=−15 | −1 | −1 | — |
| 15 | 3 | — | (−1,−5,2) | −1+5+4=8 | +1 | +1 | — |
| 16 | 4 | — | (−1,−5,2) | −1+10+2=11 | +1 | −1 | (−2,−7,1) |

**Observation:** the algorithm keeps cycling. Check separability: we need $b+w_1+w_2\ge0$, $b+2w_1-2w_2<0$, $b-w_1+2w_2\ge0$, $b-2w_1+w_2<0$. Adding (1)+(3): $2b+3w_2 \ge -0$ … a full LP shows the system **is** feasible (e.g. $w=(0,-1,1)$: $v_1=0\ \checkmark$, $v_2=-4<0\ \checkmark$, $v_3=3\ge0\ \checkmark$, $v_4=3$ ✗). Try $w=(0,-2,1)$: $v_1=-1$ ✗.
The point of this example: **not every "nice-looking" set is separable**, and the perceptron then oscillates indefinitely — motivating the pocket algorithm and, structurally, the multi-layer network of Unit II.

---

### N5. Activation-function arithmetic

Given $\mathbf x = (0.5,\,-1.0,\,2.0)^{\mathsf T}$, $\mathbf w=(0.4,\,-0.6,\,0.3)^{\mathsf T}$, $b=-0.2$.

$$v = 0.4(0.5) + (-0.6)(-1.0) + 0.3(2.0) - 0.2 = 0.2+0.6+0.6-0.2 = \mathbf{1.2}$$

| $\varphi$ | $\varphi(1.2)$ | $\varphi'(1.2)$ |
|---|---|---|
| Threshold | 1 | 0 |
| Sigmoid | $1/(1+e^{-1.2}) = 1/(1+0.30119) = 0.76852$ | $0.76852(1-0.76852)=0.17789$ |
| Tanh | $(e^{1.2}-e^{-1.2})/(e^{1.2}+e^{-1.2}) = (3.32012-0.30119)/(3.62131) = 0.83365$ | $1-0.83365^2 = 0.30502$ |
| ReLU | 1.2 | 1 |
| Leaky (0.01) | 1.2 | 1 |
| ELU ($\alpha{=}1$) | 1.2 | 1 |
| Softplus | $\ln(1+e^{1.2}) = \ln(4.32012)=1.46333$ | $\sigma(1.2)=0.76852$ |
| Swish ($\beta{=}1$) | $1.2\times0.76852 = 0.92222$ | $\sigma+v\sigma(1-\sigma) = 0.76852+1.2(0.17789)=0.98199$ |

*Consistency check:* $\tanh(1.2) = 2\sigma(2.4)-1 = 2(0.916827)-1 = 0.833655$ ✓.

---

### N6. Softmax and its Jacobian

Logits $\mathbf v = (2.0,\,1.0,\,0.1)$.
Shift by $v_{\max}=2.0$: $(0,\,-1.0,\,-1.9)$.
$e^{0}=1$, $e^{-1}=0.367879$, $e^{-1.9}=0.149569$. Sum $S = 1.517448$.

$$
\mathbf y = (0.659001,\;0.242433,\;0.098566),\qquad \textstyle\sum y_i = 1.000 \;✓
$$

Jacobian $\mathbf J = \mathrm{diag}(\mathbf y)-\mathbf y\mathbf y^{\mathsf T}$:

$$
\mathbf J=
\begin{pmatrix}
0.659001(1-0.659001) & -0.659001(0.242433) & -0.659001(0.098566)\\
-0.242433(0.659001) & 0.242433(1-0.242433) & -0.242433(0.098566)\\
-0.098566(0.659001) & -0.098566(0.242433) & 0.098566(1-0.098566)
\end{pmatrix}
$$
$$
=\begin{pmatrix}
\;\;0.224669 & -0.159773 & -0.064956\\
-0.159773 & \;\;0.183656 & -0.023897\\
-0.064956 & -0.023897 & \;\;0.088849
\end{pmatrix}
$$

Row sums: $0.224669-0.159773-0.064956 = 0.000$ ✓ (confirms $\mathbf J\mathbf 1=\mathbf 0$, rank $\le 2$).

If the true class is $t = 1$ (one-hot $(1,0,0)$):
$$\mathcal L = -\ln(0.659001) = 0.41703,\qquad \frac{\partial\mathcal L}{\partial \mathbf v} = \mathbf y - \mathbf t = (-0.340999,\;0.242433,\;0.098566).$$
Gradient sums to zero — softmax redistributes probability mass rather than creating it.

**Temperature effect:** with $T=0.5$, $\mathbf v/T = (4,2,0.2)$ ⇒ $\mathbf y \approx (0.8668,0.1173,0.0159)$ (sharper). With $T=5$, $\mathbf v/T=(0.4,0.2,0.02)$ ⇒ $\mathbf y \approx (0.3752,0.3072,0.3176)$ (flatter).

---

### N7. Vanishing gradient — quantitative
A 12-layer network with sigmoid activations, all weights $w=0.9$, all pre-activations near $0$ (so $\sigma'\approx0.25$). The gradient reaching layer 1 is scaled by

$$
\prod_{\ell=1}^{11}\sigma'(v_\ell)\,w_\ell \approx (0.25\times0.9)^{11} = 0.225^{11} = 1.05\times10^{-7}.
$$

With ReLU ($\varphi'=1$ when active): $(1\times0.9)^{11}=0.3138$ — **six orders of magnitude larger**.
If instead $w = 1.5$ with ReLU: $1.5^{11} = 86.5$ ⇒ **exploding** gradient, motivating gradient clipping and careful initialisation (Unit II).

---

### N8. Oja's rule reaches the principal eigenvector
Let $\mathbf C = \begin{pmatrix}4 & 1\\ 1& 3\end{pmatrix}$.
Eigenvalues: $\lambda^2-7\lambda+11=0 \Rightarrow \lambda = (7\pm\sqrt5)/2 = 4.61803,\,2.38197$.
Principal eigenvector for $\lambda_1=4.61803$: $(4-\lambda_1)u_1 + u_2 = 0 \Rightarrow u_2 = 0.61803\,u_1$, normalised $\mathbf u_1 = (0.85065,\,0.52573)^{\mathsf T}$.
Oja's rule $\Delta w = \eta y(x - yw)$ with $y=w^{\mathsf T}x$ converges in expectation to $\pm\mathbf u_1$ — i.e. a **single linear neuron performs PCA** (Unit IV).

---

## 10. Viva / Exam Pointers

**Likely long questions**
1. State and *prove* the perceptron convergence theorem; discuss what happens when data are not separable.
2. Show that XOR is not linearly separable and realise it with McCulloch–Pitts neurons.
3. Derive $\sigma'$, $\tanh'$ and the softmax Jacobian; explain the vanishing-gradient problem quantitatively.
4. Compare biological and artificial neurons; explain STDP and the weight-transport objection to backprop.
5. Explain Cover's theorem and the capacity $2n$ of a perceptron.

**Traps**
- $\sigma'(v) = \sigma(1-\sigma)$ is in terms of the **output**, not the input — this is why frameworks cache activations.
- Softmax Jacobian is a **matrix**, not a scalar; only when composed with cross-entropy does it collapse to $y-t$.
- The perceptron bound limits **mistakes**, not epochs or wall-clock time.
- MP neuron has **no learning** — do not write a "MP learning rule".
- "Sigmoid squashes to (0,1)" — open interval; it never *attains* 0 or 1 (hence $\log$ is safe but can underflow in FP32).

**One-line formula sheet**

$$
v=\mathbf w^{\mathsf T}\mathbf x + b \quad|\quad
\sigma' = \sigma(1-\sigma) \quad|\quad
\tanh' = 1-\tanh^2 \quad|\quad
\tanh(v)=2\sigma(2v)-1
$$
$$
\mathbf J_{\text{softmax}} = \mathrm{diag}(\mathbf y)-\mathbf y\mathbf y^{\mathsf T}\quad|\quad
\partial\mathcal L_{\text{CE}}/\partial \mathbf v = \mathbf y - \mathbf t \quad|\quad
k_{\max}\le (R/\gamma)^2 \quad|\quad
C(N,n)=2\!\sum_{k=0}^{n-1}\!\binom{N-1}{k}
$$

---

*End of Unit I — proceed to [Unit II: Feedforward Neural Networks](./Unit-2.md)*
