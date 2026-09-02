# Unit V — Advanced Neural Network Architectures

> **INT511 – Neural Networks** | M.Tech Level Notes
> **Coverage:** Self-Organizing Maps (introduction, architecture, applications) · Learning Vector Quantization (overview, algorithm, applications) · Radial Basis Function networks (structure, training methods, applications)

---

## Table of Contents

1. [Competitive Learning and Vector Quantization](#1-competitive-learning-and-vector-quantization)
2. [Self-Organizing Maps — Introduction](#2-self-organizing-maps--introduction)
3. [SOM Architecture](#3-som-architecture)
4. [SOM Algorithm and Analysis](#4-som-algorithm-and-analysis)
5. [SOM Quality Measures, Visualization, and Applications](#5-som-quality-measures-visualization-and-applications)
6. [Learning Vector Quantization](#6-learning-vector-quantization)
7. [Radial Basis Function Networks — Foundations](#7-radial-basis-function-networks--foundations)
8. [RBF Structure and Training Methods](#8-rbf-structure-and-training-methods)
9. [RBF vs MLP, and Applications](#9-rbf-vs-mlp-and-applications)
10. [Solved Numericals](#10-solved-numericals)
11. [Viva / Exam Pointers](#11-viva--exam-pointers)

---

## 1. Competitive Learning and Vector Quantization

### 1.1 Competitive learning

In a **competitive** layer, output neurons compete; only the **winner** (and possibly its neighbours) updates. This is unsupervised and implements clustering.

**Winner selection.** For input $\mathbf x$ and weight vectors $\{\mathbf w_j\}$:

$$
c = \arg\min_j \|\mathbf x - \mathbf w_j\|_2 \qquad\text{(minimum-distance criterion)}
$$

If all $\|\mathbf w_j\|$ are equal, this is equivalent to the **maximum-inner-product** criterion:

$$
\|\mathbf x-\mathbf w_j\|^2 = \|\mathbf x\|^2 - 2\mathbf w_j^{\mathsf T}\mathbf x + \|\mathbf w_j\|^2
\;\;\Longrightarrow\;\; \arg\min_j\|\mathbf x-\mathbf w_j\| = \arg\max_j \mathbf w_j^{\mathsf T}\mathbf x .
$$

**Standard competitive (WTA) update:**
$$
\Delta \mathbf w_j = \begin{cases}\eta(\mathbf x - \mathbf w_j), & j = c\\ \mathbf 0, & j\ne c\end{cases}
$$
The winner's weight vector moves **toward** the input — an instance of Haykin's third learning rule.

**Dead units.** A neuron whose initial weight is far from all data never wins and never learns. Fixes:
- **Conscience mechanism** (DeSieno): bias the distance by how often each unit has won, $d_j' = d_j - \gamma(1/M - f_j)$.
- **Leaky learning:** losers also update, with a much smaller rate $\eta_{\text{loser}} \ll \eta_{\text{winner}}$.
- **Frequency-sensitive competitive learning**, or initialising weights from randomly chosen data points.

### 1.2 Vector quantization

**VQ** partitions $\mathbb R^D$ into $M$ regions with a **codebook** $\mathcal W = \{\mathbf w_1,\dots,\mathbf w_M\}$ (the *code vectors*). The induced partition is the **Voronoi tessellation**:

$$
\mathcal V_j = \left\{\mathbf x\in\mathbb R^D \;:\; \|\mathbf x-\mathbf w_j\| \le \|\mathbf x-\mathbf w_k\|\;\forall k\right\}
$$

```
        Voronoi tessellation of the plane
        ┌──────┬───────┬──────────┐
        │  ●w₁ │       │    ●w₄   │      Each cell = all points
        │      │  ●w₂  ├─────┬────┤      closer to that code
        ├──────┴───┬───┤     │●w₅ │      vector than any other.
        │   ●w₃    │   │ ●w₆ │    │      Boundaries are
        │          │   │     │    │      perpendicular bisectors.
        └──────────┴───┴─────┴────┘
```

**Objective — quantization distortion:**
$$
\boxed{\;E = \frac12\sum_{j=1}^{M}\int_{\mathcal V_j}\|\mathbf x - \mathbf w_j\|^2 p(\mathbf x)\,d\mathbf x
\;\;\approx\;\; \frac{1}{2N}\sum_{i=1}^{N}\|\mathbf x_i - \mathbf w_{c(i)}\|^2\;}
$$

**Necessary conditions for optimality (Lloyd conditions):**
1. **Nearest-neighbour condition:** given $\mathcal W$, the optimal partition is the Voronoi tessellation.
2. **Centroid condition:** given the partition, the optimal code vector is the conditional mean $\mathbf w_j = \mathbb E[\mathbf x\mid \mathbf x\in\mathcal V_j]$.

Alternating the two is **Lloyd's algorithm = k-means**. Online competitive learning is its **stochastic-gradient** version:
$$
\frac{\partial E}{\partial \mathbf w_j} = -(\mathbf x - \mathbf w_j)\mathbb 1[j = c] \;\Longrightarrow\; \Delta\mathbf w_c = \eta(\mathbf x-\mathbf w_c) \;✓
$$

**Magnification / density matching.** For an optimal $D$-dimensional quantizer, the code-vector density satisfies
$$
\rho(\mathbf w) \propto p(\mathbf x)^{\frac{D}{D+2}} .
$$
The SOM in 1-D achieves $\rho \propto p^{2/3}$ — it *under-represents* high-density regions relative to the optimum. This is the **magnification factor** result (Ritter & Schulten).

### 1.3 Where the three topics of this unit sit

| | Supervision | Topology preserved | Purpose |
|---|---|---|---|
| **SOM** | Unsupervised | **Yes** (ordered lattice) | clustering + visualisation + topology-preserving mapping |
| **LVQ** | **Supervised** | No | classification by codebook refinement |
| **RBF** | Supervised | No | function approximation / classification via localised basis |

SOM/LVQ are *competitive*; RBF is *localised but cooperative* (all units contribute, weighted by distance).

---

## 2. Self-Organizing Maps — Introduction

### 2.1 Biological motivation

The cerebral cortex contains **topographic maps**: the somatosensory homunculus, retinotopic maps in V1, tonotopic maps in A1, and orientation/ocular-dominance columns. Neighbouring cortical neurons respond to neighbouring stimuli. Kohonen (1982) abstracted this into an algorithm.

**Key principle:** a high-dimensional input space is mapped onto a low-dimensional (usually 2-D) neural lattice such that **similar inputs excite nearby neurons** — a *topology-preserving* map.

### 2.2 The three principles of self-organization (Haykin)

1. **Self-amplification / competition** — modification of synaptic weights tends to self-amplify (Hebbian).
2. **Competition** — limited resources force neurons to compete; only the winner and its neighbourhood are strengthened.
3. **Cooperation** — the winner shares its success with topological neighbours, producing spatial order.

Plus **redundancy in the input data** as the raw material from which structure is extracted.

### 2.3 The "Mexican hat" lateral interaction

Biological cortical neurons excite near neighbours and inhibit distant ones:

```
   Lateral
   coupling
   strength
      │      ╱╲
      │     ╱  ╲            EXCITATORY (short range)
   ───┼────╱────╲──────────────────────► lateral distance
      │  ╱        ╲___╱‾‾‾╲___          INHIBITORY (medium)
      │ ╱                                weak/none (long)
```

The SOM simplifies this to a purely **excitatory, monotonically decreasing neighbourhood function** $h_{j,c}$, which is computationally equivalent for the purpose of ordering.

---

## 3. SOM Architecture

```
                    OUTPUT LAYER — 2-D lattice of neurons
                    (each neuron j has weight vector w_j ∈ R^D)

              ┌────┬────┬────┬────┬────┐
              │ ○  │ ○  │ ○  │ ○  │ ○  │       Hexagonal or
              ├────┼────┼────┼────┼────┤       rectangular grid.
              │ ○  │ ○  │ ●  │ ○  │ ○  │  ← BMU (winner) c
              ├────┼────┼────┼────┼────┤
              │ ○  │ ○  │ ○  │ ○  │ ○  │       Grid distance
              ├────┼────┼────┼────┼────┤       d(j,c) defines the
              │ ○  │ ○  │ ○  │ ○  │ ○  │       neighbourhood.
              └────┴────┴────┴────┴────┘
                 ▲    ▲    ▲    ▲    ▲
                 └────┴────┼────┴────┘   fully connected
                           │
              ┌────────────┴────────────┐
              │  x₁   x₂   x₃  ...  x_D │   INPUT LAYER (D-dim)
              └─────────────────────────┘
```

**Two distinct distance concepts — do not confuse them:**

| | Symbol | Space | Used for |
|---|---|---|---|
| **Input-space distance** | $\Vert \mathbf x - \mathbf w_j\Vert $ | $\mathbb R^D$ | finding the winner (BMU) |
| **Lattice/grid distance** | $d(j,c) = \Vert \mathbf r_j - \mathbf r_c\Vert $ | 2-D grid coordinates | computing the neighbourhood $h_{j,c}$ |

**Lattice topologies.** Rectangular (4 or 8 neighbours), **hexagonal** (6 neighbours — preferred, no directional bias), or toroidal (wrap-around, eliminates edge effects). Grid distance is Euclidean on lattice coordinates, or Manhattan/Chebyshev for discrete "bubble" neighbourhoods.

---

## 4. SOM Algorithm and Analysis

### 4.1 The algorithm

> **Algorithm: Kohonen SOM**
> **Initialisation:** small random $\mathbf w_j(0)$, or (better) sample from the data, or (best) linear initialisation along the first two principal components.
> Repeat for $t = 0,1,2,\dots,T$:
> 1. **Sampling:** draw $\mathbf x(t)$ from the training data.
> 2. **Competition (find the BMU):**
> $$c(\mathbf x) = \arg\min_{j}\big\|\mathbf x(t) - \mathbf w_j(t)\big\|$$
> 3. **Cooperation (neighbourhood):**
> $$h_{j,c}(t) = \exp\!\left(-\frac{d^2(j,c)}{2\sigma^2(t)}\right)$$
> 4. **Adaptation:**
> $$\boxed{\;\mathbf w_j(t+1) = \mathbf w_j(t) + \eta(t)\,h_{j,c}(t)\big[\mathbf x(t) - \mathbf w_j(t)\big]\;}$$
> 5. Decay $\eta(t)$ and $\sigma(t)$; repeat.

### 4.2 Neighbourhood functions

| Type | $h_{j,c}$ | Notes |
|---|---|---|
| **Gaussian** | $\exp\!\left(-d^2/2\sigma^2\right)$ | smooth, standard choice |
| Bubble / rectangular | $\mathbb 1[d \le \sigma]$ | cheap, but causes discontinuous ordering |
| Mexican hat | $\left(1-\frac{d^2}{\sigma^2}\right)e^{-d^2/2\sigma^2}$ | biologically faithful, rarely needed |
| Epanechnikov | $\max(0,\,1-d^2/\sigma^2)$ | compact support, fast |

**Requirements on $h$:** (i) maximal at $j=c$, (ii) symmetric and monotonically decreasing in $d$, (iii) $h\to0$ as $d\to\infty$, (iv) $\sigma(t)$ must shrink over time.

### 4.3 Decay schedules

$$
\eta(t) = \eta_0\exp\!\left(-\frac{t}{\tau_1}\right)\quad\text{or}\quad \eta_0\left(1-\frac{t}{T}\right),
\qquad
\sigma(t) = \sigma_0\exp\!\left(-\frac{t}{\tau_2}\right),\quad \tau_2 = \frac{T_{\text{ord}}}{\log\sigma_0}
$$

Typical: $\eta_0 = 0.1$, $\sigma_0 = \max(\text{lattice dims})/2$ (so the neighbourhood initially covers the whole map).

### 4.4 The two phases of self-organization

| Phase | Duration | $\eta$ | $\sigma$ | Effect |
|---|---|---|---|---|
| **1. Ordering (self-organizing)** | ~1000 iterations | $0.1 \to 0.01$ | $\sigma_0 \to 1$ | Topological ordering emerges; the map "unfolds" |
| **2. Convergence (fine-tuning)** | $\ge 500\times M$ iterations | $\approx0.01$ (small, constant) | $\approx 1$ (or 0) | Accurate statistical quantization; weights converge to local means |

**Critical:** if $\sigma$ is shrunk too fast, the map gets **topologically twisted/knotted** and cannot untangle — a classic failure mode. If it shrinks too slowly, the map stays over-smoothed.

```
   Ordering phase progression on 2-D uniform data (5×5 map):

   t=0            t=100           t=1000          t=10000
   ✳ tangled      ↗ unfolding     ▦ ordered       ▦ refined
   ●●●●●          ●─●─●           ●─●─●─●─●       ●─●─●─●─●
   (all near      │╲│╱│           │ │ │ │ │       │ │ │ │ │
    the centre)   ●─●─●           ●─●─●─●─●       ●─●─●─●─●
```

### 4.5 Batch SOM

$$
\boxed{\;\mathbf w_j = \frac{\displaystyle\sum_{i=1}^{N} h_{j,c(\mathbf x_i)}\,\mathbf x_i}{\displaystyle\sum_{i=1}^{N} h_{j,c(\mathbf x_i)}}\;}
$$

A weighted-mean update applied to all neurons after a full pass. **No learning rate**, deterministic, faster, parallelisable — the version used in most modern implementations. (Kohonen himself recommended it in later work.) It is a generalisation of the k-means centroid step where $h$ replaces the hard assignment.

### 4.6 Theoretical status

- **There is no global cost function** whose stochastic gradient is the online SOM update (Erwin, Obermayer & Schulten 1992) — for continuous input distributions with a general neighbourhood. This is a genuine theoretical gap.
- A related energy exists for the **discrete-data / batch** case:

  $$
  E(\mathbf W) = \frac12\sum_{i=1}^{N}\sum_{j=1}^{M} h_{j,c(\mathbf x_i)}\left\|\mathbf x_i - \mathbf w_j\right\|^2
  $$

Batch SOM performs coordinate descent on this. (Note $c(\cdot)$ depends on $\mathbf W$, which is why the online case is subtle.)
- **Convergence proofs** exist only for the 1-D map with 1-D input (Cottrell & Fort 1987): ordering is an absorbing state of the Markov chain.
- **Generative alternatives** with proper likelihoods: the **Generative Topographic Mapping (GTM)** of Bishop, Svensén & Williams (1998) — a constrained mixture of Gaussians trained by EM, giving a principled probabilistic SOM.

### 4.7 SOM as a nonlinear generalisation of PCA

A 1-D SOM chain fits a **principal curve** through the data; a 2-D SOM fits a **principal surface**. If $\sigma \to \infty$ (all neurons in one neighbourhood), all weight vectors collapse to the data mean. If $\sigma\to0$, SOM degenerates to plain competitive learning / k-means. **PCA ⊂ SOM ⊂ (nonlinear manifold learning).**

---

## 5. SOM Quality Measures, Visualization, and Applications

### 5.1 Quality measures

**(a) Quantization error** (accuracy):
$$
QE = \frac1N\sum_{i=1}^{N}\left\|\mathbf x_i - \mathbf w_{c(\mathbf x_i)}\right\|
$$

**(b) Topographic error** (topology preservation):
$$
TE = \frac1N\sum_{i=1}^{N}u(\mathbf x_i),\qquad
u(\mathbf x_i) = \begin{cases}1 & \text{if BMU and 2nd-BMU are \textbf{not} adjacent on the lattice}\\ 0 & \text{otherwise}\end{cases}
$$

**Trade-off:** more neurons ⇒ lower $QE$ but higher $TE$. Report both.

**(c) Topographic product**, **(d) trustworthiness & continuity** — more refined measures of neighbourhood preservation.

### 5.2 Visualization tools

| Tool | What it shows |
|---|---|
| **U-matrix** (unified distance matrix) | average distance between each neuron's weight vector and those of its lattice neighbours; **ridges = cluster boundaries**, valleys = clusters |
| **Component planes** | one heat-map per input feature $x_k$, showing $w_{jk}$ across the lattice; reveals feature correlations |
| **Hit histogram** | number of data points mapped to each neuron |
| **Labelled map** | majority class label per neuron (if labels available, post hoc) |

### 5.3 Applications

- **WEBSOM** — organising ~1M documents into a browsable semantic map.
- **PicSOM** — content-based image retrieval.
- **Process monitoring / fault detection** — a trajectory drifting off the "normal" region of the map signals a fault (used in steel, paper, and power plants).
- **Speech recognition** — the original *phonetic typewriter* (Kohonen 1988).
- **Bioinformatics** — gene-expression clustering, protein classification.
- **Financial analysis** — country/company profiling (the "poverty map" of world nations from 39 welfare indicators is a classic figure).
- **Robotics** — sensorimotor map learning, inverse kinematics.
- **Data mining / EDA** — clustering + visualisation in one model.
- **Image compression** — codebook design for VQ.

---

## 6. Learning Vector Quantization

### 6.1 Motivation

SOM/VQ are unsupervised — Voronoi boundaries are placed to minimise distortion, **not** classification error. **LVQ (Kohonen)** takes labelled data and *moves* code vectors to improve the decision boundary. It is a **supervised, prototype-based (nearest-prototype) classifier**.

```
   Before LVQ (VQ boundary)        After LVQ (refined boundary)
   ● ● ●  │  ○ ○ ○                 ● ● ●   │ ○ ○ ○
   ● ● ●●│●○ ○ ○                   ● ● ● ● │ ○ ○ ○
       ↑ misclassified                     ↑ boundary pushed
         region                              toward Bayes
```

### 6.2 LVQ1

> **Algorithm LVQ1**
> Given labelled data $(\mathbf x, T)$ and prototypes $\mathbf w_j$ each pre-assigned a class $C_j$:
> 1. Find the winner: $c = \arg\min_j\|\mathbf x - \mathbf w_j\|$.
> 2. Update **only the winner**:
> $$
> \boxed{\;\mathbf w_c(t+1) = \begin{cases}
> \mathbf w_c(t) + \eta(t)\left[\mathbf x - \mathbf w_c(t)\right], & \text{if } C_c = T \quad\text{(ATTRACT)}\\[4pt]
> \mathbf w_c(t) - \eta(t)\left[\mathbf x - \mathbf w_c(t)\right], & \text{if } C_c \ne T \quad\text{(REPEL)}
> \end{cases}\;}
> $$
> 3. All other prototypes unchanged. Decay $\eta(t)$ (small, monotonically decreasing, e.g. $\eta_0 \le 0.1$).

**Initialisation:** run SOM or k-means first, then label each prototype by the majority class of the data in its Voronoi cell. Prototypes whose cell is class-ambiguous should be split.

**Convergence intuition.** Writing the correct-class update as $\mathbf w_c \leftarrow (1-\eta)\mathbf w_c + \eta\mathbf x$, prototypes are exponentially-weighted moving averages of their correctly-classified inputs; the repulsion term pushes wrong prototypes out of foreign territory, shifting the Voronoi boundary toward the Bayes boundary.

### 6.3 LVQ2.1 — the window rule

LVQ1 can make the boundary drift away from Bayes-optimal. LVQ2.1 updates the **two** nearest prototypes $\mathbf w_i,\mathbf w_j$ **only when**:

1. one of them has the correct class and the other does not, **and**
2. $\mathbf x$ falls inside a **window** around the midplane:

$$
\boxed{\;\min\!\left(\frac{d_i}{d_j},\;\frac{d_j}{d_i}\right) > s,\qquad s = \frac{1-w}{1+w}\;}
$$

with relative window width $w \in [0.2, 0.3]$ typically. Then

$$
\mathbf w_{\text{correct}} \leftarrow \mathbf w_{\text{correct}} + \eta(\mathbf x - \mathbf w_{\text{correct}}),\qquad
\mathbf w_{\text{wrong}} \leftarrow \mathbf w_{\text{wrong}} - \eta(\mathbf x - \mathbf w_{\text{wrong}})
$$

**Rationale:** updating only near the boundary directly optimises the boundary and mimics the Bayes decision surface. **Danger:** LVQ2.1 does not converge — prototypes can drift indefinitely — so run it for a *short* time after LVQ1.

### 6.4 LVQ3

Adds a stabilising term: if **both** of the two nearest prototypes have the **correct** class and $\mathbf x$ is in the window,

$$
\mathbf w_k \leftarrow \mathbf w_k + \epsilon\,\eta\,(\mathbf x - \mathbf w_k),\qquad k \in \{i,j\},\quad \epsilon \in [0.1, 0.5]
$$

This keeps prototypes anchored near their class distributions, restoring convergence while retaining the boundary refinement.

### 6.5 OLVQ1 — optimised learning rate

Give each prototype its **own** adaptive rate:

$$
\eta_c(t) = \frac{\eta_c(t-1)}{1 + s(t)\,\eta_c(t-1)},\qquad
s(t) = \begin{cases}+1 & \text{correct classification}\\ -1 & \text{incorrect}\end{cases}
$$

Derived by requiring that each sample contributes equally to the final prototype position. Converges roughly an order of magnitude faster. Cap $\eta_c$ (e.g. $\le \eta_0$) to keep it stable.

### 6.6 Summary and modern successors

| Variant | Updates | Convergence | Purpose |
|---|---|---|---|
| **LVQ1** | winner only | stable | coarse placement |
| **OLVQ1** | winner, adaptive $\eta_c$ | stable, fast | fast coarse placement |
| **LVQ2.1** | two nearest, in window | **unstable** | boundary refinement (short run) |
| **LVQ3** | two nearest + same-class term | stable | refinement without drift |
| **GLVQ** | gradient descent on $\mu(\mathbf x)=\frac{d^+-d^-}{d^++d^-}$ | provable | cost-function-based LVQ |
| **GRLVQ / GMLVQ** | GLVQ + learned (relevance / full matrix) metric | provable | learns $\Vert \mathbf x-\mathbf w\Vert _\Lambda^2 = (\mathbf x-\mathbf w)^{\mathsf T}\boldsymbol\Lambda(\mathbf x-\mathbf w)$; gives feature relevance — interpretable |

**GLVQ cost function** (Sato & Yamada 1996) — the fix for LVQ's lack of an objective:
$$
E = \sum_i \Phi\!\left(\mu(\mathbf x_i)\right),\qquad \mu(\mathbf x) = \frac{d^{+}(\mathbf x) - d^{-}(\mathbf x)}{d^{+}(\mathbf x)+d^{-}(\mathbf x)} \in [-1,1]
$$
where $d^+$ = distance to nearest **same**-class prototype, $d^-$ = to nearest **different**-class prototype, $\Phi$ monotone (e.g. sigmoid). $\mu<0$ ⇔ correct classification, and $|\mu|$ is a margin. Gradient descent on $E$ gives a well-founded LVQ.

**Advantages of LVQ generally:** interpretable (prototypes are points in input space, inspectable by a domain expert), fast inference $O(M D)$, naturally multi-class, incremental, and much smaller than storing all training data (unlike k-NN).
**Disadvantages:** needs a good prototype count and initialisation; sensitive to the metric; no probabilistic output; LVQ1/2.1 lack a cost function.

### 6.7 SOM vs LVQ

| | SOM | LVQ |
|---|---|---|
| Supervision | Unsupervised | Supervised |
| Units updated | Winner **+ neighbourhood** | Winner only (or two nearest) |
| Update sign | Always toward $\mathbf x$ | Toward **or away** depending on label |
| Output topology | Ordered lattice | None (prototypes are independent) |
| Goal | Topology-preserving quantization | Minimise classification error |
| Typical use | Visualisation, clustering | Classification |

---

## 7. Radial Basis Function Networks — Foundations

### 7.1 The core idea

An RBF network solves classification/regression as a **curve-fitting (interpolation) problem in a high-dimensional space**: map the input nonlinearly into a space where the problem becomes *linearly* solvable, then fit a linear model.

$$
F(\mathbf x) = \sum_{i=1}^{M} w_i\,\varphi\!\left(\|\mathbf x - \boldsymbol\mu_i\|\right) + b
$$

The basis functions depend only on the **radial distance** from a centre — hence *radial* basis functions.

### 7.2 Cover's theorem on the separability of patterns (1965)

**Statement.** A complex pattern-classification problem cast in a high-dimensional space **nonlinearly** is more likely to be linearly separable than in a low-dimensional space, provided the space is not densely populated.

Formally, given $N$ patterns in $\mathbb R^D$ mapped by $\boldsymbol\varphi(\mathbf x) = [\varphi_1(\mathbf x),\dots,\varphi_{M_1}(\mathbf x)]^{\mathsf T}$ into $\mathbb R^{M_1}$, the probability that a random dichotomy is $\boldsymbol\varphi$-separable is

$$
\boxed{\;P(N,M_1) = \left(\frac12\right)^{N-1}\sum_{m=0}^{M_1-1}\binom{N-1}{m}\;}
$$

**Consequences:**
- If $N \le M_1$: $P = 1$ — **every** dichotomy is separable.
- **Separating capacity** $= 2M_1$: at $N = 2M_1$, $P = 1/2$ (median).
- Increasing $M_1$ (more hidden RBF units) monotonically increases separability.

**Two ingredients Cover's theorem requires:** (i) the map $\boldsymbol\varphi$ must be **nonlinear**; (ii) $M_1$ must be **large** relative to $D$. This is the exact theoretical justification for the RBF hidden layer.

### 7.3 The exact interpolation problem

Given $N$ distinct points $\{\mathbf x_i\}$ and targets $\{d_i\}$, find $F$ with $F(\mathbf x_i)=d_i$ for all $i$. Take one basis function per data point:

$$
F(\mathbf x) = \sum_{i=1}^{N} w_i\,\varphi\!\left(\|\mathbf x - \mathbf x_i\|\right)
$$

Imposing the interpolation conditions gives the linear system

$$
\begin{pmatrix}
\varphi_{11} & \varphi_{12} & \cdots & \varphi_{1N}\\
\varphi_{21} & \varphi_{22} & \cdots & \varphi_{2N}\\
\vdots & & \ddots & \vdots\\
\varphi_{N1} & \cdots & & \varphi_{NN}
\end{pmatrix}
\begin{pmatrix}w_1\\ w_2\\ \vdots\\ w_N\end{pmatrix}
=
\begin{pmatrix}d_1\\ d_2\\ \vdots\\ d_N\end{pmatrix}
\qquad \varphi_{ij} = \varphi(\|\mathbf x_i-\mathbf x_j\|)
$$

$$
\boxed{\;\boldsymbol\Phi\mathbf w = \mathbf d \quad\Longrightarrow\quad \mathbf w = \boldsymbol\Phi^{-1}\mathbf d\;}
$$

**Micchelli's theorem (1986).** For a large class of functions $\varphi$ (including the Gaussian, multiquadric and inverse multiquadric), if the points $\{\mathbf x_i\}_{i=1}^N$ are **distinct**, then $\boldsymbol\Phi$ is **non-singular** — regardless of $N$ or $D$. So exact interpolation always has a unique solution.

**Problem:** exact interpolation with $M=N$ **overfits badly** (it fits the noise), and $\boldsymbol\Phi$ becomes severely ill-conditioned for large $N$. Hence regularization and $M \ll N$.

### 7.4 Regularization theory (Tikhonov / Poggio & Girosi)

Minimise
$$
\mathcal E(F) = \underbrace{\frac12\sum_{i=1}^{N}\left(d_i - F(\mathbf x_i)\right)^2}_{\text{standard error}} + \underbrace{\frac{\lambda}{2}\left\|\mathbf P F\right\|^2}_{\text{smoothness}}
$$
where $\mathbf P$ is a linear differential operator (a *stabiliser*) penalising roughness.

Applying the Euler–Lagrange condition gives $\tilde{\mathbf P}\mathbf P F = \frac1\lambda\sum_i(d_i - F(\mathbf x_i))\delta(\mathbf x-\mathbf x_i)$, whose solution is expressed via the **Green's function** $G$ of the operator $\tilde{\mathbf P}\mathbf P$:

$$
\boxed{\;F_\lambda(\mathbf x) = \frac{1}{\lambda}\sum_{i=1}^{N}\left[d_i - F(\mathbf x_i)\right]G(\mathbf x,\mathbf x_i) = \sum_{i=1}^{N}w_i\,G(\mathbf x,\mathbf x_i)\;}
$$

with weights $\;\mathbf w = (\mathbf G + \lambda\mathbf I)^{-1}\mathbf d$.

**Key result:** if $\mathbf P$ is translation- and rotation-invariant, $G(\mathbf x,\mathbf x_i)=G(\|\mathbf x-\mathbf x_i\|)$ — **a radial basis function**. For the specific stabiliser
$$
\|\mathbf PF\|^2 = \sum_{n=0}^{\infty}\frac{\sigma^{2n}}{n!\,2^n}\int\left\|D^{n}F(\mathbf x)\right\|^2 d\mathbf x
$$
the Green's function is exactly the **Gaussian** $G(r) = e^{-r^2/2\sigma^2}$.

**So the Gaussian RBF is not an arbitrary choice — it is the optimal solution of a well-posed regularization problem.** This also shows $\lambda$ plays the role of ridge regularization on the RBF weights.

### 7.5 Common radial basis functions

| Name | $\varphi(r)$ | Localised? | Notes |
|---|---|---|---|
| **Gaussian** | $\exp\!\left(-\dfrac{r^2}{2\sigma^2}\right)$ | **Yes** | most used; $\varphi\to0$ as $r\to\infty$ |
| Multiquadric | $\sqrt{r^2 + c^2}$ | No (grows) | excellent interpolation accuracy |
| Inverse multiquadric | $\left(r^2+c^2\right)^{-1/2}$ | Yes | |
| Thin-plate spline | $r^2\ln r$ | No | 2-D surface fitting, minimises bending energy |
| Linear | $r$ | No | |
| Cubic | $r^3$ | No | |

Only **localised** $\varphi$ give the characteristic "local receptive field" behaviour; non-localised ones still satisfy Micchelli's theorem and interpolate well.

---

## 8. RBF Structure and Training Methods

### 8.1 Architecture

```
   INPUT layer          HIDDEN layer (RBF units)        OUTPUT layer
   (D units,            (M units, NONLINEAR,             (K units, LINEAR)
    no computation)      localised receptive fields)

     x₁ ●──────┐   ┌───────────────────────┐
               ├──►│ φ₁ = exp(-‖x-μ₁‖²/2σ₁²)│───w₁───┐
     x₂ ●──────┤   ├───────────────────────┤          │      ┌──────┐
               ├──►│ φ₂ = exp(-‖x-μ₂‖²/2σ₂²)│───w₂───┼─────►│  Σ   │──► y
       ⋮       │   ├───────────────────────┤          │      └──────┘
               ├──►│         ⋮              │   ⋮      │         ▲
     x_D ●─────┘   ├───────────────────────┤          │         │
                   │ φ_M                    │───w_M───┘      bias b
                   └───────────────────────┘

   Distance-based, NOT inner-product based.  Exactly ONE hidden layer.
```

$$
\boxed{\;y_k(\mathbf x) = \sum_{i=1}^{M} w_{ki}\,\varphi_i(\mathbf x) + b_k,\qquad
\varphi_i(\mathbf x) = \exp\!\left(-\frac{\|\mathbf x-\boldsymbol\mu_i\|^2}{2\sigma_i^2}\right)\;}
$$

**Generalised form with a full covariance (Mahalanobis) metric:**
$$
\varphi_i(\mathbf x) = \exp\!\left(-\tfrac12(\mathbf x-\boldsymbol\mu_i)^{\mathsf T}\boldsymbol\Sigma_i^{-1}(\mathbf x-\boldsymbol\mu_i)\right)
$$
allowing elliptical, oriented receptive fields.

**Normalised RBF** (partition of unity — makes the network a smooth interpolator of the weights):
$$
y(\mathbf x) = \frac{\sum_i w_i\varphi_i(\mathbf x)}{\sum_i \varphi_i(\mathbf x)}
$$
This is exactly the **Nadaraya–Watson kernel regression** estimator, and its classification version is a **Probabilistic Neural Network (PNN)** / Parzen-window classifier.

**Parameters to learn:** centres $\boldsymbol\mu_i$ ($MD$), widths $\sigma_i$ ($M$ or $MD$ or $MD^2$), weights $w_{ki}$ ($KM$) and biases ($K$).

### 8.2 Training Strategy 1 — Fixed random centres

Choose $M$ centres **randomly from the training data**, with a common width

$$
\boxed{\;\sigma = \frac{d_{\max}}{\sqrt{2M}}\;}
$$

($d_{\max}$ = maximum distance between chosen centres). This heuristic makes the Gaussians neither too peaked nor too flat.

Then solve for the weights by **linear least squares** — the only step needed:

$$
\boldsymbol\Phi\mathbf w = \mathbf d \;\Longrightarrow\;
\boxed{\;\mathbf w = \boldsymbol\Phi^{+}\mathbf d = \left(\boldsymbol\Phi^{\mathsf T}\boldsymbol\Phi\right)^{-1}\boldsymbol\Phi^{\mathsf T}\mathbf d\;}
$$

with regularization: $\mathbf w = (\boldsymbol\Phi^{\mathsf T}\boldsymbol\Phi + \lambda\mathbf I)^{-1}\boldsymbol\Phi^{\mathsf T}\mathbf d$.

**Advantage:** extremely fast, no iterative optimisation, **globally optimal weights** (convex problem). **Disadvantage:** centre placement ignores the data distribution.

### 8.3 Training Strategy 2 — Hybrid (the standard method)

> **Stage 1 (unsupervised) — place the centres.**
> Run **k-means** on the inputs (ignoring labels):
> $$\boldsymbol\mu_i = \frac{1}{|\mathcal V_i|}\sum_{\mathbf x\in\mathcal V_i}\mathbf x$$
> Alternatives: SOM, GMM/EM (gives $\boldsymbol\Sigma_i$ for free), or orthogonal least squares (§8.5).
>
> **Stage 2 — set the widths.** Common heuristics:
> - $\sigma_i = \dfrac{1}{|\mathcal V_i|}\sum_{\mathbf x\in\mathcal V_i}\|\mathbf x-\boldsymbol\mu_i\|$ (mean intra-cluster distance)
> - $\sigma_i = \alpha\cdot\min_{j\ne i}\|\boldsymbol\mu_i-\boldsymbol\mu_j\|$, $\alpha \in [1,2]$ (nearest-centre)
> - $\sigma_i = \left(\frac{1}{p}\sum_{j\in\text{p-NN}(i)}\|\boldsymbol\mu_i-\boldsymbol\mu_j\|^2\right)^{1/2}$ ($p$-nearest centres)
> - Global $\sigma = d_{\max}/\sqrt{2M}$
>
> **Stage 3 (supervised) — solve for the weights** by pseudo-inverse / ridge regression (closed form) or by LMS if online learning is needed:
> $$\Delta w_{ki} = \eta\left(d_k - y_k\right)\varphi_i(\mathbf x)$$

**Why this works so well:** the problem is decomposed into a nonlinear part (centres/widths, solved by cheap unsupervised methods) and a **convex linear** part (weights, solved exactly). **No local minima in stage 3, no backpropagation needed.**

### 8.4 Training Strategy 3 — Fully supervised (gradient descent on all parameters)

Minimise $\mathcal E = \frac12\sum_j e_j^2$, $e_j = d_j - F(\mathbf x_j)$, with respect to everything:

$$
\frac{\partial\mathcal E}{\partial w_i} = -\sum_j e_j\,\varphi_i(\mathbf x_j)
$$
$$
\frac{\partial\mathcal E}{\partial \boldsymbol\mu_i} = -\frac{w_i}{\sigma_i^2}\sum_j e_j\,\varphi_i(\mathbf x_j)\left(\mathbf x_j - \boldsymbol\mu_i\right)
$$
$$
\frac{\partial\mathcal E}{\partial \sigma_i} = -\frac{w_i}{\sigma_i^3}\sum_j e_j\,\varphi_i(\mathbf x_j)\left\|\mathbf x_j-\boldsymbol\mu_i\right\|^2
$$

**Derivation of the centre gradient.** With $\varphi_i = \exp\!\left(-\frac{\|\mathbf x-\boldsymbol\mu_i\|^2}{2\sigma_i^2}\right)$,
$$
\frac{\partial\varphi_i}{\partial\boldsymbol\mu_i} = \varphi_i\cdot\left(-\frac{1}{2\sigma_i^2}\right)\cdot\frac{\partial}{\partial\boldsymbol\mu_i}\left\|\mathbf x - \boldsymbol\mu_i\right\|^2
= \varphi_i\left(-\frac{1}{2\sigma_i^2}\right)(-2)(\mathbf x-\boldsymbol\mu_i) = \frac{\varphi_i}{\sigma_i^2}(\mathbf x-\boldsymbol\mu_i)
$$
Chain rule with $\partial\mathcal E/\partial\varphi_i = -\sum_j e_j w_i$ gives the result. ∎

**Pros:** best accuracy, fewest units. **Cons:** non-convex ⇒ local minima; slow; can produce degenerate centres (a centre drifting far from all data, $\sigma\to0$ or $\to\infty$). Usually used as a *fine-tuning* stage after the hybrid method.

### 8.5 Choosing $M$ — Orthogonal Least Squares (OLS)

A **forward-selection** method (Chen, Cowan & Grant 1991): treat every training point as a candidate centre, orthogonalise the candidate regressors (Gram–Schmidt/QR), and greedily add the one maximising the **error reduction ratio**

$$
[\text{err}]_i = \frac{g_i^2\,\mathbf q_i^{\mathsf T}\mathbf q_i}{\mathbf d^{\mathsf T}\mathbf d}
$$

($\mathbf q_i$ = orthogonalised regressor, $g_i$ = its coefficient). Stop when $1-\sum_i[\text{err}]_i < \rho$ (tolerance). Produces a **parsimonious** network automatically.

Alternatives: cross-validation over $M$; RBF with pruning; Bayesian model selection; Relevance Vector Machine (which yields extremely sparse RBF-like models).

---

## 9. RBF vs MLP, and Applications

### 9.1 Detailed comparison

| Aspect | RBF Network | Multi-Layer Perceptron |
|---|---|---|
| **Hidden layers** | Exactly **one** | One or many |
| **Hidden unit argument** | **Distance** $\Vert \mathbf x-\boldsymbol\mu_i\Vert $ | **Inner product** $\mathbf w^{\mathsf T}\mathbf x$ |
| **Activation** | Radial, localised (Gaussian) | Sigmoidal/ReLU, monotone |
| **Approximation type** | **Local** (basis with compact-ish support) | **Global/distributed** |
| **Decision boundaries** | Hyper-ellipsoids around centres | Intersecting hyperplanes |
| **Response to far-away input** | $\to 0$ (knows it doesn't know) | Arbitrary, often confidently wrong |
| **Training** | Hybrid: unsupervised + **linear** (convex) | Fully supervised backprop (non-convex) |
| **Speed of training** | Very fast | Slow |
| **Local minima** | None (in the linear stage) | Many |
| **Interpretability** | High (centres are prototypes in input space) | Low |
| **Units needed in high $D$** | Often **exponential** (curse of dimensionality) | Scales better |
| **Extrapolation** | Poor (output $\to b$) | Better |
| **Online adaptation** | Easy (add a centre) | Hard (catastrophic forgetting) |
| **Universal approximator** | **Yes** (Park & Sandberg 1991) | Yes |

**The fundamental trade-off:** local basis ⇒ fast, stable, interpretable learning; but covering a $D$-dimensional space with localised bumps needs $O(k^D)$ of them. MLPs use *distributed* representations that scale better with $D$ — the reason deep MLPs/CNNs dominate high-dimensional problems while RBFs remain excellent for low-to-moderate $D$.

### 9.2 RBF ≈ kernel methods

With centres at all data points and Gaussian $\varphi$, the RBF network with ridge regularization
$$
\mathbf w = (\boldsymbol\Phi + \lambda\mathbf I)^{-1}\mathbf d
$$
is **identical to kernel ridge regression** with the Gaussian kernel, and closely related to Gaussian Process regression (whose posterior mean is exactly this expression). An SVM with an RBF kernel is an RBF network whose centres are the **support vectors**, chosen automatically by the margin criterion.

### 9.3 Applications

- **Function approximation & time-series prediction** (chaotic series, Mackey–Glass — the classic benchmark).
- **Nonlinear system identification and adaptive control** — RBFs are the standard function approximator in adaptive/neuro control because the linear-in-parameters structure permits **Lyapunov stability proofs**.
- **Channel equalisation** in digital communications (the Bayesian equaliser has an exact RBF form).
- **Face/speech/pattern recognition** with moderate feature dimensions.
- **Interpolation and meshless PDE solvers** (RBF-FD, Kansa's method) — a large field in computational mathematics.
- **Anomaly / novelty detection** — since $y\to b$ for far-away inputs, low total activation $\sum_i\varphi_i(\mathbf x)$ is a natural novelty score.
- **Medical diagnosis, financial forecasting**, and any setting where prototypes must be inspectable.

---

## 10. Solved Numericals

### N1. SOM — winner selection and weight update (Kohonen's classic)

**Setup.** 4 input units, 2 output neurons (a 1×2 lattice), $\eta = 0.6$.
$$
\mathbf w_1 = (0.2,\,0.6,\,0.5,\,0.9),\qquad \mathbf w_2 = (0.8,\,0.4,\,0.7,\,0.3)
$$

**Input 1:** $\mathbf x = (1,1,0,0)$.

$$
d_1^2 = (0.2-1)^2+(0.6-1)^2+(0.5-0)^2+(0.9-0)^2 = 0.64+0.16+0.25+0.81 = \mathbf{1.86}
$$
$$
d_2^2 = (0.8-1)^2+(0.4-1)^2+(0.7-0)^2+(0.3-0)^2 = 0.04+0.36+0.49+0.09 = \mathbf{0.98}
$$

**BMU = neuron 2** (smaller distance). Update (winner only, since $M=2$ and $\sigma$ is small):
$$
\mathbf w_2^{\text{new}} = \mathbf w_2 + 0.6\left(\mathbf x - \mathbf w_2\right)
$$
$$
= (0.8,0.4,0.7,0.3) + 0.6\,(0.2,\,0.6,\,-0.7,\,-0.3)
$$
$$
= (0.8+0.12,\;0.4+0.36,\;0.7-0.42,\;0.3-0.18) = \boxed{(0.92,\;0.76,\;0.28,\;0.12)}
$$

**Input 2:** $\mathbf x = (0,0,0,1)$.
$$
d_1^2 = 0.04+0.36+0.25+0.01 = \mathbf{0.66}
$$
$$
d_2^2 = 0.8464+0.5776+0.0784+0.7744 = \mathbf{2.2768}
$$
**BMU = neuron 1.**
$$
\mathbf w_1^{\text{new}} = (0.2,0.6,0.5,0.9) + 0.6(-0.2,\,-0.6,\,-0.5,\,0.1) = \boxed{(0.08,\;0.24,\;0.20,\;0.96)}
$$

**Observation:** neuron 2 is specialising on patterns with mass in the first two components; neuron 1 on the last component. The map is self-organizing into a clustering of the input space.

---

### N2. SOM — Gaussian neighbourhood on a 3×3 lattice

Lattice coordinates $(r,c)$, $r,c\in\{1,2,3\}$. BMU at $(2,2)$. Neighbourhood width $\sigma = 1.0$, $\eta = 0.5$.

$$
h_{j,c} = \exp\!\left(-\frac{d^2(j,c)}{2\sigma^2}\right)
$$

| Neuron | Grid pos | $d^2$ | $h$ | Effective rate $\eta h$ |
|---|---|---|---|---|
| $(2,2)$ **BMU** | centre | 0 | $e^{0}=1.0000$ | 0.5000 |
| $(1,2),(3,2),(2,1),(2,3)$ | edge-adjacent | 1 | $e^{-0.5}=0.6065$ | 0.3033 |
| $(1,1),(1,3),(3,1),(3,3)$ | diagonal | 2 | $e^{-1}=0.3679$ | 0.1839 |

If $\mathbf x = (1.0,\,2.0)$ and the corner neuron $(1,1)$ has $\mathbf w = (0.4,\,0.5)$:
$$
\mathbf w^{\text{new}} = (0.4,0.5) + 0.1839\left[(1.0,2.0)-(0.4,0.5)\right] = (0.4,0.5)+0.1839(0.6,1.5)
$$
$$
= (0.4+0.11034,\;0.5+0.27585) = \boxed{(0.51034,\;0.77585)}
$$

**Effect of shrinking $\sigma$.** At $\sigma = 0.5$: $h(d^2{=}1) = e^{-2} = 0.1353$, $h(d^2{=}2)=e^{-4}=0.0183$. At $\sigma = 0.2$: $h(d^2{=}1)=e^{-12.5}=3.7\times10^{-6}$ — effectively pure winner-take-all (k-means). This is precisely how the convergence phase differs from the ordering phase.

---

### N3. SOM decay schedules

$\eta_0 = 0.5$, $\tau_1 = 1000$; $\sigma_0 = 5$, $T_{\text{ord}} = 1000$ so $\tau_2 = T_{\text{ord}}/\ln\sigma_0 = 1000/1.60944 = 621.3$.

| $t$ | $\eta(t)=0.5e^{-t/1000}$ | $\sigma(t) = 5e^{-t/621.3}$ |
|---|---|---|
| 0 | 0.5000 | 5.000 |
| 250 | $0.5e^{-0.25}=0.3894$ | $5e^{-0.4024}=3.343$ |
| 500 | $0.5e^{-0.5}=0.3033$ | $5e^{-0.8047}=2.236$ |
| 1000 | $0.5e^{-1}=0.1839$ | $5e^{-1.6094}=1.000$ |
| 2000 | $0.5e^{-2}=0.0677$ | $5e^{-3.2189}=0.200$ |
| 5000 | $0.5e^{-5}=0.0034$ | $5e^{-8.047}=0.0016$ |

At $t = 1000$ the neighbourhood radius has fallen to exactly 1 lattice unit — the designed end of the ordering phase. From here the convergence phase should use a small constant $\eta\approx0.01$.

---

### N4. LVQ1 — complete worked example

**Training data** (4-dim, 2 classes):

| $\mathbf x$ | class $T$ |
|---|---|
| $(1,1,0,0)$ | 1 |
| $(0,0,0,1)$ | 2 |
| $(0,0,1,1)$ | 2 |
| $(1,0,0,0)$ | 1 |
| $(0,1,1,0)$ | 2 |

**Initialisation:** use the first vector of each class as a prototype.
$$
\mathbf w_1 = (1,1,0,0),\;C_1 = 1;\qquad \mathbf w_2 = (0,0,0,1),\;C_2 = 2
$$
Remaining training vectors: rows 3, 4, 5. $\eta = 0.1$.

**Iteration 1:** $\mathbf x = (0,0,1,1)$, $T = 2$.
$$
d_1^2 = 1+1+1+1 = 4,\qquad d_2^2 = 0+0+1+0 = 1
$$
Winner = $\mathbf w_2$; $C_2 = 2 = T$ ⇒ **ATTRACT**:
$$
\mathbf w_2 \leftarrow (0,0,0,1) + 0.1\left[(0,0,1,1)-(0,0,0,1)\right] = (0,0,0,1)+0.1(0,0,1,0) = \boxed{(0,\,0,\,0.1,\,1)}
$$

**Iteration 2:** $\mathbf x = (1,0,0,0)$, $T = 1$.
$$
d_1^2 = 0+1+0+0 = 1,\qquad d_2^2 = 1+0+0.01+1 = 2.01
$$
Winner = $\mathbf w_1$; $C_1 = 1 = T$ ⇒ **ATTRACT**:
$$
\mathbf w_1 \leftarrow (1,1,0,0)+0.1\left[(1,0,0,0)-(1,1,0,0)\right] = (1,1,0,0)+0.1(0,-1,0,0) = \boxed{(1,\,0.9,\,0,\,0)}
$$

**Iteration 3:** $\mathbf x = (0,1,1,0)$, $T = 2$.
$$
d_1^2 = (0-1)^2+(1-0.9)^2+(1-0)^2+0 = 1+0.01+1+0 = 2.01
$$
$$
d_2^2 = 0+(1-0)^2+(1-0.1)^2+(0-1)^2 = 0+1+0.81+1 = 2.81
$$
Winner = $\mathbf w_1$; but $C_1 = 1 \ne T = 2$ ⇒ **REPEL**:
$$
\mathbf w_1 \leftarrow \mathbf w_1 - 0.1\left[\mathbf x - \mathbf w_1\right] = (1,0.9,0,0) - 0.1\left[(0,1,1,0)-(1,0.9,0,0)\right]
$$
$$
= (1,0.9,0,0) - 0.1(-1,\,0.1,\,1,\,0) = \boxed{(1.1,\;0.89,\;-0.1,\;0)}
$$

**Interpretation.** The class-1 prototype was pulled *away* from the class-2 pattern — note the negative third component, which is fine (prototypes need not lie in the data range). Over many epochs this pushes the Voronoi boundary toward the true class boundary.

---

### N5. LVQ2.1 window test

Prototypes: $\mathbf w_i$ (class 1) at distance $d_i = 2.0$; $\mathbf w_j$ (class 2) at $d_j = 2.5$. True class $T = 2$. Window width $w = 0.25$.

$$
s = \frac{1-w}{1+w} = \frac{0.75}{1.25} = 0.6
$$
$$
\min\!\left(\frac{d_i}{d_j},\;\frac{d_j}{d_i}\right) = \min\!\left(\frac{2.0}{2.5},\;\frac{2.5}{2.0}\right) = \min(0.8,\,1.25) = 0.8
$$
Since $0.8 > 0.6$, $\mathbf x$ **is inside the window**, and exactly one of the two prototypes has the correct class ⇒ **update both**:
$$
\mathbf w_j \leftarrow \mathbf w_j + \eta(\mathbf x - \mathbf w_j)\;\;(\text{correct, attract}),\qquad
\mathbf w_i \leftarrow \mathbf w_i - \eta(\mathbf x - \mathbf w_i)\;\;(\text{wrong, repel})
$$

**Counter-example:** if $d_i = 1.0$, $d_j = 4.0$, then $\min(0.25, 4.0) = 0.25 < 0.6$ ⇒ **outside the window, no update** — the point is far from the boundary and carries no useful boundary information.

---

### N6. RBF solves XOR exactly (the classic example)

**Setup.** Two Gaussian hidden units with centres at the two "class-0" corners:
$$
\boldsymbol\mu_1 = (0,0),\qquad \boldsymbol\mu_2 = (1,1),\qquad \varphi_i(\mathbf x) = e^{-\|\mathbf x-\boldsymbol\mu_i\|^2}
$$
(i.e. $2\sigma^2 = 1$). Output $y = w_1\varphi_1 + w_2\varphi_2 + b$.

**Step 1 — Hidden-layer outputs**

| $\mathbf x$ | $\Vert \mathbf x-\boldsymbol\mu_1\Vert ^2$ | $\varphi_1$ | $\Vert \mathbf x-\boldsymbol\mu_2\Vert ^2$ | $\varphi_2$ | target $d$ |
|---|---|---|---|---|---|
| $(0,0)$ | 0 | $e^{0}=1.000000$ | 2 | $e^{-2}=0.135335$ | 0 |
| $(0,1)$ | 1 | $e^{-1}=0.367879$ | 1 | $e^{-1}=0.367879$ | 1 |
| $(1,0)$ | 1 | $0.367879$ | 1 | $0.367879$ | 1 |
| $(1,1)$ | 2 | $0.135335$ | 0 | $1.000000$ | 0 |

**Step 2 — Feature-space picture**

```
   φ₂
   1.0 ┤ ●(1,1)  d=0
       │
   0.5 ┤
       │      ○(0,1),(1,0)  d=1     ← the two class-1 points COLLAPSE
   0.37┤      ○  onto one point
       │
   0.14┤                    ● (0,0)  d=0
       └──────────────────────────────► φ₁
        0.14      0.37          1.0

   A single straight line now separates ○ from ●  →  LINEARLY SEPARABLE ✓
```

The nonlinear map has made XOR separable — **Cover's theorem in action**.

**Step 3 — Solve for the weights.** By the symmetry $\varphi_1 \leftrightarrow \varphi_2$ we may set $w_1 = w_2 = w$:

Row 1: $\;w(1.000000 + 0.135335) + b = 0 \;\Rightarrow\; 1.135335\,w + b = 0$
Row 2: $\;w(0.367879+0.367879) + b = 1 \;\Rightarrow\; 0.735759\,w + b = 1$

Subtracting:
$$
(1.135335 - 0.735759)\,w = -1 \;\Rightarrow\; 0.399576\,w = -1 \;\Rightarrow\; \boxed{w = -2.502650}
$$
$$
b = -1.135335\,w = -1.135335(-2.502650) = \boxed{2.841440}
$$

**Step 4 — Verification (all four patterns)**

| $\mathbf x$ | $y = -2.502650(\varphi_1+\varphi_2) + 2.841440$ | target |
|---|---|---|
| $(0,0)$ | $-2.502650(1.135335)+2.841440 = -2.841440+2.841440 = \mathbf{0.000}$ | 0 ✓ |
| $(0,1)$ | $-2.502650(0.735759)+2.841440 = -1.841441+2.841440 = \mathbf{1.000}$ | 1 ✓ |
| $(1,0)$ | $\mathbf{1.000}$ | 1 ✓ |
| $(1,1)$ | $-2.502650(1.135335)+2.841440 = \mathbf{0.000}$ | 0 ✓ |

**Exact solution with ZERO error, obtained in closed form — no iterative training, no local minima.** Compare with the MLP of Unit II, which needs gradient descent through a saddle-rich landscape to reach the same result.

---

### N7. RBF exact interpolation (1-D, pseudo-inverse)

**Data:** $x = 0, 1, 2$ with targets $d = 1, 3, 2$. Centres at the data points, Gaussian $\varphi(r)=e^{-r^2/2}$ ($\sigma = 1$).

**Step 1 — Interpolation matrix**
$$
\varphi(0)=1,\qquad \varphi(1)=e^{-0.5}=0.606531,\qquad \varphi(2)=e^{-2}=0.135335
$$
$$
\boldsymbol\Phi = \begin{pmatrix}
1 & 0.606531 & 0.135335\\
0.606531 & 1 & 0.606531\\
0.135335 & 0.606531 & 1
\end{pmatrix}
$$
Symmetric and (by Micchelli's theorem) non-singular ✓

**Step 2 — Solve $\boldsymbol\Phi\mathbf w = (1,3,2)^{\mathsf T}$.**

Subtract row 3 from row 1:
$$
(1-0.135335)w_1 + 0 + (0.135335-1)w_3 = 1-2 \;\Rightarrow\; 0.864665(w_1-w_3) = -1
$$
$$
w_1 - w_3 = -1.156518
$$
Add rows 1 and 3, with $S = w_1+w_3$:
$$
1.135335\,S + 1.213062\,w_2 = 3
$$
From row 2: $\;0.606531\,S + w_2 = 3 \Rightarrow w_2 = 3 - 0.606531S$. Substituting:
$$
1.135335S + 1.213062(3 - 0.606531S) = 3
$$
$$
1.135335S + 3.639186 - 0.735759S = 3 \;\Rightarrow\; 0.399576S = -0.639186 \;\Rightarrow\; S = -1.599685
$$
$$
w_2 = 3 - 0.606531(-1.599685) = 3+0.970255 = \mathbf{3.970255}
$$
$$
w_1 = \tfrac12\big(S + (w_1{-}w_3)\big) = \tfrac12(-1.599685-1.156518) = \mathbf{-1.378102}
$$
$$
w_3 = \tfrac12\big(S - (w_1{-}w_3)\big) = \tfrac12(-1.599685+1.156518) = \mathbf{-0.221584}
$$

**Step 3 — Verify**
$$
F(0) = -1.378102(1) + 3.970255(0.606531) + (-0.221584)(0.135335)
$$
$$
= -1.378102 + 2.407993 - 0.029987 = \mathbf{0.9999} \approx 1 ✓
$$
$$
F(1) = -1.378102(0.606531)+3.970255(1)+(-0.221584)(0.606531) = -0.835859+3.970255-0.134404 = \mathbf{3.0000} ✓
$$
$$
F(2) = -1.378102(0.135335)+3.970255(0.606531)+(-0.221584)(1) = -0.186508+2.407993-0.221584 = \mathbf{1.9999} ✓
$$

**Step 4 — Interpolate at $x = 0.5$**
$$
\varphi_1 = e^{-0.25/2} = e^{-0.125} = 0.882497,\quad \varphi_2 = e^{-0.125} = 0.882497,\quad \varphi_3 = e^{-2.25/2} = e^{-1.125} = 0.324652
$$
$$
F(0.5) = 0.882497(-1.378102 + 3.970255) + 0.324652(-0.221584)
$$
$$
= 0.882497(2.592153) - 0.071938 = 2.287571 - 0.071938 = \mathbf{2.2156}
$$
Sensible — between $F(0)=1$ and $F(1)=3$, with the Gaussian smoothing pulling it up.

**Note on conditioning:** $\det\boldsymbol\Phi$ here is well away from zero, but as $\sigma$ grows all $\varphi_{ij}\to1$ and $\boldsymbol\Phi$ becomes nearly rank-1 (severely ill-conditioned). Always regularise: $\mathbf w = (\boldsymbol\Phi + \lambda\mathbf I)^{-1}\mathbf d$.

---

### N8. RBF width heuristic and Cover's theorem

**(a) Width heuristic.** $M = 8$ centres, maximum inter-centre distance $d_{\max} = 6.0$:
$$
\sigma = \frac{d_{\max}}{\sqrt{2M}} = \frac{6.0}{\sqrt{16}} = \frac{6.0}{4} = \mathbf{1.5}
$$
A point $2.0$ from a centre gets activation $\varphi = e^{-4/(2\times2.25)} = e^{-0.8889} = 0.411$; at distance $5.0$, $\varphi = e^{-25/4.5} = e^{-5.556} = 0.0039$ — effectively silent. The receptive fields are well-localised but overlapping.

**(b) Cover's theorem.** $N = 30$ patterns, $M_1 = 10$ hidden units:
$$
P(30,10) = \left(\tfrac12\right)^{29}\sum_{m=0}^{9}\binom{29}{m}
$$
$$
\sum_{m=0}^{9}\binom{29}{m} = 1+29+406+3654+23751+118755+475020+1560780+4292145+10015005 = 16\,489\,546
$$
$$
P = \frac{16\,489\,546}{2^{29}} = \frac{16\,489\,546}{536\,870\,912} = \mathbf{0.0307}
$$

Now raise the hidden layer to $M_1 = 20$:
$$
P(30,20) = \left(\tfrac12\right)^{29}\sum_{m=0}^{19}\binom{29}{m}
$$
By symmetry $\sum_{m=0}^{29}\binom{29}{m}=2^{29}$ and $\sum_{m=0}^{14}\binom{29}{m} = 2^{28}$, so
$$
\sum_{m=0}^{19}\binom{29}{m} = 2^{28} + \sum_{m=15}^{19}\binom{29}{m} = 268\,435\,456 + (77\,558\,760+67\,863\,915+51\,895\,935+34\,597\,290+20\,030\,010)
$$
$$
= 268\,435\,456 + 251\,945\,910 = 520\,381\,366 \;\Rightarrow\; P = \frac{520\,381\,366}{536\,870\,912} = \mathbf{0.969}
$$

**From 3 % to 97 % separability by doubling the hidden layer.** And with $M_1 \ge N = 30$, $P = 1$ exactly. This is the quantitative justification for RBF hidden layers being wide.

**(c) Separating capacity:** $2M_1 = 2(10) = 20$ patterns — beyond that, separability degrades. Consistent with $P(20,10) = 0.5$.

---

### N9. k-means centre placement for an RBF network

Data (1-D): $\;1,\,2,\,3,\,8,\,9,\,10$. Want $M = 2$ centres. Initialise $\mu_1 = 2$, $\mu_2 = 9$.

**Iteration 1 — assign:**
$\{1,2,3\}\to\mu_1$ (distances $1,0,1$ vs $8,7,6$); $\{8,9,10\}\to\mu_2$.
**Update:** $\mu_1 = (1+2+3)/3 = 2$; $\mu_2 = (8+9+10)/3 = 9$. **Converged immediately.**

**Widths** (mean intra-cluster distance):
$$
\sigma_1 = \frac{|1-2|+|2-2|+|3-2|}{3} = \frac{2}{3} = 0.667,\qquad \sigma_2 = 0.667
$$
Alternative (nearest-centre heuristic, $\alpha = 0.5$): $\sigma = 0.5|9-2| = 3.5$ — much wider. The two heuristics differ by a factor of 5; **width selection matters more than centre selection** in practice, and should be cross-validated.

**Design matrix** for these 6 points with $\sigma = 0.667$ ($2\sigma^2 = 0.8889$), plus a bias column:

| $x$ | $\varphi_1 = e^{-(x-2)^2/0.8889}$ | $\varphi_2 = e^{-(x-9)^2/0.8889}$ |
|---|---|---|
| 1 | $e^{-1.125}=0.3247$ | $e^{-72}\approx 0$ |
| 2 | $1.0000$ | $\approx 0$ |
| 3 | $0.3247$ | $\approx 0$ |
| 8 | $\approx 0$ | $0.3247$ |
| 9 | $\approx 0$ | $1.0000$ |
| 10 | $\approx 0$ | $0.3247$ |

The block-diagonal structure shows perfect locality: each cluster activates exactly one unit. Any targets can now be fitted by 3 free parameters $(w_1,w_2,b)$ in closed form.

---

## 11. Viva / Exam Pointers

**Likely long questions**
1. Describe the SOM architecture and algorithm; explain competition, cooperation and adaptation; give the update equation and the two training phases.
2. Explain the U-matrix and how SOM is used for visualisation; define quantization and topographic error.
3. Explain LVQ1, LVQ2.1 (window rule) and LVQ3; contrast LVQ with SOM.
4. State and explain Cover's theorem; use it to justify the RBF hidden layer.
5. Derive the exact-interpolation formulation of an RBF network and state Micchelli's theorem.
6. Show how regularization theory yields the Gaussian as a Green's function.
7. Solve XOR with a Gaussian RBF network (full numerical).
8. Compare RBF networks and MLPs across at least eight dimensions.
9. Describe the hybrid training procedure for RBFs and derive the gradients for fully supervised training.

**Traps**
- In SOM, the **winner is found in input space** but the **neighbourhood is measured on the lattice**. Mixing these up is the single most common error.
- SOM updates the winner **and its neighbours**; LVQ updates the winner **only** (LVQ2.1: the two nearest). SOM never moves a weight *away* from the input; LVQ does.
- SOM has **no global cost function** for the online continuous case — say this if asked about convergence proofs.
- RBF hidden units compute a **distance**, not $\mathbf w^{\mathsf T}\mathbf x$; there is no "weight vector" into a hidden unit in the MLP sense — the centre plays that role.
- RBF has **exactly one** hidden layer; the output layer is **linear**, which is why the weight-solving step is convex.
- The number of RBF units needed grows exponentially with input dimension — do not claim RBFs beat MLPs in high dimensions.
- $\sigma$ too small ⇒ spiky, overfitting, poor coverage; $\sigma$ too large ⇒ all units respond identically, ill-conditioned $\boldsymbol\Phi$, underfitting.

**One-line formula sheet**

$$
\text{SOM: } c = \arg\min_j\|\mathbf x - \mathbf w_j\| \;\;|\;\;
h_{j,c} = e^{-d^2(j,c)/2\sigma^2(t)} \;\;|\;\;
\Delta\mathbf w_j = \eta(t)h_{j,c}(t)(\mathbf x-\mathbf w_j)
$$
$$
\text{Batch SOM: } \mathbf w_j = \frac{\sum_i h_{j,c(i)}\mathbf x_i}{\sum_i h_{j,c(i)}} \;\;|\;\;
\eta(t)=\eta_0e^{-t/\tau_1},\;\; \sigma(t)=\sigma_0e^{-t/\tau_2}
$$
$$
\text{LVQ1: } \mathbf w_c \leftarrow \mathbf w_c \pm \eta(\mathbf x - \mathbf w_c) \;\;|\;\;
\text{LVQ2.1 window: } \min(d_i/d_j,\,d_j/d_i) > \tfrac{1-w}{1+w}
$$
$$
\text{RBF: } y = \textstyle\sum_i w_i e^{-\|\mathbf x-\boldsymbol\mu_i\|^2/2\sigma_i^2} + b \;\;|\;\;
\mathbf w = (\boldsymbol\Phi^{\mathsf T}\boldsymbol\Phi + \lambda\mathbf I)^{-1}\boldsymbol\Phi^{\mathsf T}\mathbf d \;\;|\;\;
\sigma = d_{\max}/\sqrt{2M}
$$
$$
\text{Cover: } P(N,M_1) = 2^{-(N-1)}\textstyle\sum_{m=0}^{M_1-1}\binom{N-1}{m},\quad \text{capacity } = 2M_1
$$

---

*Previous: [Unit IV](./Unit-4.md) · Next: [Unit VI: Optimization Techniques for Neural Networks](./Unit-6.md)*
