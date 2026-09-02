# Unit III — Feedbackward (Feedback / Recurrent) Neural Networks

> **INT511 – Neural Networks** | M.Tech Level Notes
> **Coverage:** Introduction to feedbackward networks · Feedforward vs feedbackward · Hopfield networks · Boltzmann machines

---

## Table of Contents

1. [Introduction to Feedback Networks](#1-introduction-to-feedback-networks)
2. [Feedforward vs Feedbackward — Systematic Comparison](#2-feedforward-vs-feedbackward--systematic-comparison)
3. [Dynamics, Attractors and Stability](#3-dynamics-attractors-and-stability)
4. [Discrete Hopfield Network](#4-discrete-hopfield-network)
5. [Energy Function and the Convergence Theorem](#5-energy-function-and-the-convergence-theorem)
6. [Storage Capacity of the Hopfield Network](#6-storage-capacity-of-the-hopfield-network)
7. [Spurious States and Remedies](#7-spurious-states-and-remedies)
8. [Continuous Hopfield Network and Optimization (TSP)](#8-continuous-hopfield-network-and-optimization-tsp)
9. [Bidirectional Associative Memory (BAM)](#9-bidirectional-associative-memory-bam)
10. [Stochastic Neurons and Simulated Annealing](#10-stochastic-neurons-and-simulated-annealing)
11. [Boltzmann Machine](#11-boltzmann-machine)
12. [Restricted Boltzmann Machine and Contrastive Divergence](#12-restricted-boltzmann-machine-and-contrastive-divergence)
13. [Deep Belief Networks](#13-deep-belief-networks)
14. [Solved Numericals](#14-solved-numericals)
15. [Viva / Exam Pointers](#15-viva--exam-pointers)

---

## 1. Introduction to Feedback Networks

### 1.1 What "feedbackward" means

A **feedback (recurrent) network** contains at least one **directed cycle** in its connectivity graph. The output of a neuron can (directly or indirectly) influence its own future input. Consequently the network is a **dynamical system**, not a function:

$$
\mathbf s(t+1) = \mathcal F\!\left(\mathbf s(t),\, \mathbf x(t);\,\boldsymbol\theta\right)
$$

and the "answer" is an **attractor** of this dynamics rather than the value of a single forward pass.

```
   FEEDFORWARD (DAG)                 FEEDBACKWARD (cyclic)

   x ─►○─►○─►○─► y                    ┌─────────────┐
       └─►○─►○──┘                     ▼             │
                                    ○────────────►○ │
   Depth = fixed                     │▲            ││
   Computation = 1 pass              ││            ││
   Memoryless                        ▼│            ▼│
                                     ○◄────────────○
                                     Depth = unbounded (in time)
                                     Computation = until convergence
                                     Has internal state / memory
```

### 1.2 Taxonomy of feedback networks

| Family | Recurrence | Purpose |
|---|---|---|
| **Hopfield (discrete/continuous)** | fully connected, symmetric $W=W^{\mathsf T}$, $w_{ii}=0$ | content-addressable memory, optimization |
| **BAM** | two layers, bidirectional | hetero-associative memory |
| **Boltzmann machine** | symmetric, stochastic units | generative model, probability distribution |
| **RBM** | bipartite (no intra-layer links) | tractable generative building block |
| **Brain-State-in-a-Box** | symmetric, clipped-linear | clustering |
| **Elman / Jordan (SRN)** | context units | sequence processing |
| **LSTM / GRU** | gated recurrent | long sequences |
| **Echo State / Reservoir** | random fixed recurrent + linear readout | cheap temporal processing |

**Two distinct roles of recurrence.** Hopfield/Boltzmann use recurrence to *relax to an attractor* on a **static** input (associative memory / inference). Elman/LSTM use recurrence to *carry state across time* on a **sequential** input. This unit is about the first family; the second is covered in the context of BPTT (§3.5).

### 1.3 Associative memory: the core idea

**Content-addressable memory (CAM).** A conventional RAM is *address*-addressable: you supply an address, get contents. A CAM is addressed by (partial or noisy) **content**:

$$
\text{noisy/partial } \tilde{\boldsymbol\xi} \;\longrightarrow\; \text{network dynamics} \;\longrightarrow\; \text{stored } \boldsymbol\xi^\mu
$$

Two flavours:
- **Auto-associative:** $\tilde{\boldsymbol\xi}^\mu \mapsto \boldsymbol\xi^\mu$ (Hopfield). Input and output spaces coincide.
- **Hetero-associative:** $\mathbf x^\mu \mapsto \mathbf y^\mu$ with $\mathbf x,\mathbf y$ different (BAM).

**Basin of attraction.** The set $\mathcal B(\boldsymbol\xi^\mu) = \{\mathbf s : \lim_{t\to\infty}\Phi_t(\mathbf s) = \boldsymbol\xi^\mu\}$. Error correction = the radius of this basin in Hamming distance.

---

## 2. Feedforward vs Feedbackward — Systematic Comparison

| Aspect | Feedforward (MLP/CNN) | Feedbackward (Hopfield/BM/RNN) |
|---|---|---|
| **Graph** | Directed **acyclic** graph | Contains directed **cycles** |
| **Signal flow** | Strictly input → output | Bidirectional / looping |
| **State** | Stateless (memoryless) | Has internal state $\mathbf s(t)$ |
| **Computation** | Single pass, fixed cost | Iterate until convergence / for $T$ steps |
| **Mathematical object** | Function $f:\mathbb R^n\to\mathbb R^m$ | Dynamical system / Markov chain |
| **Output determined by** | Current input only | Input **and** history/initial state |
| **Stability** | Trivially stable | Must be **proved** (Lyapunov/energy) |
| **Learning** | Backpropagation | Hebbian outer product; BPTT; contrastive learning |
| **Typical objective** | Empirical risk minimisation | Energy minimisation / likelihood |
| **Weight symmetry** | Not required | Required ($W = W^{\mathsf T}$) for guaranteed convergence in Hopfield/BM |
| **Self-connections** | N/A | Usually forbidden ($w_{ii}=0$) |
| **Typical use** | Classification, regression | Associative memory, combinatorial optimization, generative modelling, sequences |
| **Convergence issue** | Optimisation only | Optimisation **and** dynamical stability |
| **Biological realism** | Low (brain is highly recurrent) | Higher |

**Key theoretical point.** A feedforward network computes $y = f(x)$ in $O(L)$ steps with $L$ fixed. A recurrent network with $T$ unrolled steps is equivalent to a feedforward network of depth $T$ **with shared weights** — so recurrence buys *unbounded effective depth at constant parameter cost*, at the price of vanishing/exploding gradients over time and no convergence guarantee.

---

## 3. Dynamics, Attractors and Stability

### 3.1 Types of attractors

```
  Point attractor        Limit cycle           Strange attractor
  (fixed point)          (periodic)            (chaotic)

     →→→ ●               ┌──►──┐                    ~~~~
     →→→ ●               │     ▼                  ~ fractal ~
     →→→ ●               └──◄──┘                    ~~~~

  Hopfield/BM aim        Asymmetric W,          Sensitive to
  for these ONLY         async→ possible        initial conditions
```

A Hopfield network with a symmetric $W$ and zero diagonal, updated asynchronously, is **guaranteed** to have only point attractors. Break symmetry or use synchronous updates and you can get 2-cycles.

### 3.2 Lyapunov stability

**Definition.** $E:\mathcal S \to \mathbb R$ is a **Lyapunov (energy) function** for the dynamics $\mathbf s(t+1)=\mathcal F(\mathbf s(t))$ if
1. $E$ is bounded below on $\mathcal S$;
2. $E(\mathbf s(t+1)) \le E(\mathbf s(t))$ for all $t$, with equality only at fixed points.

**Lyapunov's theorem** then guarantees convergence to a fixed point. This is *exactly* the structure of the Hopfield convergence proof (§5).

### 3.3 Update schedules

| Schedule | Rule | Convergence |
|---|---|---|
| **Asynchronous (serial)** | update one randomly chosen $i$ per step | Guaranteed to a fixed point (symmetric $W$, $w_{ii}\ge0$) |
| **Synchronous (parallel)** | update all $i$ simultaneously | Converges to a fixed point **or a 2-cycle** |
| **Block-sequential** | update disjoint blocks | intermediate |

**Why synchronous can cycle.** For the anti-ferromagnetic pair $w_{12}=w_{21}=-1$: state $(+1,-1) \to (+1,-1)$? Compute $h_1 = -x_2 = +1$, $h_2 = -x_1 = -1$ — stable. But $(+1,+1)\to(h_1,h_2)=(-1,-1)\to(+1,+1)$ — a **2-cycle**. Asynchronously, updating neuron 1 first from $(+1,+1)$ gives $(-1,+1)$, then neuron 2: $h_2 = -x_1 = +1$, stays — fixed point reached.

### 3.4 Contraction / stability condition for general recurrent maps

For $\mathbf s(t+1) = \varphi(\mathbf W\mathbf s(t) + \mathbf b)$ with $\varphi$ $\rho$-Lipschitz, the Banach fixed-point theorem gives a unique globally attracting fixed point when
$$
\rho\,\|\mathbf W\|_2 < 1 .
$$
Hopfield deliberately violates this ($\|W\|$ is large) — **multiple attractors are the whole point**.

### 3.5 Backpropagation Through Time (for completeness)

Unroll $\mathbf s_t = \varphi(\mathbf W_s\mathbf s_{t-1} + \mathbf W_x\mathbf x_t)$ over $T$ steps and apply standard backprop with **shared** weights:

$$
\frac{\partial \mathcal L}{\partial \mathbf W_s} = \sum_{t=1}^{T}\frac{\partial \mathcal L_t}{\partial \mathbf W_s},\qquad
\frac{\partial \mathcal L_T}{\partial \mathbf s_k} = \left(\prod_{t=k+1}^{T}\frac{\partial \mathbf s_t}{\partial \mathbf s_{t-1}}\right)^{\!\mathsf T}\frac{\partial\mathcal L_T}{\partial \mathbf s_T}
$$
$$
\frac{\partial \mathbf s_t}{\partial \mathbf s_{t-1}} = \mathrm{diag}\!\big(\varphi'(\cdot)\big)\mathbf W_s
\;\;\Longrightarrow\;\;
\left\|\frac{\partial\mathcal L_T}{\partial \mathbf s_k}\right\| \le \big(\rho\,\|\mathbf W_s\|\big)^{T-k}\|\cdot\| .
$$
Same exponential law as Unit II §3.8, now in **time** rather than depth: $\|W_s\|<1/\rho$ ⇒ vanishing; $>1/\rho$ ⇒ exploding. LSTM's constant-error carousel ($c_t = f_t\odot c_{t-1} + i_t\odot g_t$, Jacobian $\approx \mathrm{diag}(f_t)\approx \mathbf I$) is the standard fix.

---

## 4. Discrete Hopfield Network

### 4.1 Architecture

```
                    ┌──────────────────────────┐
                    │                          │
            ┌───────▼───┐   w₁₂   ┌───────────┐│
            │  Neuron 1 │◄───────►│  Neuron 2 ││
            └───┬───▲───┘         └───┬───▲───┘│
         w₁₄ │  │   │ w₁₃         w₂₃ │   │    │
            ┌▼──┴───┴───┐         ┌───▼───┴───┐│
            │  Neuron 4 │◄───────►│  Neuron 3 ││
            └───────────┘   w₃₄   └───────────┘│
                    │                          │
                    └──────────────────────────┘

   Fully connected · SYMMETRIC  w_ij = w_ji · NO self-loops w_ii = 0
   Every neuron is simultaneously an input and an output unit
```

**Neuron state:** bipolar $s_i \in \{-1,+1\}$ (preferred — the mathematics is cleaner) or binary $\{0,1\}$.

**Update rule (asynchronous):**

$$
\boxed{\;
s_i(t+1) = \mathrm{sgn}\!\left(\sum_{j\ne i} w_{ij}s_j(t) + I_i - \theta_i\right)
= \begin{cases}
+1 & h_i > 0\\
s_i(t) & h_i = 0 \quad (\text{stay})\\
-1 & h_i < 0
\end{cases}\;}
$$

where $h_i = \sum_j w_{ij}s_j + I_i - \theta_i$ is the local field, $I_i$ the external input, $\theta_i$ the threshold (usually 0).

### 4.2 Storage — the Hebbian outer-product rule

Given $P$ patterns $\{\boldsymbol\xi^1,\dots,\boldsymbol\xi^P\}$, $\boldsymbol\xi^\mu \in\{-1,+1\}^N$:

$$
\boxed{\;w_{ij} = \frac{1}{N}\sum_{\mu=1}^{P}\xi^\mu_i \xi^\mu_j \;\;(i \ne j),\qquad w_{ii}=0\;}
$$

Matrix form:
$$
\mathbf W = \frac{1}{N}\left(\sum_{\mu=1}^{P}\boldsymbol\xi^\mu\boldsymbol\xi^{\mu\mathsf T} - P\,\mathbf I\right)
= \frac1N\left(\boldsymbol\Xi\boldsymbol\Xi^{\mathsf T} - P\mathbf I\right),\quad \boldsymbol\Xi = [\boldsymbol\xi^1 \cdots \boldsymbol\xi^P]\in\mathbb R^{N\times P}.
$$

**Properties:** $\mathbf W = \mathbf W^{\mathsf T}$ ✓; $\mathrm{diag}(\mathbf W)=\mathbf 0$ ✓; **one-shot learning** (no iteration); **incremental** (adding a pattern only adds a term); **local** (each $w_{ij}$ depends only on units $i,j$) — genuinely Hebbian.

### 4.3 Why the Hebbian rule makes patterns stable

Take state $\mathbf s = \boldsymbol\xi^\nu$ and compute the local field:

$$
h_i = \sum_{j\ne i} w_{ij}\xi^\nu_j
= \frac1N\sum_{j\ne i}\sum_{\mu}\xi^\mu_i \xi^\mu_j \xi^\nu_j
$$

Split $\mu = \nu$ from $\mu\ne\nu$:

$$
h_i = \underbrace{\frac1N \xi^\nu_i \sum_{j\ne i}\underbrace{\left(\xi^\nu_j\right)^2}_{=1}}_{\text{signal}} \;+\; \underbrace{\frac1N\sum_{\mu\ne\nu}\xi^\mu_i \sum_{j\ne i}\xi^\mu_j\xi^\nu_j}_{\text{crosstalk } C_i^\nu}
$$

$$
\boxed{\;h_i = \frac{N-1}{N}\,\xi^\nu_i \;+\; C^\nu_i \;\approx\; \xi^\nu_i + C^\nu_i \;}
$$

- If the patterns are **mutually orthogonal**, $\sum_j \xi^\mu_j\xi^\nu_j = 0$ for $\mu\ne\nu$ ⇒ $C^\nu_i = 0$ ⇒ $\mathrm{sgn}(h_i) = \xi^\nu_i$ **exactly**: every stored pattern is a fixed point.
- Otherwise the crosstalk term may flip a bit. Define the **crosstalk indicator**

  $$
  \mathcal C^\nu_i \triangleq -\xi^\nu_i C^\nu_i = -\frac{1}{N}\xi^\nu_i\sum_{\mu\ne\nu}\xi^\mu_i\sum_{j\ne i}\xi^\mu_j\xi^\nu_j .
  $$

Bit $i$ of pattern $\nu$ is **stable iff $\mathcal C^\nu_i < 1$**. This inequality is the basis of the capacity calculation in §6.

---

## 5. Energy Function and the Convergence Theorem

### 5.1 The energy (Lyapunov) function

$$
\boxed{\;E(\mathbf s) = -\frac12\sum_{i}\sum_{j\ne i} w_{ij}s_i s_j - \sum_i I_i s_i + \sum_i \theta_i s_i
= -\tfrac12\,\mathbf s^{\mathsf T}\mathbf W\mathbf s - \mathbf I^{\mathsf T}\mathbf s + \boldsymbol\theta^{\mathsf T}\mathbf s \;}
$$

This is exactly the **Ising-model Hamiltonian** of statistical physics (spin glass with couplings $w_{ij}$ and external field $I_i$) — Hopfield's 1982 insight that launched the field's revival.

**Boundedness:** $|E| \le \tfrac12\sum_{ij}|w_{ij}| + \sum_i(|I_i|+|\theta_i|) < \infty$ since $\mathcal S = \{-1,1\}^N$ is finite. ✓ (condition 1 of Lyapunov)

### 5.2 Convergence theorem (asynchronous update)

**Theorem.** If $\mathbf W = \mathbf W^{\mathsf T}$ and $w_{ii} = 0$, then under asynchronous updates the energy is non-increasing, and the network converges to a stable state in a finite number of steps.

**Proof.** Suppose neuron $k$ is updated: $s_k \to s_k' = s_k + \Delta s_k$, all other units fixed. Only terms containing $s_k$ change. Collect them:

$$
E = -s_k\underbrace{\left(\sum_{j\ne k}w_{kj}s_j + I_k - \theta_k\right)}_{h_k} + \underbrace{(\text{terms without } s_k)}_{\text{const}}
$$

Here we used symmetry: the double sum contributes $-\tfrac12(s_k\sum_j w_{kj}s_j + \sum_i s_i w_{ik}s_k) = -s_k\sum_j w_{kj}s_j$ because $w_{kj}=w_{jk}$. Hence

$$
\Delta E = E' - E = -\,\Delta s_k\, h_k \quad (5.1)
$$

Now examine the three cases of the update rule:

| Case | $h_k$ | $s_k$ old | $s_k'$ | $\Delta s_k$ | $\Delta E = -\Delta s_k h_k$ |
|---|---|---|---|---|---|
| 1 | $>0$ | $+1$ | $+1$ | 0 | 0 |
| 2 | $>0$ | $-1$ | $+1$ | $+2$ | $-2h_k < 0$ ✔ |
| 3 | $<0$ | $+1$ | $-1$ | $-2$ | $+2h_k < 0$ ✔ |
| 4 | $<0$ | $-1$ | $-1$ | 0 | 0 |
| 5 | $=0$ | — | unchanged | 0 | 0 |

In every case $\Delta E \le 0$, and $\Delta E < 0$ strictly whenever the state actually changes. Since $E$ takes values in a **finite set** (at most $2^N$ values) and is bounded below, it cannot decrease forever ⇒ after finitely many steps no update changes the state ⇒ a fixed point is reached. $\blacksquare$

**Why $w_{ii}=0$ matters.** With $w_{kk}\ne0$ the collected term becomes $-s_k h_k - \tfrac12 w_{kk}s_k^2$; since $s_k^2=1$ always, a positive $w_{kk}$ merely adds a constant *if* the update uses $h_k$ excluding self-input. But if self-input is included, $\Delta E = -\Delta s_k h_k - \tfrac12 w_{kk}(\Delta s_k)^2$ and a **negative** $w_{kk}$ can make $\Delta E>0$. Setting $w_{ii}=0$ removes the issue entirely and also removes spurious self-sustaining states.

**Why symmetry matters.** Without $w_{kj}=w_{jk}$, the cross terms do not collapse into $-s_kh_k$ and no such energy function exists; asymmetric Hopfield nets can exhibit limit cycles and chaos.

### 5.3 Energy landscape picture

```
 E
 │ ╱╲          ╱╲        ╱╲
 │╱  ╲   ╱╲   ╱  ╲  ╱╲  ╱  ╲
 │    ╲ ╱  ╲ ╱    ╲╱  ╲╱    ╲
 │     V    V      V         V
 │     ↑    ↑      ↑         ↑
 │  spurious ξ¹  spurious    ξ²
 └──────────────────────────────► state space (2^N corners of hypercube)

 A noisy probe lands on the slope; asynchronous updates roll it downhill
 into the nearest basin. Stored patterns = designed minima.
```

**Important:** Hopfield dynamics is a **greedy local** descent — it finds the nearest local minimum, **not** the global one. This is the motivation for the stochastic Boltzmann machine (§10–11).

---

## 6. Storage Capacity of the Hopfield Network

### 6.1 Statistical derivation

Assume $P$ random patterns with i.i.d. $\xi_i^\mu = \pm1$ with probability $\tfrac12$. From §4.3, bit $i$ of pattern $\nu$ is unstable iff

$$
\mathcal C_i^\nu = -\frac1N \sum_{\mu\ne\nu}\sum_{j\ne i}\xi_i^\nu\xi_i^\mu\xi_j^\mu\xi_j^\nu \;>\; 1 .
$$

The sum contains $(P-1)(N-1) \approx PN$ terms, each an independent $\pm1$ with mean 0. By the CLT:

$$
\mathcal C_i^\nu \;\sim\; \mathcal N\!\left(0,\;\sigma^2\right),\qquad
\sigma^2 = \frac{1}{N^2}(P-1)(N-1) \approx \frac{P}{N} .
$$

Define the **load parameter** $\alpha = P/N$. Then $\sigma = \sqrt{\alpha}$ and the probability of a bit error is

$$
\boxed{\;P_{\text{err}} = \Pr[\mathcal C > 1] = \int_{1}^{\infty}\frac{1}{\sqrt{2\pi\alpha}}e^{-x^2/2\alpha}\,dx
= \frac12\left[1 - \mathrm{erf}\!\left(\frac{1}{\sqrt{2\alpha}}\right)\right] = Q\!\left(\frac{1}{\sqrt\alpha}\right)\;}
$$

### 6.2 Capacity table

| $P_{\text{err}}$ | $\alpha_{\max}=P_{\max}/N$ | $P_{\max}$ for $N=100$ |
|---|---|---|
| 0.001 | 0.105 | 10 |
| 0.0036 | 0.138 | 13 |
| 0.01 | 0.185 | 18 |
| 0.05 | 0.37 | 37 |
| 0.10 | 0.61 | 61 |

**Two classical results:**

1. **$\alpha_c \approx 0.138$** (Amit–Gutfreund–Sompolinsky, replica method): above this load the retrieval states abruptly disappear in a **first-order phase transition** to a spin-glass phase. This is the number quoted as "the" Hopfield capacity: $P_{\max} \approx 0.138\,N$.
2. **$P_{\max} = \dfrac{N}{2\ln N}$** if we require *all* patterns to be recalled **perfectly** (zero errors) with probability $\to1$ (McEliece et al.). If a vanishing fraction of errors is tolerated: $P_{\max} = \dfrac{N}{4\ln N}$.

**Information-theoretic view.** $\mathbf W$ has $N(N-1)/2$ free parameters. Storing $0.138N$ patterns of $N$ bits each is $0.138N^2$ bits of content, i.e. roughly $0.276$ bits per synapse — very inefficient compared to the theoretical $\approx 2$ bits/synapse achievable by optimal (e.g. pseudo-inverse / perceptron-rule) storage, which reaches $\alpha = 2$.

### 6.3 Improving capacity

| Method | Rule | Capacity |
|---|---|---|
| Hebbian | $\mathbf W = \frac1N\boldsymbol\Xi\boldsymbol\Xi^{\mathsf T}$ | $0.138N$ |
| **Pseudo-inverse (projection)** | $\mathbf W = \boldsymbol\Xi(\boldsymbol\Xi^{\mathsf T}\boldsymbol\Xi)^{-1}\boldsymbol\Xi^{\mathsf T}$ | $N$ (but non-local, smaller basins) |
| **Storkey rule** | $w_{ij}^{new}=w_{ij}+\frac1N\xi_i\xi_j - \frac1N\xi_ih_{ji}-\frac1N h_{ij}\xi_j$ | $\approx N/\sqrt{2\ln N}$ |
| **Perceptron/Gardner** | iterative, maximise stability margin | $2N$ (theoretical optimum) |
| **Modern (dense) Hopfield** | $E = -\sum_\mu F(\boldsymbol\xi^{\mu\mathsf T}\mathbf s)$, $F(x)=x^n$ or $e^x$ | $O(N^{n-1})$ / exponential |

> **Research link:** the exponential-interaction dense Hopfield network with $F=\exp$ has an update rule mathematically **identical to the attention mechanism of Transformers** ("Hopfield Networks is All You Need", Ramsauer et al. 2020): $\mathbf s^{new} = \boldsymbol\Xi\,\text{softmax}(\beta \boldsymbol\Xi^{\mathsf T}\mathbf s)$. Excellent viva material.

---

## 7. Spurious States and Remedies

Stable states that were **not** stored:

| Type | Description | Energy |
|---|---|---|
| **Reversed states** $-\boldsymbol\xi^\mu$ | Always stable, since $E(-\mathbf s)=E(\mathbf s)$ when $\mathbf I=\boldsymbol\theta=\mathbf 0$ | same as $\boldsymbol\xi^\mu$ |
| **Mixture states** | $\mathbf s = \mathrm{sgn}(\pm\boldsymbol\xi^{\mu_1}\pm\boldsymbol\xi^{\mu_2}\pm\boldsymbol\xi^{\mu_3})$ — odd numbers only | higher |
| **Spin-glass states** | uncorrelated with any $\boldsymbol\xi^\mu$; proliferate for $\alpha>\alpha_c$ | varies |

**Why only odd mixtures?** For an even mixture, e.g. $\mathrm{sgn}(\xi^1_i + \xi^2_i)$, the argument is 0 whenever $\xi^1_i = -\xi^2_i$ (half the sites) — undefined. Odd mixtures always have a non-zero majority.

**Remedies:**
- **Stochastic updates / simulated annealing** — thermal noise lets the system escape shallow spurious minima (see §10).
- **Unlearning** (Hopfield–Feinstein–Palmer 1983): let the net relax from random states and apply *anti*-Hebbian $\Delta w_{ij} = -\epsilon\, s_is_j$ at the reached attractor. Deepens genuine memories relative to spurious ones. (Later interpreted as an early "dreaming"/REM-sleep model.)
- **Pseudo-inverse rule** removes the crosstalk term altogether.
- **Sparse coding** — patterns with low activity $\langle\xi\rangle \ll 1$ have far fewer spurious mixtures and higher capacity $\propto N/(a\ln a)$.

---

## 8. Continuous Hopfield Network and Optimization (TSP)

### 8.1 Continuous model (Hopfield 1984)

Graded-response neurons with RC dynamics:

$$
C_i\frac{du_i}{dt} = -\frac{u_i}{R_i} + \sum_{j\ne i}w_{ij}V_j + I_i,\qquad V_i = g(u_i) = \tfrac12\left[1+\tanh\!\left(\frac{u_i}{u_0}\right)\right]
$$

$u_i$ = internal (membrane) potential, $V_i \in (0,1)$ = output firing rate, $g$ a monotone-increasing sigmoid, $u_0$ = gain parameter.

### 8.2 Energy function and proof of descent

$$
\boxed{\;E = -\frac12\sum_i\sum_{j\ne i} w_{ij}V_iV_j - \sum_i I_iV_i + \sum_i \frac{1}{R_i}\int_0^{V_i} g^{-1}(V)\,dV \;}
$$

**Proof that $dE/dt \le 0$:**

$$
\frac{dE}{dt} = \sum_i \frac{\partial E}{\partial V_i}\frac{dV_i}{dt}
= \sum_i \left(-\sum_{j}w_{ij}V_j - I_i + \frac{u_i}{R_i}\right)\frac{dV_i}{dt}
$$

The bracket is exactly $-C_i\,du_i/dt$ from the dynamics. Hence

$$
\frac{dE}{dt} = -\sum_i C_i \frac{du_i}{dt}\frac{dV_i}{dt}
= -\sum_i C_i\, g^{-1}{}'(V_i)\left(\frac{dV_i}{dt}\right)^2
$$

using $du_i/dt = \frac{d}{dt}g^{-1}(V_i) = g^{-1}{}'(V_i)\,dV_i/dt$. Since $g$ is monotonically increasing, $g^{-1}{}'>0$, and $C_i>0$:

$$
\boxed{\;\frac{dE}{dt} \le 0,\quad \text{with equality iff } \frac{dV_i}{dt}=0\ \forall i\;} \qquad\blacksquare
$$

**High-gain limit ($u_0 \to 0$):** the integral term vanishes and the minima migrate to the hypercube corners — recovering the discrete model.

### 8.3 Travelling Salesman Problem (Hopfield–Tank 1985)

**Encoding.** For $n$ cities use an $n\times n$ matrix of neurons; $V_{Xi} = 1$ means "city $X$ is visited at position $i$ in the tour". A valid tour = a **permutation matrix**.

**Energy function** — constraints as penalty terms plus the objective:

$$
E = \underbrace{\frac{A}{2}\sum_X\sum_i\sum_{j\ne i} V_{Xi}V_{Xj}}_{\text{(1) each city ≤ once}}
+ \underbrace{\frac{B}{2}\sum_i\sum_X\sum_{Y\ne X}V_{Xi}V_{Yi}}_{\text{(2) each position ≤ one city}}
+ \underbrace{\frac{C}{2}\left(\sum_X\sum_i V_{Xi} - n\right)^{2}}_{\text{(3) exactly } n \text{ ones}}
+ \underbrace{\frac{D}{2}\sum_X\sum_{Y\ne X}\sum_i d_{XY}V_{Xi}\left(V_{Y,i+1}+V_{Y,i-1}\right)}_{\text{(4) minimise total tour length}}
$$

Terms (1)–(3) are **zero exactly on permutation matrices** and positive otherwise; term (4) is the tour length.

**Reading off the weights.** Matching to $E=-\tfrac12\sum w_{Xi,Yj}V_{Xi}V_{Yj} - \sum I_{Xi}V_{Xi}$:

$$
w_{Xi,Yj} = -A\,\delta_{XY}(1-\delta_{ij}) - B\,\delta_{ij}(1-\delta_{XY}) - C - D\,d_{XY}\left(\delta_{j,i+1}+\delta_{j,i-1}\right)
$$
$$
I_{Xi} = +C\,n
$$

All connections are **inhibitory** except the bias — a classic winner-take-all-like architecture.

**Critical assessment (needed for M.Tech answers):**
- ✔ Massively parallel, $O(n^2)$ neurons, converges in ~$O(1)$ analogue time.
- ✘ Finds only **local** minima; solution quality depends strongly on $A,B,C,D$ and initial conditions.
- ✘ Wilson & Pawley (1988) showed only ~8 % of runs produced valid tours for $n=10$ with the original parameters.
- ✘ Scales badly: $n^2$ neurons, $n^4$ weights.
- ➜ Modern usage: as a *relaxation* heuristic, or replaced by simulated annealing / mean-field annealing / learned heuristics.

**Other problems mappable to Hopfield energy:** graph bipartitioning, MaxCut, job-shop scheduling, A/D conversion, linear programming, image restoration (MRF). Any problem expressible as **QUBO** ($\min \mathbf x^{\mathsf T}\mathbf Q\mathbf x$, $\mathbf x\in\{0,1\}^n$) maps directly — the same formulation used by modern quantum annealers.

---

## 9. Bidirectional Associative Memory (BAM)

**Kosko (1988)** — a two-layer hetero-associative recurrent memory.

```
    Layer X (n units)          Layer Y (m units)
        x₁ ●◄──────  W  ──────►● y₁
        x₂ ●◄──────  Wᵀ ──────►● y₂
         ⋮                       ⋮
        xₙ ●◄─────────────────►● y_m
```

**Storage** (bipolar pairs $(\mathbf x^\mu,\mathbf y^\mu)$):
$$
\mathbf W = \sum_{\mu=1}^{P}\mathbf x^\mu \mathbf y^{\mu\mathsf T} \quad (n\times m)
$$

**Recall (bidirectional resonance):**
$$
\mathbf y(t+1) = \mathrm{sgn}\!\left(\mathbf W^{\mathsf T}\mathbf x(t)\right),\qquad
\mathbf x(t+1) = \mathrm{sgn}\!\left(\mathbf W\mathbf y(t+1)\right)
$$

**Energy:** $E(\mathbf x,\mathbf y) = -\mathbf x^{\mathsf T}\mathbf W\mathbf y$. Kosko's theorem: **any** real matrix $\mathbf W$ is bidirectionally stable — the network always converges (no learning-induced instability). Capacity is weak: $P_{\max} \approx \min(n,m)$, and in practice $\ll$ that.

**BAM as a special Hopfield net:** stacking $\mathbf s = [\mathbf x;\mathbf y]$ with block weight matrix $\begin{pmatrix}\mathbf 0 & \mathbf W\\ \mathbf W^{\mathsf T} & \mathbf 0\end{pmatrix}$ gives a symmetric, zero-diagonal Hopfield network — hence the convergence guarantee. **This bipartite structure is precisely the structure of an RBM** (§12).

---

## 10. Stochastic Neurons and Simulated Annealing

### 10.1 The problem with deterministic descent

Hopfield dynamics is **zero-temperature**: it only ever goes downhill, so it becomes trapped in the first local minimum encountered. To find the *global* minimum we must sometimes accept uphill moves.

### 10.2 Stochastic (Glauber) update

Replace $s_i = \mathrm{sgn}(h_i)$ by a probabilistic rule:

$$
\boxed{\;\Pr(s_i = +1) = \sigma\!\left(\frac{2h_i}{T}\right) = \frac{1}{1+e^{-2h_i/T}},\qquad
\Pr(s_i=-1) = 1-\Pr(s_i=+1)\;}
$$

Equivalently, in terms of the energy change $\Delta E_i = 2 s_i h_i$ for flipping unit $i$:
$$
\Pr(\text{flip}) = \frac{1}{1+e^{\Delta E_i/T}} \quad\text{(heat bath / Glauber)},
$$
or the Metropolis form $\Pr(\text{accept}) = \min\left(1, e^{-\Delta E/T}\right)$.

**Limits:** $T\to0$ recovers the deterministic Hopfield rule; $T\to\infty$ gives a fair coin (pure random walk).

### 10.3 Stationary distribution: the Boltzmann–Gibbs distribution

The Markov chain defined above satisfies **detailed balance** with respect to
$$
\boxed{\;P(\mathbf s) = \frac{1}{Z}\exp\!\left(-\frac{E(\mathbf s)}{T}\right),\qquad Z = \sum_{\mathbf s'}e^{-E(\mathbf s')/T}\;}
$$

*Verification of detailed balance* for a single-site flip $\mathbf s \leftrightarrow \mathbf s'$:
$$
\frac{P(\mathbf s)\,q(\mathbf s\to\mathbf s')}{P(\mathbf s')\,q(\mathbf s'\to\mathbf s)}
= \frac{e^{-E/T}}{e^{-E'/T}}\cdot\frac{1/(1+e^{\Delta E/T})}{1/(1+e^{-\Delta E/T})}
= e^{\Delta E/T}\cdot\frac{1+e^{-\Delta E/T}}{1+e^{\Delta E/T}} = e^{\Delta E/T}\cdot e^{-\Delta E/T} = 1 \;✓
$$
Hence $P$ is the unique stationary distribution (the chain is irreducible and aperiodic for $T>0$). **This is the theoretical bridge to the Boltzmann machine.**

### 10.4 Simulated annealing

> **Algorithm (Kirkpatrick, Gelatt & Vecchi, 1983)**
> 1. Start at high temperature $T_0$ (system explores freely).
> 2. Run the stochastic update to (approximate) equilibrium.
> 3. Lower $T$ according to a schedule; repeat.
> 4. As $T\to0$, $P(\mathbf s)$ concentrates on the **global** minima of $E$.

**Why it works, formally.** As $T\to 0^+$,
$$
P(\mathbf s) = \frac{e^{-E(\mathbf s)/T}}{\sum_{\mathbf s'}e^{-E(\mathbf s')/T}} \;\longrightarrow\;
\begin{cases}1/|\mathcal G| & \mathbf s \in \mathcal G = \arg\min E\\ 0 & \text{else}\end{cases}
$$
because the ratio $e^{-(E(\mathbf s)-E_{\min})/T}\to 0$ for any $E(\mathbf s)>E_{\min}$.

**Geman & Geman (1984) guarantee:** convergence to the global optimum w.p. 1 requires the *logarithmic* schedule
$$
T(k) \ge \frac{C}{\log(1+k)} .
$$
This is impractically slow, so in practice one uses geometric cooling $T_{k+1}=\gamma T_k$ with $\gamma \in [0.8,0.99]$ and accepts approximate optimality.

---

## 11. Boltzmann Machine

### 11.1 Definition

A **Boltzmann machine** is a stochastic Hopfield network with **hidden units**, interpreted as a generative probabilistic model.

```
        VISIBLE units v (clamped to data)      HIDDEN units h (latent)
        ┌──────────────────────┐               ┌──────────────────┐
        │  v₁ ●───●  v₂        │               │   h₁ ●───● h₂    │
        │      \ / \           │               │       \ /        │
        │  v₃ ● ─ ● v₄ ────────┼───── L ───────┼──── ● ─ ●  h₃    │
        └──────────────────────┘               └──────────────────┘
             intra-visible J                        intra-hidden K
   General BM: ALL of J, K, L present  →  inference is intractable
```

State $\mathbf s = (\mathbf v, \mathbf h)$, $s_i\in\{0,1\}$ (or $\pm1$).

$$
E(\mathbf v,\mathbf h) = -\sum_{i<j} w_{ij}s_is_j - \sum_i b_i s_i,\qquad
P(\mathbf v,\mathbf h) = \frac{e^{-E(\mathbf v,\mathbf h)}}{Z},\quad Z = \sum_{\mathbf v,\mathbf h}e^{-E(\mathbf v,\mathbf h)}
$$

(Absorb $T$ into the weights, i.e. set $T=1$.)

The model's distribution over the **observed** variables is the marginal
$$
P(\mathbf v) = \sum_{\mathbf h}P(\mathbf v,\mathbf h) = \frac{1}{Z}\sum_{\mathbf h}e^{-E(\mathbf v,\mathbf h)} .
$$

### 11.2 Learning rule — full derivation

Maximise the log-likelihood of the data $\mathcal D = \{\mathbf v^{(n)}\}_{n=1}^{N}$:

$$
\ell(\mathbf w) = \frac1N\sum_n \log P(\mathbf v^{(n)})
= \frac1N\sum_n\left[\log\sum_{\mathbf h}e^{-E(\mathbf v^{(n)},\mathbf h)} - \log Z\right]
$$

**Differentiate the first term.**
$$
\frac{\partial}{\partial w_{ij}}\log\sum_{\mathbf h}e^{-E(\mathbf v,\mathbf h)}
= \frac{\sum_{\mathbf h}e^{-E}\left(-\frac{\partial E}{\partial w_{ij}}\right)}{\sum_{\mathbf h}e^{-E}}
= \sum_{\mathbf h}P(\mathbf h|\mathbf v)\,s_is_j
= \big\langle s_is_j\big\rangle_{P(\mathbf h|\mathbf v)}
$$
using $-\partial E/\partial w_{ij} = s_is_j$.

**Differentiate the second term.**
$$
\frac{\partial \log Z}{\partial w_{ij}} = \frac1Z\sum_{\mathbf v,\mathbf h}e^{-E}\,s_is_j
= \sum_{\mathbf v,\mathbf h}P(\mathbf v,\mathbf h)s_is_j = \big\langle s_is_j\big\rangle_{P(\mathbf v,\mathbf h)}
$$

**Combining:**

$$
\boxed{\;\frac{\partial \ell}{\partial w_{ij}} = \underbrace{\big\langle s_is_j\big\rangle_{\text{data}}}_{\text{positive phase } \rho^+_{ij}} - \underbrace{\big\langle s_is_j\big\rangle_{\text{model}}}_{\text{negative phase } \rho^-_{ij}}
\qquad\Longrightarrow\qquad \Delta w_{ij} = \eta\left(\rho^+_{ij} - \rho^-_{ij}\right)\;}
$$

Similarly $\Delta b_i = \eta(\langle s_i\rangle_{\text{data}} - \langle s_i\rangle_{\text{model}})$.

### 11.3 Interpretation: the two phases

| Phase | Also called | Procedure | Effect |
|---|---|---|---|
| **Positive** $\rho^+$ | "clamped", "wake" | clamp $\mathbf v$ to data, sample $\mathbf h$ to equilibrium, record correlations | **raises** $P$ of the data (Hebbian) |
| **Negative** $\rho^-$ | "free", "sleep", "dreaming" | let the whole net run free to equilibrium, record correlations | **lowers** $P$ of the model's own fantasies (anti-Hebbian) |

At the optimum the two correlation matrices are equal — the model's fantasies statistically match reality. **This is a beautifully local rule** (each synapse needs only $\langle s_is_j\rangle$ of its two endpoints), unlike backprop.

**Equivalent KL formulation.** The gradient above is exactly $-\partial D_{\text{KL}}\!\left(P_{\text{data}}(\mathbf v)\,\|\,P_{\text{model}}(\mathbf v)\right)/\partial w_{ij}$: learning minimises KL divergence.

### 11.4 Why the general BM is impractical

Both phases require **Gibbs sampling to equilibrium**:
$$
\Pr(s_i = 1 \mid \mathbf s_{-i}) = \sigma\!\left(\sum_{j\ne i}w_{ij}s_j + b_i\right).
$$
Mixing time can be exponential; $Z$ requires summing $2^{|\mathbf v|+|\mathbf h|}$ terms. Both phases are noisy, the gradient is a difference of two noisy quantities, and learning is glacially slow. **Solution: restrict the connectivity.**

---

## 12. Restricted Boltzmann Machine and Contrastive Divergence

### 12.1 The restriction

```
       h₁ ●    h₂ ●    h₃ ●        HIDDEN  (no h–h connections)
           \  / | \   /  |
            \/  |  \ /   |         bipartite W  (m × n)
            /\  |  / \   |
           /  \ | /   \  |
       v₁ ●    ●v₂    ● v₃          VISIBLE (no v–v connections)
```

$$
E(\mathbf v,\mathbf h) = -\mathbf a^{\mathsf T}\mathbf v - \mathbf b^{\mathsf T}\mathbf h - \mathbf h^{\mathsf T}\mathbf W\mathbf v
= -\sum_i a_iv_i - \sum_j b_jh_j - \sum_{i,j}w_{ji}h_jv_i
$$

### 12.2 Conditional independence (the crucial property)

$$
P(\mathbf h|\mathbf v) = \frac{P(\mathbf v,\mathbf h)}{P(\mathbf v)} \propto \exp\!\left(\sum_j\left[b_j + \sum_i w_{ji}v_i\right]h_j\right) = \prod_j \exp\!\big(h_j\,\tilde b_j\big)
$$
with $\tilde b_j = b_j + \mathbf w_j^{\mathsf T}\mathbf v$. The joint factorises over $j$, so

$$
\boxed{\;P(h_j = 1\mid\mathbf v) = \sigma\!\left(b_j + \sum_i w_{ji}v_i\right),\qquad
P(v_i = 1\mid\mathbf h) = \sigma\!\left(a_i + \sum_j w_{ji}h_j\right)\;}
$$

**All hidden units can be sampled in parallel given $\mathbf v$, and vice versa** — *block Gibbs sampling*. This single fact is what makes RBMs usable.

### 12.3 Free energy

$$
P(\mathbf v) = \frac{1}{Z}e^{-F(\mathbf v)},\qquad F(\mathbf v) = -\log\sum_{\mathbf h}e^{-E(\mathbf v,\mathbf h)}
$$

Derivation (binary $h_j\in\{0,1\}$):
$$
\sum_{\mathbf h}e^{-E} = e^{\mathbf a^{\mathsf T}\mathbf v}\prod_j\sum_{h_j\in\{0,1\}}e^{h_j(b_j + \mathbf w_j^{\mathsf T}\mathbf v)}
= e^{\mathbf a^{\mathsf T}\mathbf v}\prod_j\left(1 + e^{b_j+\mathbf w_j^{\mathsf T}\mathbf v}\right)
$$

$$
\boxed{\;F(\mathbf v) = -\mathbf a^{\mathsf T}\mathbf v - \sum_{j=1}^{m}\log\!\left(1+e^{\,b_j+\mathbf w_j^{\mathsf T}\mathbf v}\right)
= -\mathbf a^{\mathsf T}\mathbf v - \sum_j \text{softplus}\!\left(b_j+\mathbf w_j^{\mathsf T}\mathbf v\right)\;}
$$

Note this is **exactly a one-hidden-layer neural network with softplus activations** — the RBM/MLP correspondence.

### 12.4 Contrastive Divergence (CD-$k$)

Exact $\rho^-$ needs an infinitely long Gibbs chain. **Hinton's trick:** start the chain **at the data** and run only $k$ steps (usually $k=1$).

```
   v⁰  ──►  h⁰  ──►  v¹  ──►  h¹  ──► ... ──► v^k ──► h^k
   ↑data    sample   recon    sample
   └── POSITIVE ──┘           └────── NEGATIVE (after k steps) ──────┘
```

> **Algorithm CD-1 (one training sample $\mathbf v^0$)**
> 1. $\mathbf p^0 = \sigma(\mathbf W\mathbf v^0 + \mathbf b)$; sample $\mathbf h^0 \sim \text{Bernoulli}(\mathbf p^0)$
> 2. $\mathbf q^1 = \sigma(\mathbf W^{\mathsf T}\mathbf h^0 + \mathbf a)$; sample $\mathbf v^1 \sim \text{Bernoulli}(\mathbf q^1)$   *(reconstruction)*
> 3. $\mathbf p^1 = \sigma(\mathbf W\mathbf v^1 + \mathbf b)$   *(use probabilities, not samples, to reduce noise)*
> 4. Updates:
> $$\Delta \mathbf W = \eta\left(\mathbf p^0\mathbf v^{0\mathsf T} - \mathbf p^1\mathbf v^{1\mathsf T}\right),\quad
> \Delta \mathbf a = \eta(\mathbf v^0 - \mathbf v^1),\quad
> \Delta \mathbf b = \eta(\mathbf p^0 - \mathbf p^1)$$

**What CD actually optimises.** It approximately follows the gradient of

$$
\text{CD}_k = D_{\text{KL}}\!\left(P_0\,\|\,P_\infty\right) - D_{\text{KL}}\!\left(P_k\,\|\,P_\infty\right) \;\ge\; 0,
$$

the difference of two KL divergences ("contrastive"). It is **not** the exact likelihood gradient — a small bias term is dropped — but it works remarkably well in practice.

**Variants:** PCD / Persistent CD (keep a persistent fantasy chain across mini-batches — much better for generative quality), Fast PCD, Parallel Tempering.

### 12.5 RBM variants

| Variant | Visible units | Use |
|---|---|---|
| Bernoulli–Bernoulli | binary | binary images |
| **Gaussian–Bernoulli** | $v_i\sim\mathcal N$; $E$ has $\frac{(v_i-a_i)^2}{2\sigma_i^2}$ | real-valued data (needs data standardisation and small $\eta$) |
| Replicated softmax | word counts | document modelling |
| Conditional RBM | + autoregressive links | time series, motion capture |

---

## 13. Deep Belief Networks

**Greedy layer-wise pretraining (Hinton, Osindero & Teh 2006)** — the trick that launched deep learning:

```
   Step 1: train RBM₁ on raw data v
   Step 2: freeze W₁, compute h₁ = σ(W₁v); train RBM₂ on h₁ as its "data"
   Step 3: repeat for RBM₃, ...
   Step 4: stack; add a supervised output layer;
           FINE-TUNE the whole stack with backpropagation (or wake-sleep)
```

A DBN's joint distribution is a hybrid: the top two layers form an undirected RBM, all lower layers are a directed sigmoid belief network:

$$
P(\mathbf v,\mathbf h^1,\dots,\mathbf h^L) = \underbrace{P(\mathbf h^{L-1},\mathbf h^{L})}_{\text{RBM}}\prod_{\ell=0}^{L-2}\underbrace{P(\mathbf h^{\ell}|\mathbf h^{\ell+1})}_{\text{directed}}
$$

**Why pretraining helped (historically):** it initialises weights in a region of parameter space from which backprop converges to better minima; it acts as an unsupervised regulariser; it exploits abundant unlabelled data.

**Why it is rarely used today:** ReLU + He initialisation + BatchNorm + Adam + residual connections + large labelled datasets solved the optimisation problem directly. But the *idea* — unsupervised pretraining then supervised fine-tuning — is exactly the paradigm behind modern self-supervised foundation models.

**Deep Boltzmann Machine (DBM):** all layers undirected. Better generative model, much harder inference (requires mean-field variational inference for the positive phase + persistent chains for the negative phase).

---

## 14. Solved Numericals

### N1. Hopfield network — construct, verify, recall, and track energy

**Store two orthogonal patterns in a 6-neuron Hopfield network.**
$$
\boldsymbol\xi^1 = (+1,+1,+1,-1,-1,-1)^{\mathsf T},\qquad
\boldsymbol\xi^2 = (+1,+1,-1,+1,+1,-1)^{\mathsf T}
$$

**Orthogonality check:** $\boldsymbol\xi^{1\mathsf T}\boldsymbol\xi^2 = 1+1-1-1-1+1 = \mathbf 0$ ✓

**Step 1 — Weight matrix.** Use the unnormalised Hebbian rule $w_{ij}=\sum_\mu \xi^\mu_i\xi^\mu_j$ ($i\ne j$), $w_{ii}=0$.

| $(i,j)$ | $\xi^1_i\xi^1_j$ | $\xi^2_i\xi^2_j$ | $w_{ij}$ |
|---|---|---|---|
| (1,2) | $+1$ | $+1$ | **2** |
| (1,3) | $+1$ | $-1$ | 0 |
| (1,4) | $-1$ | $+1$ | 0 |
| (1,5) | $-1$ | $+1$ | 0 |
| (1,6) | $-1$ | $-1$ | **−2** |
| (2,3) | $+1$ | $-1$ | 0 |
| (2,4) | $-1$ | $+1$ | 0 |
| (2,5) | $-1$ | $+1$ | 0 |
| (2,6) | $-1$ | $-1$ | **−2** |
| (3,4) | $-1$ | $-1$ | **−2** |
| (3,5) | $-1$ | $-1$ | **−2** |
| (3,6) | $-1$ | $+1$ | 0 |
| (4,5) | $+1$ | $+1$ | **2** |
| (4,6) | $+1$ | $-1$ | 0 |
| (5,6) | $+1$ | $-1$ | 0 |

$$
\mathbf W = \begin{pmatrix}
0 & 2 & 0 & 0 & 0 & -2\\
2 & 0 & 0 & 0 & 0 & -2\\
0 & 0 & 0 & -2 & -2 & 0\\
0 & 0 & -2 & 0 & 2 & 0\\
0 & 0 & -2 & 2 & 0 & 0\\
-2 & -2 & 0 & 0 & 0 & 0
\end{pmatrix}
$$
Symmetric ✓, zero diagonal ✓.

**Step 2 — Verify both patterns are fixed points.**

For $\boldsymbol\xi^1$: $\mathbf h = \mathbf W\boldsymbol\xi^1$
$$
h_1 = 2(1) + (-2)(-1) = 4,\quad h_2 = 2(1)+(-2)(-1) = 4,\quad h_3 = -2(-1)-2(-1)=4
$$
$$
h_4 = -2(1)+2(-1) = -4,\quad h_5 = -2(1)+2(-1) = -4,\quad h_6 = -2(1)-2(1) = -4
$$
$\mathrm{sgn}(\mathbf h) = (+,+,+,-,-,-) = \boldsymbol\xi^1$ ✓

For $\boldsymbol\xi^2 = (+1,+1,-1,+1,+1,-1)$:
$$
h_1 = 2(1)+(-2)(-1)=4,\; h_2 = 4,\; h_3 = -2(1)-2(1) = -4,
$$
$$
h_4 = -2(-1)+2(1)=4,\; h_5 = -2(-1)+2(1)=4,\; h_6 = -2(1)-2(1)=-4
$$
$\mathrm{sgn}(\mathbf h) = (+,+,-,+,+,-) = \boldsymbol\xi^2$ ✓

**Step 3 — Recall from a corrupted probe.** Take $\boldsymbol\xi^1$ and flip bits 1 and 4:
$$
\mathbf s(0) = (-1,\,+1,\,+1,\,+1,\,-1,\,-1)
$$
Hamming distance: $d(\mathbf s,\boldsymbol\xi^1)=2$, $d(\mathbf s,\boldsymbol\xi^2)=3$ — closer to $\boldsymbol\xi^1$.

Asynchronous sweep in order $1,2,3,4,5,6$:

| Step | Unit $k$ | $h_k$ | $\mathrm{sgn}$ | Old $s_k$ | Action | State after |
|---|---|---|---|---|---|---|
| 1 | 1 | $2s_2 - 2s_6 = 2(1)-2(-1)= 4$ | $+1$ | $-1$ | **flip** | $(+1,+1,+1,+1,-1,-1)$ |
| 2 | 2 | $2s_1 - 2s_6 = 2+2 = 4$ | $+1$ | $+1$ | — | unchanged |
| 3 | 3 | $-2s_4-2s_5 = -2(1)-2(-1)=0$ | tie | $+1$ | **stay** | unchanged |
| 4 | 4 | $-2s_3+2s_5 = -2(1)+2(-1) = -4$ | $-1$ | $+1$ | **flip** | $(+1,+1,+1,-1,-1,-1)$ |
| 5 | 5 | $-2s_3+2s_4 = -2(1)+2(-1)=-4$ | $-1$ | $-1$ | — | unchanged |
| 6 | 6 | $-2s_1-2s_2 = -4$ | $-1$ | $-1$ | — | unchanged |

**Final state $= (+1,+1,+1,-1,-1,-1) = \boldsymbol\xi^1$** — the 2-bit error was corrected in a single sweep ✓

**Step 4 — Energy trajectory.** $E = -\tfrac12\mathbf s^{\mathsf T}\mathbf W\mathbf s$.

*Initial* $\mathbf s(0)=(-1,1,1,1,-1,-1)$:
$\mathbf h = (4,\,0,\,0,\,-4,\,0,\,0)$; $\mathbf s^{\mathsf T}\mathbf h = (-1)(4)+(1)(-4)\cdot\!$…
$$
\mathbf s^{\mathsf T}\mathbf h = (-1)(4)+(1)(0)+(1)(0)+(1)(-4)+(-1)(0)+(-1)(0) = -8 \Rightarrow E_0 = +4
$$

*After unit 1 flips* ($\Delta s_1 = +2$, $h_1 = 4$): $\Delta E = -\Delta s_1 h_1 = -8$ ⇒ $E_1 = -4$.
*After unit 4 flips* ($\Delta s_4 = -2$, $h_4 = -4$): $\Delta E = -(-2)(-4) = -8$ ⇒ $E_2 = -12$.

*Direct check at $\boldsymbol\xi^1$:* $\mathbf h = (4,4,4,-4,-4,-4)$, $\mathbf s^{\mathsf T}\mathbf h = 6\times4 = 24 \Rightarrow E = -12$ ✓

$$
\boxed{E:\; +4 \;\to\; -4 \;\to\; -12 \quad\text{monotonically decreasing ✓}}
$$

Both stored patterns (and their reversals $-\boldsymbol\xi^1,-\boldsymbol\xi^2$) sit at the minimum energy $-12$ — the four global minima of this landscape.

---

### N2. Capacity calculation

A Hopfield net with $N = 500$ neurons.

(a) **Standard capacity:** $P_{\max} = 0.138N = 0.138(500) = \mathbf{69}$ patterns.

(b) **Perfect recall (all patterns, zero error):**
$$
P_{\max} = \frac{N}{2\ln N} = \frac{500}{2\ln 500} = \frac{500}{2(6.2146)} = \frac{500}{12.429} = \mathbf{40.2} \approx 40
$$

(c) **Bit-error probability if we store $P = 100$:**
$$
\alpha = \frac{100}{500} = 0.2,\qquad P_{\text{err}} = Q\!\left(\frac{1}{\sqrt{0.2}}\right) = Q(2.2361)
$$
$$
Q(2.2361) = \tfrac12\left[1-\mathrm{erf}(2.2361/\sqrt2)\right] = \tfrac12[1-\mathrm{erf}(1.5811)] = \tfrac12[1-0.97517] = \mathbf{0.0124}
$$
So ~1.24 % of bits are unstable — about $0.0124\times500 \approx 6$ wrong bits per recalled pattern. Since $\alpha=0.2 > \alpha_c = 0.138$, in reality the network is already in the spin-glass phase and retrieval fails catastrophically.

(d) **Storage efficiency:** $\#$synapses $= N(N-1)/2 = 124\,750$; bits stored at capacity $= 69\times500 = 34\,500$ ⇒ $\mathbf{0.277}$ bits/synapse.

---

### N3. Synchronous update produces a 2-cycle

$N=2$, $w_{12}=w_{21}=-1$, $w_{11}=w_{22}=0$, no bias.

Start at $\mathbf s(0) = (+1,+1)$.
**Synchronous:** $h_1 = -s_2 = -1$, $h_2 = -s_1 = -1$ ⇒ $\mathbf s(1) = (-1,-1)$.
Then $h_1 = -(-1) = +1$, $h_2 = +1$ ⇒ $\mathbf s(2) = (+1,+1)$. **2-cycle** ✗

**Asynchronous** (update unit 1 first): $h_1 = -s_2 = -1 \Rightarrow s_1 = -1$, state $(-1,+1)$.
Then $h_2 = -s_1 = +1 \Rightarrow s_2$ stays $+1$. **Fixed point** ✓
Energy check: $E(-1,+1) = -\tfrac12[2w_{12}s_1s_2] = -w_{12}s_1s_2 = -(-1)(-1)(1) = -1$; $E(+1,+1) = -(-1)(1)(1)=+1$. Descent confirmed.

---

### N4. Stochastic neuron probabilities at different temperatures

Local field $h_i = 1.5$.

$$
\Pr(s_i=+1) = \sigma(2h_i/T) = \frac{1}{1+e^{-3/T}}
$$

| $T$ | $3/T$ | $\Pr(s_i=+1)$ | Behaviour |
|---|---|---|---|
| 0.1 | 30 | $1 - 9.4\times10^{-14} \approx 1.0000$ | deterministic (Hopfield limit) |
| 0.5 | 6 | $1/(1+0.002479) = 0.99753$ | nearly deterministic |
| 1.0 | 3 | $1/(1+0.049787) = 0.95257$ | mostly follows the field |
| 3.0 | 1 | $1/(1+0.367879) = 0.73106$ | noticeable noise |
| 10 | 0.3 | $1/(1+0.740818) = 0.57444$ | mostly random |
| $\infty$ | 0 | $0.5$ | pure coin flip |

**Annealing schedule illustration** ($T_0=10$, $\gamma=0.9$): $T_k = 10(0.9)^k$. To reach $T=0.1$: $0.9^k = 0.01 \Rightarrow k = \ln0.01/\ln0.9 = -4.6052/-0.10536 = \mathbf{43.7} \approx 44$ cooling steps.

---

### N5. Metropolis acceptance probabilities

$\Delta E = +2$ (an uphill move).

| $T$ | $e^{-\Delta E/T}$ | $\Pr(\text{accept})$ |
|---|---|---|
| 5.0 | $e^{-0.4} = 0.6703$ | 0.670 |
| 2.0 | $e^{-1} = 0.3679$ | 0.368 |
| 1.0 | $e^{-2} = 0.1353$ | 0.135 |
| 0.5 | $e^{-4} = 0.0183$ | 0.018 |
| 0.1 | $e^{-20} = 2.06\times10^{-9}$ | ≈ 0 |

Downhill moves ($\Delta E<0$) are accepted with probability 1 at every temperature.

---

### N6. RBM — one full CD-1 step (all arithmetic shown)

**Model:** 3 visible, 2 hidden, binary units.
$$
\mathbf W = \begin{pmatrix}0.2 & -0.1 & 0.4\\ 0.3 & 0.5 & -0.2\end{pmatrix}\;(2\times3),\qquad
\mathbf a = (0.1,\,-0.2,\,0.0)^{\mathsf T},\qquad \mathbf b = (0.0,\,0.1)^{\mathsf T}
$$
Training sample $\mathbf v^0 = (1,0,1)^{\mathsf T}$, $\eta = 0.1$.

**(i) Positive phase — hidden probabilities**
$$
b_1 + \mathbf w_1^{\mathsf T}\mathbf v^0 = 0.0 + 0.2(1) + (-0.1)(0) + 0.4(1) = 0.6 \Rightarrow p^0_1 = \sigma(0.6) = \mathbf{0.645656}
$$
$$
b_2 + \mathbf w_2^{\mathsf T}\mathbf v^0 = 0.1 + 0.3(1)+0.5(0)+(-0.2)(1) = 0.2 \Rightarrow p^0_2 = \sigma(0.2) = \mathbf{0.549834}
$$
Sample (with random draws $0.51, 0.72$): $\mathbf h^0 = (1,\,0)^{\mathsf T}$.

**(ii) Reconstruction — visible probabilities given $\mathbf h^0$**
$$
q^1_1 = \sigma(a_1 + w_{11}h_1 + w_{21}h_2) = \sigma(0.1 + 0.2(1) + 0.3(0)) = \sigma(0.3) = \mathbf{0.574443}
$$
$$
q^1_2 = \sigma(-0.2 + (-0.1)(1) + 0.5(0)) = \sigma(-0.3) = \mathbf{0.425557}
$$
$$
q^1_3 = \sigma(0.0 + 0.4(1) + (-0.2)(0)) = \sigma(0.4) = \mathbf{0.598688}
$$
Sample (draws $0.30, 0.88, 0.70$): $\mathbf v^1 = (1,\,0,\,0)^{\mathsf T}$.

**(iii) Negative phase — hidden probabilities given $\mathbf v^1$**
$$
p^1_1 = \sigma(0.0 + 0.2(1)) = \sigma(0.2) = \mathbf{0.549834},\qquad
p^1_2 = \sigma(0.1 + 0.3(1)) = \sigma(0.4) = \mathbf{0.598688}
$$

**(iv) Gradients**

Positive correlations $\mathbf p^0\mathbf v^{0\mathsf T}$:
$$
\begin{pmatrix}0.645656\\0.549834\end{pmatrix}(1\;\;0\;\;1) =
\begin{pmatrix}0.645656 & 0 & 0.645656\\ 0.549834 & 0 & 0.549834\end{pmatrix}
$$

Negative correlations $\mathbf p^1\mathbf v^{1\mathsf T}$:
$$
\begin{pmatrix}0.549834\\0.598688\end{pmatrix}(1\;\;0\;\;0) =
\begin{pmatrix}0.549834 & 0 & 0\\ 0.598688 & 0 & 0\end{pmatrix}
$$

$$
\Delta\mathbf W = 0.1\begin{pmatrix}0.095822 & 0 & 0.645656\\ -0.048854 & 0 & 0.549834\end{pmatrix}
= \begin{pmatrix}0.0095822 & 0 & 0.0645656\\ -0.0048854 & 0 & 0.0549834\end{pmatrix}
$$
$$
\Delta\mathbf a = 0.1\left[(1,0,1)-(1,0,0)\right] = (0,\,0,\,\mathbf{0.1})
$$
$$
\Delta\mathbf b = 0.1\left[(0.645656,0.549834)-(0.549834,0.598688)\right] = (\mathbf{0.0095822},\,\mathbf{-0.0048854})
$$

**Updated weights:**
$$
\mathbf W' = \begin{pmatrix}0.209582 & -0.1 & 0.464566\\ 0.295115 & 0.5 & -0.145017\end{pmatrix},\quad
\mathbf a' = (0.1,\,-0.2,\,0.1),\quad \mathbf b' = (0.0095822,\,0.0951146)
$$

**Interpretation:** $w_{13}$ and $w_{23}$ (connections to $v_3$) increased sharply — the reconstruction wrongly turned $v_3$ off, so the update strengthens exactly the couplings that would have kept it on. $a_3$ also increased for the same reason.

**(v) Free energy of $\mathbf v^0$ (before the update)**
$$
F(\mathbf v^0) = -\mathbf a^{\mathsf T}\mathbf v^0 - \sum_j \log\!\left(1+e^{b_j+\mathbf w_j^{\mathsf T}\mathbf v^0}\right)
$$
$$
= -[0.1(1)+(-0.2)(0)+0(1)] - \left[\log(1+e^{0.6}) + \log(1+e^{0.2})\right]
$$
$$
= -0.1 - \left[\log(2.822119) + \log(2.221403)\right] = -0.1 - [1.037840 + 0.798139] = \mathbf{-1.935979}
$$

Lower free energy ⇒ higher $P(\mathbf v)$. Training pushes $F$ down on data and up on fantasies.

---

### N7. BAM storage and recall

Store one pair: $\mathbf x^1 = (1,-1,1,-1)^{\mathsf T}$ (n=4), $\mathbf y^1 = (1,1,-1)^{\mathsf T}$ (m=3).

$$
\mathbf W = \mathbf x^1\mathbf y^{1\mathsf T} =
\begin{pmatrix}1\\-1\\1\\-1\end{pmatrix}(1\;\;1\;\;-1)
=\begin{pmatrix}1&1&-1\\ -1&-1&1\\ 1&1&-1\\ -1&-1&1\end{pmatrix}
$$

**Recall from a corrupted $\tilde{\mathbf x} = (1,1,1,-1)$ (bit 2 flipped):**
$$
\mathbf W^{\mathsf T}\tilde{\mathbf x} = \begin{pmatrix}1&-1&1&-1\\ 1&-1&1&-1\\ -1&1&-1&1\end{pmatrix}\begin{pmatrix}1\\1\\1\\-1\end{pmatrix}
= \begin{pmatrix}1-1+1+1\\ 1-1+1+1\\ -1+1-1-1\end{pmatrix} = \begin{pmatrix}2\\2\\-2\end{pmatrix}
$$
$$
\mathbf y = \mathrm{sgn}(2,2,-2) = (1,1,-1) = \mathbf y^1 \;✓
$$
**Back-propagate to $\mathbf x$:**
$$
\mathbf W\mathbf y = \begin{pmatrix}1+1+1\\ -1-1-1\\ 1+1+1\\ -1-1-1\end{pmatrix} = \begin{pmatrix}3\\-3\\3\\-3\end{pmatrix}
\Rightarrow \mathbf x = (1,-1,1,-1) = \mathbf x^1 \;✓
$$
The corrupted bit was corrected after one bidirectional pass.
Energy: $E = -\mathbf x^{\mathsf T}\mathbf W\mathbf y = -(1)(3)-(-1)(-3)-(1)(3)-(-1)(-3) = -12$ (minimum).

---

### N8. Boltzmann distribution over states

A 3-neuron Boltzmann machine with $w_{12}=1$, $w_{13}=-1$, $w_{23}=1$, all biases 0, bipolar states, $T=1$.
$E(\mathbf s) = -(w_{12}s_1s_2 + w_{13}s_1s_3 + w_{23}s_2s_3)$.

| $\mathbf s$ | $s_1s_2$ | $s_1s_3$ | $s_2s_3$ | $E$ | $e^{-E}$ | $P$ |
|---|---|---|---|---|---|---|
| $(+,+,+)$ | +1 | +1 | +1 | $-(1-1+1) = -1$ | 2.71828 | 0.1737 |
| $(+,+,-)$ | +1 | −1 | −1 | $-(1+1-1) = -1$ | 2.71828 | 0.1737 |
| $(+,-,+)$ | −1 | +1 | −1 | $-(-1-1-1)=3$ | 0.04979 | 0.0032 |
| $(+,-,-)$ | −1 | −1 | +1 | $-(-1+1+1)=-1$ | 2.71828 | 0.1737 |
| $(-,+,+)$ | −1 | −1 | +1 | $-1$ | 2.71828 | 0.1737 |
| $(-,+,-)$ | −1 | +1 | −1 | $3$ | 0.04979 | 0.0032 |
| $(-,-,+)$ | +1 | −1 | −1 | $-1$ | 2.71828 | 0.1737 |
| $(-,-,-)$ | +1 | +1 | +1 | $-1$ | 2.71828 | 0.1737 |

$$
Z = 6(2.71828) + 2(0.04979) = 16.30968 + 0.09958 = \mathbf{16.40926}
$$
Check: $6(0.17370) + 2(0.00303) = 1.0422 + 0.0061 \approx 1.00$ ✓ (rounding).

**Observation:** the two "frustrated" states $(+,-,+)$ and $(-,+,-)$ have far lower probability — this is a 3-spin **frustrated** triangle (product of couplings $= 1\cdot(-1)\cdot1 = -1 < 0$), so no state can satisfy all three constraints; six states tie for the minimum energy.

At $T = 0.1$: $e^{-E/T}$ for $E=-1$ is $e^{10}=22026$, for $E=3$ is $e^{-30}=9.4\times10^{-14}$ — the frustrated states become essentially unreachable, and the distribution is uniform over the six ground states.

---

## 15. Viva / Exam Pointers

**Likely long questions**
1. Derive the Hopfield energy function and prove convergence under asynchronous updates.
2. Derive the crosstalk term and hence the capacity $P_{\max}\approx0.138N$.
3. Compare feedforward and feedbackward networks across at least eight dimensions.
4. Prove $dE/dt \le 0$ for the continuous Hopfield network.
5. Formulate the TSP as a Hopfield energy function and state its shortcomings.
6. Derive the Boltzmann machine learning rule $\Delta w_{ij}=\eta(\rho^+_{ij}-\rho^-_{ij})$ from the log-likelihood.
7. Derive $P(h_j=1|\mathbf v)$ for an RBM and the free energy $F(\mathbf v)$; explain CD-1.
8. Explain simulated annealing and why the Boltzmann distribution concentrates on global minima as $T\to0$.

**Traps**
- Convergence requires **symmetric $W$ AND asynchronous updates AND $w_{ii}=0$**. Drop any one and the guarantee is gone.
- $\Delta E = -\Delta s_k h_k$, *not* $-s_k h_k$. The factor comes from $\Delta s_k = \pm2$.
- Capacity $0.138N$ assumes **random, roughly orthogonal** patterns. Correlated patterns give far lower capacity.
- $-\boldsymbol\xi^\mu$ is *always* stored alongside $\boldsymbol\xi^\mu$ — you cannot avoid it in a zero-field Hopfield network.
- In CD, the positive phase uses the **data** $\mathbf v^0$; the negative phase uses the **reconstruction** $\mathbf v^1$. Getting the sign or the order wrong reverses learning.
- RBM has no $v$–$v$ or $h$–$h$ connections; a general BM does. Only the restriction makes block Gibbs sampling possible.

**One-line formula sheet**

$$
w_{ij} = \tfrac1N\textstyle\sum_\mu \xi^\mu_i\xi^\mu_j \;\;|\;\;
E = -\tfrac12\mathbf s^{\mathsf T}\mathbf W\mathbf s - \mathbf I^{\mathsf T}\mathbf s \;\;|\;\;
\Delta E = -\Delta s_k h_k \;\;|\;\;
P_{\max}\approx 0.138N
$$
$$
P(\mathbf s) = e^{-E(\mathbf s)/T}/Z \;\;|\;\;
\Pr(s_i{=}1) = \sigma(2h_i/T) \;\;|\;\;
\Delta w_{ij} = \eta(\langle s_is_j\rangle_{\text{data}} - \langle s_is_j\rangle_{\text{model}})
$$
$$
P(h_j{=}1|\mathbf v) = \sigma(b_j + \mathbf w_j^{\mathsf T}\mathbf v) \;\;|\;\;
F(\mathbf v) = -\mathbf a^{\mathsf T}\mathbf v - \textstyle\sum_j\log(1+e^{b_j+\mathbf w_j^{\mathsf T}\mathbf v})
$$

---

*Previous: [Unit II](./Unit-2.md) · Next: [Unit IV: Dimensionality Reduction Techniques](./Unit-4.md)*
