# Unit IV — Dimensionality Reduction Techniques

> **INT511 – Neural Networks** | M.Tech Level Notes
> **Coverage:** Concept of dimensionality reduction · Principal Component Analysis · Linear Discriminant Analysis · Kernel PCA · Independent Component Analysis

---

## Table of Contents

1. [The Concept of Dimensionality Reduction](#1-the-concept-of-dimensionality-reduction)
2. [The Curse of Dimensionality](#2-the-curse-of-dimensionality)
3. [Principal Component Analysis](#3-principal-component-analysis)
4. [PCA via SVD, Probabilistic PCA, and Neural PCA](#4-pca-via-svd-probabilistic-pca-and-neural-pca)
5. [Linear Discriminant Analysis](#5-linear-discriminant-analysis)
6. [Kernel PCA](#6-kernel-pca)
7. [Independent Component Analysis](#7-independent-component-analysis)
8. [Comparison and Selection Guide](#8-comparison-and-selection-guide)
9. [Solved Numericals](#9-solved-numericals)
10. [Viva / Exam Pointers](#10-viva--exam-pointers)

---

## 1. The Concept of Dimensionality Reduction

### 1.1 Formal statement

Given $\mathbf X = \{\mathbf x_1,\dots,\mathbf x_N\} \subset \mathbb R^{D}$, find a map

$$
g : \mathbb R^{D} \to \mathbb R^{d},\qquad d \ll D,
$$

such that the reduced representation $\mathbf z_i = g(\mathbf x_i)$ preserves some property of interest $\mathcal P$ (variance, class separation, pairwise distances, statistical independence, likelihood).

Usually paired with a reconstruction map $f:\mathbb R^d\to\mathbb R^D$, and one minimises

$$
\min_{f,g}\;\sum_{i=1}^{N}\big\|\mathbf x_i - f(g(\mathbf x_i))\big\|^2 .
$$

- If $f,g$ are constrained to be **linear** ⇒ the optimum is **PCA** (Eckart–Young theorem).
- If they are neural networks ⇒ **autoencoder**.

### 1.2 Taxonomy

```
                 DIMENSIONALITY REDUCTION
                            │
        ┌───────────────────┴──────────────────┐
   FEATURE SELECTION                    FEATURE EXTRACTION
   (keep a subset of                    (construct new features
    original features)                   as functions of old)
   filter / wrapper / embedded                    │
   (mutual info, LASSO,          ┌────────────────┴────────────────┐
    recursive elimination)     LINEAR                         NONLINEAR
                                 │                                 │
                    ┌────────────┼──────────┐         ┌────────────┼─────────────┐
                  PCA          LDA         ICA     Kernel PCA   Manifold      Autoencoder
              (variance)   (separation) (independ.)  (kernel     (Isomap, LLE,   (deep,
                                                      trick)    t-SNE, UMAP)   learned)
                  │            │           │
             unsupervised  SUPERVISED  unsupervised
```

### 1.3 The manifold hypothesis

Real high-dimensional data (images, speech, text) concentrates near a **low-dimensional manifold** $\mathcal M \subset \mathbb R^D$ with $\dim\mathcal M = d \ll D$.

*Evidence:* a $32\times32$ RGB image lives in $\mathbb R^{3072}$, but a uniformly random point in that space is white noise — never a natural image. The set of natural images is a vanishingly small, highly structured subset.

**Intrinsic dimension estimators:**
- **Correlation dimension:** $C(r) = \frac{2}{N(N-1)}\sum_{i<j}\mathbb 1[\|x_i-x_j\|<r]$, then $d \approx \dfrac{d\log C(r)}{d\log r}$.
- **MLE (Levina–Bickel):** $\hat d_k(x_i) = \left[\frac{1}{k-1}\sum_{j=1}^{k-1}\log\frac{T_k(x_i)}{T_j(x_i)}\right]^{-1}$, $T_j$ = distance to the $j$-th neighbour.
- **PCA-based:** number of eigenvalues above a threshold (only valid for locally linear manifolds).

### 1.4 Why reduce dimension

| Motivation | Explanation |
|---|---|
| **Statistical** | Sample complexity grows with $D$; reduces variance / overfitting |
| **Computational** | Many algorithms are $O(ND^2)$ or $O(D^3)$ |
| **Storage** | Compression; PCA on MNIST: 784 → 50 dims retains ~82 % variance |
| **Noise removal** | Signal concentrates in high-variance directions; noise spreads uniformly |
| **Visualisation** | $d = 2,3$ for human inspection |
| **Multicollinearity** | PCA outputs are orthogonal ⇒ regression becomes well-conditioned |
| **Preprocessing** | Whitening is a prerequisite for ICA and helps NN optimisation |

---

## 2. The Curse of Dimensionality

### 2.1 Sampling density collapse

To cover $[0,1]^D$ at resolution $\varepsilon$ per axis requires $(1/\varepsilon)^D$ points.

| $D$ | Points needed at $\varepsilon = 0.1$ |
|---|---|
| 1 | 10 |
| 2 | 100 |
| 3 | 1 000 |
| 10 | $10^{10}$ |
| 20 | $10^{20}$ |

Equivalently, with $N$ samples the *effective* edge length of a neighbourhood containing a fraction $r$ of the data is $e_D(r) = r^{1/D}$. For $D=10$, $r=0.01$ ⇒ $e = 0.63$ — a "1 % local neighbourhood" spans **63 %** of the range of every feature. **Local methods (k-NN, RBF, kernel density) break down.**

### 2.2 Volume concentrates in the shell

Fraction of a $D$-ball's volume lying in the outer shell of relative thickness $\epsilon$:

$$
\frac{V(1) - V(1-\epsilon)}{V(1)} = 1 - (1-\epsilon)^{D}
$$

| $D$ | $\epsilon = 0.01$ |
|---|---|
| 10 | 0.096 |
| 100 | 0.634 |
| 500 | 0.993 |
| 1000 | 0.99996 |

In $D=1000$, essentially **all** the volume is within 1 % of the surface. "Interior" is a low-dimensional intuition.

### 2.3 The hypersphere vanishes inside the hypercube

$$
\frac{V_{\text{sphere}}(D)}{V_{\text{cube}}(D)} = \frac{\pi^{D/2}}{2^{D}\,\Gamma\!\left(\frac D2 + 1\right)}
$$

| $D$ | Ratio |
|---|---|
| 2 | 0.7854 |
| 3 | 0.5236 |
| 5 | 0.1645 |
| 10 | $2.5\times10^{-3}$ |
| 20 | $2.46\times10^{-8}$ |

All the volume of a high-dimensional cube is in its $2^D$ **corners**.

### 2.4 Distance concentration

For i.i.d. components with finite moments,

$$
\lim_{D\to\infty}\;\mathbb E\!\left[\frac{\max_i \|x_i - q\| - \min_i\|x_i-q\|}{\min_i\|x_i-q\|}\right] \longrightarrow 0 .
$$

All points become approximately **equidistant** ⇒ "nearest neighbour" loses meaning. Formally, if $\mathrm{Var}(\|x\|^2)/\mathbb E[\|x\|^2]^2 \to 0$ then relative contrast vanishes. Mitigations: use fractional $\ell_p$ norms ($p<1$), cosine similarity, or reduce dimension first.

### 2.5 Other consequences

- **Gaussian mass moves to a shell:** for $\mathbf x\sim\mathcal N(0,\mathbf I_D)$, $\|\mathbf x\|^2\sim\chi^2_D$ with mean $D$ and SD $\sqrt{2D}$ ⇒ mass concentrates on the sphere of radius $\sqrt D$, **not** near the mode at the origin.
- **Random vectors are nearly orthogonal:** for random unit $\mathbf u,\mathbf v \in \mathbb R^D$, $\mathbb E[\mathbf u^{\mathsf T}\mathbf v] = 0$ and $\mathrm{Var} = 1/D$, so $|\cos\theta| \approx 1/\sqrt D$.
- **Johnson–Lindenstrauss (the good news):** any $N$ points can be embedded in $d = O(\varepsilon^{-2}\log N)$ dimensions with all pairwise distances preserved to within $(1\pm\varepsilon)$ — **independent of $D$**. Random projection works.

---

## 3. Principal Component Analysis

### 3.1 Setup

Data matrix $\mathbf X\in\mathbb R^{N\times D}$ (rows = samples). Centre it:
$$
\bar{\mathbf x} = \frac1N\sum_i \mathbf x_i,\qquad \mathbf X_c = \mathbf X - \mathbf 1_N\bar{\mathbf x}^{\mathsf T} = \mathbf H\mathbf X,\quad \mathbf H = \mathbf I - \tfrac1N\mathbf 1\mathbf 1^{\mathsf T}.
$$
Sample covariance:
$$
\mathbf S = \frac{1}{N-1}\mathbf X_c^{\mathsf T}\mathbf X_c \in \mathbb R^{D\times D},\qquad \mathbf S = \mathbf S^{\mathsf T} \succeq 0 .
$$

### 3.2 Derivation 1 — Maximum variance

Find the unit vector $\mathbf u_1$ maximising the variance of the projections $z_i = \mathbf u_1^{\mathsf T}\mathbf x_{c,i}$:

$$
\mathrm{Var}(z) = \frac{1}{N-1}\sum_i \left(\mathbf u_1^{\mathsf T}\mathbf x_{c,i}\right)^2 = \mathbf u_1^{\mathsf T}\mathbf S\mathbf u_1 .
$$

Without the constraint $\|\mathbf u_1\|=1$ the objective is unbounded, so form the Lagrangian:

$$
\mathcal J(\mathbf u_1,\lambda) = \mathbf u_1^{\mathsf T}\mathbf S\mathbf u_1 - \lambda\left(\mathbf u_1^{\mathsf T}\mathbf u_1 - 1\right)
$$

$$
\frac{\partial \mathcal J}{\partial \mathbf u_1} = 2\mathbf S\mathbf u_1 - 2\lambda\mathbf u_1 = \mathbf 0
\qquad\Longrightarrow\qquad \boxed{\;\mathbf S\mathbf u_1 = \lambda\mathbf u_1\;}
$$

So $\mathbf u_1$ must be an **eigenvector** of $\mathbf S$. Left-multiplying by $\mathbf u_1^{\mathsf T}$:

$$
\mathbf u_1^{\mathsf T}\mathbf S\mathbf u_1 = \lambda\,\mathbf u_1^{\mathsf T}\mathbf u_1 = \lambda .
$$

**The variance captured equals the eigenvalue**, so the maximum is attained by the eigenvector of the **largest** eigenvalue.

**Induction for subsequent components.** Maximise $\mathbf u_2^{\mathsf T}\mathbf S\mathbf u_2$ subject to $\|\mathbf u_2\|=1$ **and** $\mathbf u_2^{\mathsf T}\mathbf u_1 = 0$:
$$
\mathcal J = \mathbf u_2^{\mathsf T}\mathbf S\mathbf u_2 - \lambda(\mathbf u_2^{\mathsf T}\mathbf u_2 - 1) - \mu\,\mathbf u_2^{\mathsf T}\mathbf u_1
$$
$$
2\mathbf S\mathbf u_2 - 2\lambda\mathbf u_2 - \mu\mathbf u_1 = 0 .
$$
Pre-multiply by $\mathbf u_1^{\mathsf T}$: $2\mathbf u_1^{\mathsf T}\mathbf S\mathbf u_2 - 0 - \mu = 0$. But $\mathbf u_1^{\mathsf T}\mathbf S\mathbf u_2 = (\mathbf S\mathbf u_1)^{\mathsf T}\mathbf u_2 = \lambda_1\mathbf u_1^{\mathsf T}\mathbf u_2 = 0$, hence $\mu = 0$ and again $\mathbf S\mathbf u_2 = \lambda\mathbf u_2$ — the **second-largest** eigenvector. By induction, the $d$ principal directions are the top-$d$ eigenvectors. $\blacksquare$

### 3.3 Derivation 2 — Minimum reconstruction error

Represent each centred point using an orthonormal basis $\{\mathbf u_j\}_{j=1}^D$:
$$
\mathbf x_{c,i} = \sum_{j=1}^{D}\left(\mathbf u_j^{\mathsf T}\mathbf x_{c,i}\right)\mathbf u_j .
$$
Approximate with the first $d$ terms plus constants $b_j$ for the rest:
$$
\tilde{\mathbf x}_i = \sum_{j=1}^{d}z_{ij}\mathbf u_j + \sum_{j=d+1}^{D}b_j\mathbf u_j .
$$
The reconstruction error
$$
J = \frac1N\sum_i\|\mathbf x_{c,i}-\tilde{\mathbf x}_i\|^2 .
$$
Setting $\partial J/\partial z_{ij}=0$ gives $z_{ij} = \mathbf u_j^{\mathsf T}\mathbf x_{c,i}$; $\partial J/\partial b_j = 0$ gives $b_j = \mathbf u_j^{\mathsf T}\bar{\mathbf x}_c = 0$. Substituting back:

$$
J = \frac1N\sum_i\sum_{j=d+1}^{D}\left(\mathbf u_j^{\mathsf T}\mathbf x_{c,i}\right)^2 = \sum_{j=d+1}^{D}\mathbf u_j^{\mathsf T}\mathbf S\mathbf u_j \;\;\propto\;\; \sum_{j=d+1}^{D}\lambda_j .
$$

$$
\boxed{\;J_{\min} = \sum_{j=d+1}^{D}\lambda_j\;}
$$

Minimising the **discarded** eigenvalues = maximising the **retained** ones. The two derivations are equivalent because $\sum_j \lambda_j = \mathrm{tr}(\mathbf S)$ is fixed.

### 3.4 The PCA algorithm

> 1. Centre: $\mathbf X_c = \mathbf X - \mathbf 1\bar{\mathbf x}^{\mathsf T}$. (Optionally standardise each feature: use the correlation matrix instead of covariance if features have different units.)
> 2. Covariance: $\mathbf S = \frac{1}{N-1}\mathbf X_c^{\mathsf T}\mathbf X_c$.
> 3. Eigendecompose: $\mathbf S = \mathbf U\boldsymbol\Lambda\mathbf U^{\mathsf T}$, sort $\lambda_1\ge\lambda_2\ge\cdots\ge\lambda_D \ge 0$.
> 4. Select $d$: cumulative explained variance $\ge$ threshold.
> 5. Project: $\mathbf Z = \mathbf X_c\mathbf U_d$, $\;\mathbf U_d = [\mathbf u_1\cdots\mathbf u_d]$.
> 6. Reconstruct: $\hat{\mathbf X} = \mathbf Z\mathbf U_d^{\mathsf T} + \mathbf 1\bar{\mathbf x}^{\mathsf T}$.

**Complexity:** $O(ND^2 + D^3)$ via covariance eigendecomposition; $O(ND\min(N,D))$ via SVD; $O(NDd)$ via randomised/truncated SVD or power iteration.

### 3.5 Choosing $d$

| Criterion | Rule |
|---|---|
| **Cumulative variance** | smallest $d$ with $\sum_{j\le d}\lambda_j / \sum_j\lambda_j \ge 0.90$–$0.99$ |
| **Scree plot / elbow** | visual knee in $\lambda_j$ vs $j$ |
| **Kaiser criterion** | keep $\lambda_j > 1$ (correlation matrix; average eigenvalue is 1) |
| **Broken-stick** | keep $\lambda_j$ exceeding the expected value under a random split |
| **Cross-validation** | reconstruct held-out entries, pick $d$ minimising error |
| **Minka's MLE / BIC** | Bayesian model selection under probabilistic PCA |

### 3.6 Properties and pitfalls

**Properties.**
- Components are **uncorrelated**: $\mathrm{Cov}(\mathbf Z) = \boldsymbol\Lambda_d$ (diagonal).
- Total variance is preserved: $\sum_j \lambda_j = \mathrm{tr}(\mathbf S) = \sum_j \mathrm{Var}(x_j)$.
- Optimal linear rank-$d$ approximation in Frobenius **and** spectral norm (Eckart–Young–Mirsky).
- **Whitening/sphering:** $\mathbf Z_w = \mathbf X_c \mathbf U\boldsymbol\Lambda^{-1/2}$ gives $\mathrm{Cov}(\mathbf Z_w)=\mathbf I$. (Whitening is defined only up to an arbitrary rotation — this indeterminacy is exactly what ICA resolves.)

**Pitfalls.**
- **Scale-sensitive:** a feature measured in millimetres dominates one in metres. Standardise when units differ.
- **Variance ≠ information:** the discriminative direction may have small variance (see §5 and the classic figure below).
- Assumes **linear** structure; fails on a Swiss roll or a circle.
- Assumes the interesting structure is captured by **second-order statistics only** (true only if data are Gaussian).
- Components are **not interpretable** in general (dense linear combinations). Sparse PCA fixes this.
- **Outlier-sensitive** (squared error). Use robust PCA: $\min \|\mathbf L\|_* + \lambda\|\mathbf S\|_1$ s.t. $\mathbf X = \mathbf L + \mathbf S$.

```
   PCA picks max-variance direction        LDA picks max-separation direction
        │        ● ●                              │      ● ●
        │      ● ● ● ○ ○                          │    ● ● ●  ○ ○
   PC1 ➚      ● ●   ○ ○ ○                    LD1 ─┼──►● ●    ○ ○ ○
        │    ● ●     ○ ○                          │  ● ●      ○ ○
        │                                         │
   Projecting on PC1 MIXES the classes      Projecting on LD1 SEPARATES them
```

---

## 4. PCA via SVD, Probabilistic PCA, and Neural PCA

### 4.1 SVD formulation (numerically preferred)

$$
\mathbf X_c = \mathbf U_{\!s}\,\boldsymbol\Sigma\,\mathbf V^{\mathsf T},\qquad \mathbf U_{\!s}\in\mathbb R^{N\times r},\;\boldsymbol\Sigma = \mathrm{diag}(\sigma_1\ge\cdots\ge\sigma_r),\;\mathbf V\in\mathbb R^{D\times r}
$$

Then
$$
\mathbf S = \frac{\mathbf X_c^{\mathsf T}\mathbf X_c}{N-1} = \frac{\mathbf V\boldsymbol\Sigma\mathbf U_{\!s}^{\mathsf T}\mathbf U_{\!s}\boldsymbol\Sigma\mathbf V^{\mathsf T}}{N-1} = \mathbf V\frac{\boldsymbol\Sigma^2}{N-1}\mathbf V^{\mathsf T}
$$

$$
\boxed{\;\text{principal directions } = \mathbf V,\qquad \lambda_j = \frac{\sigma_j^2}{N-1},\qquad \text{scores } \mathbf Z = \mathbf X_c\mathbf V = \mathbf U_{\!s}\boldsymbol\Sigma \;}
$$

**Why SVD is better:** forming $\mathbf X_c^{\mathsf T}\mathbf X_c$ squares the condition number ($\kappa(\mathbf S) = \kappa(\mathbf X_c)^2$), losing half the significant digits. SVD works on $\mathbf X_c$ directly.

**The $N \ll D$ trick (e.g. eigenfaces, $D=10^4$, $N=100$).** Eigendecompose the $N\times N$ Gram matrix $\mathbf G = \mathbf X_c\mathbf X_c^{\mathsf T}$ instead: if $\mathbf G\mathbf w = \lambda\mathbf w$, then left-multiplying by $\mathbf X_c^{\mathsf T}$ gives $\mathbf X_c^{\mathsf T}\mathbf X_c(\mathbf X_c^{\mathsf T}\mathbf w) = \lambda(\mathbf X_c^{\mathsf T}\mathbf w)$ — so $\mathbf u = \mathbf X_c^{\mathsf T}\mathbf w/\|\cdot\|$ is the corresponding eigenvector of the $D\times D$ covariance. Cost drops from $O(D^3)$ to $O(N^3)$. **This dual formulation is exactly what makes Kernel PCA possible (§6).**

### 4.2 Probabilistic PCA (Tipping & Bishop, 1999)

Latent-variable generative model:
$$
\mathbf z \sim \mathcal N(\mathbf 0,\mathbf I_d),\qquad \mathbf x \mid \mathbf z \sim \mathcal N(\mathbf W\mathbf z + \boldsymbol\mu,\;\sigma^2\mathbf I_D)
$$
Marginal:
$$
\mathbf x \sim \mathcal N(\boldsymbol\mu,\;\mathbf C),\qquad \mathbf C = \mathbf W\mathbf W^{\mathsf T} + \sigma^2\mathbf I .
$$

**Maximum-likelihood solution:**
$$
\boxed{\;\mathbf W_{\text{ML}} = \mathbf U_d\left(\boldsymbol\Lambda_d - \sigma^2\mathbf I\right)^{1/2}\mathbf R,\qquad
\sigma^2_{\text{ML}} = \frac{1}{D-d}\sum_{j=d+1}^{D}\lambda_j\;}
$$
($\mathbf R$ = arbitrary orthogonal rotation — the solution is identifiable only up to rotation.)

As $\sigma^2\to0$, PPCA $\to$ classical PCA. **Benefits:** principled model selection (BIC/Bayesian), handles missing data via EM, gives $p(\mathbf x)$ for novelty detection, extends to mixtures of PPCA (a nonlinear model). **Factor analysis** generalises it by allowing $\boldsymbol\Psi = \mathrm{diag}(\psi_1..\psi_D)$ instead of $\sigma^2\mathbf I$ — i.e. per-feature noise.

### 4.3 Neural-network implementations of PCA

**(a) Oja's rule (single neuron, first PC).** From Unit I:
$$
\Delta \mathbf w = \eta\, y(\mathbf x - y\mathbf w),\qquad y = \mathbf w^{\mathsf T}\mathbf x
$$
At equilibrium $\mathbf C\mathbf w = \lambda\mathbf w$, $\|\mathbf w\| = 1$ ⇒ $\mathbf w \to \pm\mathbf u_1$.

**(b) Sanger's rule / Generalized Hebbian Algorithm (all $d$ PCs).**
$$
\boxed{\;\Delta w_{ij} = \eta\,y_i\!\left(x_j - \sum_{k\le i} w_{kj}y_k\right)\;}
$$
The sum up to $k\le i$ performs online **Gram–Schmidt deflation**: neuron $i$ sees the input with the first $i-1$ components already subtracted, so $\mathbf w_i \to \mathbf u_i$ in order.

**(c) Linear autoencoder.** A $D$–$d$–$D$ network with linear activations and squared loss has a global minimum spanning exactly the top-$d$ principal subspace (Baldi & Hornik 1989). It recovers the **subspace**, not the individual ordered components, unless orthogonality/deflation is imposed.

**(d) Nonlinear autoencoder.** Replacing activations with nonlinearities gives a nonlinear generalisation; a deep autoencoder with a bottleneck is the modern successor to PCA. Note: an autoencoder with nonlinear *encoder/decoder* but only one hidden layer with linear output still reduces to PCA — **the nonlinearity must be in the right place.**

---

## 5. Linear Discriminant Analysis

### 5.1 Objective

PCA is unsupervised and can destroy class information. **LDA (Fisher, 1936)** finds the projection maximising *between-class* separation relative to *within-class* scatter.

### 5.2 Two-class Fisher criterion — derivation

Class means and scatters:
$$
\boldsymbol\mu_k = \frac{1}{N_k}\sum_{\mathbf x\in\mathcal C_k}\mathbf x,\qquad
\mathbf S_k = \sum_{\mathbf x\in\mathcal C_k}(\mathbf x-\boldsymbol\mu_k)(\mathbf x - \boldsymbol\mu_k)^{\mathsf T}
$$
$$
\mathbf S_W = \mathbf S_1 + \mathbf S_2 \quad(\text{within-class}),\qquad
\mathbf S_B = (\boldsymbol\mu_1-\boldsymbol\mu_2)(\boldsymbol\mu_1-\boldsymbol\mu_2)^{\mathsf T}\quad(\text{between-class})
$$

Project $y = \mathbf w^{\mathsf T}\mathbf x$. Then $\tilde\mu_k = \mathbf w^{\mathsf T}\boldsymbol\mu_k$ and $\tilde s_k^2 = \mathbf w^{\mathsf T}\mathbf S_k\mathbf w$. Fisher's criterion:

$$
\boxed{\;J(\mathbf w) = \frac{(\tilde\mu_1-\tilde\mu_2)^2}{\tilde s_1^2 + \tilde s_2^2}
= \frac{\mathbf w^{\mathsf T}\mathbf S_B\mathbf w}{\mathbf w^{\mathsf T}\mathbf S_W\mathbf w}\;}
$$

This is a **generalised Rayleigh quotient**. Differentiate using the quotient rule:

$$
\frac{\partial J}{\partial \mathbf w}
= \frac{2\mathbf S_B\mathbf w\left(\mathbf w^{\mathsf T}\mathbf S_W\mathbf w\right) - 2\mathbf S_W\mathbf w\left(\mathbf w^{\mathsf T}\mathbf S_B\mathbf w\right)}{\left(\mathbf w^{\mathsf T}\mathbf S_W\mathbf w\right)^2} = \mathbf 0
$$

$$
\Longrightarrow\quad \mathbf S_B\mathbf w\left(\mathbf w^{\mathsf T}\mathbf S_W\mathbf w\right) = \mathbf S_W\mathbf w\left(\mathbf w^{\mathsf T}\mathbf S_B\mathbf w\right)
\quad\Longrightarrow\quad
\boxed{\;\mathbf S_B\mathbf w = J(\mathbf w)\,\mathbf S_W\mathbf w\;}
$$

a **generalised eigenvalue problem** $\mathbf S_B\mathbf w = \lambda\mathbf S_W\mathbf w$.

**Closed form for two classes.** Note $\mathbf S_B\mathbf w = (\boldsymbol\mu_1-\boldsymbol\mu_2)\underbrace{(\boldsymbol\mu_1-\boldsymbol\mu_2)^{\mathsf T}\mathbf w}_{\text{scalar}}$, so $\mathbf S_B\mathbf w \parallel (\boldsymbol\mu_1-\boldsymbol\mu_2)$. Since direction (not magnitude) matters:

$$
\boxed{\;\mathbf w^\star \;\propto\; \mathbf S_W^{-1}(\boldsymbol\mu_1 - \boldsymbol\mu_2)\;}
$$

and the optimum value is
$$
J(\mathbf w^\star) = (\boldsymbol\mu_1-\boldsymbol\mu_2)^{\mathsf T}\mathbf S_W^{-1}(\boldsymbol\mu_1-\boldsymbol\mu_2)
$$
— the squared **Mahalanobis distance** between the class means.

**Special case:** if $\mathbf S_W \propto \mathbf I$ (isotropic within-class scatter), $\mathbf w^\star \propto \boldsymbol\mu_1-\boldsymbol\mu_2$ — the naive mean-difference direction. LDA's contribution is the $\mathbf S_W^{-1}$ *decorrelation/whitening* factor.

### 5.3 Multi-class LDA

For $C$ classes with total mean $\boldsymbol\mu$:

$$
\mathbf S_W = \sum_{k=1}^{C}\sum_{\mathbf x\in\mathcal C_k}(\mathbf x-\boldsymbol\mu_k)(\mathbf x-\boldsymbol\mu_k)^{\mathsf T},\qquad
\mathbf S_B = \sum_{k=1}^{C}N_k(\boldsymbol\mu_k-\boldsymbol\mu)(\boldsymbol\mu_k-\boldsymbol\mu)^{\mathsf T}
$$

with the decomposition $\mathbf S_T = \mathbf S_W + \mathbf S_B$ (total scatter).

Maximise $J(\mathbf W) = \dfrac{|\mathbf W^{\mathsf T}\mathbf S_B\mathbf W|}{|\mathbf W^{\mathsf T}\mathbf S_W\mathbf W|}$ (determinant / trace ratio). Solution: the top eigenvectors of

$$
\boxed{\;\mathbf S_W^{-1}\mathbf S_B\,\mathbf w_j = \lambda_j \mathbf w_j\;}
$$

**Rank limitation.** $\mathbf S_B$ is a sum of $C$ outer products of vectors that satisfy one linear constraint ($\sum_k N_k(\boldsymbol\mu_k-\boldsymbol\mu)=\mathbf 0$), hence

$$
\mathrm{rank}(\mathbf S_B) \le C-1 \quad\Longrightarrow\quad \boxed{\;d_{\max} = C-1\;}
$$

For a 10-class problem LDA can produce at most 9 discriminant directions, no matter how large $D$ is. **PCA has no such limit** — a key exam distinction.

### 5.4 Connection to Bayes-optimal classification

Assume $p(\mathbf x|\mathcal C_k) = \mathcal N(\boldsymbol\mu_k,\boldsymbol\Sigma)$ with a **shared** covariance. Then

$$
\log\frac{P(\mathcal C_1|\mathbf x)}{P(\mathcal C_2|\mathbf x)}
= \underbrace{(\boldsymbol\mu_1-\boldsymbol\mu_2)^{\mathsf T}\boldsymbol\Sigma^{-1}}_{\mathbf w^{\mathsf T}}\mathbf x
\;-\;\tfrac12\left(\boldsymbol\mu_1^{\mathsf T}\boldsymbol\Sigma^{-1}\boldsymbol\mu_1 - \boldsymbol\mu_2^{\mathsf T}\boldsymbol\Sigma^{-1}\boldsymbol\mu_2\right) + \log\frac{P(\mathcal C_1)}{P(\mathcal C_2)}
$$

The decision boundary is **linear** and its normal is exactly $\boldsymbol\Sigma^{-1}(\boldsymbol\mu_1-\boldsymbol\mu_2)$ — identical to the Fisher direction with $\mathbf S_W \propto \hat{\boldsymbol\Sigma}$. **So under the Gaussian equal-covariance assumption, LDA is Bayes-optimal.** Dropping the shared-covariance assumption gives **QDA**, whose boundary is quadratic.

Optimal threshold on the projection (equal priors, equal variances): $\;y_0 = \tfrac12\mathbf w^{\mathsf T}(\boldsymbol\mu_1 + \boldsymbol\mu_2)$.

### 5.5 Practical issues

| Problem | Cause | Fix |
|---|---|---|
| $\mathbf S_W$ singular | $N < D$ (small-sample-size problem, e.g. face recognition) | **PCA+LDA (Fisherfaces):** first reduce to $N-C$ dims with PCA; or regularised LDA $\mathbf S_W + \gamma\mathbf I$; or Null-space / direct LDA |
| Non-Gaussian classes | assumption violated | Kernel LDA (GDA), or use a neural net |
| Multimodal class | one mean is not representative | Subclass discriminant analysis, mixture discriminant analysis |
| $d$ capped at $C-1$ | rank of $\mathbf S_B$ | use PCA or a nonlinear method for more dims |
| Unequal covariances | LDA assumption violated | QDA |

### 5.6 PCA vs LDA

| | PCA | LDA |
|---|---|---|
| Supervision | Unsupervised | **Supervised** |
| Criterion | Max variance / min reconstruction error | Max between/within scatter ratio |
| Matrix decomposed | $\mathbf S$ (covariance) | $\mathbf S_W^{-1}\mathbf S_B$ |
| Max components | $\min(N-1, D)$ | $C-1$ |
| Component orthogonality | Orthogonal ($\mathbf S$ symmetric) | **Not** orthogonal in general ($\mathbf S_W^{-1}\mathbf S_B$ is not symmetric); orthogonal in the $\mathbf S_W$ metric |
| Best when | Structure = variance | Classes are Gaussian, equal covariance |
| Small-sample | Robust | Fails ($\mathbf S_W$ singular) |
| Typical use | compression, denoising, visualisation | classification preprocessing |

---

## 6. Kernel PCA

### 6.1 Motivation

PCA can only find **linear** subspaces. If the data lies on a circle, a parabola, or two concentric rings, no linear projection separates the structure. **Idea:** map into a high-dimensional feature space $\boldsymbol\Phi:\mathbb R^D\to\mathcal F$, do linear PCA *there*, and never compute $\boldsymbol\Phi$ explicitly.

```
   Input space (not linearly separable)     Feature space Φ(x)   (separable)
        ○ ○ ○                                       ●●●
      ○  ● ●  ○           Φ                    ────────────────
      ○  ● ●  ○      ──────────►                    ○○○○○
        ○ ○ ○                                     (a linear PC
   concentric rings                              now captures it)
```

### 6.2 Derivation

Assume for the moment that the mapped data is centred: $\sum_i \boldsymbol\Phi(\mathbf x_i)=\mathbf 0$. The covariance in $\mathcal F$:

$$
\mathbf C = \frac1N\sum_{i=1}^{N}\boldsymbol\Phi(\mathbf x_i)\boldsymbol\Phi(\mathbf x_i)^{\mathsf T}
$$

We need $\mathbf C\mathbf v = \lambda\mathbf v$. Substituting:

$$
\lambda\mathbf v = \frac1N\sum_i \boldsymbol\Phi(\mathbf x_i)\underbrace{\left(\boldsymbol\Phi(\mathbf x_i)^{\mathsf T}\mathbf v\right)}_{\text{scalar}}
$$

**Key observation:** $\mathbf v$ lies in the span of the mapped data (for $\lambda\ne0$). Therefore

$$
\mathbf v = \sum_{j=1}^{N}\alpha_j\,\boldsymbol\Phi(\mathbf x_j) .
$$

Substituting and taking the inner product with $\boldsymbol\Phi(\mathbf x_k)$ for every $k$:

$$
\lambda\sum_j\alpha_j\,\boldsymbol\Phi(\mathbf x_k)^{\mathsf T}\boldsymbol\Phi(\mathbf x_j)
= \frac1N\sum_i \boldsymbol\Phi(\mathbf x_k)^{\mathsf T}\boldsymbol\Phi(\mathbf x_i)\sum_j\alpha_j\,\boldsymbol\Phi(\mathbf x_i)^{\mathsf T}\boldsymbol\Phi(\mathbf x_j)
$$

Define the **kernel (Gram) matrix** $K_{ij} = \boldsymbol\Phi(\mathbf x_i)^{\mathsf T}\boldsymbol\Phi(\mathbf x_j) = k(\mathbf x_i,\mathbf x_j)$. Then

$$
\lambda N\,\mathbf K\boldsymbol\alpha = \mathbf K^2\boldsymbol\alpha
\qquad\Longrightarrow\qquad
\boxed{\;\mathbf K\boldsymbol\alpha = N\lambda\,\boldsymbol\alpha = \tilde\lambda\,\boldsymbol\alpha\;}
$$

(dropping the extra $\mathbf K$ is valid because solutions in $\ker\mathbf K$ contribute nothing to $\mathbf v$).

**Normalisation.** We need $\|\mathbf v\|=1$:
$$
1 = \mathbf v^{\mathsf T}\mathbf v = \sum_{i,j}\alpha_i\alpha_j\boldsymbol\Phi(\mathbf x_i)^{\mathsf T}\boldsymbol\Phi(\mathbf x_j) = \boldsymbol\alpha^{\mathsf T}\mathbf K\boldsymbol\alpha = \tilde\lambda\,\|\boldsymbol\alpha\|^2
$$
$$
\boxed{\;\|\boldsymbol\alpha\|^2 = \frac{1}{\tilde\lambda}\;}
$$

**Projection of a new point $\mathbf x$ onto component $p$:**

$$
\boxed{\;y_p(\mathbf x) = \boldsymbol\Phi(\mathbf x)^{\mathsf T}\mathbf v_p = \sum_{j=1}^{N}\alpha^{(p)}_j\,k(\mathbf x, \mathbf x_j)\;}
$$

$\boldsymbol\Phi$ never appears — only the kernel. **This is the kernel trick.**

### 6.3 Centering in feature space

We assumed $\sum_i\boldsymbol\Phi(\mathbf x_i)=0$, which is generally false. Define $\tilde{\boldsymbol\Phi}(\mathbf x_i) = \boldsymbol\Phi(\mathbf x_i) - \frac1N\sum_k\boldsymbol\Phi(\mathbf x_k)$. Then

$$
\tilde K_{ij} = \tilde{\boldsymbol\Phi}(\mathbf x_i)^{\mathsf T}\tilde{\boldsymbol\Phi}(\mathbf x_j)
= K_{ij} - \frac1N\sum_k K_{kj} - \frac1N\sum_k K_{ik} + \frac{1}{N^2}\sum_{k,l}K_{kl}
$$

In matrix form, with $\mathbf 1_N = \frac1N\mathbf 1\mathbf 1^{\mathsf T}$ (all entries $1/N$):

$$
\boxed{\;\tilde{\mathbf K} = \mathbf K - \mathbf 1_N\mathbf K - \mathbf K\mathbf 1_N + \mathbf 1_N\mathbf K\mathbf 1_N = \mathbf H\mathbf K\mathbf H,\quad \mathbf H = \mathbf I - \tfrac1N\mathbf 1\mathbf 1^{\mathsf T}\;}
$$

For a **test** point, the projection must use the same centring:
$$
\tilde k(\mathbf x, \mathbf x_j) = k(\mathbf x,\mathbf x_j) - \tfrac1N\sum_l k(\mathbf x_l,\mathbf x_j) - \tfrac1N\sum_l k(\mathbf x,\mathbf x_l) + \tfrac{1}{N^2}\sum_{l,m}k(\mathbf x_l,\mathbf x_m).
$$

### 6.4 Algorithm

> 1. Choose a kernel $k$ and its hyperparameters.
> 2. Build $\mathbf K \in\mathbb R^{N\times N}$, $K_{ij} = k(\mathbf x_i,\mathbf x_j)$.
> 3. Centre: $\tilde{\mathbf K} = \mathbf H\mathbf K\mathbf H$.
> 4. Eigendecompose: $\tilde{\mathbf K}\boldsymbol\alpha_p = \tilde\lambda_p\boldsymbol\alpha_p$, sort descending.
> 5. Normalise: rescale so $\|\boldsymbol\alpha_p\|^2 = 1/\tilde\lambda_p$.
> 6. Project: $y_p(\mathbf x) = \sum_j\alpha^{(p)}_j\tilde k(\mathbf x,\mathbf x_j)$.

### 6.5 Common kernels (Mercer conditions)

| Kernel | $k(\mathbf x,\mathbf y)$ | Feature space |
|---|---|---|
| Linear | $\mathbf x^{\mathsf T}\mathbf y$ | $\mathbb R^D$ — recovers ordinary PCA |
| Polynomial | $(\mathbf x^{\mathsf T}\mathbf y + c)^p$ | all monomials up to degree $p$; $\dim = \binom{D+p}{p}$ |
| **Gaussian (RBF)** | $\exp\!\left(-\dfrac{\Vert \mathbf x-\mathbf y\Vert ^2}{2\sigma^2}\right)$ | **infinite**-dimensional |
| Laplacian | $\exp(-\Vert \mathbf x-\mathbf y\Vert /\sigma)$ | infinite |
| Sigmoid | $\tanh(\kappa\mathbf x^{\mathsf T}\mathbf y + c)$ | not always PSD (conditionally Mercer) |

**Mercer's theorem:** $k$ is a valid kernel iff the Gram matrix $\mathbf K$ is symmetric positive semi-definite for every finite sample. **Closure properties:** $k_1+k_2$, $ck_1$ ($c>0$), $k_1k_2$, $f(\mathbf x)k_1f(\mathbf y)$, $\exp(k_1)$ are all valid kernels — use these to construct new ones.

**Why the Gaussian kernel is infinite-dimensional (sketch):** expand $e^{\mathbf x^{\mathsf T}\mathbf y/\sigma^2} = \sum_{n=0}^\infty \frac{(\mathbf x^{\mathsf T}\mathbf y)^n}{n!\sigma^{2n}}$ — a sum of polynomial kernels of *every* degree.

### 6.6 Properties and limitations

**Properties.**
- Number of usable components can be up to $N$ (not $D$) — kernel PCA can *increase* dimension.
- Recovers nonlinear structure a linear PCA cannot.
- Linear kernel reduces exactly to PCA (useful sanity check).

**Limitations.**
- $\mathbf K$ is $N\times N$: memory $O(N^2)$, eigendecomposition $O(N^3)$ ⇒ infeasible for large $N$. Fixes: Nyström approximation, random Fourier features, incremental kernel PCA.
- **Pre-image problem:** given $\mathbf v \in \mathcal F$, there may be *no* $\mathbf x$ with $\boldsymbol\Phi(\mathbf x)=\mathbf v$ ⇒ **reconstruction is ill-posed** (must be solved by a separate optimisation, e.g. fixed-point iteration for the Gaussian kernel).
- Hyperparameter $\sigma$ (or $p$) must be tuned and strongly affects the result.
- No explicit loadings ⇒ poor interpretability.

---

## 7. Independent Component Analysis

### 7.1 The problem: blind source separation

**Cocktail party problem.** $n$ speakers, $n$ microphones; each mic records an instantaneous linear mixture:

$$
\boxed{\;\mathbf x = \mathbf A\mathbf s\;}
$$

$\mathbf s\in\mathbb R^n$ = unknown independent sources, $\mathbf A\in\mathbb R^{n\times n}$ = unknown mixing matrix, $\mathbf x$ = observations. Find $\mathbf W \approx \mathbf A^{-1}$ so that

$$
\hat{\mathbf s} = \mathbf W\mathbf x
$$

recovers the sources. Everything is unknown — hence *blind*.

### 7.2 Assumptions and ambiguities

**Assumptions.**
1. Sources are **statistically independent**: $p(s_1,\dots,s_n)=\prod_i p_i(s_i)$.
2. **At most one** source is Gaussian.
3. Mixing is linear, instantaneous, and $\mathbf A$ is invertible (square case).

**Why not more than one Gaussian?** If $\mathbf s\sim\mathcal N(\mathbf 0,\mathbf I)$ and $\mathbf Q$ is any orthogonal matrix, $\mathbf s' = \mathbf Q\mathbf s$ has the *same* distribution $\mathcal N(\mathbf 0,\mathbf I)$. Then $\mathbf x = \mathbf A\mathbf s = (\mathbf A\mathbf Q^{\mathsf T})(\mathbf Q\mathbf s)$ is an equally valid factorisation — the model is **unidentifiable**. Gaussians are fully described by second-order statistics, and second-order statistics cannot pin down a rotation. **ICA fundamentally needs higher-order statistics.**

**Inherent ambiguities (cannot be resolved, and don't matter in practice):**
1. **Scale/sign:** $\mathbf A\mathbf s = (\mathbf A\mathbf D^{-1})(\mathbf D\mathbf s)$ for any diagonal $\mathbf D$. Fixed by convention $\mathbb E[s_i^2]=1$.
2. **Permutation:** $\mathbf A\mathbf s = (\mathbf A\mathbf P^{-1})(\mathbf P\mathbf s)$ for a permutation $\mathbf P$. The order of recovered sources is arbitrary.

### 7.3 Preprocessing

**(1) Centering:** $\mathbf x \leftarrow \mathbf x - \mathbb E[\mathbf x]$.

**(2) Whitening (via PCA):** find $\mathbf V$ with $\tilde{\mathbf x} = \mathbf V\mathbf x$, $\mathbb E[\tilde{\mathbf x}\tilde{\mathbf x}^{\mathsf T}]=\mathbf I$:
$$
\mathbf V = \boldsymbol\Lambda^{-1/2}\mathbf U^{\mathsf T},\qquad \mathbb E[\mathbf x\mathbf x^{\mathsf T}] = \mathbf U\boldsymbol\Lambda\mathbf U^{\mathsf T}.
$$

**Why whitening halves the problem.** After whitening, $\tilde{\mathbf x} = \mathbf V\mathbf A\mathbf s = \tilde{\mathbf A}\mathbf s$ and
$$
\mathbf I = \mathbb E[\tilde{\mathbf x}\tilde{\mathbf x}^{\mathsf T}] = \tilde{\mathbf A}\,\mathbb E[\mathbf s\mathbf s^{\mathsf T}]\,\tilde{\mathbf A}^{\mathsf T} = \tilde{\mathbf A}\tilde{\mathbf A}^{\mathsf T}
$$
so $\tilde{\mathbf A}$ is **orthogonal**. Parameters to estimate drop from $n^2$ to $n(n-1)/2$ (the dimension of the orthogonal group). **ICA = PCA (whitening) + finding the right rotation.**

### 7.4 Contrast functions — measuring non-Gaussianity

**Central Limit Theorem intuition.** Any mixture $y = \mathbf w^{\mathsf T}\mathbf x = (\mathbf w^{\mathsf T}\mathbf A)\mathbf s = \mathbf q^{\mathsf T}\mathbf s$ is a *sum* of independent variables, hence **more Gaussian** than any single source. Therefore: **maximising non-Gaussianity of $\mathbf w^{\mathsf T}\mathbf x$ recovers a single source** (i.e. drives $\mathbf q$ toward a unit vector).

**(a) Kurtosis (4th cumulant).**
$$
\boxed{\;\mathrm{kurt}(y) = \mathbb E[y^4] - 3\left(\mathbb E[y^2]\right)^2 \;\;\overset{\text{if }\mathrm{Var}(y)=1}{=}\;\; \mathbb E[y^4]-3\;}
$$

| Sign | Name | Shape | Examples |
|---|---|---|---|
| $>0$ | super-Gaussian / leptokurtic | peaked, heavy tails, sparse | speech, Laplace, natural-image wavelets |
| $=0$ | Gaussian | — | noise |
| $<0$ | sub-Gaussian / platykurtic | flat-topped, light tails | uniform, binary signals |

Kurtosis is **linear in cumulants**: $\mathrm{kurt}(x_1+x_2)=\mathrm{kurt}(x_1)+\mathrm{kurt}(x_2)$ and $\mathrm{kurt}(\alpha x)=\alpha^4\mathrm{kurt}(x)$. *Drawback:* very sensitive to outliers (4th power).

**(b) Negentropy (theoretically optimal).**
$$
\boxed{\;J(y) = H(y_{\text{gauss}}) - H(y) \;\ge 0,\;\; = 0 \text{ iff } y \text{ is Gaussian}\;}
$$
where $y_{\text{gauss}}$ is Gaussian with the same variance (the maximum-entropy distribution for fixed variance). Negentropy is invariant to invertible linear transforms — the *right* measure, but requires the unknown pdf.

**Practical approximations (Hyvärinen):**
$$
J(y) \approx \frac{1}{12}\mathbb E[y^3]^2 + \frac{1}{48}\mathrm{kurt}(y)^2
\qquad\text{or, more robustly,}\qquad
J(y) \propto \Big(\mathbb E[G(y)] - \mathbb E[G(\nu)]\Big)^2,\;\; \nu\sim\mathcal N(0,1)
$$

with non-quadratic $G$:
$$
G_1(u) = \frac1{a}\log\cosh(a u),\quad G_1'(u)=g(u)=\tanh(au),\; a\in[1,2] \quad (\text{general purpose})
$$
$$
G_2(u) = -e^{-u^2/2},\quad g(u) = u\,e^{-u^2/2} \quad (\text{robust, highly super-Gaussian})
$$
$$
G_3(u) = u^4/4,\quad g(u)=u^3 \quad (\text{kurtosis; sub-Gaussian, outlier-sensitive})
$$

**(c) Mutual information (equivalent formulation).**
$$
I(y_1,\dots,y_n) = \sum_i H(y_i) - H(\mathbf y) = \sum_i J(y_i) - J(\mathbf y) + \text{const}
$$
For a fixed whitening (so $\mathbf W$ orthogonal, $J(\mathbf y)$ constant), **minimising mutual information $\equiv$ maximising the sum of negentropies.** All three formulations coincide.

**(d) Maximum likelihood / Infomax.** With $p_i$ the assumed source densities:
$$
\ell(\mathbf W) = \sum_{t}\left[\sum_i \log p_i(\mathbf w_i^{\mathsf T}\mathbf x_t)\right] + T\log|\det \mathbf W|
$$
Natural-gradient update (Amari):
$$
\Delta\mathbf W \propto \left[\mathbf I - \boldsymbol\varphi(\mathbf y)\mathbf y^{\mathsf T}\right]\mathbf W,\qquad \varphi_i(y) = -\frac{d\log p_i(y)}{dy}
$$
Bell–Sejnowski **Infomax** with logistic nonlinearity is the special case $\varphi(y)=\tanh(y)$, and is *equivalent to ML with a super-Gaussian source prior*.

### 7.5 FastICA — derivation and algorithm

Maximise $J_G(\mathbf w) = \left(\mathbb E[G(\mathbf w^{\mathsf T}\tilde{\mathbf x})] - \mathbb E[G(\nu)]\right)^2$ subject to $\|\mathbf w\|=1$ (valid because the data are whitened).

**KKT condition.** The optima satisfy
$$
\mathbb E\!\left[\tilde{\mathbf x}\,g(\mathbf w^{\mathsf T}\tilde{\mathbf x})\right] - \beta\mathbf w = \mathbf 0 \;\triangleq\; F(\mathbf w)
$$
Apply **Newton's method**: $\mathbf w^+ = \mathbf w - \mathbf J_F^{-1}F(\mathbf w)$ with
$$
\mathbf J_F = \mathbb E\!\left[\tilde{\mathbf x}\tilde{\mathbf x}^{\mathsf T}g'(\mathbf w^{\mathsf T}\tilde{\mathbf x})\right] - \beta\mathbf I
\;\approx\; \mathbb E[\tilde{\mathbf x}\tilde{\mathbf x}^{\mathsf T}]\,\mathbb E[g'(\mathbf w^{\mathsf T}\tilde{\mathbf x})] - \beta\mathbf I
= \left(\mathbb E[g'] - \beta\right)\mathbf I
$$
(using whiteness $\mathbb E[\tilde{\mathbf x}\tilde{\mathbf x}^{\mathsf T}]=\mathbf I$ and an independence approximation — this is the trick that makes the Jacobian **diagonal and trivially invertible**). Substituting and multiplying through by $(\beta - \mathbb E[g'])$ (a scale factor removed by the subsequent normalisation):

$$
\boxed{\;\mathbf w^{+} = \mathbb E\!\left[\tilde{\mathbf x}\,g(\mathbf w^{\mathsf T}\tilde{\mathbf x})\right] - \mathbb E\!\left[g'(\mathbf w^{\mathsf T}\tilde{\mathbf x})\right]\mathbf w,
\qquad \mathbf w \leftarrow \frac{\mathbf w^{+}}{\|\mathbf w^{+}\|}\;}
$$

> **Algorithm: FastICA (deflationary, $n$ components)**
> 1. Centre and whiten the data → $\tilde{\mathbf X}$.
> 2. For $p = 1,\dots,n$:
> &nbsp;&nbsp;&nbsp;**a.** Initialise $\mathbf w_p$ randomly, normalise.
> &nbsp;&nbsp;&nbsp;**b.** Repeat until convergence ($|\mathbf w_p^{+\mathsf T}\mathbf w_p| \to 1$):
> &nbsp;&nbsp;&nbsp;&nbsp;&bull; $\mathbf w_p \leftarrow \mathbb E[\tilde{\mathbf x}g(\mathbf w_p^{\mathsf T}\tilde{\mathbf x})] - \mathbb E[g'(\mathbf w_p^{\mathsf T}\tilde{\mathbf x})]\mathbf w_p$
> &nbsp;&nbsp;&nbsp;&nbsp;&bull; **Gram–Schmidt deflation:** $\mathbf w_p \leftarrow \mathbf w_p - \sum_{j<p}(\mathbf w_p^{\mathsf T}\mathbf w_j)\mathbf w_j$
> &nbsp;&nbsp;&nbsp;&nbsp;&bull; Normalise: $\mathbf w_p \leftarrow \mathbf w_p/\|\mathbf w_p\|$
> 3. Output $\mathbf W = [\mathbf w_1\cdots\mathbf w_n]^{\mathsf T}$, sources $\hat{\mathbf s} = \mathbf W\tilde{\mathbf X}$.
>
> **Symmetric variant:** update all $\mathbf w_p$ in parallel, then orthogonalise the whole matrix via $\mathbf W \leftarrow (\mathbf W\mathbf W^{\mathsf T})^{-1/2}\mathbf W$ — avoids accumulating deflation error.

**Convergence:** cubic (or at least quadratic) — far faster than gradient-based ICA, and no learning rate to tune.

### 7.6 PCA vs ICA

| | PCA | ICA |
|---|---|---|
| Criterion | Maximise variance | Maximise non-Gaussianity / minimise mutual info |
| Statistics used | 2nd order only (covariance) | **Higher order** (kurtosis, negentropy) |
| Output property | **Uncorrelated** | **Independent** (stronger: independence ⇒ uncorrelated, not conversely) |
| Components | Orthogonal, **ordered** by variance | Not orthogonal (in original space); **unordered** |
| Scale | Determined | Ambiguous (fixed by convention) |
| Gaussian data | Works fine | **Fails** (unidentifiable) |
| Dimension reduction | Native | Not its purpose (usually $n$ in, $n$ out; reduce with PCA first) |
| Typical use | Compression, denoising, whitening | Source separation: EEG/MEG artefact removal, fMRI, audio, finance |

**Independence vs uncorrelatedness.** Uncorrelated: $\mathbb E[y_iy_j]=\mathbb E[y_i]\mathbb E[y_j]$ (2nd order only). Independent: $\mathbb E[f(y_i)g(y_j)]=\mathbb E[f(y_i)]\mathbb E[g(y_j)]$ for **all** measurable $f,g$. Only for jointly Gaussian variables are the two equivalent.

### 7.7 Applications

- **EEG/MEG artefact removal:** separate eye-blink, ECG, and muscle artefacts from brain signals (the canonical application).
- **fMRI:** spatial ICA to identify functionally connected networks.
- **Audio:** blind source separation, denoising.
- **Finance:** extracting driving factors from portfolio returns.
- **Feature extraction:** ICA on natural image patches yields **Gabor-like, oriented, localised edge filters** — matching V1 simple cells (Bell & Sejnowski 1997; Olshausen & Field's sparse coding). A major result linking ICA to biological vision.

---

## 8. Comparison and Selection Guide

| Method | Type | Supervised | Objective | Max dims | Complexity | Handles nonlinearity |
|---|---|---|---|---|---|---|
| **PCA** | Linear | No | max variance | $\min(N{-}1,D)$ | $O(ND^2{+}D^3)$ | No |
| **LDA** | Linear | **Yes** | max $S_B/S_W$ | $C-1$ | $O(ND^2{+}D^3)$ | No |
| **ICA** | Linear | No | max non-Gaussianity | $n$ (=inputs) | $O(N n^2)$/iter | No |
| **Kernel PCA** | Nonlinear | No | max variance in $\mathcal F$ | $N$ | $O(N^3)$ | **Yes** |
| Isomap | Nonlinear | No | preserve geodesic distances | — | $O(N^3)$ | Yes (global) |
| LLE | Nonlinear | No | preserve local linear weights | — | $O(N^2)$ | Yes (local) |
| t-SNE | Nonlinear | No | match neighbour distributions (KL) | 2–3 | $O(N\log N)$ (Barnes-Hut) | Yes (visualisation only) |
| UMAP | Nonlinear | Semi | fuzzy topological structure | any | $O(N^{1.14})$ | Yes |
| Autoencoder | Nonlinear | No | min reconstruction error | any | $O(\text{epochs}\cdot N)$ | **Yes, learned** |

**Decision guide**

```
Do you have labels and want better classification?     → LDA (or PCA+LDA if N < D)
Do you want compression / denoising / whitening?       → PCA
Do you need statistically independent sources?         → ICA (whiten with PCA first)
Is the structure nonlinear, N small (< 5000)?          → Kernel PCA
Is it purely for 2-D visualisation?                    → t-SNE / UMAP
Large N, complex nonlinearity, need an inverse map?    → Autoencoder
```

**Note on t-SNE:** excellent for visualisation, but distances and cluster sizes in the embedding are **not** meaningful, and it has no out-of-sample extension. Never use t-SNE output as features for a downstream model.

---

## 9. Solved Numericals

### N1. PCA — complete worked example

**Data** ($N=4$, $D=2$): $\;\mathbf x_1=(4,11),\;\mathbf x_2=(8,4),\;\mathbf x_3=(13,5),\;\mathbf x_4=(7,14)$

**Step 1 — Mean**
$$
\bar{\mathbf x} = \left(\frac{4+8+13+7}{4},\;\frac{11+4+5+14}{4}\right) = \left(\frac{32}{4},\frac{34}{4}\right) = \mathbf{(8,\,8.5)}
$$

**Step 2 — Centre**

| $i$ | $x_1-\bar x_1$ | $x_2-\bar x_2$ |
|---|---|---|
| 1 | $-4$ | $2.5$ |
| 2 | $0$ | $-4.5$ |
| 3 | $5$ | $-3.5$ |
| 4 | $-1$ | $5.5$ |

**Step 3 — Covariance** ($\div (N-1)=3$)
$$
s_{11} = \frac{16+0+25+1}{3} = \frac{42}{3} = 14
$$
$$
s_{22} = \frac{6.25+20.25+12.25+30.25}{3} = \frac{69}{3} = 23
$$
$$
s_{12} = \frac{(-4)(2.5)+(0)(-4.5)+(5)(-3.5)+(-1)(5.5)}{3} = \frac{-10+0-17.5-5.5}{3} = \frac{-33}{3} = -11
$$
$$
\mathbf S = \begin{pmatrix}14 & -11\\ -11 & 23\end{pmatrix}
$$

**Step 4 — Eigenvalues**
$$
\det(\mathbf S - \lambda\mathbf I) = (14-\lambda)(23-\lambda) - 121 = \lambda^2 - 37\lambda + (322-121) = \lambda^2 - 37\lambda + 201 = 0
$$
$$
\lambda = \frac{37 \pm\sqrt{1369 - 804}}{2} = \frac{37\pm\sqrt{565}}{2} = \frac{37 \pm 23.7697}{2}
$$
$$
\boxed{\lambda_1 = 30.3849,\qquad \lambda_2 = 6.6151}
$$
*Check:* $\lambda_1+\lambda_2 = 37 = \mathrm{tr}(\mathbf S)$ ✓; $\lambda_1\lambda_2 = 201 = \det(\mathbf S)$ ✓

**Step 5 — Eigenvectors**
For $\lambda_1$: $(14-30.3849)u_1 - 11u_2 = 0 \Rightarrow -16.3849u_1 = 11u_2 \Rightarrow u_2 = -1.48954u_1$.
Take $(1, -1.48954)$, norm $=\sqrt{1+2.21874}=1.79408$:
$$
\mathbf u_1 = (0.55739,\;-0.83023)^{\mathsf T},\qquad \mathbf u_2 = (0.83023,\;0.55739)^{\mathsf T}
$$
*Check orthogonality:* $0.55739(0.83023) + (-0.83023)(0.55739) = 0$ ✓
*Check eigen-equation:* $\mathbf S\mathbf u_1 = (14(0.55739)+(-11)(-0.83023),\; -11(0.55739)+23(-0.83023)) = (16.9360,\,-25.2266)$, and $\lambda_1\mathbf u_1 = 30.3849(0.55739,-0.83023) = (16.9361,\,-25.2266)$ ✓

**Step 6 — Explained variance**
$$
\frac{\lambda_1}{\lambda_1+\lambda_2} = \frac{30.3849}{37} = \mathbf{82.12\%},\qquad \text{PC2}: 17.88\%
$$

**Step 7 — Projections onto PC1**

| $i$ | Centred | $z_{i1} = \mathbf u_1^{\mathsf T}\mathbf x_c$ | $z_{i2} = \mathbf u_2^{\mathsf T}\mathbf x_c$ |
|---|---|---|---|
| 1 | $(-4, 2.5)$ | $-2.22956 - 2.07558 = -4.30514$ | $-3.32092+1.39348 = -1.92744$ |
| 2 | $(0,-4.5)$ | $0 + 3.73604 = 3.73604$ | $0 - 2.50826 = -2.50826$ |
| 3 | $(5,-3.5)$ | $2.78695+2.90581 = 5.69276$ | $4.15115 - 1.95087 = 2.20028$ |
| 4 | $(-1,5.5)$ | $-0.55739-4.56627 = -5.12366$ | $-0.83023+3.06565 = 2.23542$ |

*Check:* $\sum_i z_{i1} = -4.30514+3.73604+5.69276-5.12366 \approx 0$ ✓ (scores are centred)
*Check variance:* $\frac{1}{3}\left(18.534+13.958+32.408+26.252\right) = \frac{91.152}{3} = 30.384 = \lambda_1$ ✓

**Step 8 — Reconstruction with $d=1$**
$$
\hat{\mathbf x}_1 = \bar{\mathbf x} + z_{11}\mathbf u_1 = (8,8.5) + (-4.30514)(0.55739,-0.83023) = (8 - 2.39965,\; 8.5+3.57432)
$$
$$
= \mathbf{(5.60035,\; 12.07432)} \quad\text{vs original } (4,11)
$$
Error vector $= (1.60035, -1.07432)$, squared error $= 2.5611+1.1542 = 3.7153 = z_{12}^2$ ✓

**Total reconstruction error:**
$$
\frac{1}{N-1}\sum_i z_{i2}^2 = \frac{3.7150+6.2914+4.8412+4.9971}{3} = \frac{19.8447}{3} = 6.6149 = \lambda_2 \;✓
$$
confirming $J_{\min} = \sum_{j>d}\lambda_j$.

---

### N2. LDA — complete worked example

**Class 1:** $(4,1),(2,4),(2,3),(3,6),(4,4)$  **Class 2:** $(9,10),(6,8),(9,5),(8,7),(10,8)$

**Step 1 — Class means**
$$
\boldsymbol\mu_1 = \left(\tfrac{15}{5},\tfrac{18}{5}\right) = \mathbf{(3,\,3.6)},\qquad
\boldsymbol\mu_2 = \left(\tfrac{42}{5},\tfrac{38}{5}\right) = \mathbf{(8.4,\,7.6)}
$$

**Step 2 — Within-class scatter $\mathbf S_1$**

Deviations from $\boldsymbol\mu_1$: $(1,-2.6),(-1,0.4),(-1,-0.6),(0,2.4),(1,0.4)$
$$
S_{1,11} = 1+1+1+0+1 = 4
$$
$$
S_{1,22} = 6.76+0.16+0.36+5.76+0.16 = 13.2
$$
$$
S_{1,12} = (1)(-2.6)+(-1)(0.4)+(-1)(-0.6)+(0)(2.4)+(1)(0.4) = -2.6-0.4+0.6+0+0.4 = -2.0
$$
$$
\mathbf S_1 = \begin{pmatrix}4 & -2\\ -2 & 13.2\end{pmatrix}
$$

**Step 3 — $\mathbf S_2$**

Deviations from $\boldsymbol\mu_2$: $(0.6,2.4),(-2.4,0.4),(0.6,-2.6),(-0.4,-0.6),(1.6,0.4)$
$$
S_{2,11} = 0.36+5.76+0.36+0.16+2.56 = 9.2
$$
$$
S_{2,22} = 5.76+0.16+6.76+0.36+0.16 = 13.2
$$
$$
S_{2,12} = 1.44-0.96-1.56+0.24+0.64 = -0.20
$$
$$
\mathbf S_2 = \begin{pmatrix}9.2 & -0.2\\ -0.2 & 13.2\end{pmatrix}
$$

**Step 4 — Pooled within-class scatter**
$$
\mathbf S_W = \mathbf S_1+\mathbf S_2 = \begin{pmatrix}13.2 & -2.2\\ -2.2 & 26.4\end{pmatrix}
$$

**Step 5 — Invert**
$$
\det\mathbf S_W = 13.2(26.4) - (-2.2)^2 = 348.48 - 4.84 = 343.64
$$
$$
\mathbf S_W^{-1} = \frac{1}{343.64}\begin{pmatrix}26.4 & 2.2\\ 2.2 & 13.2\end{pmatrix}
$$

**Step 6 — Discriminant direction**
$$
\boldsymbol\mu_1 - \boldsymbol\mu_2 = (3-8.4,\; 3.6-7.6) = (-5.4,\; -4.0)
$$
$$
\mathbf w = \mathbf S_W^{-1}(\boldsymbol\mu_1-\boldsymbol\mu_2) = \frac{1}{343.64}\begin{pmatrix}26.4(-5.4)+2.2(-4.0)\\ 2.2(-5.4)+13.2(-4.0)\end{pmatrix}
= \frac{1}{343.64}\begin{pmatrix}-151.36\\ -64.68\end{pmatrix}
$$
$$
\mathbf w = (-0.440449,\; -0.188223)^{\mathsf T}
$$
Normalising (and flipping sign so class 2 projects higher):
$$
\|\mathbf w\| = \sqrt{0.193995+0.035428} = \sqrt{0.229423} = 0.478981
$$
$$
\boxed{\;\hat{\mathbf w} = (0.919555,\; 0.392965)^{\mathsf T}\;}
$$

**Step 7 — Fisher criterion value**
$$
J(\mathbf w^\star) = (\boldsymbol\mu_1-\boldsymbol\mu_2)^{\mathsf T}\mathbf S_W^{-1}(\boldsymbol\mu_1-\boldsymbol\mu_2)
= (-5.4)(-0.440449)+(-4.0)(-0.188223) = 2.378425+0.752892 = \mathbf{3.1313}
$$

**Step 8 — Projections $y = \hat{\mathbf w}^{\mathsf T}\mathbf x$**

| Class 1 | $y$ | | Class 2 | $y$ |
|---|---|---|---|---|
| $(4,1)$ | $3.67822+0.39297 = 4.0712$ | | $(9,10)$ | $8.27600+3.92965 = 12.2056$ |
| $(2,4)$ | $1.83911+1.57186 = 3.4110$ | | $(6,8)$ | $5.51733+3.14372 = 8.6611$ |
| $(2,3)$ | $1.83911+1.17890 = 3.0180$ | | $(9,5)$ | $8.27600+1.96483 = 10.2408$ |
| $(3,6)$ | $2.75867+2.35779 = 5.1165$ | | $(8,7)$ | $7.35644+2.75076 = 10.1072$ |
| $(4,4)$ | $3.67822+1.57186 = 5.2501$ | | $(10,8)$ | $9.19555+3.14372 = 12.3393$ |
| **mean** | **4.1733** | | **mean** | **10.7108** |

*Check:* $\hat{\mathbf w}^{\mathsf T}\boldsymbol\mu_1 = 0.919555(3)+0.392965(3.6) = 2.758665+1.414674 = 4.173339$ ✓

**Result:** class-1 maximum $=5.25$, class-2 minimum $=8.66$ ⇒ **perfectly separated in 1-D**.
Decision threshold: $y_0 = \tfrac12(4.1733+10.7108) = \mathbf{7.442}$. Classify $y > 7.442$ as class 2.

**Compare with PCA on the same data.** The pooled data has its largest variance roughly along $(1,1)$ (because both classes are spread and offset diagonally), which mixes the classes far more than $\hat{\mathbf w}$ does. This is the concrete version of the diagram in §3.6.

---

### N3. Kernel PCA — complete worked example

**Data:** $\mathbf x_1 = (1,0),\;\mathbf x_2=(0,1),\;\mathbf x_3=(-1,0)$.
**Kernel:** $k(\mathbf x,\mathbf y) = (\mathbf x^{\mathsf T}\mathbf y)^2$ (homogeneous quadratic).

**Step 1 — Gram matrix**
$$
K_{11}=(1)^2=1,\quad K_{12}=(0)^2=0,\quad K_{13}=(-1)^2=1
$$
$$
K_{22}=1,\quad K_{23}=(0)^2=0,\quad K_{33}=1
$$
$$
\mathbf K = \begin{pmatrix}1&0&1\\ 0&1&0\\ 1&0&1\end{pmatrix}
$$

**Step 2 — Centre in feature space.** Row/column sums: $(2,1,2)$; grand sum $=5$.
$$
\tilde K_{ij} = K_{ij} - \tfrac13(\text{colsum}_j) - \tfrac13(\text{rowsum}_i) + \tfrac{5}{9}
$$
$$
\tilde K_{11} = 1 - \tfrac23 - \tfrac23 + \tfrac59 = \tfrac{9-6-6+5}{9} = \tfrac29
$$
$$
\tilde K_{12} = 0 - \tfrac13 - \tfrac23 + \tfrac59 = \tfrac{0-3-6+5}{9} = -\tfrac49
$$
$$
\tilde K_{22} = 1 - \tfrac13-\tfrac13+\tfrac59 = \tfrac{9-3-3+5}{9} = \tfrac89,\qquad
\tilde K_{13} = \tfrac29,\quad \tilde K_{23} = -\tfrac49,\quad \tilde K_{33} = \tfrac29
$$
$$
\tilde{\mathbf K} = \frac19\begin{pmatrix}2&-4&2\\ -4&8&-4\\ 2&-4&2\end{pmatrix}
$$
*Check:* every row sums to zero ✓ (a necessary property of $\mathbf H\mathbf K\mathbf H$).

**Step 3 — Eigendecomposition.** Notice $\tilde{\mathbf K} = \frac{2}{9}\mathbf v\mathbf v^{\mathsf T}$ with $\mathbf v = (1,-2,1)^{\mathsf T}$, $\mathbf v^{\mathsf T}\mathbf v = 6$. A rank-1 matrix, so
$$
\tilde\lambda_1 = \tfrac29 (6) = \tfrac{12}{9} = \mathbf{1.3333},\qquad \tilde\lambda_2=\tilde\lambda_3 = 0
$$
*Check:* $\mathrm{tr}(\tilde{\mathbf K}) = (2+8+2)/9 = 12/9$ ✓

**Step 4 — Normalise.** Require $\|\boldsymbol\alpha_1\|^2 = 1/\tilde\lambda_1 = 0.75$.
Unit eigenvector: $\mathbf v/\sqrt6 = (0.40825,-0.81650,0.40825)$. Scale by $\sqrt{0.75}=0.86603$:
$$
\boldsymbol\alpha_1 = (0.353553,\; -0.707107,\; 0.353553)^{\mathsf T}
$$
*Check:* $0.125+0.5+0.125 = 0.75$ ✓

**Step 5 — Projections** $y_1(\mathbf x_i) = \sum_j \alpha_{1j}\tilde K_{ji}$

$$
y_1(\mathbf x_1) = 0.353553\left(\tfrac29\right) + (-0.707107)\left(-\tfrac49\right) + 0.353553\left(\tfrac29\right)
$$
$$
= 0.078567 + 0.314270 + 0.078567 = \mathbf{0.471404}
$$
$$
y_1(\mathbf x_2) = 0.353553\left(-\tfrac49\right)+(-0.707107)\left(\tfrac89\right)+0.353553\left(-\tfrac49\right) = -0.157135-0.628540-0.157135 = \mathbf{-0.942809}
$$
$$
y_1(\mathbf x_3) = \mathbf{0.471404}
$$

*Checks:* sum $= 0$ ✓ (centred); variance $=\frac13(0.22222+0.88889+0.22222) = 0.44444 = \tilde\lambda_1/N$ ✓

**Interpretation.** $\mathbf x_1$ and $\mathbf x_3 = -\mathbf x_1$ receive **identical** projections. This is correct: $k(\mathbf x,\mathbf y)=(\mathbf x^{\mathsf T}\mathbf y)^2$ is invariant under $\mathbf x \to -\mathbf x$, so the feature map cannot distinguish antipodal points. Kernel PCA has discovered the **sign-invariant** structure — something linear PCA can never do. Explicit feature map for this kernel in 2-D: $\boldsymbol\Phi(x_1,x_2) = (x_1^2,\sqrt2 x_1x_2, x_2^2)$; indeed $\boldsymbol\Phi(1,0)=\boldsymbol\Phi(-1,0)=(1,0,0)$ and $\boldsymbol\Phi(0,1)=(0,0,1)$.

---

### N4. Whitening matrix

Using $\mathbf S$ from N1: $\lambda_1 = 30.3849$, $\lambda_2 = 6.6151$, $\mathbf U = \begin{pmatrix}0.55739 & 0.83023\\ -0.83023 & 0.55739\end{pmatrix}$.

$$
\boldsymbol\Lambda^{-1/2} = \mathrm{diag}\!\left(\tfrac{1}{\sqrt{30.3849}},\;\tfrac{1}{\sqrt{6.6151}}\right)
= \mathrm{diag}(0.181414,\; 0.388804)
$$

$$
\mathbf V = \boldsymbol\Lambda^{-1/2}\mathbf U^{\mathsf T}
= \begin{pmatrix}0.181414 & 0\\ 0 & 0.388804\end{pmatrix}
\begin{pmatrix}0.55739 & -0.83023\\ 0.83023 & 0.55739\end{pmatrix}
= \begin{pmatrix}0.101115 & -0.150614\\ 0.322805 & 0.216718\end{pmatrix}
$$

Whitened score of $\mathbf x_1$: $\mathbf V(-4,2.5)^{\mathsf T} = (-0.404460-0.376535,\;-1.291220+0.541795) = (-0.780995,\,-0.749425)$.
*Sanity:* whitened PC1 score should be $z_{11}/\sqrt{\lambda_1} = -4.30514/5.51225 = -0.78100$ ✓

**Why this matters for ICA:** after applying $\mathbf V$, the covariance is $\mathbf I$, so the remaining unknown is a pure rotation — exactly what FastICA searches over.

---

### N5. Kurtosis and source identification

**(a) Sub-Gaussian source.** $y = (-2,-1,0,1,2)$.
$$
\mathbb E[y]=0,\qquad \mathbb E[y^2] = \tfrac{4+1+0+1+4}{5} = 2,\qquad \mathbb E[y^4] = \tfrac{16+1+0+1+16}{5} = 6.8
$$
$$
\mathrm{kurt}(y) = 6.8 - 3(2)^2 = 6.8 - 12 = \mathbf{-5.2} \quad (<0 \Rightarrow \text{sub-Gaussian, uniform-like})
$$
Normalised (excess kurtosis of the standardised variable): $6.8/2^2 - 3 = 1.7-3 = -1.3$.

**(b) Super-Gaussian (sparse) source.** $y = (0,0,0,0,0,0,0,0,-5,5)$, $N=10$.
$$
\mathbb E[y^2] = \tfrac{25+25}{10} = 5,\qquad \mathbb E[y^4] = \tfrac{625+625}{10} = 125
$$
$$
\mathrm{kurt}(y) = 125 - 3(25) = \mathbf{+50} \quad (>0 \Rightarrow \text{super-Gaussian, sparse/spiky})
$$
Normalised: $125/25 - 3 = 5-3 = +2$.

**(c) Gaussian reference.** $\mathbb E[y^4]=3\sigma^4$ ⇒ $\mathrm{kurt}=3\sigma^4-3\sigma^4 = 0$ ✓

**(d) Mixing destroys non-Gaussianity.** Let $s_1,s_2$ be independent copies of the sub-Gaussian source above, and $y = (s_1+s_2)/\sqrt2$ (unit-scaled). By additivity and the $\alpha^4$ scaling law:
$$
\mathrm{kurt}(y) = \frac{1}{(\sqrt2)^4}\left[\mathrm{kurt}(s_1)+\mathrm{kurt}(s_2)\right] = \frac{-5.2-5.2}{4} = \mathbf{-2.6}
$$
$|\mathrm{kurt}|$ has **halved** — the mixture is closer to Gaussian, exactly as the CLT predicts. This is why maximising $|\mathrm{kurt}(\mathbf w^{\mathsf T}\mathbf x)|$ recovers an unmixed source.

---

### N6. Curse of dimensionality calculations

**(a)** Fraction of a $D$-ball's volume within 1 % of the surface, $D=200$:
$$
1-(0.99)^{200} = 1 - e^{200\ln0.99} = 1 - e^{-2.0101} = 1-0.13398 = \mathbf{0.86602}
$$

**(b)** Edge length of a hypercube neighbourhood containing 1 % of uniformly distributed data:
$$
e_D(0.01) = 0.01^{1/D}
$$
$D=1$: $0.01$;  $D=10$: $0.6310$;  $D=50$: $0.9120$;  $D=100$: $0.9550$.

**(c)** Sphere-in-cube ratio at $D=10$:
$$
\frac{\pi^{5}}{2^{10}\,\Gamma(6)} = \frac{306.0197}{1024\times120} = \frac{306.0197}{122880} = \mathbf{2.49\times10^{-3}}
$$

**(d)** Johnson–Lindenstrauss dimension for $N=10^6$ points at $\varepsilon = 0.1$:
$$
d \ge \frac{8\ln N}{\varepsilon^2} = \frac{8(13.8155)}{0.01} = \mathbf{11\,052}
$$
Independent of the original $D$ (which might be $10^9$).

---

### N7. Choosing the number of components

Eigenvalues of a $D=8$ covariance matrix:
$$
\lambda = (12.4,\; 6.8,\; 3.1,\; 1.9,\; 0.6,\; 0.3,\; 0.1,\; 0.05),\qquad \textstyle\sum\lambda = 25.25
$$

| $d$ | $\lambda_d$ | Cumulative | % variance |
|---|---|---|---|
| 1 | 12.40 | 12.40 | 49.11 |
| 2 | 6.80 | 19.20 | 76.04 |
| 3 | 3.10 | 22.30 | 88.32 |
| 4 | 1.90 | 24.20 | **95.84** |
| 5 | 0.60 | 24.80 | 98.22 |
| 6 | 0.30 | 25.10 | 99.41 |
| 7 | 0.10 | 25.20 | 99.80 |
| 8 | 0.05 | 25.25 | 100.00 |

- **95 % criterion** ⇒ $d = 4$ (compression $8\to4$, 50 % reduction).
- **Kaiser** (on a correlation matrix, average eigenvalue $=1$) ⇒ keep $\lambda>1$ ⇒ $d=4$ ✓ consistent.
- **Scree elbow:** the drop $1.90\to0.60$ is the largest relative fall ⇒ $d=4$ ✓
- **Reconstruction error** at $d=4$: $\sum_{j>4}\lambda_j = 0.6+0.3+0.1+0.05 = 1.05$, i.e. $4.16$ % of total variance.
- If PPCA is used: $\sigma^2_{\text{ML}} = \frac{1}{8-4}(1.05) = 0.2625$.

---

### N8. LDA rank limitation

A face-recognition problem: $D = 10\,000$ pixels, $C = 40$ subjects, $N = 400$ images.

- Maximum LDA dimensions: $C-1 = \mathbf{39}$.
- $\mathrm{rank}(\mathbf S_W) \le N - C = 400-40 = 360 \ll 10\,000$ ⇒ **$\mathbf S_W$ is singular**; plain LDA cannot be applied.
- **Fisherfaces fix:** first apply PCA to reduce $10\,000 \to N-C = 360$ dims (guaranteeing $\mathbf S_W$ is now full rank), then LDA to $360\to39$.
- Alternative: regularised LDA with $\mathbf S_W + \gamma\mathbf I$, $\gamma$ chosen by cross-validation.
- PCA alone could keep up to $\min(N-1, D) = 399$ components — far more than LDA's 39, but without any guarantee of discriminative power.

---

## 10. Viva / Exam Pointers

**Likely long questions**
1. Derive PCA from (a) maximum variance with Lagrange multipliers and (b) minimum reconstruction error; show the two are equivalent.
2. Derive the Fisher criterion and show $\mathbf w^\star \propto \mathbf S_W^{-1}(\boldsymbol\mu_1-\boldsymbol\mu_2)$; explain the rank limit $C-1$.
3. Derive Kernel PCA including the centring formula $\tilde{\mathbf K} = \mathbf H\mathbf K\mathbf H$ and the normalisation $\|\boldsymbol\alpha\|^2 = 1/\tilde\lambda$.
4. Explain why ICA fails for Gaussian sources and derive the FastICA fixed-point update.
5. Compare PCA, LDA, ICA and Kernel PCA across objective, supervision, statistics used, and output properties.
6. Explain the curse of dimensionality with at least three quantitative illustrations.
7. Prove that whitening reduces ICA to finding an orthogonal matrix.

**Traps**
- **Always centre before PCA.** Skipping this makes PC1 point at the mean, not at maximum variance.
- Eigenvalue = variance along that component, *not* the standard deviation.
- LDA gives at most $C-1$ components — a very common exam question.
- PCA components are orthogonal; LDA directions are only $\mathbf S_W$-orthogonal.
- ICA needs whitening **first**; running FastICA on unwhitened data breaks the Newton derivation.
- Uncorrelated $\ne$ independent (except for jointly Gaussian variables).
- Kernel PCA has **no exact inverse** (pre-image problem) — do not claim you can reconstruct exactly.
- Kernel centring must also be applied to test points, using the *training* statistics.

**One-line formula sheet**

$$
\mathbf S\mathbf u = \lambda\mathbf u \;\;|\;\;
J_{\min} = \textstyle\sum_{j>d}\lambda_j \;\;|\;\;
\lambda_j = \sigma_j^2/(N{-}1) \;\;|\;\;
\text{whiten: } \mathbf V = \boldsymbol\Lambda^{-1/2}\mathbf U^{\mathsf T}
$$
$$
J(\mathbf w) = \frac{\mathbf w^{\mathsf T}\mathbf S_B\mathbf w}{\mathbf w^{\mathsf T}\mathbf S_W\mathbf w} \;\;|\;\;
\mathbf w^\star \propto \mathbf S_W^{-1}(\boldsymbol\mu_1-\boldsymbol\mu_2) \;\;|\;\;
\mathbf S_W^{-1}\mathbf S_B\mathbf w = \lambda\mathbf w,\;\; d\le C-1
$$
$$
\tilde{\mathbf K} = \mathbf H\mathbf K\mathbf H \;\;|\;\;
\tilde{\mathbf K}\boldsymbol\alpha = \tilde\lambda\boldsymbol\alpha,\;\|\boldsymbol\alpha\|^2 = 1/\tilde\lambda \;\;|\;\;
y_p(\mathbf x)=\textstyle\sum_j\alpha_j^{(p)}k(\mathbf x,\mathbf x_j)
$$
$$
\mathrm{kurt}(y)=\mathbb E[y^4]-3\mathbb E[y^2]^2 \;\;|\;\;
J(y)=H(y_{\text{gauss}})-H(y) \;\;|\;\;
\mathbf w^+ = \mathbb E[\mathbf xg(\mathbf w^{\mathsf T}\mathbf x)] - \mathbb E[g'(\mathbf w^{\mathsf T}\mathbf x)]\mathbf w
$$

---

*Previous: [Unit III](./Unit-3.md) · Next: [Unit V: Advanced Neural Network Architectures](./Unit-5.md)*
