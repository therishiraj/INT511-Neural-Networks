# Unit VI — Optimization Techniques for Neural Networks

> **INT511 – Neural Networks** | M.Tech Level Notes
> **Coverage:** SGD · Mini-batch · Momentum · Adam · RMSprop · AdaGrad

---

## Table of Contents

1. [The Optimization Problem](#1-the-optimization-problem)
2. [Geometry of the Loss Landscape](#2-geometry-of-the-loss-landscape)
3. [Gradient Descent and Its Convergence](#3-gradient-descent-and-its-convergence)
4. [Stochastic Gradient Descent](#4-stochastic-gradient-descent)
5. [Mini-Batch Gradient Descent](#5-mini-batch-gradient-descent)
6. [Momentum](#6-momentum)
7. [Nesterov Accelerated Gradient](#7-nesterov-accelerated-gradient)
8. [AdaGrad](#8-adagrad)
9. [RMSProp and AdaDelta](#9-rmsprop-and-adadelta)
10. [Adam and Its Variants](#10-adam-and-its-variants)
11. [Learning-Rate Schedules](#11-learning-rate-schedules)
12. [Second-Order and Natural-Gradient Methods](#12-second-order-and-natural-gradient-methods)
13. [Comparison and Practical Guidance](#13-comparison-and-practical-guidance)
14. [Applications of Neural Networks](#14-applications-of-neural-networks)
15. [Solved Numericals](#15-solved-numericals)
16. [Viva / Exam Pointers](#16-viva--exam-pointers)

---

## 1. The Optimization Problem

### 1.1 Statement

$$
\theta^\star = \arg\min_{\theta\in\mathbb R^{P}}\; \mathcal L(\theta),\qquad
\mathcal L(\theta) = \frac1N\sum_{i=1}^{N}\ell\big(f_\theta(\mathbf x_i),\mathbf y_i\big) + \lambda\Omega(\theta)
$$

**What makes neural-network optimization hard:**

| Difficulty | Description |
|---|---|
| **Non-convexity** | Many critical points; no global guarantee |
| **High dimension** | $P$ up to $10^{11}$; Hessian ($P^2$ entries) cannot be stored |
| **Ill-conditioning** | $\kappa(\mathbf H) = \lambda_{\max}/\lambda_{\min}$ can exceed $10^{6}$ |
| **Stochasticity** | Only noisy gradient estimates available |
| **Saddle points & plateaus** | Dominant in high dimension |
| **Cliffs / exploding gradients** | Especially in RNNs |
| **Poor correspondence** | We minimise empirical risk, but care about expected risk |
| **Symmetries** | Permutation and scaling symmetries create flat directions and degenerate manifolds |

**Important distinction.** Pure optimization minimises $\mathcal L$; machine learning wants low **test** error. Driving training loss to exactly zero is often *not* optimal — this is why early stopping, implicit regularization by SGD noise, and flat-minima seeking matter.

### 1.2 Local quadratic model — the master equation

Taylor-expand around $\theta_0$:

$$
\mathcal L(\theta_0 + \Delta) \approx \mathcal L(\theta_0) + \mathbf g^{\mathsf T}\Delta + \tfrac12\Delta^{\mathsf T}\mathbf H\Delta,
\qquad \mathbf g = \nabla\mathcal L(\theta_0),\;\; \mathbf H = \nabla^2\mathcal L(\theta_0)
$$

For a gradient step $\Delta = -\eta\mathbf g$:

$$
\boxed{\;\mathcal L(\theta_0 - \eta\mathbf g) \approx \mathcal L(\theta_0) - \eta\|\mathbf g\|^2 + \tfrac{\eta^2}{2}\mathbf g^{\mathsf T}\mathbf H\mathbf g\;}
$$

**Reading this equation is the key to the whole unit:**
- If $\mathbf g^{\mathsf T}\mathbf H\mathbf g \le 0$ (negative curvature): the step always decreases loss; larger $\eta$ is better.
- If $\mathbf g^{\mathsf T}\mathbf H\mathbf g > 0$: the optimal step size is

  $$
  \eta^\star = \frac{\|\mathbf g\|^2}{\mathbf g^{\mathsf T}\mathbf H\mathbf g}
  $$

and progress *stalls* when $\mathbf g^{\mathsf T}\mathbf H\mathbf g$ grows faster than $\|\mathbf g\|^2$ — the classic "loss stops decreasing even though the gradient is large" phenomenon in high-curvature directions.

---

## 2. Geometry of the Loss Landscape

### 2.1 Critical-point taxonomy

At $\nabla\mathcal L = \mathbf 0$, classify by the Hessian eigenvalues:

| All $\lambda_i$ | Type |
|---|---|
| $>0$ | local **minimum** |
| $<0$ | local **maximum** |
| mixed signs | **saddle point** |
| some $=0$ | degenerate (plateau / flat valley) |

### 2.2 Why saddle points, not local minima, are the problem

Model the Hessian eigenvalues at a random critical point of a high-dimensional random function as i.i.d. symmetric about 0. Then

$$
\Pr[\text{all } P \text{ eigenvalues} > 0] \approx 2^{-P}
$$

For $P = 10^6$ this is astronomically small. **Random critical points are almost surely saddles.**

**Random matrix theory refinement (Dauphin et al. 2014, Bray & Dean):** for Gaussian random fields, the *index* (fraction of negative eigenvalues) of a critical point is tightly coupled to its loss value:
- high loss ⇒ many negative directions ⇒ easy to escape;
- low loss ⇒ index $\to 0$ ⇒ genuine minima cluster near the global value.

**Consequence:** most local minima found by SGD have loss close to the global minimum. The real enemy is **slow escape from saddle plateaus**, where $\|\mathbf g\|\approx0$ but $\mathcal L$ is still high. Plain gradient descent can take exponentially long to escape; adding noise (SGD) or curvature information escapes in polynomial time.

### 2.3 Ill-conditioning and the ravine

For a quadratic $\mathcal L = \tfrac12\theta^{\mathsf T}\mathbf H\theta$ in the eigenbasis $\tilde\theta = \mathbf Q^{\mathsf T}\theta$:

$$
\tilde\theta_i^{(k)} = (1-\eta\lambda_i)^{k}\,\tilde\theta_i^{(0)}
$$

- **Stability:** $|1-\eta\lambda_i|<1\;\forall i \;\Rightarrow\; \boxed{0 < \eta < 2/\lambda_{\max}}$.
- $\eta$ is therefore capped by the **largest** eigenvalue, but convergence along the smallest direction takes $\sim\lambda_{\max}/\lambda_{\min} = \kappa$ times longer.

```
        Ill-conditioned quadratic (κ large): the "ravine"

           ╭──────────────────────────────╮
          ╱   ↘ ↗ ↘ ↗ ↘ ↗ ↘ ↗ ↘           ╲     GD zig-zags across the
         │      (steep direction)          │    narrow axis while crawling
          ╲   →→→→→→→→→→→→→→→→→→→ ✳       ╱     along the flat one.
           ╰──────────────────────────────╯
              (flat direction, slow)

        Momentum: oscillations cancel across the steep axis,
                  accelerate along the consistent flat axis.
```

### 2.4 Flat vs sharp minima

Empirically, minima with small $\|\mathbf H\|$ ("flat") generalise better. Intuition (MDL / PAC-Bayes): a flat minimum can be specified with fewer bits and is robust to the train→test distribution shift.

- **Large-batch training** tends to find sharp minima (less gradient noise to escape them) — the *generalization gap* of large-batch SGD.
- **SAM (Sharpness-Aware Minimization):** minimise $\max_{\|\epsilon\|_2\le\rho}\mathcal L(\theta+\epsilon)$, implemented as a two-step gradient (ascent then descent).
- **Mode connectivity:** independently found minima are typically joined by low-loss curves — the solution set is a connected manifold, not isolated basins.

---

## 3. Gradient Descent and Its Convergence

### 3.1 Batch gradient descent

$$
\theta_{k+1} = \theta_k - \eta\,\nabla\mathcal L(\theta_k),\qquad \nabla\mathcal L = \frac1N\sum_{i=1}^N\nabla\ell_i
$$

### 3.2 Descent lemma and the non-convex rate

**Assumption ($L$-smoothness):** $\|\nabla\mathcal L(x)-\nabla\mathcal L(y)\|\le L\|x-y\|$, equivalently
$$
\mathcal L(y) \le \mathcal L(x) + \nabla\mathcal L(x)^{\mathsf T}(y-x) + \tfrac{L}{2}\|y-x\|^2 .
$$

Set $y = x - \eta\nabla\mathcal L(x)$:
$$
\mathcal L(x^+) \le \mathcal L(x) - \eta\|\nabla\mathcal L\|^2 + \tfrac{L\eta^2}{2}\|\nabla\mathcal L\|^2 = \mathcal L(x) - \eta\!\left(1-\tfrac{L\eta}{2}\right)\!\|\nabla\mathcal L\|^2
$$

Guaranteed descent requires $\eta < 2/L$; the best bound is at $\eta = 1/L$:
$$
\mathcal L(x^+) \le \mathcal L(x) - \frac{1}{2L}\|\nabla\mathcal L(x)\|^2 .
$$

Summing over $K$ iterations and telescoping:
$$
\boxed{\;\min_{k<K}\|\nabla\mathcal L(\theta_k)\|^2 \le \frac{2L\left(\mathcal L(\theta_0)-\mathcal L^\star\right)}{K}\;}
$$
i.e. $\|\nabla\mathcal L\| = O(1/\sqrt K)$ — the standard non-convex guarantee (a stationary point, not a minimum).

### 3.3 Convex and strongly convex rates

| Setting | Rate | Iterations for $\epsilon$ |
|---|---|---|
| Non-convex, smooth | $\min_k\Vert \nabla\mathcal L\Vert ^2 = O(1/K)$ | $O(1/\epsilon^2)$ |
| Convex, smooth | $\mathcal L(\theta_K)-\mathcal L^\star = O(1/K)$ | $O(1/\epsilon)$ |
| Convex + Nesterov | $O(1/K^2)$ | $O(1/\sqrt\epsilon)$ **(optimal)** |
| $\mu$-strongly convex | $\left(1-\tfrac1\kappa\right)^K$ (linear) | $O(\kappa\log\tfrac1\epsilon)$ |
| Strongly convex + momentum | $\left(1-\tfrac{1}{\sqrt\kappa}\right)^K$ | $O(\sqrt\kappa\log\tfrac1\epsilon)$ **(optimal)** |

**The headline result:** momentum improves the dependence on the condition number from $\kappa$ to $\sqrt\kappa$. For $\kappa = 10^4$ that is a **100×** speed-up.

**Exact quadratic analysis.**
$$
\eta^\star = \frac{2}{\lambda_{\min}+\lambda_{\max}},\qquad
\text{contraction factor } = \frac{\kappa-1}{\kappa+1}.
$$

---

## 4. Stochastic Gradient Descent

### 4.1 The update

$$
\boxed{\;\theta_{k+1} = \theta_k - \eta_k\,\nabla\ell_{i_k}(\theta_k),\qquad i_k \sim \text{Uniform}\{1,\dots,N\}\;}
$$

**Unbiasedness:** $\;\mathbb E_{i}[\nabla\ell_i(\theta)] = \frac1N\sum_i\nabla\ell_i(\theta) = \nabla\mathcal L(\theta)$ ✓
**Variance:** $\;\mathrm{Var} = \boldsymbol\Sigma(\theta) = \frac1N\sum_i\left(\nabla\ell_i - \nabla\mathcal L\right)\left(\nabla\ell_i-\nabla\mathcal L\right)^{\mathsf T}$.

### 4.2 Convergence — Robbins–Monro conditions (1951)

$$
\boxed{\;\sum_{k=1}^{\infty}\eta_k = \infty \quad\text{(can travel any distance)},\qquad
\sum_{k=1}^{\infty}\eta_k^2 < \infty \quad\text{(noise is averaged out)}\;}
$$

| Schedule | $\sum\eta_k$ | $\sum\eta_k^2$ | Valid? |
|---|---|---|---|
| $\eta_k = \eta_0$ (constant) | $\infty$ ✓ | $\infty$ ✗ | No (converges to a noise ball) |
| $\eta_k = \eta_0/k$ | $\infty$ ✓ | $\pi^2\eta_0^2/6 <\infty$ ✓ | **Yes** |
| $\eta_k = \eta_0/\sqrt k$ | $\infty$ ✓ | $\infty$ ✗ | No (but gives the optimal $O(1/\sqrt K)$ convex rate) |
| $\eta_k = \eta_0/k^{0.75}$ | $\infty$ ✓ | $<\infty$ ✓ | Yes |

**Rate (convex, bounded variance $\sigma^2$):**
$$
\mathbb E\left[\mathcal L(\bar\theta_K)\right] - \mathcal L^\star = O\!\left(\frac{\sigma}{\sqrt K}\right)
$$
**Sub-linear and independent of $\kappa$** — noise, not curvature, is the bottleneck. Constant $\eta$ converges only to a ball of radius $O(\eta\sigma^2/\mu)$ around the optimum.

### 4.3 Why the noise is useful

- **Escapes saddles and sharp minima:** the stochastic term acts like thermal noise, allowing uphill moves (cf. simulated annealing, Unit III).
- **Implicit regularization:** the SDE approximation

  $$
  d\theta = -\nabla\mathcal L(\theta)\,dt + \sqrt{\tfrac{\eta}{B}}\,\boldsymbol\Sigma^{1/2}\,d\mathbf W_t
  $$

shows the *temperature* is $\eta/B$. High temperature (large $\eta$, small $B$) biases toward **flat** minima, which generalise better.
- **Cheap:** $O(1)$ per update instead of $O(N)$ ⇒ many more updates per epoch. On large datasets SGD reaches a good solution before batch GD has completed one step.

### 4.4 SGD vs Batch GD

| | Batch GD | SGD |
|---|---|---|
| Cost/update | $O(N)$ | $O(1)$ |
| Updates/epoch | 1 | $N$ |
| Gradient | exact | noisy, unbiased |
| Convergence | smooth, deterministic | noisy, "random walk downhill" |
| Escapes saddles | poorly | well |
| Final accuracy | exact optimum | noise ball (unless $\eta_k\to0$) |
| Parallelism | high | low (sequential) |
| Online/streaming | no | **yes** |

---

## 5. Mini-Batch Gradient Descent

### 5.1 The update

$$
\boxed{\;\theta_{k+1} = \theta_k - \eta\,\frac{1}{B}\sum_{i\in\mathcal B_k}\nabla\ell_i(\theta_k)\;}
$$

### 5.2 Variance reduction — the $1/\sqrt B$ law

For a mini-batch of $B$ i.i.d. samples,
$$
\mathrm{Var}\!\left(\hat{\mathbf g}_B\right) = \frac{\boldsymbol\Sigma}{B}
\qquad\Longrightarrow\qquad
\text{SD}\left(\hat{\mathbf g}_B\right) = \frac{\sigma}{\sqrt B}
$$

**Diminishing returns:** going from $B=1$ to $B=100$ reduces noise 10×, but costs 100× more computation per step. Going $B=100\to400$ costs 4× more for only 2× less noise. This is why very large batches give diminishing (and eventually negative) returns.

**Gradient noise scale (McCandlish et al.):** $B_{\text{crit}} = \dfrac{\mathrm{tr}(\mathbf H\boldsymbol\Sigma)}{\mathbf g^{\mathsf T}\mathbf H\mathbf g}$. Below $B_{\text{crit}}$, doubling the batch nearly halves the number of steps required (perfect scaling); above it, extra samples are wasted.

### 5.3 Learning-rate scaling rules

**Linear scaling rule (Goyal et al. 2017):** when multiplying the batch size by $k$, multiply the learning rate by $k$.
*Justification:* $k$ small steps with rate $\eta$ approximately equal one big step with rate $k\eta$, provided the gradient does not change much over those steps. Requires a **warmup** phase because the assumption fails early in training.

**Square-root scaling:** $\eta \propto \sqrt B$ — keeps the SGD "temperature" $\eta/B$ … more precisely keeps the noise magnitude $\eta\sigma/\sqrt B$ constant. Often preferred for adaptive optimizers (Adam).

### 5.4 Practical notes

- Choose $B$ as a power of 2 (32, 64, 128, 256) for GPU/SIMD efficiency.
- Small $B$ (2–32) often generalises better (more noise) but under-uses hardware.
- **Shuffle every epoch** — otherwise a fixed order introduces correlated bias. *Random reshuffling* actually converges faster than with-replacement sampling ($O(1/K^2)$ vs $O(1/K)$ in the strongly convex case).
- BatchNorm couples samples within a batch; $B$ becomes a model hyperparameter, not just an optimization one. Use GroupNorm/LayerNorm for tiny batches.
- Gradient accumulation simulates a large $B$ under memory constraints.

---

## 6. Momentum

### 6.1 The update (Polyak, 1964 — "heavy ball")

$$
\boxed{\;\mathbf v_{k+1} = \beta\mathbf v_k + \nabla\mathcal L(\theta_k),\qquad \theta_{k+1} = \theta_k - \eta\,\mathbf v_{k+1}\;}
$$

Equivalent form used in many texts/frameworks:
$$
\mathbf v_{k+1} = \beta\mathbf v_k - \eta\nabla\mathcal L(\theta_k),\qquad \theta_{k+1}=\theta_k + \mathbf v_{k+1}
$$
(Identical up to where $\eta$ is applied; be consistent.)

Typical $\beta = 0.9$ ($0.99$, $0.999$ for very noisy problems).

### 6.2 Physical interpretation

$\theta$ = position, $\mathbf v$ = velocity, $-\nabla\mathcal L$ = force, $\beta$ = $(1-\text{friction})$. The particle has **inertia**: it does not stop at every local wiggle and rolls through small bumps and plateaus.

### 6.3 Effective learning rate — derivation

Unroll the recursion (with $\mathbf v_0 = \mathbf 0$):
$$
\mathbf v_{k+1} = \sum_{j=0}^{k}\beta^{\,k-j}\,\mathbf g_j \qquad (\text{exponentially weighted sum of past gradients})
$$

If the gradient is approximately constant, $\mathbf g_j \approx \mathbf g$:
$$
\mathbf v_\infty = \mathbf g\sum_{j=0}^{\infty}\beta^{j} = \frac{\mathbf g}{1-\beta}
\qquad\Longrightarrow\qquad
\boxed{\;\Delta\theta = -\frac{\eta}{1-\beta}\,\mathbf g \;\;\Longrightarrow\;\; \eta_{\text{eff}} = \frac{\eta}{1-\beta}\;}
$$

| $\beta$ | $\eta_{\text{eff}}/\eta$ | Effective averaging window $\approx 1/(1-\beta)$ |
|---|---|---|
| 0.5 | 2× | 2 gradients |
| 0.9 | **10×** | 10 gradients |
| 0.99 | 100× | 100 gradients |
| 0.999 | 1000× | 1000 gradients |

**Practical consequence:** when you increase $\beta$, you must usually *decrease* $\eta$ to keep $\eta/(1-\beta)$ roughly constant, or training will diverge.

### 6.4 Why momentum fixes the ravine

Decompose in the Hessian eigenbasis. Along a **high-curvature** direction the gradient alternates sign each step, so consecutive terms in $\sum\beta^{k-j}\mathbf g_j$ **cancel** ⇒ oscillation is damped. Along a **low-curvature** direction the gradient keeps the same sign, so terms **accumulate** ⇒ effective step is amplified by $1/(1-\beta)$. Momentum thus performs *automatic per-direction rescaling toward $\kappa^{-1/2}$ behaviour*.

**Optimal parameters for a quadratic (Polyak):**
$$
\boxed{\;\beta^\star = \left(\frac{\sqrt\kappa - 1}{\sqrt\kappa+1}\right)^{2},\qquad
\eta^\star = \frac{4}{\left(\sqrt{\lambda_{\max}}+\sqrt{\lambda_{\min}}\right)^{2}},\qquad
\text{rate} = \frac{\sqrt\kappa-1}{\sqrt\kappa+1}\;}
$$
compared with GD's $\dfrac{\kappa-1}{\kappa+1}$. **$\kappa \to \sqrt\kappa$.**

---

## 7. Nesterov Accelerated Gradient

### 7.1 The update (Nesterov, 1983)

$$
\boxed{\;\mathbf v_{k+1} = \beta\mathbf v_k + \nabla\mathcal L\!\left(\theta_k - \eta\beta\mathbf v_k\right),\qquad
\theta_{k+1} = \theta_k - \eta\mathbf v_{k+1}\;}
$$

**The only change:** the gradient is evaluated at the **look-ahead** point $\theta_k - \eta\beta\mathbf v_k$ — where momentum is about to take us — rather than at the current point.

```
   Classical momentum                  Nesterov (NAG)

   θ ●───────► βv (momentum step)      θ ●───────► βv
     │                                            │
     ▼ g(θ)  (gradient at current)                ▼ g(θ - ηβv)
     ╲                                              ╲   (gradient at look-ahead)
      ╲──────► actual step                           ╲──► actual step
                                                    "correction" — if the
   overshoots more                                  momentum overshoots, the
                                                    look-ahead gradient pulls back
```

### 7.2 Why it is better

The look-ahead acts as a **first-order correction term** resembling curvature information: expanding $\nabla\mathcal L(\theta - \eta\beta\mathbf v)\approx \nabla\mathcal L(\theta) - \eta\beta\mathbf H\mathbf v$, NAG implicitly uses $\mathbf H\mathbf v$ (a Hessian–vector product) without ever forming $\mathbf H$. This is why it damps overshoot.

**Theoretical guarantee (convex, smooth):** $\mathcal L(\theta_K)-\mathcal L^\star = O(1/K^2)$, which **matches the lower bound** for first-order methods — hence "optimal" / "accelerated".

### 7.3 Bengio's reformulation (used in PyTorch/TF)

To avoid evaluating the gradient at a different point, substitute $\tilde\theta_k = \theta_k - \eta\beta\mathbf v_k$:
$$
\mathbf v_{k+1} = \beta\mathbf v_k + \mathbf g_k,\qquad
\tilde\theta_{k+1} = \tilde\theta_k - \eta\left(\beta\mathbf v_{k+1} + \mathbf g_k\right)
$$
Now the gradient is evaluated at the iterate itself, so it drops into standard code.

---

## 8. AdaGrad

### 8.1 Motivation

A single global $\eta$ is wrong when features have very different frequencies/scales — e.g. in NLP, rare words need large steps, frequent words small ones. **Give every parameter its own learning rate**, adapted from its gradient history.

### 8.2 The update (Duchi, Hazan & Singer, 2011)

$$
\boxed{\;\mathbf G_k = \mathbf G_{k-1} + \mathbf g_k\odot\mathbf g_k = \sum_{j=1}^{k}\mathbf g_j^2,
\qquad
\theta_{k+1} = \theta_k - \frac{\eta}{\sqrt{\mathbf G_k}+\epsilon}\odot\mathbf g_k\;}
$$

(All operations element-wise; $\epsilon \approx 10^{-8}$ prevents division by zero.)

Per-coordinate:
$$
\theta_{k+1,i} = \theta_{k,i} - \frac{\eta}{\sqrt{\sum_{j=1}^{k}g_{j,i}^2}+\epsilon}\,g_{k,i}
$$

### 8.3 Properties

- **Automatic per-parameter rates:** coordinates with historically large gradients get small effective rates and vice versa.
- **Sparse-data friendly:** rare features accumulate little $G$, so keep large steps. Excellent for NLP/recommender systems with sparse one-hot inputs.
- **Scale invariance:** the update magnitude is roughly $\eta$ regardless of the gradient's units.
- **Regret bound (online convex optimization):**

  $$
  R(T) = \sum_{t=1}^{T}\left[f_t(\theta_t)-f_t(\theta^\star)\right] = O\!\left(\sqrt T\right),
  $$

with a constant that is much better than plain online GD when the gradients are sparse — this is the theoretical selling point.
- **Fatal flaw:** $\mathbf G_k$ is **monotonically increasing** and never forgets, so the effective learning rate $\eta/\sqrt{G_k}$ decays to zero **too aggressively**; deep-network training stalls before convergence.

$$
\text{Effective rate} \sim \frac{\eta}{\sqrt{k}\,\bar g}\;\longrightarrow\; 0
$$

This single defect motivates RMSProp and Adam.

---

## 9. RMSProp and AdaDelta

### 9.1 RMSProp (Hinton, Coursera lecture 6e, 2012 — never formally published)

**Fix:** replace the cumulative sum with an **exponentially weighted moving average** so old gradients are forgotten.

$$
\boxed{\;\mathbf E[\mathbf g^2]_k = \rho\,\mathbf E[\mathbf g^2]_{k-1} + (1-\rho)\,\mathbf g_k^2,\qquad
\theta_{k+1} = \theta_k - \frac{\eta}{\sqrt{\mathbf E[\mathbf g^2]_k}+\epsilon}\odot\mathbf g_k\;}
$$

Defaults: $\rho = 0.9$, $\eta = 0.001$, $\epsilon = 10^{-8}$.

The denominator $\sqrt{\mathbf E[g^2]}$ is the **root mean square** of recent gradients — hence *RMSProp*.

**Why it works:** the EWMA has an effective window of $1/(1-\rho) = 10$ steps, so the accumulator **stabilises** instead of growing without bound. The learning rate can shrink *and grow* as the landscape changes — essential for non-convex, non-stationary problems.

**Comparison with AdaGrad:**

| | AdaGrad | RMSProp |
|---|---|---|
| Accumulator | $\sum_{j\le k} g_j^2$ (all history) | $\rho E + (1-\rho)g_k^2$ (recent) |
| Effective rate | monotonically $\downarrow 0$ | can rise and fall |
| Suits | convex, sparse | **non-convex, deep nets, RNNs** |
| Failure mode | premature stalling | none major (but no momentum) |

### 9.2 AdaDelta (Zeiler, 2012)

Two extra ideas: (i) drop the global $\eta$ entirely; (ii) fix the **unit mismatch** — in SGD/AdaGrad the update has units of $1/[\theta]$ rather than $[\theta]$.

$$
\mathbf E[\mathbf g^2]_k = \rho\mathbf E[\mathbf g^2]_{k-1} + (1-\rho)\mathbf g_k^2
$$
$$
\Delta\theta_k = -\frac{\sqrt{\mathbf E[\Delta\theta^2]_{k-1}+\epsilon}}{\sqrt{\mathbf E[\mathbf g^2]_k + \epsilon}}\odot\mathbf g_k
$$
$$
\mathbf E[\Delta\theta^2]_k = \rho\,\mathbf E[\Delta\theta^2]_{k-1} + (1-\rho)\,\Delta\theta_k^2
$$

The RMS of past *updates* supplies the numerator, so the ratio is dimensionally consistent and **no learning rate is needed**. In practice a small $\eta$ is often still used.

---

## 10. Adam and Its Variants

### 10.1 Adam = Adaptive Moment Estimation (Kingma & Ba, 2015)

**Idea: momentum (1st moment) + RMSProp (2nd moment) + bias correction.**

> **Algorithm: Adam**
> Defaults: $\eta = 0.001$, $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$. Init $\mathbf m_0 = \mathbf v_0 = \mathbf 0$.
> For $k = 1,2,\dots$:
> $$\mathbf g_k = \nabla\mathcal L(\theta_{k-1})$$
> $$\mathbf m_k = \beta_1\mathbf m_{k-1} + (1-\beta_1)\mathbf g_k \qquad \text{(1st moment: mean)}$$
> $$\mathbf v_k = \beta_2\mathbf v_{k-1} + (1-\beta_2)\mathbf g_k^2 \qquad \text{(2nd moment: uncentred variance)}$$
> $$\hat{\mathbf m}_k = \frac{\mathbf m_k}{1-\beta_1^{k}},\qquad \hat{\mathbf v}_k = \frac{\mathbf v_k}{1-\beta_2^{k}} \qquad\text{(bias correction)}$$
> $$\boxed{\;\theta_k = \theta_{k-1} - \eta\,\frac{\hat{\mathbf m}_k}{\sqrt{\hat{\mathbf v}_k}+\epsilon}\;}$$

### 10.2 Derivation of the bias correction

Unroll $\mathbf m_k$ with $\mathbf m_0 = \mathbf 0$:
$$
\mathbf m_k = (1-\beta_1)\sum_{j=1}^{k}\beta_1^{\,k-j}\mathbf g_j
$$
Take expectations, assuming $\mathbb E[\mathbf g_j]\approx\mathbb E[\mathbf g]$ is stationary:
$$
\mathbb E[\mathbf m_k] = \mathbb E[\mathbf g]\,(1-\beta_1)\sum_{j=1}^{k}\beta_1^{\,k-j}
= \mathbb E[\mathbf g]\,(1-\beta_1)\cdot\frac{1-\beta_1^{k}}{1-\beta_1}
= \mathbb E[\mathbf g]\left(1-\beta_1^{k}\right)
$$

$$
\boxed{\;\mathbb E[\mathbf m_k] = (1-\beta_1^{k})\,\mathbb E[\mathbf g]\;\;\Longrightarrow\;\; \hat{\mathbf m}_k = \frac{\mathbf m_k}{1-\beta_1^{k}} \text{ is unbiased}\;}
$$

Identically, $\mathbb E[\mathbf v_k] = (1-\beta_2^{k})\mathbb E[\mathbf g^2]$.

**Why this matters concretely.** At $k=1$: $\mathbf m_1 = 0.1\mathbf g$, $\mathbf v_1 = 0.001\mathbf g^2$, so
$$
\frac{m_1}{\sqrt{v_1}} = \frac{0.1g}{0.0316|g|} = 3.16\,\text{sign}(g)
$$
— a step **3.16× too large**. With correction: $\hat m_1 = g$, $\hat v_1 = g^2$, ratio $=1$. Since $\beta_2 = 0.999$, the bias persists for **hundreds** of iterations; without correction, early Adam steps are wildly mis-scaled.

### 10.3 Properties

- **Bounded step size:** $\left|\eta\hat m/\sqrt{\hat v}\right| \lesssim \eta$ (exactly $\le \eta\cdot\frac{1-\beta_1}{\sqrt{1-\beta_2}}$ in the worst case) — a "trust region" of roughly $\eta$ per parameter per step.
- **Scale invariant:** rescaling the loss by $c$ scales $\mathbf m$ by $c$ and $\sqrt{\mathbf v}$ by $c$ — the update is unchanged.
- **Signed-gradient limit:** for very noisy gradients $\hat m/\sqrt{\hat v}\to \text{sign}(g)$ — Adam behaves like **signSGD**.
- Works well out of the box; the de-facto default for Transformers, GANs and RNNs.

### 10.4 Known problems and fixes

| Problem | Fix |
|---|---|
| **Non-convergence** — Reddi et al. (2018) exhibited a convex problem where Adam fails, because $\hat{\mathbf v}$ can *decrease*, making the effective rate increase | **AMSGrad:** $\hat{\mathbf v}_k \leftarrow \max(\hat{\mathbf v}_{k-1},\mathbf v_k)$, enforcing a non-increasing rate |
| **Weight decay ≠ $L_2$ under adaptive scaling** — the $L_2$ gradient $\lambda\theta$ gets divided by $\sqrt{\hat v}$, so decay is weaker for large-gradient parameters | **AdamW:** decouple — $\theta \leftarrow \theta - \eta\!\left(\frac{\hat m}{\sqrt{\hat v}+\epsilon} + \lambda\theta\right)$ |
| **High variance of the adaptive rate early on** | **RAdam** (rectified: use plain SGD-with-momentum until $\hat v$ is reliable), or a **warmup** schedule |
| **Worse generalization than SGD+momentum on CNNs** | SWATS (switch Adam→SGD), or just use SGD+momentum for vision |
| **Memory 3× parameters** ($\theta,\mathbf m,\mathbf v$) | **Adafactor** (factored 2nd moment), 8-bit Adam, **Lion** (sign-based, stores only momentum) |

### 10.5 Other modern optimizers (worth naming at M.Tech level)

| Optimizer | Idea |
|---|---|
| **Nadam** | Adam + Nesterov look-ahead on the first moment |
| **LARS / LAMB** | **Layer-wise** adaptive rate $\eta_\ell \propto \frac{\Vert \theta_\ell\Vert }{\Vert \mathbf g_\ell\Vert }$ — enables batch sizes of 32k+ |
| **Lookahead** | maintain slow + fast weights: $\phi \leftarrow \phi + \alpha(\theta_k - \phi)$ every $k$ steps |
| **Shampoo / K-FAC** | approximate full-matrix preconditioning via Kronecker factors |
| **Lion** | $\theta\leftarrow\theta - \eta\,\text{sign}(\beta_1 m + (1-\beta_1)g)$; discovered by program search |
| **SAM** | ascend to $\theta+\rho\frac{\mathbf g}{\Vert \mathbf g\Vert }$, then descend using that gradient ⇒ flat minima |

---

## 11. Learning-Rate Schedules

The learning rate is consistently reported as **the single most important hyperparameter**.

| Schedule | Formula | Notes |
|---|---|---|
| Constant | $\eta_k = \eta_0$ | baseline; converges to a noise ball |
| Step decay | $\eta_k = \eta_0\gamma^{\lfloor k/s\rfloor}$ | e.g. $\times0.1$ every 30 epochs (ResNet recipe) |
| Exponential | $\eta_k = \eta_0 e^{-\alpha k}$ | smooth |
| $1/t$ | $\eta_k = \eta_0/(1+\alpha k)$ | satisfies Robbins–Monro |
| Polynomial | $\eta_k = \eta_0(1-k/K)^{p}$ | $p=1$ is linear decay |
| **Cosine annealing** | $\eta_k = \eta_{\min} + \tfrac12(\eta_0-\eta_{\min})\left(1+\cos\frac{k\pi}{K}\right)$ | very widely used |
| Cosine + warm restarts (SGDR) | restart $\eta$ periodically | escapes minima, gives free ensembles |
| **Warmup** | $\eta_k = \eta_0\,k/K_w$ for $k<K_w$ | essential for large batches and Transformers |
| One-cycle (Smith) | ramp $\eta$ up then down; momentum inversely | fast "super-convergence" |
| ReduceLROnPlateau | $\times\gamma$ when val loss stalls | adaptive, no schedule design |

**Why warmup is needed.** Early in training, $\hat{\mathbf v}$ in Adam is estimated from very few samples and has huge variance; and with large batches the linear-scaling assumption is violated because the loss surface changes rapidly. A short linear ramp (typically 1–5 % of total steps) fixes both.

**Transformer schedule (Vaswani et al.):**
$$
\eta_k = d_{\text{model}}^{-0.5}\cdot\min\!\left(k^{-0.5},\; k\cdot K_w^{-1.5}\right)
$$
— linear warmup then inverse-square-root decay.

**LR range test (Smith).** Train for a few hundred iterations, increasing $\eta$ exponentially from $10^{-7}$ to $10$; plot loss vs $\eta$. Pick $\eta_{\max}$ just before the loss diverges, and $\eta_0 \approx \eta_{\max}/10$.

---

## 12. Second-Order and Natural-Gradient Methods

### 12.1 Newton's method

$$
\theta_{k+1} = \theta_k - \mathbf H^{-1}\mathbf g_k
$$

Derived by minimising the quadratic model exactly: $\nabla[\mathbf g^{\mathsf T}\Delta + \tfrac12\Delta^{\mathsf T}\mathbf H\Delta] = 0 \Rightarrow \Delta = -\mathbf H^{-1}\mathbf g$.

**Advantage:** *quadratic* convergence near a minimum; invariant to linear reparameterisation ⇒ **immune to ill-conditioning** ($\kappa$ disappears).
**Problems for deep nets:**
- $\mathbf H$ is $P\times P$: storage $O(P^2)$, inversion $O(P^3)$ — impossible for $P=10^8$.
- $\mathbf H$ is **indefinite** in non-convex problems ⇒ Newton steps can move *toward* saddle points. Fix: **saddle-free Newton** using $|\mathbf H|$ (absolute eigenvalues), or damping $(\mathbf H + \alpha\mathbf I)^{-1}$ (Levenberg–Marquardt).

### 12.2 Gauss–Newton and the Fisher information matrix

For a sum of squares $\mathcal L = \tfrac12\sum_i r_i^2$ with Jacobian $\mathbf J$:
$$
\mathbf H = \mathbf J^{\mathsf T}\mathbf J + \sum_i r_i\nabla^2 r_i \;\approx\; \mathbf J^{\mathsf T}\mathbf J = \mathbf G \quad\text{(Gauss–Newton, always PSD)}
$$

**Fisher information matrix:**
$$
\mathbf F = \mathbb E_{\mathbf x\sim p_{\text{data}},\,\mathbf y\sim p_\theta}\left[\nabla\log p_\theta(\mathbf y|\mathbf x)\,\nabla\log p_\theta(\mathbf y|\mathbf x)^{\mathsf T}\right]
$$
For the standard loss/activation pairings of Unit II, $\mathbf F$ equals the generalised Gauss–Newton matrix.

**Natural gradient (Amari):**
$$
\boxed{\;\theta_{k+1} = \theta_k - \eta\,\mathbf F^{-1}\mathbf g_k\;}
$$
This is steepest descent in the space of **distributions** (KL geometry) rather than parameters — invariant to reparameterisation.

### 12.3 Practical approximations

| Method | Approximation |
|---|---|
| **L-BFGS** | low-rank $\mathbf H^{-1}$ from the last $m$ (gradient, step) pairs; $O(mP)$ memory. Great for full-batch, deterministic problems; **poor with mini-batch noise** |
| **Hessian-free / truncated Newton** | solve $\mathbf H\mathbf d = -\mathbf g$ by conjugate gradients using Hessian-**vector** products $\mathbf H\mathbf v = \nabla(\mathbf g^{\mathsf T}\mathbf v)$ — never forms $\mathbf H$ |
| **K-FAC** | $\mathbf F_\ell \approx \mathbb E[\mathbf a\mathbf a^{\mathsf T}]\otimes\mathbb E[\boldsymbol\delta\boldsymbol\delta^{\mathsf T}]$ (Kronecker factorisation per layer) ⇒ invert two small matrices instead of one huge one |
| **Shampoo** | Kronecker preconditioners per tensor dimension |
| **Adam (viewed this way)** | diagonal approximation to $\mathbf F^{1/2}$ — this is why adaptive methods are called "diagonal second-order" |

---

## 13. Comparison and Practical Guidance

### 13.1 Master comparison table

| Optimizer | Update rule (core) | Extra state | Adaptive LR | Momentum | Key strength | Key weakness |
|---|---|---|---|---|---|---|
| **SGD** | $\theta \!-\! \eta\mathbf g$ | none | ✗ | ✗ | simple; best final generalization (vision) | slow, LR-sensitive, ravines |
| **Mini-batch SGD** | $\theta\!-\!\eta\bar{\mathbf g}_B$ | none | ✗ | ✗ | hardware-efficient, variance $\sigma^2/B$ | still needs tuning |
| **Momentum** | $\mathbf v\!=\!\beta\mathbf v\!+\!\mathbf g$ | $\mathbf v$ ($1\times P$) | ✗ | ✓ | $\kappa\to\sqrt\kappa$; damps oscillation | one more hyperparameter |
| **NAG** | look-ahead gradient | $\mathbf v$ | ✗ | ✓ | $O(1/K^2)$, optimal first-order | marginal gain in practice |
| **AdaGrad** | $\eta/\sqrt{\sum \mathbf g^2}$ | $\mathbf G$ | ✓ | ✗ | sparse features, convex, no LR tuning | LR $\to 0$; stalls |
| **RMSProp** | EWMA of $\mathbf g^2$ | $\mathbf E[\mathbf g^2]$ | ✓ | ✗ | fixes AdaGrad decay; great for RNNs | no momentum; unpublished |
| **AdaDelta** | RMS(Δθ)/RMS(g) | 2 accumulators | ✓ | ✗ | **no learning rate** | can be slow |
| **Adam** | $\hat m/\sqrt{\hat v}$ | $\mathbf m,\mathbf v$ ($2\times P$) | ✓ | ✓ | robust default, fast | may generalise worse; convergence gap |
| **AdamW** | Adam + decoupled decay | $\mathbf m,\mathbf v$ | ✓ | ✓ | correct weight decay | — |
| **AMSGrad** | $\max$ of $\hat v$ | $\mathbf m,\mathbf v,\hat v_{\max}$ | ✓ | ✓ | fixes convergence proof | often no empirical gain |
| **L-BFGS** | quasi-Newton | $2m$ vectors | (implicit) | — | fast on full-batch, small $P$ | breaks with mini-batch noise |

### 13.2 The unified view

Every method above has the form

$$
\boxed{\;\theta_{k+1} = \theta_k - \eta\,\mathbf P_k^{-1}\,\mathbf s_k\;}
$$

where $\mathbf s_k$ = a **search direction** (raw gradient, or momentum-smoothed) and $\mathbf P_k$ = a **preconditioner**.

| Method | $\mathbf s_k$ | $\mathbf P_k$ |
|---|---|---|
| SGD | $\mathbf g_k$ | $\mathbf I$ |
| Momentum | $\text{EWMA}(\mathbf g)$ | $\mathbf I$ |
| AdaGrad | $\mathbf g_k$ | $\mathrm{diag}\!\big(\sqrt{\textstyle\sum \mathbf g^2}\big)$ |
| RMSProp | $\mathbf g_k$ | $\mathrm{diag}\!\big(\sqrt{\text{EWMA}(\mathbf g^2)}\big)$ |
| Adam | $\text{EWMA}(\mathbf g)$ (bias-corrected) | $\mathrm{diag}\!\big(\sqrt{\text{EWMA}(\mathbf g^2)}\big)$ |
| Newton | $\mathbf g_k$ | $\mathbf H$ |
| Natural gradient | $\mathbf g_k$ | $\mathbf F$ |

**One sentence to remember:** *momentum improves the search direction; adaptive methods improve the preconditioner; Adam does both.*

### 13.3 Practical recipe

```
1. Optimizer choice
     Computer vision / CNNs   → SGD + momentum(0.9) + weight decay + cosine LR
     NLP / Transformers       → AdamW (β₁=0.9, β₂=0.999 or 0.98) + warmup + cosine
     RNNs / LSTMs             → Adam or RMSProp + gradient clipping (‖g‖ ≤ 1–5)
     GANs                     → Adam with β₁ = 0.5, β₂ = 0.999
     Sparse features / NLP embeddings → AdaGrad / sparse Adam
     Full-batch, small P, deterministic → L-BFGS
     Batch size ≥ 8k          → LARS / LAMB + warmup

2. Learning rate
     Run an LR range test; take η just below the divergence point ÷ 10.
     Typical: SGD 0.1 (with BN), Adam 1e-3, AdamW fine-tuning 2e-5 – 5e-5.

3. Schedule
     Linear warmup (1–5 % of steps) → cosine decay to ~0.

4. Batch size
     As large as memory allows, then apply the linear (or √) scaling rule.

5. Regularization
     Weight decay 1e-4 (SGD) / 0.01–0.1 (AdamW); do NOT decay biases or
     BatchNorm γ, β.

6. Stability
     Gradient clipping for RNNs/Transformers; mixed precision with loss scaling.

7. Diagnose
     loss = NaN            → LR too high, or log(0); clip and lower η
     loss flat from step 0 → LR too low, or dead units, or bad init
     loss ↓ then ↑         → LR too high late; add decay
     train ↓ val ↑         → overfitting; more regularization / data
     spiky loss            → batch too small or LR too high
```

### 13.4 Gradient clipping

$$
\text{by norm: } \mathbf g \leftarrow \mathbf g\cdot\min\!\left(1,\;\frac{c}{\|\mathbf g\|_2}\right);
\qquad
\text{by value: } g_i \leftarrow \mathrm{clip}(g_i, -c, c)
$$

Clipping by **norm** preserves the gradient direction and is preferred. Essential wherever cliffs occur (RNNs, Transformers, RL). It provably allows convergence under relaxed smoothness assumptions.

---

## 14. Applications of Neural Networks

*(CO6 also asks for applications — a compact catalogue.)*

| Domain | Task | Typical architecture |
|---|---|---|
| **Computer vision** | classification, detection, segmentation | CNN (ResNet, EfficientNet), ViT |
| | image generation, super-resolution | GAN, diffusion model, VAE |
| **NLP** | translation, summarisation, QA, dialogue | Transformer / LLM |
| | sequence labelling | BiLSTM-CRF, encoder Transformer |
| **Speech** | ASR, TTS, speaker ID | Conformer, WaveNet, RNN-T |
| **Healthcare** | diagnosis from imaging, drug discovery, protein folding | CNN, GNN, AlphaFold-style |
| **Finance** | fraud detection, algorithmic trading, credit scoring | LSTM, GBM+NN ensembles |
| **Control / robotics** | manipulation, locomotion, autonomous driving | Deep RL (PPO, SAC), imitation learning |
| **Recommender systems** | ranking, CTR prediction | embeddings + MLP, two-tower, transformers |
| **Signal / comms** | equalisation, denoising, modulation classification | RBF, CNN |
| **Science** | PDE surrogates, climate, materials | PINNs, GNNs, neural operators |
| **Associative memory / optimization** | CAM, QUBO, scheduling | Hopfield / Boltzmann (Unit III) |
| **Data mining / EDA** | clustering, visualisation, anomaly detection | SOM, autoencoder (Units IV–V) |

---

## 15. Solved Numericals

### N1. All six optimizers on the same gradient sequence

**Setup.** A single scalar parameter $\theta$, $\theta_0 = 1.0$, $\eta = 0.1$.
Observed gradients: $g_1 = 1.0$, $g_2 = 0.5$, $g_3 = 0.25$.
Hyperparameters: momentum $\beta = 0.9$; RMSProp $\rho = 0.9$; Adam $\beta_1 = 0.9$, $\beta_2 = 0.999$; $\epsilon$ negligible.

---

**(a) Plain SGD.** $\Delta\theta = -\eta g$

| $k$ | $g_k$ | $\Delta\theta$ | $\theta$ |
|---|---|---|---|
| 1 | 1.00 | $-0.1(1.00) = -0.100000$ | 0.900000 |
| 2 | 0.50 | $-0.1(0.50) = -0.050000$ | 0.850000 |
| 3 | 0.25 | $-0.1(0.25) = -0.025000$ | 0.825000 |

Total movement: $-0.175000$.

---

**(b) Momentum.** $v_k = \beta v_{k-1} + g_k$, $\Delta\theta = -\eta v_k$, $v_0 = 0$

| $k$ | $v_k$ | $\Delta\theta$ | $\theta$ |
|---|---|---|---|
| 1 | $0.9(0)+1.00 = 1.000$ | $-0.100000$ | 0.900000 |
| 2 | $0.9(1.000)+0.50 = 1.400$ | $-0.140000$ | 0.760000 |
| 3 | $0.9(1.400)+0.25 = 1.510$ | $-0.151000$ | 0.609000 |

Total: $-0.391000$ — **2.2× further than SGD**, because the consistently-signed gradients accumulate. Note $\Delta\theta$ is *growing* even though $g$ is *shrinking*: that is inertia.

---

**(c) AdaGrad.** $G_k = G_{k-1}+g_k^2$, $\Delta\theta = -\eta g_k/\sqrt{G_k}$

| $k$ | $G_k$ | $\sqrt{G_k}$ | $\Delta\theta$ | $\theta$ |
|---|---|---|---|---|
| 1 | $1.0000$ | 1.000000 | $-0.1(1.00)/1.000000 = -0.100000$ | 0.900000 |
| 2 | $1.2500$ | 1.118034 | $-0.1(0.50)/1.118034 = -0.044721$ | 0.855279 |
| 3 | $1.3125$ | 1.145644 | $-0.1(0.25)/1.145644 = -0.021822$ | 0.833457 |

Total: $-0.166543$. Steps shrink faster than SGD's — the accumulator only grows.

---

**(d) RMSProp.** $E_k = 0.9E_{k-1} + 0.1g_k^2$, $\Delta\theta = -\eta g_k/\sqrt{E_k}$, $E_0=0$

| $k$ | $E_k$ | $\sqrt{E_k}$ | $\Delta\theta$ | $\theta$ |
|---|---|---|---|---|
| 1 | $0.1(1.00) = 0.100000$ | 0.316228 | $-0.1(1.00)/0.316228 = -0.316228$ | 0.683772 |
| 2 | $0.9(0.1)+0.1(0.25) = 0.115000$ | 0.339116 | $-0.1(0.50)/0.339116 = -0.147441$ | 0.536331 |
| 3 | $0.9(0.115)+0.1(0.0625) = 0.109750$ | 0.331285 | $-0.1(0.25)/0.331285 = -0.075463$ | 0.460868 |

Total: $-0.539132$. **Huge first step** — because $E_1$ is biased toward zero (no bias correction in RMSProp). This is exactly the instability Adam's bias correction removes.

---

**(e) Adam (with bias correction).**

*Step 1:*
$$
m_1 = 0.9(0)+0.1(1.00) = 0.100000,\qquad v_1 = 0.999(0)+0.001(1.00) = 0.001000
$$
$$
\hat m_1 = \frac{0.100000}{1-0.9} = \frac{0.1}{0.1} = 1.000000,\qquad
\hat v_1 = \frac{0.001000}{1-0.999} = \frac{0.001}{0.001} = 1.000000
$$
$$
\Delta\theta = -0.1\frac{1.000000}{\sqrt{1.000000}} = \mathbf{-0.100000},\qquad \theta = 0.900000
$$

*Step 2:*
$$
m_2 = 0.9(0.100000)+0.1(0.50) = 0.090000+0.050000 = 0.140000
$$
$$
v_2 = 0.999(0.001000)+0.001(0.250000) = 0.000999+0.000250 = 0.001249
$$
$$
\hat m_2 = \frac{0.140000}{1-0.81} = \frac{0.140000}{0.190000} = 0.736842
$$
$$
\hat v_2 = \frac{0.001249}{1-0.998001} = \frac{0.001249}{0.001999} = 0.624812,\qquad \sqrt{\hat v_2} = 0.790451
$$
$$
\Delta\theta = -0.1\frac{0.736842}{0.790451} = \mathbf{-0.093217},\qquad \theta = 0.806783
$$

*Step 3:*
$$
m_3 = 0.9(0.140000)+0.1(0.25) = 0.126000+0.025000 = 0.151000
$$
$$
v_3 = 0.999(0.001249)+0.001(0.0625) = 0.001247751+0.000062500 = 0.001310251
$$
$$
\hat m_3 = \frac{0.151000}{1-0.729} = \frac{0.151000}{0.271000} = 0.557196
$$
$$
\hat v_3 = \frac{0.001310251}{1-0.997003} = \frac{0.001310251}{0.002997} = 0.437189,\qquad \sqrt{\hat v_3} = 0.661203
$$
$$
\Delta\theta = -0.1\frac{0.557196}{0.661203} = \mathbf{-0.084271},\qquad \theta = 0.722512
$$

Total: $-0.277488$.

---

**(f) Summary comparison**

| Optimizer | $\Delta\theta_1$ | $\Delta\theta_2$ | $\Delta\theta_3$ | Total | Final $\theta$ |
|---|---|---|---|---|---|
| SGD | −0.100000 | −0.050000 | −0.025000 | −0.175000 | 0.825000 |
| Momentum | −0.100000 | −0.140000 | −0.151000 | **−0.391000** | 0.609000 |
| AdaGrad | −0.100000 | −0.044721 | −0.021822 | −0.166543 | 0.833457 |
| RMSProp | **−0.316228** | −0.147441 | −0.075463 | −0.539132 | 0.460868 |
| Adam | −0.100000 | −0.093217 | −0.084271 | −0.277488 | 0.722512 |

**Reading the table.**
- **Momentum** accelerates monotonically on consistent gradients.
- **AdaGrad** decelerates fastest — the accumulator never forgets.
- **RMSProp** takes a dangerously large first step (no bias correction).
- **Adam** keeps its steps within $\approx\eta = 0.1$ throughout — the bounded-step property — and decays gracefully. This stability is why Adam works out of the box.

---

### N2. Adam without bias correction

Same first step, but skipping the correction:
$$
\Delta\theta = -\eta\frac{m_1}{\sqrt{v_1}} = -0.1\frac{0.100000}{\sqrt{0.001000}} = -0.1\frac{0.100000}{0.031623} = \mathbf{-0.316228}
$$

That is **3.16× larger** than the corrected step of $-0.1$ — and exactly matches RMSProp's uncorrected first step.

**How long does the bias last?** The correction factor for the second moment is $1/(1-\beta_2^k)$:

| $k$ | $\beta_2^k = 0.999^k$ | $1-\beta_2^k$ | Correction factor $1/\sqrt{1-\beta_2^k}$ |
|---|---|---|---|
| 1 | 0.999000 | 0.001000 | 31.62 |
| 10 | 0.990045 | 0.009955 | 10.02 |
| 100 | 0.904792 | 0.095208 | 3.24 |
| 1000 | 0.367695 | 0.632305 | 1.258 |
| 5000 | 0.006738 | 0.993262 | 1.003 |

**The bias persists for thousands of iterations.** Bias correction is not a cosmetic detail.

---

### N3. Learning-rate stability bound and optimal $\eta$

Quadratic loss $\mathcal L(\theta) = \tfrac12\theta^{\mathsf T}\mathbf H\theta$ with $\mathbf H = \mathrm{diag}(100,\,10,\,1)$.

**(a) Stability:** need $|1-\eta\lambda_i|<1$ for all $i$:
$$
\eta < \frac{2}{\lambda_{\max}} = \frac{2}{100} = \mathbf{0.02}
$$
With $\eta = 0.025$: $|1-0.025(100)| = |1-2.5| = 1.5 > 1$ ⇒ **diverges** along the first coordinate.

**(b) Optimal $\eta$:**
$$
\eta^\star = \frac{2}{\lambda_{\min}+\lambda_{\max}} = \frac{2}{101} = \mathbf{0.019802}
$$

**(c) Per-coordinate contraction at $\eta^\star$:**

| $\lambda_i$ | $1-\eta^\star\lambda_i$ |
|---|---|
| 100 | $1-1.980198 = -0.980198$ |
| 10 | $1-0.198020 = 0.801980$ |
| 1 | $1-0.019802 = 0.980198$ |

The extremes both have $|1-\eta\lambda| = 0.980198 = \dfrac{\kappa-1}{\kappa+1} = \dfrac{99}{101}$ ✓ (the minimax property), while the middle eigenvalue converges much faster.

**(d) Iterations for a $10^{-3}$ reduction:**
$$
0.980198^{K} = 10^{-3} \;\Rightarrow\; K = \frac{\ln 10^{-3}}{\ln 0.980198} = \frac{-6.907755}{-0.020005} = 345.3 \;\Rightarrow\; \mathbf{346}
$$

---

### N4. Momentum turns $\kappa$ into $\sqrt\kappa$

Same $\mathbf H$: $\lambda_{\min}=1$, $\lambda_{\max}=100$, so $\kappa = 100$, $\sqrt\kappa = 10$.

**Optimal Polyak parameters:**
$$
\beta^\star = \left(\frac{\sqrt\kappa - 1}{\sqrt\kappa+1}\right)^{2} = \left(\frac{10-1}{10+1}\right)^2 = \left(\frac{9}{11}\right)^2 = 0.818182^2 = \mathbf{0.669421}
$$
$$
\eta^\star = \frac{4}{\left(\sqrt{\lambda_{\max}}+\sqrt{\lambda_{\min}}\right)^2} = \frac{4}{(10+1)^2} = \frac{4}{121} = \mathbf{0.033058}
$$
$$
\text{contraction rate} = \frac{\sqrt\kappa-1}{\sqrt\kappa+1} = \frac{9}{11} = \mathbf{0.818182}
$$

**Iterations for a $10^{-3}$ reduction:**
$$
K = \frac{\ln 10^{-3}}{\ln 0.818182} = \frac{-6.907755}{-0.200671} = 34.4 \;\Rightarrow\; \mathbf{35}
$$

$$
\boxed{\;\text{Speed-up} = \frac{346}{35} \approx \mathbf{9.9\times} \;\approx\; \sqrt\kappa = 10 \;✓}
$$

Note that momentum also permits a **larger** learning rate ($0.0331$ vs $0.0198$).

**Sanity check on the effective rate:** $\eta_{\text{eff}} = \eta^\star/(1-\beta^\star) = 0.033058/0.330579 = 0.100000$ — five times the stable plain-GD rate, which is exactly the acceleration mechanism.

---

### N5. Mini-batch variance and the scaling rules

Per-sample gradient variance $\sigma^2 = 4.0$ (per coordinate).

| $B$ | $\mathrm{Var} = \sigma^2/B$ | SD | Cost/step | Noise per unit cost |
|---|---|---|---|---|
| 1 | 4.000 | 2.000 | 1 | 2.000 |
| 4 | 1.000 | 1.000 | 4 | 2.000 |
| 16 | 0.250 | 0.500 | 16 | 2.000 |
| 64 | 0.0625 | 0.250 | 64 | 2.000 |
| 256 | 0.015625 | 0.125 | 256 | 2.000 |

**Noise per unit compute is constant** — larger batches buy accuracy at exactly proportional cost. The benefit is entirely from hardware parallelism, not from statistics.

**Linear scaling.** Baseline $B = 64$, $\eta = 0.1$. Scale to $B = 1024$ (16×):
$$
\eta_{\text{new}} = 0.1 \times 16 = \mathbf{1.6},\qquad \text{with warmup over } \approx 5 \text{ epochs.}
$$
**Square-root scaling** would give $\eta_{\text{new}} = 0.1\sqrt{16} = 0.4$ — the safer choice for Adam.

**Epoch arithmetic.** $N = 50\,000$, $B = 128$ ⇒ $\lceil 50000/128\rceil = \mathbf{391}$ updates per epoch. For 100 epochs: $39\,100$ updates. With 5-epoch warmup, the warmup lasts $5\times391 = 1955$ steps.

---

### N6. Momentum's effective learning rate

$\eta = 0.01$.

| $\beta$ | $\eta_{\text{eff}} = \eta/(1-\beta)$ | Window $1/(1-\beta)$ |
|---|---|---|
| 0.0 | 0.0100 | 1 |
| 0.5 | 0.0200 | 2 |
| 0.9 | **0.1000** | 10 |
| 0.95 | 0.2000 | 20 |
| 0.99 | 1.0000 | 100 |

**Practical implication.** Switching from $\beta = 0.9$ to $\beta = 0.99$ multiplies the effective step by 10 — if you do this without lowering $\eta$ by ~10×, training will usually diverge. This is the single most common momentum-tuning mistake.

**Verification of the geometric sum.** With $\beta = 0.9$ and constant $g = 1$:
$$
v_1 = 1,\; v_2 = 1.9,\; v_3 = 2.71,\; v_5 = 4.0951,\; v_{10} = 6.5132,\; v_{50} = 9.9485,\; v_\infty = \frac{1}{1-0.9} = 10 \;✓
$$

---

### N7. AdaGrad's decay vs RMSProp's stability (10 steps of constant gradient)

Constant $g = 1$, $\eta = 0.1$, $\rho = 0.9$.

| $k$ | AdaGrad $G_k = k$ | AdaGrad step $\eta/\sqrt k$ | RMSProp $E_k = 1-0.9^k$ | RMSProp step $\eta/\sqrt{E_k}$ |
|---|---|---|---|---|
| 1 | 1 | 0.100000 | 0.100000 | 0.316228 |
| 2 | 2 | 0.070711 | 0.190000 | 0.229416 |
| 5 | 5 | 0.044721 | 0.409510 | 0.156269 |
| 10 | 10 | 0.031623 | 0.651322 | 0.123920 |
| 100 | 100 | 0.010000 | 0.999973 | 0.100001 |
| 10 000 | 10 000 | **0.001000** | 1.000000 | **0.100000** |

**AdaGrad's step has decayed 100× by iteration $10^4$ and keeps falling as $1/\sqrt k$; RMSProp's has converged to the constant $\eta$.** For a 10-epoch run this is fine; for a 300-epoch deep-network run AdaGrad simply stops learning. This table *is* the answer to "why was RMSProp invented?".

*(Derivation of the RMSProp column: with constant $g^2=1$ and $E_0=0$, $E_k = 1 - \rho^k$.)*

---

### N8. Adam step-size bound

Show $\left|\eta\,\hat m/\sqrt{\hat v}\right|$ is bounded.

Worst case: a single non-zero gradient $g$ at step 1, zeros thereafter. At step $k$:
$$
m_k = (1-\beta_1)\beta_1^{k-1}g,\quad v_k = (1-\beta_2)\beta_2^{k-1}g^2
$$
$$
\frac{\hat m_k}{\sqrt{\hat v_k}} = \frac{(1-\beta_1)\beta_1^{k-1}g/(1-\beta_1^k)}{\sqrt{(1-\beta_2)\beta_2^{k-1}g^2/(1-\beta_2^k)}}
$$

At $k=1$ this equals $\dfrac{g}{|g|} = \pm 1$, so $|\Delta\theta| = \eta$.

The general bound (Kingma & Ba, Lemma):
$$
\left|\frac{\hat m_k}{\sqrt{\hat v_k}}\right| \le \frac{1-\beta_1}{\sqrt{1-\beta_2}}
= \frac{0.1}{\sqrt{0.001}} = \frac{0.1}{0.031623} = 3.162 \quad (\text{when } \beta_1 > \sqrt{\beta_2}\text{ fails})
$$
and $\le 1$ in the usual regime $\beta_1 < \sqrt{\beta_2}$ ($0.9 < \sqrt{0.999}=0.9995$ ✓, so the bound is **1**).

$$
\boxed{\;|\Delta\theta| \le \eta = 0.001 \;\text{(default)}\;}
$$

**This is Adam's "trust region": no parameter can move more than $\eta$ per step**, regardless of gradient magnitude — which is why Adam rarely explodes and why $\eta$ has an interpretable scale (roughly "how far a weight may move per update").

---

### N9. Cosine annealing schedule

$\eta_0 = 0.1$, $\eta_{\min} = 0$, total $K = 100$ epochs.
$$
\eta_k = \frac{\eta_0}{2}\left(1+\cos\frac{k\pi}{K}\right)
$$

| $k$ | $k\pi/K$ (rad) | $\cos$ | $\eta_k$ |
|---|---|---|---|
| 0 | 0 | 1.0000 | 0.10000 |
| 10 | 0.3142 | 0.9511 | 0.09755 |
| 25 | 0.7854 | 0.7071 | 0.08536 |
| 50 | 1.5708 | 0.0000 | 0.05000 |
| 75 | 2.3562 | −0.7071 | 0.01464 |
| 90 | 2.8274 | −0.9511 | 0.00245 |
| 100 | $\pi$ | −1.0000 | 0.00000 |

**Shape:** slow decay early (lets the model explore), rapid decay in the middle, very slow decay at the end (fine polishing near the minimum). Compare with a step schedule ($\times 0.1$ at epochs 30, 60, 90), which gives the same coarse behaviour but with abrupt loss "cliffs".

**With 5-epoch linear warmup prepended:** $\eta_k = 0.1\,k/5$ for $k<5$, giving $0, 0.02, 0.04, 0.06, 0.08$, then the cosine curve from $k=5$.

---

### N10. Escaping a saddle point

Consider $\mathcal L(\theta_1,\theta_2) = \theta_1^2 - \theta_2^2$ (a canonical saddle at the origin). $\mathbf H = \mathrm{diag}(2,-2)$.

Start at $\theta = (1.0,\;10^{-6})$ — almost exactly on the stable manifold. $\eta = 0.1$.

$$
\mathbf g = (2\theta_1,\; -2\theta_2)
$$
$$
\theta_1^{(k)} = (1-0.2)^k(1.0) = 0.8^k,\qquad \theta_2^{(k)} = (1+0.2)^k(10^{-6}) = 1.2^k\times10^{-6}
$$

| $k$ | $\theta_1$ | $\theta_2$ | $\mathcal L$ | $\Vert \mathbf g\Vert $ |
|---|---|---|---|---|
| 0 | 1.000000 | $1.0\times10^{-6}$ | 1.000000 | 2.000 |
| 10 | 0.107374 | $6.19\times10^{-6}$ | 0.011529 | 0.2147 |
| 30 | 0.001238 | $2.37\times10^{-4}$ | $1.48\times10^{-6}$ | 0.00252 |
| 50 | $1.43\times10^{-5}$ | $9.10\times10^{-3}$ | $-8.28\times10^{-5}$ | 0.0182 |
| 76 | $1.06\times10^{-8}$ | 1.0083 | $-1.0167$ | 2.017 |

**Reading it.** For ~40 iterations the iterate sits on a near-zero-gradient **plateau** ($\|\mathbf g\| \approx 0.0025$ at $k=30$) with the loss barely moving — this is the saddle trap. Only the exponential growth $1.2^k$ of the tiny initial perturbation eventually eject it.

**Estimated escape time:** need $1.2^k\times10^{-6} \approx 1 \Rightarrow k = \dfrac{\ln 10^6}{\ln 1.2} = \dfrac{13.8155}{0.182322} = \mathbf{75.8}$ iterations.

**If the initial perturbation were $10^{-12}$** (e.g. a smaller initialisation): $k = \dfrac{27.631}{0.182322} = 151.6$ — **twice as long**. Escape time grows *logarithmically* in $1/\|\text{perturbation}\|$. SGD's noise continually re-injects perturbation of size $O(\eta\sigma)$, so **stochastic** methods escape in $O(\log(1/\eta\sigma))$ steps rather than potentially forever. This is the formal answer to "why does SGD escape saddles?".

---

### N11. Gradient clipping

Clipping threshold $c = 1.0$, gradient $\mathbf g = (3.0,\,4.0)$, so $\|\mathbf g\|_2 = 5.0$.

**By norm:**
$$
\mathbf g \leftarrow \mathbf g\cdot\min\!\left(1,\frac{1.0}{5.0}\right) = 0.2(3.0,\,4.0) = \mathbf{(0.6,\,0.8)},\qquad \|\mathbf g\| = 1.0 \;✓
$$
Direction preserved: $(0.6,0.8)$ is parallel to $(3,4)$ ✓

**By value:**
$$
\mathbf g \leftarrow (\mathrm{clip}(3,-1,1),\;\mathrm{clip}(4,-1,1)) = \mathbf{(1.0,\,1.0)},\qquad \|\mathbf g\| = 1.414
$$
Direction **changed** from a $53.1°$ angle to $45°$ — a distortion. **Prefer clipping by norm.**

---

## 16. Viva / Exam Pointers

**Likely long questions**
1. Derive the descent lemma and the condition $\eta < 2/L$; give the convergence rate of GD.
2. State the Robbins–Monro conditions and explain which schedules satisfy them.
3. Derive the effective learning rate $\eta/(1-\beta)$ for momentum; explain why momentum improves $\kappa \to \sqrt\kappa$.
4. Explain AdaGrad's update, its advantage for sparse features, and its fatal flaw; show how RMSProp fixes it.
5. Write the full Adam algorithm and **derive the bias-correction terms** $\hat m = m/(1-\beta_1^k)$, $\hat v = v/(1-\beta_2^k)$.
6. Compare all six optimizers in a table and place them in the unified $\theta - \eta\mathbf P^{-1}\mathbf s$ framework.
7. Explain why saddle points, not local minima, dominate high-dimensional loss landscapes.
8. Explain mini-batch variance $\sigma^2/B$ and the linear scaling rule with warmup.
9. Compute the updates of SGD/Momentum/AdaGrad/RMSProp/Adam for a given gradient sequence (numerical).

**Traps**
- AdaGrad's accumulator is a **cumulative sum**; RMSProp's is an **EWMA**. That single difference is the entire distinction.
- Bias correction exists **only in Adam**, not in RMSProp — and it matters for hundreds of steps because $\beta_2 = 0.999$.
- $\eta_{\text{eff}} = \eta/(1-\beta)$: raising $\beta$ *requires* lowering $\eta$.
- Adam's step is bounded by $\approx\eta$ — this is a feature, not a coincidence.
- Momentum is *not* an adaptive learning rate; it changes the **direction**, not the per-parameter scale. Adam does both.
- AdamW $\ne$ Adam + $L_2$: the decay must be **decoupled** from the adaptive denominator.
- Gradient clipping by **norm** preserves direction; by **value** it does not.
- SGD's noise is not purely a nuisance — it is the mechanism for escaping saddles and finding flat minima.

**One-line formula sheet**

$$
\text{SGD: } \theta \leftarrow \theta - \eta\mathbf g \;\;|\;\;
\text{Descent: } \eta < 2/L \;\;|\;\;
\text{Stability: } 0<\eta<2/\lambda_{\max} \;\;|\;\;
\eta^\star = \tfrac{2}{\lambda_{\min}+\lambda_{\max}}
$$
$$
\text{Momentum: } \mathbf v \leftarrow \beta\mathbf v + \mathbf g,\;\; \theta\leftarrow\theta-\eta\mathbf v,\;\; \eta_{\text{eff}} = \tfrac{\eta}{1-\beta},\;\; \beta^\star = \left(\tfrac{\sqrt\kappa-1}{\sqrt\kappa+1}\right)^2
$$
$$
\text{NAG: } \mathbf v\leftarrow\beta\mathbf v + \nabla\mathcal L(\theta - \eta\beta\mathbf v) \;\;|\;\;
\text{AdaGrad: } \theta\leftarrow\theta - \tfrac{\eta}{\sqrt{\sum\mathbf g^2}+\epsilon}\odot\mathbf g
$$
$$
\text{RMSProp: } E \leftarrow \rho E + (1-\rho)\mathbf g^2,\;\; \theta\leftarrow\theta-\tfrac{\eta}{\sqrt E + \epsilon}\odot\mathbf g
$$
$$
\text{Adam: } \mathbf m\leftarrow\beta_1\mathbf m + (1{-}\beta_1)\mathbf g,\;\;
\mathbf v\leftarrow\beta_2\mathbf v + (1{-}\beta_2)\mathbf g^2,\;\;
\hat{\mathbf m}=\tfrac{\mathbf m}{1-\beta_1^k},\;\;
\hat{\mathbf v}=\tfrac{\mathbf v}{1-\beta_2^k}
$$
$$
\theta\leftarrow\theta - \eta\tfrac{\hat{\mathbf m}}{\sqrt{\hat{\mathbf v}}+\epsilon}
\;\;|\;\;
\mathrm{Var}(\hat{\mathbf g}_B) = \tfrac{\sigma^2}{B}
\;\;|\;\;
\eta_k = \tfrac{\eta_0}{2}\left(1+\cos\tfrac{k\pi}{K}\right)
$$

---

*Previous: [Unit V](./Unit-5.md) · Back to [Unit I](./Unit-1.md)*
