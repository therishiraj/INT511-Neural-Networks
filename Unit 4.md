# Unit IV — Dimensionality Reduction Techniques

**INT511 – Neural Networks (M.Tech)**
**This unit is designed for Weeks 7–8 of the semester (6 lecture sessions).**

Real-world data often has *hundreds or thousands* of features (columns) — pixels in an image, genes in a biological sample, words in a document. This unit is about techniques that compress this data down to a much smaller number of meaningful features, while losing as little useful information as possible.

---

## How this unit is paced

| Lecture | Topic |
|---|---|
| 19 | Why reduce dimensions? The curse of dimensionality |
| 20 | Principal Component Analysis (PCA) — the core idea |
| 21 | PCA — full worked numerical example, step by step |
| 22 | Linear Discriminant Analysis (LDA) — with worked example |
| 23 | Kernel PCA — handling non-linear data |
| 24 | Independent Component Analysis (ICA) + unit wrap-up |

---

## Lecture 19: Why Reduce Dimensions?

### 19.1 The problem with too many features

Suppose each data point has 1,000 features (dimensions). This creates several practical problems:

1. **Storage and computation cost:** more features mean more memory and slower processing.
2. **The curse of dimensionality:** as the number of dimensions grows, the "space" the data lives in grows *exponentially* larger, so your data points become extremely spread out and sparse. Imagine trying to fill a 2D square with 100 points — it looks reasonably dense. Now imagine spreading those same 100 points across a 1000-dimensional cube — they become almost impossibly far apart from each other. This makes it much harder for machine learning models to find reliable patterns, because there's rarely enough data to "cover" such a huge space well.
3. **Redundant/correlated features:** many features often carry overlapping information (e.g., "height in cm" and "height in inches"), so a lot of the data's dimensionality is unnecessary.
4. **Difficulty visualizing:** we can easily plot data in 2D or 3D, but not in 1000 dimensions — reducing dimensions lets us actually *see* the structure of our data.

### 19.2 What dimensionality reduction does

**Goal:** take data with a large number of features (D dimensions) and represent it using a much smaller number of features (d dimensions, where d is much less than D), while keeping as much of the important information/structure as possible.

```
   Original data                  Reduced data
   (D = 1000 features)    --->    (d = 2 or 3 features)
   [x1, x2, ..., x1000]           [z1, z2]
```

This unit covers four popular techniques: **PCA**, **LDA**, **Kernel PCA**, and **ICA** — each with a different idea of "what information is important to keep."

**Takeaway for Lecture 19:** High-dimensional data is expensive to store, hard to learn from (curse of dimensionality), and impossible to visualize directly — dimensionality reduction fixes all three problems.

---

## Lecture 20: Principal Component Analysis (PCA) — The Core Idea

### 20.1 The main idea, in plain words

PCA looks at a dataset and asks: **"What direction, if I drew a line through my data, would capture the most spread-out (most varying) information?"** That direction is called the **first principal component**. Then it asks the same question again, but this time restricted to directions perpendicular (at 90°) to the first one — that gives the **second principal component**, and so on.

**Why maximum variance?** A direction along which the data varies a lot carries a lot of *information* — it tells us how the data points differ from each other. A direction along which the data barely changes carries very little useful information (all points look almost the same along that direction), so we can safely ignore it.

```
        y
        │        ●
        │      ●   ●              PC1: the direction of
        │    ●   ●   ●             maximum spread (variance)
        │  ●   ●   ●        ---->  (data varies a LOT along this line)
        │    ●   ●
        │──────────────── x
                                   PC2: perpendicular to PC1
                                   (data varies much LESS along this direction)
```

### 20.2 Two ideas you must know before computing PCA: variance and covariance

**Variance** measures how spread out a single feature's values are around its mean:
```
Variance(x) = (1/(n-1)) × Σ (xi − mean_x)²
```

**Covariance** measures how *two* features change together:
```
Covariance(x,y) = (1/(n-1)) × Σ (xi − mean_x)(yi − mean_y)
```
- Positive covariance: when x is above its mean, y also tends to be above its mean (they move together).
- Negative covariance: when x is above its mean, y tends to be below its mean (they move oppositely).
- Covariance near zero: no clear linear relationship.

For data with multiple features, we collect all the variances and covariances into a single table called the **covariance matrix**. For 2 features (x, y), it looks like:
```
Covariance Matrix =  [ Var(x)      Cov(x,y) ]
                      [ Cov(x,y)   Var(y)   ]
```

### 20.3 What are eigenvalues and eigenvectors, in plain terms?

For any square matrix like our covariance matrix, an **eigenvector** is a special direction that the matrix does not "rotate" — it only stretches or shrinks it. The amount of stretching is called the **eigenvalue**.

**In PCA, this has a beautiful meaning:** the eigenvectors of the covariance matrix are exactly the principal component directions, and the eigenvalues tell us exactly how much variance (spread/information) is captured along each of those directions. The eigenvector with the *largest* eigenvalue is PC1 (the most informative direction); the next largest is PC2, and so on.

### 20.4 The PCA algorithm, in simple steps

1. **Centre the data:** subtract the mean of each feature, so the data is centred around the origin (0,0).
2. **Compute the covariance matrix** of the centred data.
3. **Find the eigenvalues and eigenvectors** of the covariance matrix.
4. **Sort the eigenvectors** by their eigenvalues, from largest to smallest.
5. **Choose the top d eigenvectors** (the ones with the largest eigenvalues) — these are your new "axes."
6. **Project the original data** onto these new axes to get the reduced-dimension representation.

**Takeaway for Lecture 20:** PCA finds new axes (eigenvectors of the covariance matrix) that capture the most variance (spread) in the data, ranked by their eigenvalues, and lets us keep only the most informative few.

---

## Lecture 21: PCA — Full Worked Numerical Example

Let's apply all 6 steps from Lecture 20 to a tiny, concrete dataset, entirely by hand.

**Our dataset (4 points, 2 features each):**

| Point | x | y |
|---|---|---|
| P1 | 1 | 2 |
| P2 | 3 | 3 |
| P3 | 3 | 5 |
| P4 | 5 | 4 |

### Step 1 — Centre the data

**Compute the mean of each column:**
```
mean_x = (1+3+3+5)/4 = 12/4 = 3
mean_y = (2+3+5+4)/4 = 14/4 = 3.5
```

**Subtract the mean from each point:**

| Point | x_centered | y_centered |
|---|---|---|
| P1 | 1−3 = −2 | 2−3.5 = −1.5 |
| P2 | 3−3 = 0 | 3−3.5 = −0.5 |
| P3 | 3−3 = 0 | 5−3.5 = 1.5 |
| P4 | 5−3 = 2 | 4−3.5 = 0.5 |

### Step 2 — Compute the covariance matrix

**Var(x):**
```
Var(x) = [(−2)² + 0² + 0² + 2²] / (4−1)
       = [4 + 0 + 0 + 4] / 3
       = 8/3 ≈ 2.667
```

**Var(y):**
```
Var(y) = [(−1.5)² + (−0.5)² + (1.5)² + (0.5)²] / 3
       = [2.25 + 0.25 + 2.25 + 0.25] / 3
       = 5/3 ≈ 1.667
```

**Cov(x,y):**
```
Cov(x,y) = [(−2)(−1.5) + (0)(−0.5) + (0)(1.5) + (2)(0.5)] / 3
         = [3 + 0 + 0 + 1] / 3
         = 4/3 ≈ 1.333
```

**Covariance matrix:**
```
Σ = [ 2.667   1.333 ]
    [ 1.333   1.667 ]
```

### Step 3 — Find the eigenvalues

For a 2×2 matrix [[a,b],[b,d]], the eigenvalues are found by solving:
```
(a − λ)(d − λ) − b² = 0
```
Substituting a=2.667, d=1.667, b=1.333:
```
(2.667 − λ)(1.667 − λ) − (1.333)² = 0
λ² − 4.334λ + (2.667×1.667 − 1.777) = 0
λ² − 4.334λ + (4.445 − 1.777) = 0
λ² − 4.334λ + 2.668 = 0
```

**Using the quadratic formula, λ = [4.334 ± √(4.334² − 4×2.668)] / 2:**
```
4.334² = 18.784
4×2.668 = 10.672
18.784 − 10.672 = 8.112
√8.112 ≈ 2.848

λ = (4.334 ± 2.848) / 2
λ1 = (4.334 + 2.848)/2 = 7.182/2 = 3.591
λ2 = (4.334 − 2.848)/2 = 1.486/2 = 0.743
```

**Quick sanity check:** the sum of eigenvalues should equal the sum of the diagonal (the "trace") of Σ:
```
λ1 + λ2 = 3.591 + 0.743 = 4.334 = 2.667 + 1.667 ✓
```

### Step 4 — Find the eigenvectors

**Eigenvector for λ1 = 3.591** (this will be our PC1 — the most important direction): solve (Σ − λ1·I)v = 0:
```
(2.667 − 3.591)v1 + 1.333·v2 = 0
−0.924·v1 + 1.333·v2 = 0
v2 = (0.924/1.333)·v1 = 0.693·v1
```
So the direction is (1, 0.693). Normalize it (divide by its length) to get a unit vector:
```
length = √(1² + 0.693²) = √(1 + 0.480) = √1.480 ≈ 1.217
PC1 direction ≈ (1/1.217, 0.693/1.217) ≈ (0.822, 0.570)
```

**Eigenvector for λ2 = 0.743** (this will be PC2):
```
(2.667 − 0.743)v1 + 1.333·v2 = 0
1.924·v1 + 1.333·v2 = 0
v2 = −(1.924/1.333)·v1 = −1.443·v1
```
Direction is (1, −1.443). Normalize:
```
length = √(1² + 1.443²) = √(1 + 2.082) = √3.082 ≈ 1.756
PC2 direction ≈ (1/1.756, −1.443/1.756) ≈ (0.570, −0.822)
```

**Check:** PC1 and PC2 should be perpendicular — their dot product should be (close to) zero:
```
(0.822 × 0.570) + (0.570 × −0.822) = 0.469 − 0.469 = 0.000 ✓
```

### Step 5 — How much information does PC1 capture?

```
Explained variance ratio of PC1 = λ1 / (λ1 + λ2) = 3.591 / 4.334 ≈ 0.829
```
**PC1 alone captures about 82.9% of the total variation in the data** — so if we only keep PC1 and drop PC2, we still retain most of the useful information, while cutting our number of features in half (from 2 down to 1).

### Step 6 — Project the data onto PC1 (reduce from 2D to 1D)

To project a centred point onto PC1, we simply take the dot product of the point with the PC1 direction vector (0.822, 0.570):

**Point P1 (centred: −2, −1.5):**
```
projection = (−2 × 0.822) + (−1.5 × 0.570) = −1.644 − 0.855 = −2.499
```

**Point P2 (centred: 0, −0.5):**
```
projection = (0 × 0.822) + (−0.5 × 0.570) = 0 − 0.285 = −0.285
```

**Point P3 (centred: 0, 1.5):**
```
projection = (0 × 0.822) + (1.5 × 0.570) = 0 + 0.855 = 0.855
```

**Point P4 (centred: 2, 0.5):**
```
projection = (2 × 0.822) + (0.5 × 0.570) = 1.644 + 0.285 = 1.929
```

**Final reduced (1D) dataset:** P1 → −2.499, P2 → −0.285, P3 → 0.855, P4 → 1.929

We've successfully reduced our data from 2 dimensions to 1, while keeping about 83% of the original information!

**Takeaway for Lecture 21:** Following the same six steps — centre, covariance, eigenvalues, eigenvectors, sort, project — will work for *any* PCA problem, no matter how many original dimensions you start with.

---

## Lecture 22: Linear Discriminant Analysis (LDA)

### 22.1 How LDA is different from PCA

PCA doesn't care about class labels at all — it just looks for the direction of maximum spread in the whole dataset. But what if our real goal is **classification**, and the direction of maximum overall spread is *not* the best direction for telling two classes apart?

**LDA's idea:** find a direction where, after projecting the data onto it:
- points from the **same class** are as *close together* as possible (small within-class spread), and
- points from **different classes** are as *far apart* as possible (large between-class separation).

```
   PCA might choose this direction         LDA chooses THIS direction
   (max overall spread, but classes           (classes clearly separated,
    overlap badly):                            even though overall spread
                                                is smaller)
   ●○●○●○●○  <- classes mixed              ●●●●        ○○○○
```

### 22.2 The two key quantities: within-class and between-class scatter

- **Within-class scatter (S_W):** measures how spread out each class is around its own mean. We want this to be *small*.
- **Between-class scatter (S_B):** measures how far apart the class means are from each other. We want this to be *large*.

**LDA's goal (Fisher's criterion), in words:** find a projection direction w that makes the ratio (between-class scatter) / (within-class scatter) as large as possible.

### 22.3 Step-by-step worked example (two classes, two features)

**Class 1 points:** (4, 1) and (2, 4)
**Class 2 points:** (4, 6) and (6, 4)

**Step 1 — Compute the mean of each class:**
```
mean1 = ((4+2)/2, (1+4)/2) = (3, 2.5)
mean2 = ((4+6)/2, (6+4)/2) = (5, 5)
```

**Step 2 — Compute the within-class scatter for Class 1:**

For each point, compute (point − mean1) and then its "outer product" matrix:

Point (4,1): diff = (1, −1.5)
```
outer product = [ 1×1     1×(−1.5) ]  = [ 1     −1.5 ]
                [ −1.5×1  −1.5×(−1.5)]   [ −1.5   2.25]
```
Point (2,4): diff = (−1, 1.5) — gives the exact same outer product matrix (since it's just the negative of the previous difference vector, and multiplying two negatives gives the same positive products):
```
= [ 1     −1.5 ]
  [ −1.5   2.25]
```
**Sum for Class 1:**
```
S_W(class1) = [ 2     −3   ]
              [ −3     4.5 ]
```

**Step 3 — Compute the within-class scatter for Class 2:**

Point (4,6): diff = (−1, 1)
```
outer product = [ 1   −1 ]
                [ −1   1 ]
```
Point (6,4): diff = (1, −1) — again gives the same matrix:
```
= [ 1   −1 ]
  [ −1   1 ]
```
**Sum for Class 2:**
```
S_W(class2) = [ 2   −2 ]
              [ −2   2 ]
```

**Step 4 — Total within-class scatter:**
```
S_W = S_W(class1) + S_W(class2) = [ 2+2    −3−2  ]  = [  4   −5  ]
                                    [ −3−2   4.5+2 ]    [ −5   6.5 ]
```

**Step 5 — Compute the direction w using w = S_W⁻¹ × (mean1 − mean2)**

First, mean1 − mean2 = (3−5, 2.5−5) = (−2, −2.5)

**Invert S_W:** for a 2×2 matrix [[a,b],[c,d]], the inverse is (1/(ad−bc)) × [[d,−b],[−c,a]].
```
determinant = (4)(6.5) − (−5)(−5) = 26 − 25 = 1
S_W⁻¹ = (1/1) × [ 6.5    5  ]  = [ 6.5   5 ]
                  [ 5      4 ]    [ 5     4 ]
```

**Multiply S_W⁻¹ by (mean1 − mean2):**
```
w1 = (6.5 × −2) + (5 × −2.5) = −13 − 12.5 = −25.5
w2 = (5 × −2)   + (4 × −2.5) = −10 − 10   = −20
```
So w = (−25.5, −20). Since only the *direction* matters (not the sign), we can flip both signs for convenience: **w = (25.5, 20)**.

**Step 6 — Project all four points onto w, and check that the classes separate**

```
Point (4,1):  projection = 25.5×4 + 20×1 = 102 + 20 = 122
Point (2,4):  projection = 25.5×2 + 20×4 = 51 + 80  = 131
Point (4,6):  projection = 25.5×4 + 20×6 = 102 + 120 = 222
Point (6,4):  projection = 25.5×6 + 20×4 = 153 + 80  = 233
```

**Result:** Class 1's projected values are 122 and 131 (average ≈ 126.5); Class 2's projected values are 222 and 233 (average ≈ 227.5). **The two classes are now very clearly separated** on this single new axis — exactly what LDA is designed to achieve.

**Takeaway for Lecture 22:** LDA uses class labels to find the one direction that best *separates* classes (large between-class spread, small within-class spread), which can work better for classification than PCA's label-blind "maximum overall variance" direction.

---

## Lecture 23: Kernel PCA — Handling Non-Linear Data

### 23.1 When plain PCA fails

PCA only draws **straight-line** axes. But some datasets have structure that no straight line can capture. The classic example is data arranged in **two concentric circles** (one class forms an inner ring, another class forms an outer ring):

```
        Outer ring (Class B): ● ● ● ● ● ● ● ●
                             ●                 ●
                            ●     Inner ring     ●
                            ●     (Class A):      ●
                            ●      ○ ○ ○ ○         ●
                             ●                    ●
                              ● ● ● ● ● ● ● ●
```

No straight line (and therefore no ordinary PCA axis) can separate the inner ring from the outer ring — they are tangled together in a circular pattern.

### 23.2 The "kernel trick" idea

**The idea:** first transform the data into a new, higher-dimensional space where the circular pattern *becomes* a straight-line-separable pattern, and then perform ordinary PCA in that new space.

The problem is that this transformation, φ(x), can map data into a *very* high (sometimes infinite) number of dimensions — computing it directly would be far too expensive.

**The trick:** it turns out we never actually need to compute φ(x) itself! We only ever need the **dot product** φ(x)·φ(y) between pairs of transformed points (this is all PCA's maths requires). And for many useful transformations, there's a shortcut formula — called a **kernel function** — that computes this same dot product directly from the *original* x and y, without ever forming φ(x) or φ(y) explicitly.

```
Expensive way:   x, y  --> φ(x), φ(y)  (map to huge space)  --> compute φ(x)·φ(y)
Kernel trick:    x, y  --> plug directly into kernel formula K(x,y)  --> SAME answer, much faster!
```

### 23.3 A worked numerical example proving the trick really works

Let's use the popular **polynomial kernel** of degree 2:
```
K(x, y) = (x·y + 1)²
```

**Take x = (1, 2) and y = (2, 1).**

**Method 1 — the fast kernel-trick way:**
```
x·y = (1×2) + (2×1) = 2 + 2 = 4
K(x,y) = (4 + 1)² = 5² = 25
```

**Method 2 — the slow, "explicit mapping" way, just to double-check:**

The degree-2 polynomial kernel corresponds to explicitly mapping each 2D point into this 6-dimensional space:
```
φ(x1, x2) = (x1², √2·x1·x2, x2², √2·x1, √2·x2, 1)
```

For x = (1,2): φ(x) = (1², √2×1×2, 2², √2×1, √2×2, 1) = (1, 2.828, 4, 1.414, 2.828, 1)

For y = (2,1): φ(y) = (2², √2×2×1, 1², √2×2, √2×1, 1) = (4, 2.828, 1, 2.828, 1.414, 1)

**Now compute the ordinary dot product φ(x)·φ(y):**
```
= (1×4) + (2.828×2.828) + (4×1) + (1.414×2.828) + (2.828×1.414) + (1×1)
= 4 + 8.00 + 4 + 4.00 + 4.00 + 1
= 25
```

**Both methods give exactly 25!** This confirms the kernel trick: we got the *same* answer as explicitly mapping to 6 dimensions and computing the dot product there, but the kernel formula K(x,y) = (x·y+1)² let us get it with almost no extra work — no need to ever build that 6-dimensional vector.

### 23.4 What Kernel PCA does with this trick

Kernel PCA replaces every dot-product calculation inside ordinary PCA with a kernel function (such as the polynomial kernel above, or the very popular **Gaussian/RBF kernel**, K(x,y) = exp(−‖x−y‖²/2σ²)). This lets PCA effectively operate in a much richer, non-linear feature space — which is exactly what's needed to "unroll" tangled data like the concentric-circles example, turning it into something that becomes linearly separable after the transformation.

**Takeaway for Lecture 23:** Kernel PCA solves PCA's "straight-lines-only" limitation by using a kernel function to implicitly work in a much higher-dimensional space — without ever paying the cost of actually building that space.

---

## Lecture 24: Independent Component Analysis (ICA) and Unit Wrap-Up

### 24.1 The "cocktail party problem"

Imagine you're at a party where two people are speaking simultaneously, and you have two microphones placed at different spots in the room. Each microphone picks up a **mixture** of both voices (just in different proportions, depending on distance). Question: **can we recover each person's original voice separately, using only the two mixed microphone recordings?**

This is exactly the kind of problem ICA solves. In general:

```
Observed signals (what microphones record) = some MIXTURE of unknown Source signals
```

### 24.2 The key assumption ICA relies on

ICA assumes that the original sources are **statistically independent** of each other (one source's value tells you nothing about the other source's value) and **non-Gaussian** (their distributions aren't simple bell curves). This is a *stronger* assumption than PCA's, which only requires sources to be *uncorrelated* — independence is a much stricter and more useful condition when the goal is to cleanly separate genuinely different, unrelated source signals.

### 24.3 A simple numerical example: mixing and unmixing

Suppose we have two source signals at one instant: **s = (s1, s2) = (2, −1)**. These get mixed by an (unknown, in real problems) mixing matrix A:

```
A = [ 1     0.5 ]
    [ 0.3   1   ]
```

**Step 1 — Compute the observed (mixed) signals: x = A × s**
```
x1 = (1 × 2) + (0.5 × −1) = 2 − 0.5 = 1.5
x2 = (0.3 × 2) + (1 × −1) = 0.6 − 1 = −0.4
```
So the microphones would record x = (1.5, −0.4) — a jumbled mixture of both original sources.

**Step 2 — If we knew the mixing matrix A, we could unmix perfectly using A⁻¹**

```
determinant of A = (1×1) − (0.5×0.3) = 1 − 0.15 = 0.85
A⁻¹ = (1/0.85) × [ 1     −0.5 ]  = [ 1.176   −0.588 ]
                   [ −0.3   1   ]    [ −0.353   1.176 ]
```

**Step 3 — Apply A⁻¹ to the observed signals to recover the sources:**
```
s1_recovered = (1.176 × 1.5) + (−0.588 × −0.4) = 1.764 + 0.235 = 1.999 ≈ 2.0 ✓
s2_recovered = (−0.353 × 1.5) + (1.176 × −0.4) = −0.530 − 0.470 = −1.000 ✓
```

We recovered the original sources (2, −1) almost exactly!

### 24.4 The catch — and what makes ICA clever

In this example, we *assumed we already knew* the mixing matrix A. In a real cocktail-party problem, **we don't know A at all** — we only have the mixed microphone recordings. This is where the real magic of ICA comes in: it estimates an unmixing matrix **W** directly from the mixed observations, by searching for the transformation that makes the *outputs* as statistically independent (and as non-Gaussian) as possible — without ever needing to know A in advance. Popular algorithms for this include **FastICA**.

### 24.5 Comparison: PCA vs LDA vs Kernel PCA vs ICA

| Technique | Uses labels? | Handles non-linear data? | What it optimises for |
|---|---|---|---|
| **PCA** | No | No (straight lines only) | Maximum overall variance |
| **LDA** | Yes | No (straight lines only) | Maximum class separability |
| **Kernel PCA** | No | Yes (via the kernel trick) | Maximum variance, in a transformed (non-linear) space |
| **ICA** | No | No (straight lines only) | Maximum statistical independence between components |

### 24.6 Unit summary

- High-dimensional data is costly to store, hard to learn from, and impossible to visualize directly — this motivates dimensionality reduction.
- **PCA** finds the directions (eigenvectors of the covariance matrix) of maximum variance, ranked by eigenvalue, and projects data onto the top few.
- **LDA** uses class labels to find the direction that best separates classes, by maximizing between-class scatter relative to within-class scatter.
- **Kernel PCA** uses the "kernel trick" to perform PCA in an implicit, much richer non-linear feature space, without ever explicitly computing that space.
- **ICA** separates mixed signals into statistically independent source components, useful for problems like the cocktail-party problem.

### 24.7 Practice questions for self-study

1. Repeat the PCA worked example (Lecture 21) with the dataset {(2,3), (4,5), (6,5), (8,7)}. Compute the mean, covariance matrix, eigenvalues, eigenvectors, and the projection of each point onto PC1.
2. In the LDA example (Lecture 22), verify the within-class scatter matrix for Class 2 by redoing the outer-product calculation by hand.
3. Using the polynomial kernel K(x,y) = (x·y + 1)², compute K for x=(2,0) and y=(1,1). What does this value represent conceptually?
4. In the ICA example (Lecture 24), suppose the mixing matrix is instead A = [[1, 0.2],[0.4, 1]] and sources are s=(3,1). Compute the observed mixture x = A×s, then find A⁻¹ and verify you recover s from x.

---

*End of Unit IV — proceed to Unit V: Advanced Neural Network Architectures*
