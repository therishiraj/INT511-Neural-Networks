# Unit VI — Optimization Techniques for Neural Networks

**INT511 – Neural Networks (M.Tech)**
**This unit is designed for Weeks 11–12 of the semester (6 lecture sessions).**

Every network we've built in this course learns by adjusting its weights to reduce error — this is done using **gradient descent** (Unit II, Lecture 11). This unit takes a closer look at gradient descent's practical variants and improvements: how big a "batch" of data to use per update, and smarter update rules (Momentum, AdaGrad, RMSprop, Adam) that converge faster and more reliably than plain gradient descent.

**Running example used throughout this unit:** to make every optimizer easy to compare, we will repeatedly minimise the same simple function:
```
f(w) = (w − 4)²          with gradient      f'(w) = 2(w − 4)
```
The true minimum is obviously at w = 4. We always start at w = 0 and watch how quickly (and how smoothly) each method gets there.

---

## How this unit is paced

| Lecture | Topic |
|---|---|
| 31 | The optimization problem — why it's harder than it looks |
| 32 | Batch, Stochastic, and Mini-Batch Gradient Descent (with worked example) |
| 33 | Momentum (with worked example) |
| 34 | AdaGrad (with worked example) |
| 35 | RMSprop (with worked example) |
| 36 | Adam + final comparison + unit and course wrap-up |

---

## Lecture 31: The Optimization Problem

### 31.1 Quick recap: what are we even doing?

Training a neural network means finding the weight values that make the loss/error function as small as possible. We do this with **gradient descent**:
```
w_new = w_old − η × (gradient of the loss with respect to w)
```
This works well in simple cases (as we saw in Unit II), but real neural network loss landscapes have a few tricky features that make plain gradient descent slower or less reliable than we'd like.

### 31.2 Why the loss landscape is tricky

**1. Ravines (elongated valleys):** imagine a valley that is steep on the sides but only gently sloped along its length, with the minimum far down at one end.

```
        Steep walls
        ╲          ╱
         ╲        ╱
          ╲      ╱     <- gentle slope along the valley floor,
           ╲    ╱          minimum is far down this direction
            ╲__╱
```
Plain gradient descent tends to bounce back and forth between the steep walls (making the loss decrease very slowly overall), instead of moving efficiently along the gentle floor toward the minimum.

**2. Saddle points:** a point where the surface goes *up* in one direction but *down* in another — like the middle of a horse's saddle. The gradient here is nearly zero, so plain gradient descent can slow to a crawl even though it hasn't actually reached a good solution yet.

**3. Flat regions (plateaus):** large areas where the slope is almost zero everywhere, so gradient descent makes almost no progress for many steps.

### 31.3 Why this matters for us

All the improved methods in this unit (Momentum, AdaGrad, RMSprop, Adam) exist specifically to deal with these three problems — either by "remembering" past gradient directions (to push through flat regions and dampen ravine oscillations), or by adjusting the step size separately for each individual weight (to handle weights that need very different amounts of correction).

**Takeaway for Lecture 31:** Real loss landscapes have ravines, saddle points, and flat regions that make plain gradient descent slow or unreliable — the rest of this unit is about smarter update rules that handle these problems better.

---

## Lecture 32: Batch, Stochastic, and Mini-Batch Gradient Descent

### 32.1 The question: how much data per update?

When training on a large dataset, we must decide how many training examples to look at before making one weight update.

| Method | Examples used per update | Behaviour |
|---|---|---|
| **Batch Gradient Descent** | All training examples | Very smooth, accurate direction, but slow and memory-hungry |
| **Stochastic Gradient Descent (SGD)** | Just 1 randomly chosen example | Very fast per step, but noisy/zig-zagging path |
| **Mini-Batch Gradient Descent** | A small group (e.g., 32, 64, or 128 examples) | Good balance — the standard choice in practice |

### 32.2 Why mini-batch wins in practice

- It's far cheaper than batch GD (you don't need to process the whole dataset for every single update).
- It gives a much steadier, less noisy direction than single-example SGD.
- It can use hardware (GPUs) very efficiently, since processing 64 examples together is barely slower than processing 1.

**This is why, unless stated otherwise, "SGD" in modern deep learning libraries almost always actually means mini-batch gradient descent.**

### 32.3 Worked example: Plain Gradient Descent on our running example

Let's run plain gradient descent on f(w) = (w−4)², starting at w = 0, with learning rate η = 0.1.

**Step 1:**
```
gradient at w=0: f'(0) = 2(0−4) = −8
w1 = 0 − 0.1×(−8) = 0 + 0.8 = 0.8
```

**Step 2:**
```
gradient at w=0.8: f'(0.8) = 2(0.8−4) = −6.4
w2 = 0.8 − 0.1×(−6.4) = 0.8 + 0.64 = 1.44
```

**Step 3:**
```
gradient at w=1.44: f'(1.44) = 2(1.44−4) = −5.12
w3 = 1.44 − 0.1×(−5.12) = 1.44 + 0.512 = 1.952
```

**Step 4:**
```
gradient at w=1.952: f'(1.952) = 2(1.952−4) = −4.096
w4 = 1.952 − 0.1×(−4.096) = 1.952 + 0.4096 = 2.3616
```

**After 4 steps: w = 2.3616** (still some way from the true minimum, 4). Notice the step size keeps shrinking (0.8, 0.64, 0.512, 0.4096) because the slope itself is shrinking as we approach the minimum. We will use this exact same starting conditions (w0=0, target=4) to compare every other method in the rest of this unit.

**Takeaway for Lecture 32:** Mini-batch gradient descent is the everyday, practical version of gradient descent used in real training; plain (batch) gradient descent, while simple, converges slowly and steadily, as our worked example shows.

---

## Lecture 33: Momentum

### 33.1 The intuition: a ball rolling downhill

Imagine rolling a heavy ball down the loss landscape instead of a weightless point sliding down. The ball builds up **speed (velocity)** as it rolls consistently in one direction, and this accumulated speed helps it:
- push through small flat patches or bumps that would stall a plain gradient-descent step, and
- move faster and more smoothly along the gentle floor of a ravine, since the sideways bouncing (steep-wall oscillation) partially cancels out over time while the forward (downhill) motion keeps adding up.

### 33.2 The momentum update rule

We keep a running "velocity" v, which is a blend of the previous velocity and the current gradient:
```
v_new = β × v_old + η × gradient
w_new = w_old − v_new
```
- β (beta) is the momentum coefficient (typically around 0.9) — how much of the old velocity we keep.
- η is the learning rate, same as before.

### 33.3 Worked example: Momentum on our running example

Using β = 0.9, η = 0.1, starting w0 = 0, v0 = 0.

**Step 1:**
```
gradient at w0=0: f'(0) = −8
v1 = 0.9×0 + 0.1×(−8) = −0.8
w1 = 0 − (−0.8) = 0.8
```
(Same as plain GD for this very first step, since the velocity started at zero.)

**Step 2:**
```
gradient at w1=0.8: f'(0.8) = −6.4
v2 = 0.9×(−0.8) + 0.1×(−6.4) = −0.72 − 0.64 = −1.36
w2 = 0.8 − (−1.36) = 0.8 + 1.36 = 2.16
```
**Compare:** plain GD was only at 1.44 after 2 steps — momentum is already ahead, at 2.16!

**Step 3:**
```
gradient at w2=2.16: f'(2.16) = 2(2.16−4) = −3.68
v3 = 0.9×(−1.36) + 0.1×(−3.68) = −1.224 − 0.368 = −1.592
w3 = 2.16 − (−1.592) = 2.16 + 1.592 = 3.752
```
**Compare:** plain GD was only at 1.952 after 3 steps — momentum is at 3.752, very close to the target of 4 already!

**Step 4:**
```
gradient at w3=3.752: f'(3.752) = 2(3.752−4) = −0.496
v4 = 0.9×(−1.592) + 0.1×(−0.496) = −1.4328 − 0.0496 = −1.4824
w4 = 3.752 − (−1.4824) = 3.752 + 1.4824 = 5.2344
```

**Interesting result: momentum overshot the target!** It reached w = 5.2344, past the true minimum of 4. This is a very important, realistic lesson: **momentum speeds up convergence, but the "built-up speed" can cause it to overshoot the minimum**, just like a fast-rolling ball can roll past the bottom of a valley and partway up the other side. In practice, this is usually not a problem — a few more steps will bring it back — but it does mean momentum can be less stable than plain gradient descent if the momentum coefficient β is set too high.

**Takeaway for Lecture 33:** Momentum accelerates convergence by "remembering" the direction of previous gradients, but this same memory can cause overshoot past the minimum.

---

## Lecture 34: AdaGrad

### 34.1 The intuition: give each weight its own learning rate

So far, every weight in the network has used the *same* learning rate η. But some weights (especially those connected to rarely-occurring/sparse features) might need bigger updates, while others (connected to frequently-occurring features) might need smaller, gentler updates. **AdaGrad (Adaptive Gradient)** automatically gives each weight its own, individually-adapted learning rate.

### 34.2 The AdaGrad update rule

We keep a running total, G, of the squares of all past gradients for this weight:
```
G_new = G_old + (gradient)²
w_new = w_old − (η / √(G_new + ε)) × gradient
```
(ε is a tiny constant, like 0.00000001, just to avoid dividing by zero — we can ignore it in our by-hand calculations.)

**Key idea:** the more a weight's gradient has historically been "active" (large or frequent), the bigger G becomes, and the smaller its effective learning rate (η/√G) becomes. This automatically slows down weights that have already received a lot of updates.

### 34.3 Worked example: AdaGrad on our running example

Using η = 1.0 (AdaGrad is typically used with a larger base learning rate than plain GD, since it gets divided down), starting w0 = 0, G0 = 0.

**Step 1:**
```
gradient at w0=0: f'(0) = −8
G1 = 0 + (−8)² = 64
step size = η/√G1 = 1.0/√64 = 1.0/8 = 0.125
update = 0.125 × (−8) = −1.0
w1 = 0 − (−1.0) = 1.0
```

**Step 2:**
```
gradient at w1=1.0: f'(1.0) = 2(1−4) = −6
G2 = 64 + (−6)² = 64 + 36 = 100
step size = 1.0/√100 = 1.0/10 = 0.1
update = 0.1 × (−6) = −0.6
w2 = 1.0 − (−0.6) = 1.6
```

**Step 3:**
```
gradient at w2=1.6: f'(1.6) = 2(1.6−4) = −4.8
G3 = 100 + (−4.8)² = 100 + 23.04 = 123.04
step size = 1.0/√123.04 = 1.0/11.093 ≈ 0.0901
update = 0.0901 × (−4.8) ≈ −0.4327
w3 = 1.6 − (−0.4327) = 2.0327
```

**Step 4:**
```
gradient at w3=2.0327: f'(2.0327) = 2(2.0327−4) = −3.9346
G4 = 123.04 + (−3.9346)² = 123.04 + 15.481 = 138.521
step size = 1.0/√138.521 = 1.0/11.7695 ≈ 0.0850
update = 0.0850 × (−3.9346) ≈ −0.3343
w4 = 2.0327 − (−0.3343) = 2.367
```

**Look at the effective step sizes across the 4 iterations: 0.125 → 0.1 → 0.0901 → 0.0850.** They keep shrinking, because G keeps accumulating and only ever grows. **This is exactly AdaGrad's well-known limitation:** if training continues for a very long time, G keeps growing without bound, and the effective learning rate keeps shrinking toward zero — eventually the weight updates become so tiny that learning effectively stops, even if the model hasn't reached a good solution yet.

**Takeaway for Lecture 34:** AdaGrad gives each weight its own adaptive learning rate based on how much that weight has historically changed — great for handling weights with very different update needs, but the ever-growing accumulator G can cause learning to stall in long training runs.

---

## Lecture 35: RMSprop

### 35.1 The fix for AdaGrad's problem

RMSprop (Root Mean Square Propagation) keeps AdaGrad's good idea (an individual, adaptive learning rate per weight) but fixes its main weakness. Instead of accumulating *all* past squared gradients forever (which only ever grows), RMSprop keeps an **exponentially decaying moving average** of squared gradients — old gradient information gradually "fades out," so the accumulator doesn't grow without bound.

### 35.2 The RMSprop update rule

```
S_new = β × S_old + (1 − β) × (gradient)²
w_new = w_old − (η / √(S_new + ε)) × gradient
```
β (typically 0.9) controls how quickly old information fades — this is very similar in spirit to the momentum coefficient, but here it's applied to the *squared* gradient (magnitude), not the gradient's direction.

### 35.3 Worked example: RMSprop on our running example

Using β = 0.9, η = 0.1, starting w0 = 0, S0 = 0.

**Step 1:**
```
gradient at w0=0: f'(0) = −8
S1 = 0.9×0 + 0.1×(−8)² = 0.1×64 = 6.4
step size = η/√S1 = 0.1/√6.4 = 0.1/2.530 ≈ 0.0395
update = 0.0395 × (−8) ≈ −0.3162
w1 = 0 − (−0.3162) = 0.3162
```

**Step 2:**
```
gradient at w1=0.3162: f'(0.3162) = 2(0.3162−4) = −7.3676
S2 = 0.9×6.4 + 0.1×(−7.3676)² = 5.76 + 0.1×54.28 = 5.76 + 5.428 = 11.188
step size = 0.1/√11.188 = 0.1/3.3448 ≈ 0.0299
update ≈ 0.0299 × (−7.3676) ≈ −0.2203
w2 = 0.3162 − (−0.2203) = 0.5365
```

**Step 3:**
```
gradient at w2=0.5365: f'(0.5365) = 2(0.5365−4) = −6.927
S3 = 0.9×11.188 + 0.1×(−6.927)² = 10.069 + 0.1×47.98 = 10.069 + 4.798 = 14.867
step size = 0.1/√14.867 = 0.1/3.8557 ≈ 0.0259
update ≈ 0.0259 × (−6.927) ≈ −0.1797
w3 = 0.5365 − (−0.1797) = 0.7162
```

**Step 4:**
```
gradient at w3=0.7162: f'(0.7162) = 2(0.7162−4) = −6.5676
S4 = 0.9×14.867 + 0.1×(−6.5676)² = 13.380 + 0.1×43.13 = 13.380 + 4.313 = 17.693
step size = 0.1/√17.693 = 0.1/4.2064 ≈ 0.0238
update ≈ 0.0238 × (−6.5676) ≈ −0.1561
w4 = 0.7162 − (−0.1561) = 0.8723
```

**Notice the pattern:** the effective step sizes across the 4 iterations are 0.0395, 0.0299, 0.0259, 0.0238 — they are shrinking gently, but they are **levelling off** rather than collapsing toward zero the way AdaGrad's did. This is exactly the benefit RMSprop provides: because old squared gradients "fade" instead of piling up forever, RMSprop can keep adapting and making reasonable progress even in very long training runs — it won't stall the way AdaGrad eventually does.

**Takeaway for Lecture 35:** RMSprop keeps AdaGrad's per-weight adaptive learning rate but replaces the ever-growing accumulator with a decaying moving average, preventing the learning-rate collapse problem — making it much more suitable for long training runs.

---

## Lecture 36: Adam — Combining the Best of Both Worlds

### 36.1 The idea: momentum + RMSprop, together

**Adam (Adaptive Moment Estimation)** combines two ideas we've already learned:
1. **Momentum's idea** — keep a moving average of the *gradient itself* (this is called the "first moment," m), to smooth out the direction of movement.
2. **RMSprop's idea** — keep a moving average of the *squared gradient* (this is called the "second moment," v), to adapt the learning rate per weight.

### 36.2 The Adam update rule

```
m_new = β1 × m_old + (1−β1) × gradient          (like momentum)
v_new = β2 × v_old + (1−β2) × gradient²          (like RMSprop)

Bias correction (needed because m and v start at zero, which biases early steps):
m_hat = m_new / (1 − β1^t)
v_hat = v_new / (1 − β2^t)

w_new = w_old − η × m_hat / (√v_hat + ε)
```
Typical default values: β1 = 0.9, β2 = 0.999. The "t" in β1^t and β2^t is simply the iteration number (1, 2, 3, …).

**Why bias correction?** Since m and v both start at exactly zero, in the very first few steps they are systematically too small (they haven't had time to "fill up" yet). Dividing by (1 − β1^t) and (1 − β2^t) exactly corrects for this, so the very first updates are properly scaled instead of being artificially tiny.

### 36.3 Worked example: Adam on our running example

Using β1 = 0.9, β2 = 0.999, η = 0.5, starting w0 = 0, m0 = 0, v0 = 0.

**Step 1 (t=1):**
```
gradient at w0=0: f'(0) = −8

m1 = 0.9×0 + 0.1×(−8) = −0.8
v1 = 0.999×0 + 0.001×(−8)² = 0.001×64 = 0.064

Bias correction:
m1_hat = m1 / (1 − 0.9¹) = −0.8 / 0.1 = −8.0
v1_hat = v1 / (1 − 0.999¹) = 0.064 / 0.001 = 64.0

update = η × m1_hat / √v1_hat = 0.5 × (−8.0) / 8.0 = 0.5 × (−1.0) = −0.5
w1 = 0 − (−0.5) = 0.5
```

**Interesting observation:** notice that after bias correction, m1_hat exactly recovers the raw gradient (−8), and v1_hat exactly recovers the raw squared gradient (64) — this is what bias correction is designed to do at the very first step. As a result, the very first Adam update always has a size close to η, regardless of how large or small the actual gradient was.

**Step 2 (t=2):**
```
gradient at w1=0.5: f'(0.5) = 2(0.5−4) = −7

m2 = 0.9×(−0.8) + 0.1×(−7) = −0.72 − 0.7 = −1.42
v2 = 0.999×0.064 + 0.001×(−7)² = 0.063936 + 0.049 = 0.112936

Bias correction:
m2_hat = m2 / (1 − 0.9²) = −1.42 / 0.19 ≈ −7.4737
v2_hat = v2 / (1 − 0.999²) = 0.112936 / 0.001999 ≈ 56.496

update = 0.5 × (−7.4737) / √56.496 = 0.5 × (−7.4737) / 7.5164 ≈ 0.5 × (−0.9943) ≈ −0.4972
w2 = 0.5 − (−0.4972) = 0.9972
```

**Step 3 (t=3):**
```
gradient at w2=0.9972: f'(0.9972) = 2(0.9972−4) = −6.0056

m3 = 0.9×(−1.42) + 0.1×(−6.0056) = −1.278 − 0.60056 = −1.87856
v3 = 0.999×0.112936 + 0.001×(−6.0056)² = 0.112823 + 0.036067 = 0.148890

Bias correction:
m3_hat = m3 / (1 − 0.9³) = −1.87856 / 0.271 ≈ −6.9321
v3_hat = v3 / (1 − 0.999³) = 0.148890 / 0.002997 ≈ 49.680

update = 0.5 × (−6.9321) / √49.680 = 0.5 × (−6.9321) / 7.0484 ≈ 0.5 × (−0.9835) ≈ −0.4918
w3 = 0.9972 − (−0.4918) = 1.4890
```

**Step 4 (t=4):**
```
gradient at w3=1.4890: f'(1.4890) = 2(1.4890−4) = −5.0220

m4 = 0.9×(−1.87856) + 0.1×(−5.0220) = −1.690704 − 0.50220 = −2.192904
v4 = 0.999×0.148890 + 0.001×(−5.0220)² = 0.148741 + 0.025220 = 0.173961

Bias correction:
m4_hat = m4 / (1 − 0.9⁴) = −2.192904 / 0.3439 ≈ −6.3766
v4_hat = v4 / (1 − 0.999⁴) = 0.173961 / 0.003994 ≈ 43.549

update = 0.5 × (−6.3766) / √43.549 = 0.5 × (−6.3766) / 6.5991 ≈ 0.5 × (−0.9663) ≈ −0.4831
w4 = 1.4890 − (−0.4831) = 1.9721
```

**Look at the update sizes across the 4 steps: −0.5, −0.4972, −0.4918, −0.4831.** They stay remarkably close to η = 0.5 throughout, barely shrinking at all — this is a signature feature of Adam: **it tends to take fairly consistent-sized steps**, neither collapsing toward zero (like AdaGrad eventually does) nor wildly accelerating and overshooting (like plain momentum can). This consistent, well-behaved step size is a big part of why Adam is such a popular default choice in practice.

### 36.4 Final comparison: all five methods, side by side

Here is where each method's weight w landed after exactly 4 iterations, all starting from w0 = 0 (target = 4). Note that each method used its own typically-recommended learning rate, so this is a comparison of *behaviour*, not a perfectly fair race with identical settings:

| Method | Learning rate used | w after 4 steps | Behaviour observed |
|---|---|---|---|
| Plain Gradient Descent | η=0.1 | 2.3616 | Slow, steady, shrinking steps |
| Momentum | η=0.1, β=0.9 | 5.2344 | Fast at first, but **overshot** the target (4) |
| AdaGrad | η=1.0 | 2.367 | Starts fast, but **step size keeps shrinking** (0.125→0.085) |
| RMSprop | η=0.1, β=0.9 | 0.8723 | Steady, step size **levels off** rather than collapsing |
| Adam | η=0.5, β1=0.9, β2=0.999 | 1.9721 | Steady progress, **step size stays close to η** throughout |

### 36.5 Practical guidance: which one should you use?

| Optimizer | Good for... |
|---|---|
| Plain/Mini-batch GD | Simple problems, or as a baseline; often needs a carefully hand-tuned learning rate |
| Momentum | Speeding up convergence on ravine-shaped loss surfaces; watch out for overshoot if β is too high |
| AdaGrad | Sparse data (some features rarely appear); avoid for very long training runs |
| RMSprop | A safe general-purpose choice, especially for recurrent networks and long training runs |
| **Adam** | **The most common default choice today** — combines the benefits of momentum and RMSprop with well-behaved, stable step sizes |

### 36.6 Unit summary

- Neural network loss landscapes have ravines, saddle points, and flat regions that make plain gradient descent slow.
- Batch, stochastic, and mini-batch gradient descent differ in how many examples are used per update; mini-batch is the practical standard.
- **Momentum** accelerates convergence using a running average of past gradients, but can overshoot the minimum.
- **AdaGrad** gives each weight its own adaptive learning rate based on accumulated squared gradients, but this accumulator only grows, eventually stalling learning.
- **RMSprop** fixes this by using a decaying moving average of squared gradients instead of an ever-growing sum.
- **Adam** combines momentum's smoothing with RMSprop's adaptive step size, plus bias correction, and is the most widely used default optimizer today.

### 36.7 Practice questions for self-study

1. Repeat the plain gradient descent worked example (Lecture 32) for 4 more steps (steps 5–8) and see how close w gets to 4.
2. For the same function f(w)=(w−4)², run 3 iterations of AdaGrad starting from w0=1 instead of w0=0, with η=1.0. Show all steps.
3. Explain, in your own words, why Adam's very first update (Step 1) has a size very close to the learning rate η, no matter how large the gradient is.
4. A friend complains that their model trained with AdaGrad "stopped learning" after a while, even though it hadn't reached a good accuracy yet. Explain what's likely happening and suggest one alternative optimizer to try.

---

## Course Wrap-Up: How the Six Units Fit Together

| Unit | What it taught | Where it's used later |
|---|---|---|
| I | Basic neuron, learning types, perceptron, activation functions | Every later unit builds on the "weighted sum + activation" neuron |
| II | Multi-layer networks, backpropagation, cost functions, regularization | Provides the standard training recipe used across deep learning |
| III | Feedback networks — Hopfield, Boltzmann Machines | An alternative, memory-based way of using neural networks |
| IV | PCA, LDA, Kernel PCA, ICA | Used to pre-process/compress data before feeding it into any network |
| V | SOM, LVQ, RBF | Specialised architectures for clustering, simple classification, and function approximation |
| VI | SGD, Momentum, AdaGrad, RMSprop, Adam | The optimizers that actually make Unit II's backpropagation work well in practice |

*End of Unit VI — end of the INT511 Neural Networks course notes.*
