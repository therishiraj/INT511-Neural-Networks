# Unit V — Advanced Neural Network Architectures

**INT511 – Neural Networks (M.Tech)**
**This unit is designed for Weeks 9–10 of the semester (6 lecture sessions).**

This unit introduces three more specialised network types: **Self-Organizing Maps (SOM)**, which cluster and visualize data; **Learning Vector Quantization (LVQ)**, a simple and interpretable classifier; and **Radial Basis Function (RBF) networks**, which use "closeness to a landmark" as their basic computation.

---

## How this unit is paced

| Lecture | Topic |
|---|---|
| 25 | Competitive learning and the idea behind Self-Organizing Maps |
| 26 | SOM architecture and training algorithm — full worked numerical example |
| 27 | SOM applications and quality; unit checkpoint |
| 28 | Learning Vector Quantization (LVQ) — algorithm with worked example |
| 29 | Radial Basis Function (RBF) Networks — structure and training |
| 30 | Comparing SOM, LVQ, RBF + unit wrap-up |

---

## Lecture 25: Competitive Learning and the Idea Behind SOM

### 25.1 What is "competitive learning"?

In all the networks we've seen so far (Units I–III), many neurons could be "active" at once. In a **competitive** network, neurons *compete* with each other for the right to respond to a given input — and **only the winner gets to update its weights**. This is a purely unsupervised idea: there are no labels, just input data.

### 25.2 The "winner-take-all" rule

Given an input x and several neurons, each with its own weight vector wj, the **winner** is simply the neuron whose weight vector is *closest* to the input (using ordinary distance):
```
winner = the neuron j that has the smallest distance ||x - wj||
```
Only this winning neuron (called the **Best Matching Unit, or BMU**) gets updated, moving its weight vector a little bit *closer* to the input:
```
w_winner_new = w_winner_old + η × (x − w_winner_old)
```
Over time, each neuron's weight vector settles near the centre of a cluster of similar inputs — this is a simple form of automatic clustering.

### 25.3 From competitive learning to Self-Organizing Maps

A **Self-Organizing Map (SOM)**, invented by Teuvo Kohonen, takes this idea one step further: the output neurons are arranged on a **fixed grid** (usually a 2D grid, like a chessboard), and when one neuron wins, its **neighbours on the grid** also get updated a little bit (though less strongly than the winner itself).

**Why does this matter?** This neighbourhood update is what makes a SOM special: it forces neurons that are physically close to each other *on the grid* to also become similar in terms of what kind of input they respond to. The end result is a map where similar inputs end up mapped to nearby locations — this is called **preserving the topology** of the input space, and it's exactly what makes a SOM so useful for visualization.

```
   Input space (could be very high-dimensional)     SOM grid (always low-dimensional, e.g. 2D)

     ●●●            ▲▲▲              maps to        [cluster A][cluster A][cluster B]
      ●●              ▲▲                             [cluster A][cluster A][cluster B]
       (similar inputs are near                       [cluster C][cluster C][cluster B]
        each other in input space)                  (similar inputs land on nearby
                                                       grid cells)
```

**Takeaway for Lecture 25:** Competitive learning has only one neuron update per input (the winner); SOM extends this by also gently updating the winner's grid-neighbours, which preserves the topology of the input data on a simple 2D map.

---

## Lecture 26: SOM Architecture and Training — Worked Numerical Example

### 26.1 Architecture recap

A SOM has:
- An **input layer** simply passing in the raw feature vector.
- A **grid of output neurons** (1D line, or more commonly a 2D grid), where each neuron j has its own weight vector wj of the same size as the input.

```
   Input x (say, 2D)
        │
        ▼
   ┌───────────────────────────┐
   │  N1        N2        N3   │   <- a simple 1D line of 3 neurons
   │  w1=(..)  w2=(..)  w3=(..)│      (each weight vector is 2D, matching input)
   └───────────────────────────┘
```

### 26.2 The training algorithm, step by step

1. **Initialise** all neurons' weight vectors (often small random values).
2. For each training input x:
   a. **Find the BMU** (Best Matching Unit) — the neuron with the smallest distance to x.
   b. **Determine the neighbourhood** around the BMU on the grid (this neighbourhood shrinks as training goes on).
   c. **Update the BMU and its neighbours**, moving each of their weight vectors a little toward x — the BMU moves the most, and neurons further away (on the grid) move progressively less.
3. **Gradually shrink** both the neighbourhood size and the learning rate as training progresses, so the map settles into a stable, fine-tuned arrangement.

**The update formula:**
```
wj_new = wj_old + η × h(j) × (x − wj_old)
```
where h(j) is the "neighbourhood strength" for neuron j — it equals 1 for the BMU itself, and gets smaller the further away (on the grid) neuron j is from the BMU. Neurons outside the neighbourhood radius get h(j) = 0 (no update at all).

### 26.3 Fully worked example

**Setup:** a small SOM with 3 neurons arranged in a line: N1, N2, N3. Each has a 2D weight vector.

```
Initial weights:
N1: (0.2, 0.6)
N2: (0.5, 0.5)
N3: (0.9, 0.1)
```

**Input presented:** x = (0.6, 0.4). **Learning rate:** η = 0.3. **Neighbourhood:** immediate neighbours get half-strength (h = 0.5), the BMU gets full strength (h = 1.0).

**Step 1 — Compute the distance from x to each neuron's weight vector:**
```
distance to N1 = √[(0.6−0.2)² + (0.4−0.6)²] = √[0.16 + 0.04] = √0.20 ≈ 0.447
distance to N2 = √[(0.6−0.5)² + (0.4−0.5)²] = √[0.01 + 0.01] = √0.02 ≈ 0.141
distance to N3 = √[(0.6−0.9)² + (0.4−0.1)²] = √[0.09 + 0.09] = √0.18 ≈ 0.424
```

**Step 2 — Find the BMU:** N2 has the smallest distance (0.141), so **N2 is the BMU**.

**Step 3 — Assign neighbourhood strengths:** since N2 is the BMU, h(N2) = 1.0. N1 and N3 are its immediate (only) neighbours on this 3-neuron line, so h(N1) = h(N3) = 0.5.

**Step 4 — Update N2 (the BMU):**
```
w2_new = (0.5, 0.5) + 0.3 × 1.0 × [(0.6,0.4) − (0.5,0.5)]
       = (0.5, 0.5) + 0.3 × (0.1, −0.1)
       = (0.5 + 0.03, 0.5 − 0.03)
       = (0.53, 0.47)
```

**Step 5 — Update N1 (neighbour, half strength):**
```
w1_new = (0.2, 0.6) + 0.3 × 0.5 × [(0.6,0.4) − (0.2,0.6)]
       = (0.2, 0.6) + 0.15 × (0.4, −0.2)
       = (0.2 + 0.06, 0.6 − 0.03)
       = (0.26, 0.57)
```

**Step 6 — Update N3 (neighbour, half strength):**
```
w3_new = (0.9, 0.1) + 0.3 × 0.5 × [(0.6,0.4) − (0.9,0.1)]
       = (0.9, 0.1) + 0.15 × (−0.3, 0.3)
       = (0.9 − 0.045, 0.1 + 0.045)
       = (0.855, 0.145)
```

**Result after one training step:**

| Neuron | Old weight | New weight | Moved by |
|---|---|---|---|
| N1 | (0.20, 0.60) | (0.26, 0.57) | a little, toward x |
| N2 (BMU) | (0.50, 0.50) | (0.53, 0.47) | the most, toward x |
| N3 | (0.90, 0.10) | (0.855, 0.145) | a little, toward x |

Notice how **all three** neurons moved a bit toward the input, but N2 (the winner) moved proportionally the most, while its neighbours N1 and N3 moved less. This is exactly the topology-preserving behaviour that makes SOMs special. Repeating this process for many inputs, while slowly shrinking the neighbourhood size and learning rate, causes the whole map to gradually "unfold" and organise itself to mirror the structure of the input data.

**Takeaway for Lecture 26:** Every SOM training step is: find the closest neuron (BMU), then nudge the BMU and its grid-neighbours toward the input, with the nudge strength fading with grid-distance from the BMU.

---

## Lecture 27: SOM Applications and Quality

### 27.1 What SOMs are good for

Because a SOM naturally compresses high-dimensional data onto a simple, easy-to-look-at grid (usually 2D) while preserving neighbourhood relationships, it's especially popular for:

- **Data visualization:** e.g., visualizing customer segments, document collections, or gene-expression profiles as a 2D "map."
- **Clustering:** grouping similar data points without needing labels.
- **Dimensionality reduction for exploration:** getting an intuitive first look at the structure of complex data before applying other methods.

### 27.2 Quick sanity checks for a trained SOM

- Do inputs that are similar to each other in the original data end up on nearby grid cells? (If yes, the map has learned useful topology.)
- Are there "dead" neurons that never win for any input? (This can happen if a neuron's initial weights are far from all the data — a known practical issue.)

### 27.3 Unit checkpoint

By this point, you should be able to:
- Explain the difference between competitive learning and a full SOM.
- Compute the BMU for a given input and set of neuron weights.
- Perform one full weight-update step for a small SOM, including neighbourhood effects.

---

## Lecture 28: Learning Vector Quantization (LVQ)

### 28.1 What is LVQ, and how is it different from SOM?

LVQ looks similar to SOM in that it also uses "prototype" vectors and a nearest-neighbour comparison — but there's one crucial difference: **LVQ is supervised**. Each prototype is assigned a class label in advance, and the training process uses the *true labels* of the training data to decide whether to move a prototype closer to, or further from, a given input.

### 28.2 The LVQ1 update rule

For an input x with true class label t, first find the nearest prototype (by distance), call it the "winner" w_winner, which belongs to some class c.

```
If c = t   (the winning prototype's class matches the true label — CORRECT):
       w_winner_new = w_winner_old + η × (x − w_winner_old)     [move CLOSER]

If c ≠ t   (the winning prototype's class does NOT match — INCORRECT):
       w_winner_new = w_winner_old − η × (x − w_winner_old)     [move FURTHER AWAY]
```

**Why does this make sense?** If the nearest prototype already has the correct class, we want to reinforce it by pulling it slightly closer to this input, so it represents this region of the input space even better. If the nearest prototype has the *wrong* class, we push it away, making room for the correct-class prototype to "win" for inputs like this one in the future.

### 28.3 Fully worked example

**Setup:** two class prototypes:
```
w_A (Class A) = (2, 3)
w_B (Class B) = (6, 7)
```
Learning rate η = 0.2.

**Training example 1: x = (3, 4), true label = A**

**Step 1 — Compute distances to both prototypes:**
```
distance to w_A = √[(3−2)² + (4−3)²] = √[1 + 1] = √2 ≈ 1.414
distance to w_B = √[(3−6)² + (4−7)²] = √[9 + 9] = √18 ≈ 4.243
```

**Step 2 — Find the winner:** w_A is closer, so it wins. Its class (A) matches the true label (A) — **correct classification**.

**Step 3 — Attract w_A toward x:**
```
w_A_new = (2, 3) + 0.2 × [(3,4) − (2,3)]
        = (2, 3) + 0.2 × (1, 1)
        = (2.2, 3.2)
```

**Training example 2: x = (5, 5), true label = A**

**Step 1 — Compute distances (using the updated w_A):**
```
distance to w_A(2.2, 3.2) = √[(5−2.2)² + (5−3.2)²] = √[7.84 + 3.24] = √11.08 ≈ 3.329
distance to w_B(6, 7)     = √[(5−6)² + (5−7)²]       = √[1 + 4]     = √5     ≈ 2.236
```

**Step 2 — Find the winner:** w_B is closer, so it wins. But its class (B) does **not** match the true label (A) — **incorrect classification**.

**Step 3 — Repel w_B away from x:**
```
w_B_new = (6, 7) − 0.2 × [(5,5) − (6,7)]
        = (6, 7) − 0.2 × (−1, −2)
        = (6, 7) + (0.2, 0.4)
        = (6.2, 7.4)
```

Notice: w_B moved *away* from x (its x-coordinate increased from 6 to 6.2, moving further from x's x-coordinate of 5), which is exactly the "push away on a wrong answer" behaviour we want.

**Takeaway for Lecture 28:** LVQ keeps a small set of labelled prototype vectors and, for every training example, either pulls the nearest prototype closer (if its class is correct) or pushes it away (if its class is wrong) — over time, this carves out clean, well-placed decision regions for each class.

---

## Lecture 29: Radial Basis Function (RBF) Networks

### 29.1 The core idea: "how close am I to a landmark?"

An RBF network's hidden neurons don't compute a weighted sum like a normal neuron — instead, each hidden neuron represents a **"landmark" point** (called a **centre**), and its output simply measures **how close the input is to that landmark**. The most common way to measure this is the **Gaussian (bell-curve) function**:

```
φ(x) = exp( − (distance from x to the centre)² / (2 × width²) )
```

- If x is *very close* to the centre, φ(x) is close to 1 (strong activation).
- If x is *far* from the centre, φ(x) is close to 0 (weak/no activation).
- The **width** (σ) controls how quickly the activation fades as you move away from the centre — a small width means a very "sharp," localized bump; a large width means a wide, gently-fading bump.

### 29.2 Architecture

```
    Input x
       │
       ▼
   ┌─────────────────────────────┐
   │  RBF hidden neurons          │      Each computes:
   │  φ1 (centre c1)              │      φj(x) = exp(-(x-cj)²/2σj²)
   │  φ2 (centre c2)              │
   └──────────────┬────────────────┘
                  │ (weighted sum, LINEAR this time)
                  ▼
             Output y = w1φ1 + w2φ2 + ... + b
```

Unlike a standard MLP, the hidden layer here uses "distance to a landmark" instead of "weighted sum of inputs," and only the very last step (output layer) is an ordinary weighted sum.

### 29.3 Training an RBF network: two separate stages

**Stage 1 (unsupervised): choose the centres and widths.**
A common simple approach: run a clustering algorithm (like k-means) on the training inputs, and use the resulting cluster centres as the RBF centres. The width of each RBF is usually set based on how far apart nearby centres are (so that neighbouring bumps overlap reasonably).

**Stage 2 (supervised): learn the output weights.**
Once the centres and widths are fixed, computing the hidden layer's output for every training example is straightforward. The mapping from these hidden outputs to the final answer is just a normal linear combination, so we can find the best output weights using ordinary linear regression (least squares) — a fast, simple calculation, since this final step has no non-linearity to fight against.

### 29.4 Fully worked example: computing an RBF network's output

**Setup:** a 1D input, 2 RBF centres.
```
Centre 1: c1 = 2,   width σ1 = 1.5
Centre 2: c2 = 5,   width σ2 = 1.5
Output weights (already learned): w1 = 2.0, w2 = −1.0, bias b = 0.5
Input: x = 3
```

**Step 1 — Compute the activation of the first RBF neuron:**
```
φ1 = exp( −(x−c1)² / (2×σ1²) )
   = exp( −(3−2)² / (2×1.5²) )
   = exp( −1 / 4.5 )
   = exp(−0.222)
   ≈ 0.801
```

**Step 2 — Compute the activation of the second RBF neuron:**
```
φ2 = exp( −(x−c2)² / (2×σ2²) )
   = exp( −(3−5)² / (2×1.5²) )
   = exp( −4 / 4.5 )
   = exp(−0.889)
   ≈ 0.411
```

**Step 3 — Combine using the (linear) output layer:**
```
y = w1×φ1 + w2×φ2 + b
  = (2.0 × 0.801) + (−1.0 × 0.411) + 0.5
  = 1.602 − 0.411 + 0.5
  = 1.691
```

So for input x = 3, the network's output is approximately **1.691**. Notice that x = 3 is closer to centre c1 = 2 than to c2 = 5, so φ1 (0.801) is larger than φ2 (0.411) — the network's output is dominated more by centre 1's "vote."

**Takeaway for Lecture 29:** RBF networks measure "closeness to a landmark" using Gaussian bumps in the hidden layer, then combine these closeness scores with ordinary linear weights in the output layer — trained in two easy stages: find good landmarks first (clustering), then fit a simple linear model on top.

---

## Lecture 30: Comparing SOM, LVQ, and RBF — Unit Wrap-Up

### 30.1 Side-by-side comparison

| Aspect | SOM | LVQ | RBF |
|---|---|---|---|
| Needs labels? | No (unsupervised) | Yes (supervised) | Yes, for the output-weight stage |
| Main purpose | Clustering & 2D visualization | Classification | Function approximation / classification |
| Output form | A 2D grid of prototype neurons | A set of labelled class prototypes | A continuous output value |
| Key computation | Distance to weight vector + neighbourhood update | Distance to prototype + attract/repel rule | Distance to centre → Gaussian bump → linear combination |
| Good for visualization? | Yes, this is its main strength | Not really | Not really |

### 30.2 When would you pick each one?

- **Pick SOM** when you want to explore/visualize the structure of unlabelled, high-dimensional data (e.g., "are there natural groups in my customer data?").
- **Pick LVQ** when you have labelled data and want a simple, interpretable classifier made of a small number of representative "template" examples per class.
- **Pick RBF** when you want a network that can approximate a smooth function or make a classification decision based on "how similar is this input to examples I've already seen," and you'd like fast, two-stage training instead of full backpropagation.

### 30.3 Unit summary

- **Competitive learning** has neurons compete for each input, with only the winner updating.
- **SOM** extends competitive learning by placing neurons on a grid and updating the winner's neighbours too — this preserves the topology of the input space and is excellent for visualization.
- **LVQ** is the supervised cousin of competitive learning: prototypes are pulled toward inputs of the correct class and pushed away from inputs of the wrong class.
- **RBF networks** use Gaussian "closeness to a landmark" neurons in the hidden layer, followed by an ordinary linear output layer, and are trained in two fast stages (unsupervised centres, then supervised output weights).

### 30.4 Practice questions for self-study

1. For a SOM with neurons at (0.1, 0.1), (0.5, 0.5), (0.9, 0.9) arranged in a line, and input x = (0.4, 0.6), find the BMU and perform one weight update (η=0.4, neighbour strength 0.5).
2. In the LVQ example (Lecture 28), continue training with a third example x = (1, 2), true label = A. Show the winner, whether it's correct or incorrect, and the resulting prototype update.
3. For an RBF network with a single centre c = 0, width σ = 1, output weight w = 3, bias b = 0, compute the output for inputs x = 0, x = 1, and x = 2. What pattern do you notice as x moves away from the centre?
4. In your own words, explain why SOM is unsupervised but LVQ, which looks very similar, is supervised.

---

*End of Unit V — proceed to Unit VI: Optimization Techniques for Neural Networks*
