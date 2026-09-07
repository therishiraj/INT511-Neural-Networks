# Unit II — Feedforward Neural Networks

**INT511 – Neural Networks (M.Tech)**
**This unit is designed for Weeks 3–4 of the semester (6 lecture sessions).**

In Unit I, we saw that a single neuron (M-P neuron or Perceptron) can only draw one straight line, so it fails on problems like XOR. This unit shows exactly *how* stacking neurons into multiple layers fixes this, and — most importantly — *how such a multi-layer network is actually trained*, using an algorithm called **backpropagation**.

---

## How this unit is paced

| Lecture | Topic |
|---|---|
| 7 | Multi-layer neural networks — architecture and why depth helps |
| 8 | Solving the XOR problem with a 2-layer network |
| 9 | Backpropagation — Part 1: the forward pass (with numerical example) |
| 10 | Backpropagation — Part 2: the backward pass (continuing the numerical example) |
| 11 | Cost functions and Gradient Descent (batch / stochastic / mini-batch) |
| 12 | Overfitting and Regularization techniques + unit wrap-up |

---

## Lecture 7: Multi-Layer Neural Networks

### 7.1 Why one layer is not enough

Recall from Unit I: a single neuron/perceptron can only separate data using **one straight line** (or, in higher dimensions, one flat plane). Many real-world problems — including something as simple as XOR — need a **bent or curved boundary**, which a single layer simply cannot draw.

**The fix:** stack neurons into multiple layers. Each hidden layer bends the decision boundary a little more, so a network with enough hidden neurons can approximate almost any shape of boundary.

### 7.2 The multi-layer perceptron (MLP) architecture

```
 INPUT LAYER        HIDDEN LAYER          OUTPUT LAYER

   x1 ●───┐      ┌──►● h1 ──┐         ┌──►● y1
          ├──────┤          ├─────────┤
   x2 ●───┤      ├──►● h2 ──┤         ├──►● y2
          ├──────┤          ├─────────┤
   x3 ●───┘      └──►● h3 ──┘         └──►● y3
                  (weights W1,          (weights W2,
                   bias b1)              bias b2)

     ───────────── data flows forward ─────────────►
```

- **Input layer:** just holds the raw feature values, no computation.
- **Hidden layer(s):** each hidden neuron computes a weighted sum of the *previous* layer's outputs, then applies an activation function (sigmoid, tanh, or ReLU — see Unit I, Lecture 6).
- **Output layer:** produces the final prediction (a class probability, or a number).

Because every connection only sends data **forward** (input → hidden → output, never backward), this is called a **feedforward network**.

### 7.3 Simple notation we will use in this unit

To avoid confusing formulas, we will use plain, readable notation:

| Symbol | Meaning |
|---|---|
| x1, x2, … | Input values |
| w (with a superscript layer number, e.g. w¹, w²) | Weights of a given layer |
| b¹, b² | Bias of a given layer |
| z | Weighted sum before activation ("net input") |
| a (or h for hidden, y for output) | Value *after* activation |
| f() | The activation function used |
| t (target) | The correct/desired output we want the network to learn |
| E (error/loss) | A number describing how wrong the output is |
| η (eta) | The learning rate |

**General rule for any layer:**
```
z = (weights of this layer) · (outputs of previous layer) + (bias of this layer)
a = f(z)
```
We simply repeat this, layer after layer, until we reach the output.

### 7.4 Counting parameters (a quick sanity-check skill)

For a layer that takes n inputs and produces m outputs, the number of weights is n×m, plus m biases (one bias per output neuron in that layer).

**Example:** A network with 3 inputs → 4 hidden neurons → 2 output neurons has:
- Layer 1 (input→hidden): 3×4 = 12 weights + 4 biases = 16 parameters
- Layer 2 (hidden→output): 4×2 = 8 weights + 2 biases = 10 parameters
- **Total: 26 learnable parameters**

This is a useful habit: before training any network, always work out how many numbers it actually needs to learn.

**Takeaway for Lecture 7:** An MLP is just several layers of neurons stacked together, where each layer's output feeds the next layer's input. More layers (depth) allow the network to build more complex, curved decision boundaries.

---

## Lecture 8: Solving the XOR Problem with a 2-Layer Network

### 8.1 Recap: why XOR fails for one layer

XOR truth table: (0,0)→0, (0,1)→1, (1,0)→1, (1,1)→0. As shown in Unit I, no single straight line separates the "0" outputs from the "1" outputs.

### 8.2 A worked 2-layer solution (with real numbers, step activation)

We build a small feedforward network with **2 hidden neurons** and **1 output neuron**, using the step activation function (fires 1 if z ≥ 0, else 0):

```
                     ┌─────────┐
        x1 ────┬────►│  h1 =OR │───┐
               │      └─────────┘   │      ┌──────────┐
               │                    ├─────►│  y = AND │───► output
               │      ┌─────────┐   │      └──────────┘
        x2 ────┴────►│ h2 =NAND│───┘
                      └─────────┘
```

**Hidden neuron h1 (computes OR):** weights (1, 1), bias = −0.5
```
z(h1) = 1·x1 + 1·x2 − 0.5
```

**Hidden neuron h2 (computes NAND):** weights (−1, −1), bias = 1.5
```
z(h2) = −1·x1 − 1·x2 + 1.5
```

**Output neuron y (computes AND of h1 and h2):** weights (1, 1), bias = −1.5
```
z(y) = 1·h1 + 1·h2 − 1.5
```

### 8.3 Step-by-step verification for all 4 inputs

**Input (0, 0):**
- z(h1) = 1(0)+1(0)−0.5 = −0.5 → h1 = 0 (since z<0)
- z(h2) = −1(0)−1(0)+1.5 = 1.5 → h2 = 1 (since z≥0)
- z(y) = 1(0)+1(1)−1.5 = −0.5 → y = 0
- **Expected XOR(0,0) = 0 ✓**

**Input (0, 1):**
- z(h1) = 0+1−0.5 = 0.5 → h1 = 1
- z(h2) = 0−1+1.5 = 0.5 → h2 = 1
- z(y) = 1+1−1.5 = 0.5 → y = 1
- **Expected XOR(0,1) = 1 ✓**

**Input (1, 0):** (by symmetry with the above, h1=1, h2=1, y=1) — **Expected 1 ✓**

**Input (1, 1):**
- z(h1) = 1+1−0.5 = 1.5 → h1 = 1
- z(h2) = −1−1+1.5 = −0.5 → h2 = 0
- z(y) = 1+0−1.5 = −0.5 → y = 0
- **Expected XOR(1,1) = 0 ✓**

All four cases match. **This proves that a 2-layer network can solve a problem that no single-layer network can solve.**

### 8.4 The geometric intuition

Each hidden neuron (h1, h2) draws its own straight line in the input space. The output neuron then combines these two lines using a simple AND-like rule. The overall effect is a **bent boundary** made of two straight-line pieces — something a single neuron could never draw by itself. This is exactly why adding hidden layers gives a network so much more power.

**Takeaway for Lecture 8:** By combining two simple straight-line classifiers (in the hidden layer) with one more classifier (in the output layer), we can represent shapes that a single straight line cannot — this is the core reason multi-layer networks are needed.

---

## Lecture 9: Backpropagation — Part 1 (The Forward Pass)

### 9.1 The problem backpropagation solves

In Lecture 8, we *manually chose* good weights to solve XOR. In practice, we don't know the right weights in advance — we must **learn** them from data. **Backpropagation** is simply an efficient method for computing how much each weight in a multi-layer network should change, by working backward from the error at the output.

**The overall training loop (for any network, any dataset):**

1. **Forward pass:** feed the input through the network, layer by layer, to get a prediction.
2. **Compute the error:** compare the prediction to the target/desired value.
3. **Backward pass:** work backward from the output to figure out how much each weight contributed to the error.
4. **Update weights:** nudge each weight slightly in the direction that reduces the error.
5. Repeat for many examples, many times (epochs), until the error becomes small.

This lecture covers Step 1 and 2 (forward pass + error). Lecture 10 covers Steps 3 and 4 (backward pass + update).

### 9.2 Our example network for this worked problem

We will use a small, standard network: **2 inputs → 2 hidden neurons → 1 output neuron**, all using the **sigmoid** activation function.

```
   Inputs          Hidden layer            Output layer

   i1=0.05 ──w1=0.15──┐
                       ├──► h1 (bias bh1=0.35)
   i2=0.10 ──w2=0.20──┘         │
                                 w5=0.40
   i1=0.05 ──w3=0.25──┐         │
                       ├──► h2 (bias bh2=0.35) ──► o (bias bo=0.60) ──► output
   i2=0.10 ──w4=0.30──┘         │
                                 w6=0.45
```

**All the numbers we'll use:**
```
Inputs:            i1 = 0.05,  i2 = 0.10
Input→Hidden:      w1 = 0.15,  w2 = 0.20,  w3 = 0.25,  w4 = 0.30
Hidden biases:     bh1 = 0.35,  bh2 = 0.35
Hidden→Output:     w5 = 0.40,  w6 = 0.45
Output bias:       bo = 0.60
Target output:     t = 0.01
Learning rate:     η = 0.5
```

### 9.3 Step-by-step forward pass

**Step 1 — Compute the net input to hidden neuron h1:**
```
z(h1) = w1×i1 + w2×i2 + bh1
      = (0.15×0.05) + (0.20×0.10) + 0.35
      = 0.0075 + 0.02 + 0.35
      = 0.3775
```

**Step 2 — Apply the sigmoid activation to get h1's output:**
```
h1 = 1 / (1 + e^(-0.3775))
e^(-0.3775) ≈ 0.6856
h1 = 1 / 1.6856 ≈ 0.5933
```

**Step 3 — Compute the net input to hidden neuron h2:**
```
z(h2) = w3×i1 + w4×i2 + bh2
      = (0.25×0.05) + (0.30×0.10) + 0.35
      = 0.0125 + 0.03 + 0.35
      = 0.3925
```

**Step 4 — Apply sigmoid to get h2's output:**
```
h2 = 1 / (1 + e^(-0.3925))
e^(-0.3925) ≈ 0.6754
h2 = 1 / 1.6754 ≈ 0.5968
```

**Step 5 — Compute the net input to the output neuron:**
```
z(o) = w5×h1 + w6×h2 + bo
     = (0.40×0.5933) + (0.45×0.5968) + 0.60
     = 0.2373 + 0.2686 + 0.60
     = 1.1059
```

**Step 6 — Apply sigmoid to get the final output:**
```
o = 1 / (1 + e^(-1.1059))
e^(-1.1059) ≈ 0.3308
o = 1 / 1.3308 ≈ 0.7514
```

### 9.4 Computing the error

We use the standard "half squared error" (the ½ is just there to make the derivative clean later):
```
E = ½ × (t − o)²
  = ½ × (0.01 − 0.7514)²
  = ½ × (−0.7414)²
  = ½ × 0.5497
  = 0.2749
```

Our network currently predicts 0.7514, but we wanted 0.01 — a large error. In Lecture 10, we will work backward through the network to figure out exactly how to adjust every single weight to reduce this error.

**Takeaway for Lecture 9:** The forward pass is nothing more than repeatedly applying "weighted sum → activation" layer by layer. Once we reach the output, we measure how wrong we are using an error/loss function.

---

## Lecture 10: Backpropagation — Part 2 (The Backward Pass)

We continue the exact same example from Lecture 9. Our goal now is to find out how much each of the 6 weights (w1–w6) and 3 biases should change to reduce the error E = 0.2749.

### 10.1 The key idea: the chain rule, applied one layer at a time

We cannot directly see how, say, w1 affects the final error E — there are two layers of "activation function" in between. The chain rule lets us break this long connection into small, easy steps:
```
(how E changes with a weight) = (how E changes with the neuron's output)
                               × (how the neuron's output changes with its net input z)
                               × (how the net input z changes with that weight)
```
We compute this **starting from the output layer and moving backward** — hence "back"-propagation.

### 10.2 Output layer: compute delta_o

**Step 1 — How does the error change with the output o?**
```
dE/do = −(t − o) = −(0.01 − 0.7514) = 0.7414
```

**Step 2 — How does the output o change with its net input z(o)?** (This is the sigmoid derivative: f'(z) = f(z)×(1−f(z)), using the already-computed output value.)
```
do/dz(o) = o × (1 − o) = 0.7514 × (1 − 0.7514) = 0.7514 × 0.2486 = 0.1868
```

**Step 3 — Combine these into "delta_o" (the output layer's error signal):**
```
delta_o = (dE/do) × (do/dz(o)) = 0.7414 × 0.1868 = 0.1385
```

### 10.3 Gradients for the output layer's weights and bias

The net input to the output neuron was z(o) = w5×h1 + w6×h2 + bo, so:
```
dE/dw5 = delta_o × h1 = 0.1385 × 0.5933 = 0.0822
dE/dw6 = delta_o × h2 = 0.1385 × 0.5968 = 0.0826
dE/dbo = delta_o × 1  = 0.1385
```

### 10.4 Update the output layer's weights (gradient descent step)

**Rule: new weight = old weight − η × gradient**
```
w5_new = 0.40 − 0.5×0.0822 = 0.40 − 0.0411 = 0.3589
w6_new = 0.45 − 0.5×0.0826 = 0.45 − 0.0413 = 0.4087
bo_new = 0.60 − 0.5×0.1385 = 0.60 − 0.0693 = 0.5307
```

### 10.5 Hidden layer: compute delta_h1 and delta_h2

Now we push the error signal one layer further back. Each hidden neuron's delta depends on **how much it contributed to the output neuron's error**, via the weight connecting them (using the *old* weight values, before the update above):

```
delta_h1 = delta_o × w5(old) × h1 × (1 − h1)
         = 0.1385 × 0.40 × 0.5933 × (1 − 0.5933)
         = 0.1385 × 0.40 × 0.5933 × 0.4067
         = 0.1385 × 0.40 × 0.2413
         = 0.01337

delta_h2 = delta_o × w6(old) × h2 × (1 − h2)
         = 0.1385 × 0.45 × 0.5968 × (1 − 0.5968)
         = 0.1385 × 0.45 × 0.5968 × 0.4032
         = 0.1385 × 0.45 × 0.2406
         = 0.01500
```

**Why does this formula make sense?** delta_o × w5 tells us "how much of the output's error can be traced back through the h1→output connection," and multiplying by h1×(1−h1) accounts for h1's own sigmoid activation.

### 10.6 Gradients for the hidden layer's weights and biases

Recall z(h1) = w1×i1 + w2×i2 + bh1, so:
```
dE/dw1 = delta_h1 × i1 = 0.01337 × 0.05 = 0.000669
dE/dw2 = delta_h1 × i2 = 0.01337 × 0.10 = 0.001337
dE/dbh1 = delta_h1 × 1 = 0.01337
```
And z(h2) = w3×i1 + w4×i2 + bh2, so:
```
dE/dw3 = delta_h2 × i1 = 0.01500 × 0.05 = 0.000750
dE/dw4 = delta_h2 × i2 = 0.01500 × 0.10 = 0.001500
dE/dbh2 = delta_h2 × 1 = 0.01500
```

### 10.7 Update the hidden layer's weights and biases

```
w1_new  = 0.15 − 0.5×0.000669 = 0.149666
w2_new  = 0.20 − 0.5×0.001337 = 0.199332
w3_new  = 0.25 − 0.5×0.000750 = 0.249625
w4_new  = 0.30 − 0.5×0.001500 = 0.299250
bh1_new = 0.35 − 0.5×0.01337  = 0.343315
bh2_new = 0.35 − 0.5×0.01500  = 0.342500
```

### 10.8 Did the error actually go down? Let's check!

Run the forward pass again with the **new** weights:
```
z(h1)_new = 0.149666×0.05 + 0.199332×0.10 + 0.343315 ≈ 0.37077  → h1_new ≈ 0.5917
z(h2)_new = 0.249625×0.05 + 0.299250×0.10 + 0.342500 ≈ 0.38491  → h2_new ≈ 0.5951
z(o)_new  = 0.3589×0.5917 + 0.4087×0.5951 + 0.5307   ≈ 0.98635  → o_new  ≈ 0.7284

New error: E_new = ½×(0.01 − 0.7284)² = ½×0.5161 = 0.2581
```

**Old error was 0.2749; new error is 0.2581 — it went down!** This is exactly what we expect: one small step of backpropagation always nudges the network's weights to make the error a little smaller. Repeating this process thousands of times, over many training examples, is how a real neural network is trained.

**Takeaway for Lecture 10:** Backpropagation is just the chain rule, applied one layer at a time, from the output back to the input, to find how each weight should change to reduce the error — followed by a normal gradient-descent update.

---

## Lecture 11: Cost Functions and Gradient Descent

### 11.1 Cost functions (also called loss functions)

A cost function is simply a formula that turns "how wrong the network's prediction is" into a single number we can try to minimise.

**Mean Squared Error (MSE)** — used for regression (predicting a number):
```
MSE = (1/N) × Σ (target_i − prediction_i)²
```
Squaring makes all errors positive and penalises large mistakes more heavily than small ones.

**Cross-Entropy Loss** — used for classification (predicting a category):
```
For binary classification:
CE = − [ t×log(o) + (1−t)×log(1−o) ]
```
This penalises confident wrong answers very heavily (e.g., if the true label is 1 but the model outputs 0.01, log(0.01) is a large negative number, making the loss huge) and rewards confident correct answers with a very small loss.

**Rule of thumb:** use MSE for regression problems, and cross-entropy for classification problems.

### 11.2 Gradient Descent — the big picture

Imagine you are standing on a hilly landscape in thick fog and want to reach the lowest point (minimum error). You cannot see the whole landscape, but you can feel the slope of the ground right where you're standing. The sensible strategy: **take a small step in the direction that goes downhill**, and repeat.

This is exactly what gradient descent does:
```
new_weight = old_weight − η × (gradient of error with respect to that weight)
```
- The **gradient** tells us the slope (which direction is "uphill").
- We move in the *opposite* direction (downhill), because we want to *decrease* the error.
- η (the learning rate) controls how big a step we take. Too large, and we might overshoot the minimum; too small, and training takes forever.

### 11.3 A tiny numerical example of gradient descent (without any neural network, just to build intuition)

Suppose we want to minimise a very simple function: f(w) = (w − 3)². Its minimum is obviously at w = 3, but let's *find* it using gradient descent.

The gradient (derivative) is: f'(w) = 2(w − 3)

Start at w = 0, use learning rate η = 0.2:

| Step | w (current) | f'(w) = 2(w−3) | w_new = w − η×f'(w) |
|---|---|---|---|
| 1 | 0.0 | 2(0−3) = −6.0 | 0 − 0.2×(−6.0) = 1.20 |
| 2 | 1.20 | 2(1.2−3) = −3.6 | 1.20 − 0.2×(−3.6) = 1.92 |
| 3 | 1.92 | 2(1.92−3) = −2.16 | 1.92 − 0.2×(−2.16) = 2.352 |
| 4 | 2.352 | 2(2.352−3) = −1.296 | 2.352 − 0.2×(−1.296) = 2.6112 |
| 5 | 2.6112 | 2(2.6112−3) = −0.7776 | 2.6112 − 0.2×(−0.7776) = 2.7667 |

Notice how w keeps getting closer and closer to 3 with every step, and the corrections get smaller as we approach the minimum (because the slope flattens out near the bottom). This is *exactly* the same idea we used in Lecture 10, just applied there to 9 different weights at once instead of a single w.

### 11.4 Batch, Stochastic, and Mini-Batch Gradient Descent

When training a network, we usually have thousands (or millions) of training examples. How many examples should we look at before making one weight update?

| Method | How many examples per update? | Pros | Cons |
|---|---|---|---|
| **Batch Gradient Descent** | *All* training examples at once | Very stable, accurate direction | Very slow for large datasets; needs lots of memory |
| **Stochastic Gradient Descent (SGD)** | Just *one* randomly chosen example | Very fast per update; the randomness can help escape bad spots | Noisy, "zig-zag" path toward the minimum |
| **Mini-Batch Gradient Descent** | A small random group (e.g., 32 or 64 examples) | Good balance of speed and stability; works well with GPUs | Needs to choose a good batch size |

**In practice, almost everyone uses mini-batch gradient descent** — it is the standard choice in real deep-learning systems, because it's fast, reasonably stable, and can use hardware efficiently.

**Takeaway for Lecture 11:** The cost function measures how wrong we are; gradient descent tells us how to change the weights to become less wrong; and mini-batch gradient descent is the practical, everyday version used to train real networks.

---

## Lecture 12: Overfitting and Regularization

### 12.1 What is overfitting?

Imagine a student preparing for an exam by **memorising** the exact answers to last year's question paper, instead of *understanding* the underlying concepts. This student will do great if the same questions repeat — but will fail badly on any new question.

This is exactly what **overfitting** means for a neural network: the model learns the training data so precisely (including its noise and quirks) that it performs very well on the training data but poorly on new, unseen data.

```
   Error
    │                                 ___________ Validation/Test error
    │                                /            (starts going UP - overfitting!)
    │                               /
    │       Training error \___    /
    │                       \___\_/
    │                           \________  Training error keeps going DOWN
    └──────────────────────────────────────► Training time (epochs)
                              ↑
                    Ideal stopping point
```

### 12.2 Why does overfitting happen?

- The model has **too much capacity** (too many neurons/layers) relative to how much training data is available — it has "room" to memorise instead of generalise.
- Training runs for **too long**, letting the model gradually fit noise in the data.

### 12.3 Regularization Technique 1: L2 Regularization (Weight Decay)

**Idea:** discourage the network from using very large weights, since large weights often mean the model is fitting the training data too precisely (overreacting to small changes in input).

We simply add a penalty to the cost function based on the size of the weights:
```
New Cost = Original Cost + (λ/2) × (sum of all squared weights)
```
λ (lambda) is a small positive number we choose — the bigger λ is, the more strongly we discourage large weights.

**Effect on the weight update:** working through the calculus gives this simple modified update rule:
```
w_new = w_old − η×(gradient of original cost) − η×λ×w_old
```
which can be rewritten as:
```
w_new = (1 − ηλ)×w_old − η×(gradient of original cost)
```

**Tiny numerical example:** suppose w_old = 0.5, the gradient of the original cost is 0.02, η = 0.1, λ = 0.1:
```
Without regularization: w_new = 0.5 − 0.1×0.02 = 0.5 − 0.002 = 0.498
With L2 regularization: w_new = (1 − 0.1×0.1)×0.5 − 0.1×0.02
                               = (0.99)×0.5 − 0.002
                               = 0.495 − 0.002 = 0.493
```
Notice the weight shrinks a little bit *extra*, every single update — this constant, gentle shrinkage is why L2 regularization is also called **"weight decay."**

### 12.4 Regularization Technique 2: L1 Regularization

**Idea:** similar to L2, but the penalty is based on the *absolute value* of weights instead of the square:
```
New Cost = Original Cost + λ × (sum of |weights|)
```
**Key difference from L2:** L1 regularization tends to push many weights all the way to exactly **zero**, effectively removing some connections entirely. This makes L1 useful when we want the network to automatically ignore unimportant input features (a kind of automatic feature selection). L2, on the other hand, just makes all weights *a bit smaller*, rather than eliminating them.

### 12.5 Regularization Technique 3: Dropout

**Idea:** during training, randomly "switch off" (set to zero) a fraction of the hidden neurons in a layer for each training example.

```
Normal layer:        With dropout (50%, this pass):
  ● ● ● ● ●              ● x ● x ●     (the x'd neurons are ignored this time)
```

This forces the network to *not rely too heavily* on any single neuron, since that neuron might be switched off next time. It effectively trains many slightly different "thinned" networks and averages their behaviour — this makes the final model much more robust and less likely to overfit. A typical dropout rate is 20–50%. (Dropout is only applied during training; at test time, all neurons are used.)

### 12.6 Regularization Technique 4: Early Stopping

**Idea:** keep a separate small chunk of data (the "validation set") that is *not* used for training. After every epoch, check the error on this validation set. As training continues, training error always keeps falling, but validation error will eventually start rising again (see the graph in section 12.1) — this is the moment overfitting begins. **Early stopping simply means: stop training at that point**, and keep the weights from the best validation-error epoch, rather than the final epoch.

### 12.7 Putting it all together

In practice, these techniques are often combined:
- Use a reasonably sized network (not excessively large for the amount of data available).
- Apply L2 regularization and/or dropout during training.
- Monitor validation error and use early stopping.

This combination attacks overfitting from three different angles: keeping weights small (L2), preventing over-reliance on individual neurons (dropout), and stopping training at the right time (early stopping).

### 12.8 Unit summary

- Multi-layer networks solve problems (like XOR) that single-layer models cannot, by combining multiple straight-line decisions.
- Backpropagation trains a multi-layer network by computing, layer by layer from the output backward, how much each weight should change — using the chain rule.
- Cost functions (MSE for regression, cross-entropy for classification) measure how wrong the network is.
- Gradient descent (batch / stochastic / mini-batch) uses these error signals to gradually improve the weights; mini-batch is the standard practical choice.
- Overfitting happens when a network memorises training data instead of learning general patterns; L1/L2 regularization, dropout, and early stopping are the standard defences.

### 12.9 Practice questions for self-study

1. Repeat the backpropagation numerical example from Lectures 9–10, but change the target to t = 0.99 instead of t = 0.01. Show all forward and backward steps.
2. For the function f(w) = (w−5)², perform 4 iterations of gradient descent by hand, starting at w=0 with η=0.3.
3. Explain, in your own words, why L1 regularization tends to produce exactly-zero weights while L2 does not.
4. A network gets 99% training accuracy but only 70% test accuracy. Name three techniques you could apply and explain briefly how each one helps.

---

*End of Unit II — proceed to Unit III: Feedbackward Neural Networks*
