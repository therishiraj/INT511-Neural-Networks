# Unit I — Introduction to Neural Networks

**INT511 – Neural Networks (M.Tech)**
**This unit is designed for Weeks 1–2 of the semester (6 lecture sessions, 3 lectures/week).**

This unit answers four simple questions: What is a neural network? How is it inspired by the brain? How does a network "learn"? And what is the simplest possible neuron model?

---

## How this unit is paced

| Lecture | Topic |
|---|---|
| 1 | What is a Neural Network? Why do we need them? Basic building blocks |
| 2 | Biological inspiration — from brain cell to artificial neuron |
| 3 | Types of learning (supervised, unsupervised, reinforcement, hybrid) |
| 4 | The McCulloch–Pitts neuron and simple logic gates |
| 5 | The Perceptron — model and learning algorithm (with full worked example) |
| 6 | Activation functions (threshold, sigmoid, tanh, ReLU, softmax) + unit wrap-up |

---

## Lecture 1: What Is a Neural Network?

### 1.1 The everyday problem it solves

Suppose you want a computer to look at a photo and say "this is a cat" or "this is a dog." You cannot write a simple `if-else` rule for this — there is no fixed formula that separates all cat photos from all dog photos. What you *can* do is show the computer thousands of labelled photos and let it **figure out the pattern by itself**. A neural network is a mathematical structure that is very good at exactly this kind of pattern-learning.

In one sentence: **a neural network is a machine-learning model, loosely inspired by the brain, that learns a mapping from inputs to outputs by adjusting internal numbers (called weights) based on examples.**

### 1.2 The basic building block: one neuron

Every neural network, however large, is built out of a repeating unit called a **neuron** (or **node**). A single neuron does something very simple:

1. It receives some numbers as input: x1, x2, x3, …
2. It multiplies each input by a "weight" that says how important that input is: w1, w2, w3, …
3. It adds all these weighted inputs together, plus one extra number called the **bias** (b), which just shifts the result up or down.
4. It passes this sum through a small function called the **activation function**, which decides the final output.

Written as a simple formula:

```
z = (w1 × x1) + (w2 × x2) + ... + (wn × xn) + b        <- weighted sum ("net input")
y = f(z)                                                <- activation function decides output
```

Think of it like a decision made by weighing evidence: each input xi is a piece of evidence, wi is how much you trust that evidence, b is your baseline opinion before seeing any evidence, and f is the rule that turns the total evidence into a final yes/no or a number.

**Simple picture:**

```
   x1 ---(w1)---\
   x2 ---(w2)----->  [ SUM: z = w1x1+w2x2+w3x3+b ]  --->  [ f(z) ]  --->  y (output)
   x3 ---(w3)---/
```

### 1.3 Why "network"? Stacking neurons in layers

One neuron can only make a very simple decision. To solve harder problems, we connect many neurons together in **layers**:

- **Input layer**: just holds the raw input values (no computation, just passes data forward).
- **Hidden layer(s)**: one or more layers of neurons that do the actual computation. "Hidden" simply means we don't directly observe their output — it's an internal step.
- **Output layer**: the final layer that gives us the answer (e.g., "cat" or "dog", or a number).

```
  INPUT LAYER      HIDDEN LAYER      OUTPUT LAYER

    x1  ●---------\    ● ---------\
                    \  |            \
    x2  ●------------ ●  ●--------- ●   y (final answer)
                    /  |            /
    x3  ●---------/    ● ---------/

   (every line here carries its own weight)
```

Data flows from left to right — this is why this basic type of network is called a **feedforward network**. Each connection (line) has its own weight, and every neuron has its own bias. Learning simply means: **find good values for all these weights and biases so the network's output matches what we want.**

### 1.4 The three things that make neural networks special

| Property | What it means | Why it matters |
|---|---|---|
| **Many simple units working together** | No single neuron is smart, but thousands of them together can be | If a few neurons "fail," the network still works reasonably well |
| **Information is spread out** | A concept (e.g. "roundness") isn't stored in one neuron — it's a pattern across many neurons | The network generalises well to new, unseen examples |
| **It adapts from data** | Weights change automatically as the network sees more examples | We don't have to hand-write rules |

### 1.5 A short, simple history (just for context)

You don't need to memorise this — just get a feel for the story:

- **1943** — McCulloch & Pitts proposed the first simple mathematical neuron model.
- **1958** — Rosenblatt invented the Perceptron, the first *trainable* neuron.
- **1969** — Minsky & Papert showed a single perceptron cannot solve the XOR problem — interest in neural networks dropped for a while.
- **1986** — The backpropagation algorithm (Unit II) became popular, allowing multi-layer networks to be trained — interest came back.
- **2012 onward** — With more data and powerful GPUs, "deep" networks (many layers) started beating older methods dramatically, leading to today's AI boom (image recognition, ChatGPT-like models, self-driving cars, etc.)

**Takeaway for Lecture 1:** A neural network is just layers of simple neurons, each doing "weighted sum → activation function," and the network learns by adjusting weights and biases from examples.

---

## Lecture 2: Biological Inspiration and Artificial Neurons

### 2.1 The biological neuron (in simple terms)

Your brain has about 86 billion neurons, and each one is a simple cell that communicates with others. It has four important parts:

| Part | What it does |
|---|---|
| **Dendrites** | Branch-like structures that *receive* signals from other neurons |
| **Cell body (Soma)** | *Collects and adds up* all the incoming signals |
| **Axon** | A long fibre that *carries the output signal* away from the cell |
| **Synapse** | The *junction* between one neuron's axon and another's dendrite — this is where the connection strength (like a "weight") lives |

A biological neuron "fires" (sends an output signal) only when the total incoming signal crosses some threshold — otherwise it stays quiet. This "add up inputs, then decide to fire or not" behaviour is exactly what inspired the artificial neuron.

```
   Other neurons' axons
         \   |   /
          \  |  /            (dendrites collect signals)
        ----[SOMA]----  if total signal > threshold --> fires along axon
              |
            (axon)
              |
        signal passed to next neuron's dendrites via a synapse
```

### 2.2 Mapping biology to the artificial neuron

| Biological neuron | Artificial neuron |
|---|---|
| Dendrites (receiving signals) | Inputs x1, x2, …, xn |
| Synapse strength | Weights w1, w2, …, wn |
| Soma adding up signals | Weighted sum, z = Σ wi·xi + b |
| Firing threshold | Activation function f(z) |
| Axon (output signal) | Output y = f(z) |

This is only a loose analogy — real neurons are far more complex (they use electrical spikes, chemical signals, timing patterns, etc.), but the *core idea* of "combine inputs, then decide an output" survives directly into the artificial neuron.

### 2.3 Quick comparison table

| Feature | Biological neuron | Artificial neuron |
|---|---|---|
| Signal type | Electrical spikes | A single real number |
| Speed | Slow (milliseconds) | Extremely fast (nanoseconds) |
| Number of units | ~86 billion in the brain | A few hundred to billions in modern networks |
| Learning | Connections strengthen/weaken based on activity | Weights updated using a learning algorithm (e.g. backpropagation) |
| Power usage | ~20 watts for the entire brain | Can require huge power for large models (GPU clusters) |

**Takeaway for Lecture 2:** The artificial neuron is a deliberately simplified, mathematical version of a biological neuron — same basic idea (combine inputs, decide output), executed with plain arithmetic instead of biology.

---

## Lecture 3: Types of Learning

A neural network needs to be *trained* — that is, it needs a strategy for learning from data. There are four broad categories.

### 3.1 Supervised Learning

**Idea:** You give the model input-output *pairs* — i.e., for every input, you also tell it the correct answer (called the "label"). The model's job is to learn the mapping from input to output so well that it can predict the answer for new, unseen inputs.

**Example:** Show the network 10,000 photos, each labelled "cat" or "dog." It learns to predict the label for a brand-new photo.

**Two flavours:**
- **Classification** — output is a category (cat/dog, spam/not-spam).
- **Regression** — output is a number (house price, temperature tomorrow).

### 3.2 Unsupervised Learning

**Idea:** You only give the model inputs — *no* labels. The model must find structure or patterns on its own, such as grouping similar items together.

**Example:** Given purchase histories of 10,000 customers (no labels at all), group them into customer "segments" that behave similarly. This is called **clustering**.

### 3.3 Reinforcement Learning

**Idea:** There's no fixed dataset of correct answers. Instead, an **agent** takes actions in an **environment** and receives a **reward** (or penalty) signal. Over time, it learns which actions lead to more reward.

**Example:** A robot learns to walk by trying different movements — falling down gives a low reward, walking steadily gives a high reward.

### 3.4 Hybrid Learning (Semi-supervised / Self-supervised)

**Idea:** A mix of the above — usually a *small* amount of labelled data plus a *large* amount of unlabelled data, or the model creates its own "labels" from the data itself (e.g., hide part of a sentence and ask the model to predict the missing word).

**Example:** Modern language models are first trained on huge amounts of unlabelled text (self-supervised), then fine-tuned on a small labelled dataset for a specific task.

### 3.5 Quick comparison table

| Type | Needs labels? | Goal | Example |
|---|---|---|---|
| Supervised | Yes | Predict known output for new input | Spam detection |
| Unsupervised | No | Discover hidden structure | Customer segmentation |
| Reinforcement | No (uses rewards instead) | Learn best actions over time | Game-playing agent, robot control |
| Hybrid | Partially | Combine small labelled + large unlabelled data | Pre-trained language models |

**Takeaway for Lecture 3:** The "type of learning" describes what kind of feedback the model gets while training — an exact answer (supervised), no answer at all (unsupervised), a reward score (reinforcement), or a mix (hybrid).

---

## Lecture 4: The McCulloch–Pitts (M-P) Neuron

### 4.1 The simplest possible neuron model

Before the perceptron, in 1943, McCulloch and Pitts proposed the very first mathematical model of a neuron. It is deliberately very simple:

- All inputs are **binary**: either 0 or 1.
- Inputs are of two kinds: **excitatory** (they push the neuron toward firing) and **inhibitory** (if even one inhibitory input is 1, the neuron is forced to stay off, no matter what).
- There's a fixed **threshold θ** (theta). The neuron fires (output = 1) only if the sum of excitatory inputs is ≥ θ AND no inhibitory input is active.

**Rule, in words:**
> If any inhibitory input is ON, output = 0 (always).
> Otherwise, output = 1 if (sum of excitatory inputs) ≥ θ, else output = 0.

**Important limitation to remember:** the M-P neuron has **no learning** — the weights are always fixed at 1 for excitatory inputs, and there is no procedure to adjust θ from data. It's a *fixed logic circuit*, not a trainable model.

### 4.2 Realising simple logic gates — step by step

**Example 1: AND gate** (output 1 only if both inputs are 1)

Use two excitatory inputs, set θ = 2.

| x1 | x2 | Sum (x1+x2) | Is sum ≥ θ(=2)? | Output y |
|---|---|---|---|---|
| 0 | 0 | 0 | No | 0 |
| 0 | 1 | 1 | No | 0 |
| 1 | 0 | 1 | No | 0 |
| 1 | 1 | 2 | Yes | 1 |

This exactly matches the AND truth table. ✓

**Example 2: OR gate** (output 1 if at least one input is 1)

Use two excitatory inputs, set θ = 1.

| x1 | x2 | Sum | Is sum ≥ θ(=1)? | Output y |
|---|---|---|---|---|
| 0 | 0 | 0 | No | 0 |
| 0 | 1 | 1 | Yes | 1 |
| 1 | 0 | 1 | Yes | 1 |
| 1 | 1 | 2 | Yes | 1 |

Matches OR. ✓

**Example 3: NOT gate** (output is the opposite of the input)

Use one *inhibitory* input, θ = 0.

| x (inhibitory) | Inhibitory input active? | Output y |
|---|---|---|
| 0 | No | 1 (fires, since θ=0 is satisfied and no inhibition) |
| 1 | Yes | 0 (forced off by inhibition) |

Matches NOT. ✓

**Example 4: NOR gate** (output 1 only if both inputs are 0)

Use two *inhibitory* inputs, θ = 0. If either input is 1, the inhibitory rule forces output to 0. If both are 0, no inhibition is active and θ = 0 is trivially satisfied, so output = 1.

| x1 | x2 | Output y |
|---|---|---|
| 0 | 0 | 1 |
| 0 | 1 | 0 |
| 1 | 0 | 0 |
| 1 | 1 | 0 |

Matches NOR. ✓

### 4.3 Why a *single* M-P neuron cannot do XOR

XOR's truth table is: (0,0)→0, (0,1)→1, (1,0)→1, (1,1)→0.

A single M-P neuron (or a single perceptron, as we'll see in Lecture 5) can only draw **one straight line** to separate 0s from 1s. If you plot the XOR points on a graph, the two "1" outputs sit on one diagonal and the two "0" outputs sit on the other diagonal — **no single straight line can separate them**. Try drawing it yourself: put a dot at (0,0)=0, (1,1)=0, and circles at (0,1)=1, (1,0)=1. Any line you draw will always have one dot and one circle on the same side.

**Solution:** Use **two** M-P neurons in a first layer, and combine their outputs with a third M-P neuron. The trick is:

```
XOR(x1, x2) = (x1 AND NOT x2) OR (NOT x1 AND x2)
```

- Neuron 1 computes "x1 AND NOT x2" (x1 excitatory, x2 inhibitory, θ=1)
- Neuron 2 computes "NOT x1 AND x2" (x2 excitatory, x1 inhibitory, θ=1)
- Neuron 3 computes "Neuron1 OR Neuron2" (both excitatory, θ=1)

**Verification table:**

| x1 | x2 | N1 = x1 AND NOT x2 | N2 = NOT x1 AND x2 | y = N1 OR N2 | XOR (expected) |
|---|---|---|---|---|---|
| 0 | 0 | 0 | 0 | 0 | 0 ✓ |
| 0 | 1 | 0 | 1 | 1 | 1 ✓ |
| 1 | 0 | 1 | 0 | 1 | 1 ✓ |
| 1 | 1 | 0 | 0 | 0 | 0 ✓ |

**This is the single most important idea in this unit:** a single neuron/layer can only separate data with one straight line, so problems like XOR that need a *bent/curved* boundary require **more than one layer**. This exact idea reappears as the motivation for Unit II (Multi-Layer Networks).

**Takeaway for Lecture 4:** The M-P neuron is a fixed (non-learning) binary threshold unit. Single gates like AND/OR/NOT are easy with one neuron; XOR needs two layers.

---

## Lecture 5: The Perceptron

### 5.1 From M-P neuron to Perceptron — what changed?

The Perceptron (Rosenblatt, 1958) fixed the biggest weakness of the M-P neuron: **the weights can now be learned from data**, instead of being fixed at 1.

| | M-P Neuron | Perceptron |
|---|---|---|
| Inputs | Binary only | Real numbers allowed |
| Weights | Fixed at 1 | Adjustable, learned from data |
| Learning | None | Yes — a training algorithm |
| Bias | Fixed threshold θ | Learnable bias b |

**Perceptron formula:**

```
z = w1x1 + w2x2 + ... + wnxn + b
y = +1  if z ≥ 0
y = -1  if z < 0
```

(Some textbooks use 0/1 instead of −1/+1 — the idea is identical, just a labelling choice. We will use +1/−1 here since it makes the learning rule simpler to write.)

### 5.2 The Perceptron Learning Algorithm (in plain steps)

We want to find weights w and bias b so that the perceptron's output y matches the desired/target output d for every training example.

**Step-by-step procedure:**

1. Start with small (often zero) initial weights and bias.
2. Pick a learning rate η (eta) — a small positive number, e.g. 0.1 or 1, that controls how big each correction is.
3. For each training example (x, d):
   a. Compute the current output: y = sign(w·x + b)
   b. If y is already correct (y = d), do nothing.
   c. If y is wrong, correct the weights:
      ```
      w_new = w_old + η × d × x
      b_new = b_old + η × d
      ```
4. Repeat step 3 for all examples, again and again (each full pass through all examples is called an **epoch**), until an entire epoch produces zero mistakes.

**Why does this correction make sense?** If the network was wrong, moving the weights a little in the direction of d×x makes the output slightly more likely to be correct next time we see this input. If the network was already correct, we don't want to disturb it, so we do nothing.

### 5.3 Fully worked example: Perceptron learning an AND gate

**Data (using −1/+1 labels):** treat 0 as −1, and 1 as +1 for both inputs and the desired output.

| x1 | x2 | Desired output d |
|---|---|---|
| −1 | −1 | −1 |
| −1 | +1 | −1 |
| +1 | −1 | −1 |
| +1 | +1 | +1 |

Start with w1 = 0, w2 = 0, b = 0, learning rate η = 1.

**Epoch 1**

*Example 1: x=(−1,−1), d=−1*
- z = 0(−1) + 0(−1) + 0 = 0
- sign(0) is taken as +1 (by convention) → y = +1
- y ≠ d (predicted +1, wanted −1) → update:
  - w1 = 0 + 1×(−1)×(−1) = 1
  - w2 = 0 + 1×(−1)×(−1) = 1
  - b = 0 + 1×(−1) = −1

*Example 2: x=(−1,+1), d=−1*
- z = 1(−1) + 1(1) + (−1) = −1 + 1 − 1 = −1
- y = sign(−1) = −1
- y = d ✓ → no update

*Example 3: x=(+1,−1), d=−1*
- z = 1(1) + 1(−1) + (−1) = 1 − 1 − 1 = −1
- y = −1
- y = d ✓ → no update

*Example 4: x=(+1,+1), d=+1*
- z = 1(1) + 1(1) + (−1) = 1 + 1 − 1 = 1
- y = sign(1) = +1
- y = d ✓ → no update

**End of Epoch 1:** w1 = 1, w2 = 1, b = −1. Only 1 mistake was made.

**Epoch 2 (re-check all 4 examples with the new weights):**

- x=(−1,−1): z = −1−1−1 = −3 → y=−1 = d ✓
- x=(−1,+1): z = −1+1−1 = −1 → y=−1 = d ✓
- x=(+1,−1): z = 1−1−1 = −1 → y=−1 = d ✓
- x=(+1,+1): z = 1+1−1 = 1 → y=+1 = d ✓

**Zero mistakes in Epoch 2 → training is complete!**

**Final learned rule:** y = sign(x1 + x2 − 1). This is a straight line x1 + x2 = 1 separating the "AND = +1" point from the other three points — exactly what we expect, since AND is a simple, linearly separable problem.

### 5.4 Does the Perceptron always converge?

**Perceptron Convergence Theorem (stated simply, no proof needed at this level):** If the data can be separated by a straight line (i.e., it is "linearly separable"), the perceptron learning algorithm is *guaranteed* to find such a line in a finite number of corrections, no matter where you start.

**But:** if the data is *not* linearly separable (like XOR), the algorithm will keep making corrections forever and never settle down — it will "oscillate." This is exactly the practical demonstration of the same limitation we saw with the M-P neuron in Lecture 4: **one neuron with one straight-line boundary is not enough for every problem.**

**Takeaway for Lecture 5:** The perceptron is a trainable version of the threshold neuron. Its learning rule nudges weights toward correcting mistakes, and it is guaranteed to succeed *only* when the data is linearly separable.

---

## Lecture 6: Activation Functions

### 6.1 Why do we even need an activation function?

If we simply used z = w·x + b as the final output (no activation function at all), then stacking many layers would collapse into just one big linear function — you would gain nothing by adding layers! The activation function introduces **non-linearity**, which is what allows multi-layer networks to model complicated, curved decision boundaries (like the XOR case from Lecture 4).

### 6.2 The five activation functions in this syllabus

**1. Threshold (Step) function** — the one we used for M-P neuron / perceptron.
```
f(z) = 1 if z ≥ 0
f(z) = 0 (or -1) if z < 0
```
- Output is strictly 0/1 (or −1/+1) — hard decision, no in-between.
- Problem: it has no useful "slope" (the derivative is 0 everywhere except at z=0, where it's undefined), so it cannot be used with gradient-based learning algorithms like backpropagation (Unit II). Good only for simple, single-layer models.

**2. Sigmoid function**
```
f(z) = 1 / (1 + e^(-z))
```
- Output is always between 0 and 1 — can be read as a "probability."
- Smooth and differentiable everywhere, so gradient-based learning works.
- Its derivative has a very convenient shortcut: f'(z) = f(z) × (1 − f(z))
- Downside: for very large positive or negative z, the curve becomes almost flat, so the gradient becomes tiny ("vanishing gradient") and learning slows down.

**3. Tanh (hyperbolic tangent) function**
```
f(z) = (e^z - e^(-z)) / (e^z + e^(-z))
```
- Output is between −1 and +1.
- Like sigmoid, but centred around 0, which often helps the network learn faster.
- Derivative: f'(z) = 1 − f(z)²
- Still suffers from vanishing gradients for large |z|, though usually less severely than sigmoid.

**4. ReLU (Rectified Linear Unit)**
```
f(z) = z    if z > 0
f(z) = 0    if z ≤ 0
```
- Extremely simple and fast to compute.
- Derivative is 1 for z > 0 and 0 for z ≤ 0 — no vanishing-gradient problem for positive inputs, which is why ReLU is the default choice in most modern deep networks.
- Downside: if a neuron's input is always negative, it "dies" (always outputs 0 and never updates again) — this is called the "dying ReLU" problem.

**5. Softmax function** (used only in the output layer, for multi-class classification)
```
For outputs z1, z2, ..., zk:
softmax(zi) = e^(zi) / (e^(z1) + e^(z2) + ... + e^(zk))
```
- Converts a list of raw scores into a list of probabilities that all add up to 1.
- Used when the network must choose *one* class out of many (e.g., "which digit, 0–9, is this?").

### 6.3 Quick comparison table

| Activation | Output range | Smooth (usable with gradient learning)? | Typical use |
|---|---|---|---|
| Threshold | {0,1} or {−1,+1} | No | Simple perceptron only |
| Sigmoid | (0, 1) | Yes | Binary classification output, older hidden layers |
| Tanh | (−1, 1) | Yes | Hidden layers (better than sigmoid, still can vanish) |
| ReLU | [0, ∞) | Yes (except exactly at 0) | Default choice for hidden layers today |
| Softmax | (0,1), all outputs sum to 1 | Yes | Output layer for multi-class classification |

### 6.4 Step-by-step numerical example

Let's compute all activation outputs for a single neuron with:
```
x1 = 0.5, x2 = -1.0, x3 = 2.0
w1 = 0.4, w2 = -0.6, w3 = 0.3
b = -0.2
```

**Step 1 — Compute z (the weighted sum):**
```
z = (0.4 × 0.5) + (-0.6 × -1.0) + (0.3 × 2.0) + (-0.2)
z = 0.20 + 0.60 + 0.60 - 0.20
z = 1.2
```

**Step 2 — Apply each activation function to z = 1.2:**

*Threshold:* Since z = 1.2 ≥ 0, output = 1.

*Sigmoid:*
```
f(1.2) = 1 / (1 + e^(-1.2))
e^(-1.2) ≈ 0.3012
f(1.2) = 1 / 1.3012 ≈ 0.7685
```

*Tanh:*
```
e^(1.2) ≈ 3.3201
e^(-1.2) ≈ 0.3012
f(1.2) = (3.3201 - 0.3012) / (3.3201 + 0.3012) = 3.0189 / 3.6213 ≈ 0.8337
```

*ReLU:* Since z = 1.2 > 0, output = z = 1.2

**Step 3 — Softmax example (needs more than one output, so let's use 3 raw scores):**

Suppose the output layer produces three raw scores: z1 = 2.0, z2 = 1.0, z3 = 0.1

```
e^(2.0) ≈ 7.389
e^(1.0) ≈ 2.718
e^(0.1) ≈ 1.105
Sum = 7.389 + 2.718 + 1.105 = 11.212

softmax(z1) = 7.389 / 11.212 ≈ 0.659
softmax(z2) = 2.718 / 11.212 ≈ 0.242
softmax(z3) = 1.105 / 11.212 ≈ 0.099

Check: 0.659 + 0.242 + 0.099 = 1.000 ✓
```

This tells us: the model is 65.9% confident in class 1, 24.2% confident in class 2, and 9.9% confident in class 3.

### 6.5 Unit summary

- A neural network is built from simple neurons (weighted sum + activation function) arranged in layers.
- The artificial neuron loosely copies how a biological neuron combines and fires signals.
- Networks learn using one of: supervised, unsupervised, reinforcement, or hybrid learning.
- The M-P neuron is the earliest, non-learning threshold model; it can realise simple gates but not XOR with a single unit.
- The perceptron adds learnable weights and a training rule, and is guaranteed to converge only on linearly separable data.
- Activation functions (threshold, sigmoid, tanh, ReLU, softmax) add non-linearity, which is essential for solving problems that a single straight line cannot solve.

### 6.6 Practice questions for self-study

1. Design an M-P neuron for the NAND gate. Verify your design against the full truth table.
2. Run the perceptron learning algorithm by hand for the OR gate, starting from zero weights, and show all epochs until convergence.
3. For a neuron with x=(1, 2), w=(0.5, -0.3), b=0.1, compute z, and then compute the sigmoid, tanh, and ReLU outputs step by step.
4. Explain in your own words why a single perceptron cannot learn XOR, using the "straight line" argument.

---

*End of Unit I — proceed to Unit II: Feedforward Neural Networks*
