# Unit III — Feedbackward (Feedback) Neural Networks

**INT511 – Neural Networks (M.Tech)**
**This unit is designed for Weeks 5–6 of the semester (6 lecture sessions).**

So far, every network we've seen has been **feedforward** — data flows strictly in one direction, input to output, and each output is computed fresh every time. In this unit, we look at **feedback networks**, where a neuron's output can loop back and influence the network again. These networks behave more like a "memory" that settles into a stable answer, rather than a one-shot calculator.

---

## How this unit is paced

| Lecture | Topic |
|---|---|
| 13 | What is a feedback network? Feedforward vs. feedback, associative memory |
| 14 | The Discrete Hopfield Network — architecture and storing patterns |
| 15 | The Energy Function — how a Hopfield network "settles" (with numerical example) |
| 16 | Storage capacity and spurious states |
| 17 | The Boltzmann Machine — adding randomness |
| 18 | Restricted Boltzmann Machines (RBM) + unit wrap-up |

---

## Lecture 13: What Is a Feedback Network?

### 13.1 Feedforward vs. Feedback — the key difference

In a **feedforward** network (Units I and II), information travels strictly forward: input → hidden → output. Once you compute the output, you're done — there's no looping back.

In a **feedback (recurrent)** network, a neuron's output can be fed back as an input — either to itself or to other neurons in the network — creating a loop. This means the network doesn't just compute one thing and stop; it keeps updating itself over several steps, gradually **settling down** into a stable pattern.

```
FEEDFORWARD:                       FEEDBACK:

  x --> [Neuron] --> y               x --> [Neuron] --> y
        (one pass,                          ▲            │
         then done)                         └────────────┘
                                    (output loops back as input again)
```

### 13.2 A simple analogy: associative memory

Think about how your own memory works. If a friend says just the first few notes of a song, you often "recall" the entire tune from a small hint. You aren't searching through an index — you're taking a partial, noisy cue and completing it into a full, familiar pattern.

This is exactly what a feedback network like the **Hopfield network** does: you give it a partial or noisy pattern, and it iteratively updates itself until it "recalls" (settles into) the complete, correct stored pattern. This is called **associative memory** or **content-addressable memory** — you retrieve information by its *content* (a piece of it), not by a fixed address.

### 13.3 Systematic comparison

| Aspect | Feedforward network | Feedback network |
|---|---|---|
| Connections | One direction only, no cycles | Contains loops/cycles |
| Computation | Input → single pass → output | Input → repeated updates → stable state |
| Typical use | Classification, regression | Associative memory, optimisation, pattern completion |
| Example | MLP (Unit II) | Hopfield Network, Boltzmann Machine |

**Takeaway for Lecture 13:** A feedback network loops information back on itself, so instead of producing one instant answer, it *evolves* over time toward a stable, settled state — which is exactly how associative memory works.

---

## Lecture 14: The Discrete Hopfield Network

### 14.1 Architecture

A Hopfield network is a group of neurons that are **all connected to each other** (except that no neuron connects to itself), with the following features:

- Every neuron's state is either **+1** or **−1** (we use this "bipolar" style, rather than 0/1, because the math works out more cleanly).
- The connection weight from neuron i to neuron j is the *same* as from j to i — the weights are **symmetric**: wij = wji.
- No neuron connects to itself: wii = 0.

```
         ┌──────────────────────────┐
         │                          │
       (1)◄────────────────────────►(2)
         │  ╲                    ╱  │
         │    ╲                ╱    │
         │      ╲            ╱      │
         │        ╲        ╱        │
       (4)◄─────────╲────╱─────────►(3)
                      ╳
             (every neuron connects to every other,
              weight from i to j = weight from j to i)
```

### 14.2 Storing a pattern: the Hebbian rule

To make the network "remember" a pattern, we use a very simple rule inspired by the biological idea "neurons that fire together, wire together" (Hebb's rule). For a pattern ξ = (ξ1, ξ2, …, ξN) with each ξi = +1 or −1:

```
wij = (1/N) × ξi × ξj      (for i ≠ j)
wii = 0
```

**In words:** if two neurons have the *same* sign in the stored pattern, their connecting weight becomes positive (they "excite" each other). If they have *opposite* signs, the weight becomes negative (they "inhibit" each other). This is exactly what allows the network to "remember" the pattern.

### 14.3 Step-by-step example: storing one pattern

Let's store the 4-neuron pattern: **ξ = (1, 1, −1, −1)**. Here N = 4.

**Step 1 — Compute w12:**
```
w12 = (1/4) × ξ1 × ξ2 = (1/4) × (1)×(1) = 0.25
```

**Step 2 — Compute w13:**
```
w13 = (1/4) × ξ1 × ξ3 = (1/4) × (1)×(−1) = −0.25
```

**Step 3 — Compute w14:**
```
w14 = (1/4) × ξ1 × ξ4 = (1/4) × (1)×(−1) = −0.25
```

**Step 4 — Compute w23:**
```
w23 = (1/4) × ξ2 × ξ3 = (1/4) × (1)×(−1) = −0.25
```

**Step 5 — Compute w24:**
```
w24 = (1/4) × ξ2 × ξ4 = (1/4) × (1)×(−1) = −0.25
```

**Step 6 — Compute w34:**
```
w34 = (1/4) × ξ3 × ξ4 = (1/4) × (−1)×(−1) = 0.25
```

**Final weight matrix (symmetric, diagonal = 0):**

| | 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| **1** | 0 | 0.25 | −0.25 | −0.25 |
| **2** | 0.25 | 0 | −0.25 | −0.25 |
| **3** | −0.25 | −0.25 | 0 | 0.25 |
| **4** | −0.25 | −0.25 | 0.25 | 0 |

This weight matrix now "contains" the memory of the pattern (1, 1, −1, −1). In the next lecture, we'll see how the network uses this matrix to recall the pattern even from a corrupted (noisy) version.

**Takeaway for Lecture 14:** A Hopfield network stores patterns using simple pairwise "same-sign-attracts, opposite-sign-repels" weights, computed once, with no iterative training needed.

---

## Lecture 15: The Energy Function — How the Network "Settles"

### 15.1 Retrieving a stored pattern: the update rule

Once the weights are set, we can "ask" the network to recall a pattern by giving it a starting state (possibly a corrupted/noisy version of a stored pattern) and letting it update itself. The **update rule** for one neuron i is:

```
si_new = +1   if (sum over all j of wij × sj) ≥ 0
si_new = −1   if (sum over all j of wij × sj) <  0
```

This sum (Σj wij×sj) is called the **local field** for neuron i. We update neurons **one at a time** (this is called **asynchronous updating**), and we repeat until no neuron's state changes any more — at that point, the network has "settled."

### 15.2 Step-by-step retrieval example

Using the weight matrix from Lecture 14 (which stored the pattern (1, 1, −1, −1)), suppose we present a **noisy** version where bit 2 has been flipped: **(1, −1, −1, −1)**.

**Step 1 — Compute the local field for neuron 2:**
```
local field for neuron 2 = w21×s1 + w23×s3 + w24×s4
                          = (0.25)(1) + (−0.25)(−1) + (−0.25)(−1)
                          = 0.25 + 0.25 + 0.25
                          = 0.75
```

**Step 2 — Since 0.75 ≥ 0, update neuron 2 to +1:**
```
s2_new = +1
```

**Step 3 — New state:** (1, 1, −1, −1) — **this is exactly the original stored pattern!** The network has corrected the single-bit error entirely on its own.

If we check any other neuron now, its local field will still point in the same direction (no more changes happen) — the network has **converged** (settled into a stable state).

### 15.3 The Energy Function — why does this always work?

The Hopfield network has a special quantity called the **energy function**:

```
E = −(1/2) × Σi Σj (wij × si × sj)        [summing over all i ≠ j]
```

**Key fact (which we will verify numerically, not prove formally):** every time a neuron updates using the rule from Section 15.1, this energy E either **decreases or stays the same** — it never increases. Since E cannot keep decreasing forever (it's bounded), the network is guaranteed to eventually stop changing — i.e., **converge to a stable state**. Think of E as describing a landscape with hills and valleys; the network update rule always rolls the state "downhill," and a stored pattern sits at the bottom of a valley.

### 15.4 Verifying that energy decreased in our example

**Energy of the noisy state (1, −1, −1, −1), before the update:**

We compute wij×si×sj for every pair i<j, then sum and take the negative (since −(1/2)×2×[sum over i<j] = −[sum over i<j]):

```
Pair (1,2): w12×s1×s2 = 0.25×(1)×(−1)  = −0.25
Pair (1,3): w13×s1×s3 = −0.25×(1)×(−1) =  0.25
Pair (1,4): w14×s1×s4 = −0.25×(1)×(−1) =  0.25
Pair (2,3): w23×s2×s3 = −0.25×(−1)×(−1)= −0.25
Pair (2,4): w24×s2×s4 = −0.25×(−1)×(−1)= −0.25
Pair (3,4): w34×s3×s4 = 0.25×(−1)×(−1) =  0.25

Sum = −0.25+0.25+0.25−0.25−0.25+0.25 = 0.00
E(noisy) = −(sum) = 0.00
```

**Energy of the corrected state (1, 1, −1, −1), after the update:**

```
Pair (1,2): 0.25×(1)×(1)   =  0.25
Pair (1,3): −0.25×(1)×(−1) =  0.25
Pair (1,4): −0.25×(1)×(−1) =  0.25
Pair (2,3): −0.25×(1)×(−1) =  0.25
Pair (2,4): −0.25×(1)×(−1) =  0.25
Pair (3,4): 0.25×(−1)×(−1) =  0.25

Sum = 0.25×6 = 1.50
E(corrected) = −1.50
```

**Energy went from 0.00 down to −1.50** — a clear decrease, exactly as the theory predicts! The network moved "downhill" into the valley representing the stored memory.

**Takeaway for Lecture 15:** Every update in a Hopfield network moves the system to lower (or equal) energy. The stored patterns sit at the bottom of energy "valleys," which is why the network reliably corrects noisy inputs back to the nearest stored memory.

---

## Lecture 16: Storage Capacity and Spurious States

### 16.1 How many patterns can we store?

A Hopfield network with N neurons cannot store an unlimited number of patterns reliably. As a rule of thumb (from statistical analysis, which we won't derive in detail here):

```
Maximum reliable capacity ≈ 0.15 × N patterns
```

**Example:** for a network with N = 100 neurons, we can reliably store around 0.15×100 ≈ **15 patterns**. If we try to store many more than this, the network starts making retrieval mistakes — stored patterns get corrupted or confused with each other.

### 16.2 Spurious states

Even with only a few stored patterns, the network's energy landscape can develop **extra valleys that don't correspond to any pattern we intentionally stored**. These are called **spurious states**, and the network can mistakenly settle into one of them instead of a real memory. Two common kinds:

1. **Reversed states:** if ξ = (1, 1, −1, −1) is stored, its exact opposite, (−1, −1, 1, 1), is *automatically* also a stable state. Why? Because every term in the energy formula involves the *product* si×sj — flipping the sign of every neuron doesn't change any of these products, so the energy is exactly the same. This is a harmless kind of spurious state (it's just the same memory, "flipped").

2. **Mixture states:** when multiple patterns are stored together, the network can settle into a stable state that is a kind of "blend" of an odd number of stored patterns (e.g., a combination of 3 stored patterns) — this doesn't correspond to anything meaningful and is considered a genuine retrieval error.

### 16.3 Ways to reduce these problems

- Store fewer patterns (well within the 0.15×N limit).
- Make sure stored patterns are not too similar to each other (very similar patterns are more likely to get confused).
- Use more advanced training rules (beyond the simple Hebbian rule) that specifically try to make spurious states less stable — this is an active area of improvement, but not required at this level.

**Takeaway for Lecture 16:** A Hopfield network can only reliably store roughly 15% of its neuron count as patterns; beyond that, retrieval becomes unreliable, and extra "fake" stable states (reversed or mixture states) can appear even with few stored patterns.

---

## Lecture 17: The Boltzmann Machine

### 17.1 Why add randomness at all?

A big limitation of the Hopfield network: once it starts rolling "downhill" in the energy landscape, it can only ever end up in the *nearest* valley — even if that valley is a bad (spurious) one, and a much better valley exists just a little further away. It has no way to "escape" a poor local solution.

The **Boltzmann Machine** fixes this by making neuron updates **probabilistic (random)** rather than a hard yes/no rule. This lets the network occasionally accept a move that *increases* energy slightly, which gives it a chance to escape a poor local valley and potentially find a better one.

### 17.2 The stochastic update rule

Instead of always setting a neuron deterministically, we set it to +1 with a *probability* that depends on its local field ΔEi and a parameter called **temperature (T)**:

```
P(neuron i becomes +1) = 1 / (1 + e^(−ΔEi / T))
```

This is the same sigmoid-shaped formula we saw in Unit I! Let's understand the role of temperature T with a numerical example.

### 17.3 Step-by-step example: effect of temperature

Suppose a neuron's local field is ΔEi = 2. Let's compute the probability of it turning ON (+1) at three different temperatures.

**At T = 1 (normal/moderate temperature):**
```
P = 1 / (1 + e^(−2/1)) = 1 / (1 + e^(−2)) = 1 / (1 + 0.1353) = 1 / 1.1353 ≈ 0.881
```
The neuron is quite likely (88.1%), but not certain, to turn on.

**At T = 5 (high temperature — more randomness):**
```
P = 1 / (1 + e^(−2/5)) = 1 / (1 + e^(−0.4)) = 1 / (1 + 0.6703) = 1 / 1.6703 ≈ 0.599
```
The probability is now much closer to 50-50 — the neuron's behaviour is much more random, regardless of what its local field is telling it to do.

**At T = 0.1 (very low temperature — nearly deterministic):**
```
P = 1 / (1 + e^(−2/0.1)) = 1 / (1 + e^(−20)) ≈ 1 / (1 + 0.0000000021) ≈ 1.000
```
At very low temperature, the behaviour becomes almost exactly like the deterministic Hopfield rule (fires with near-certainty since ΔEi > 0).

### 17.4 Simulated annealing: the practical training trick

Because high temperature = more exploration (escaping bad valleys), and low temperature = more precise settling, a common strategy is to **start with a high temperature and gradually lower it** during training/inference. This is called **simulated annealing** — the name comes from how blacksmiths slowly cool hot metal to help it settle into a strong, stable structure. Starting hot lets the network explore many possibilities; cooling down lets it lock in a good solution.

### 17.5 Boltzmann Machine vs. Hopfield Network — summary

| Aspect | Hopfield Network | Boltzmann Machine |
|---|---|---|
| Neuron update | Deterministic (fixed rule) | Probabilistic (depends on temperature) |
| Can escape bad local valleys? | No | Yes, especially at higher temperature |
| Extra hidden units? | No (only visible units) | Often yes (hidden units, see Lecture 18) |
| Typical use | Associative memory, simple optimisation | Learning probability distributions, harder optimisation problems |

**Takeaway for Lecture 17:** The Boltzmann Machine replaces the Hopfield network's rigid, deterministic update with a probabilistic one controlled by temperature — allowing it to escape poor local solutions, especially when combined with simulated annealing.

---

## Lecture 18: Restricted Boltzmann Machines (RBM) and Unit Wrap-Up

### 18.1 Visible and hidden units

A full Boltzmann Machine connects *every* neuron to *every other* neuron, which makes it very slow to train. A **Restricted Boltzmann Machine (RBM)** simplifies this by splitting neurons into two groups:

- **Visible units** — represent the actual data we feed in (e.g., pixel values of an image).
- **Hidden units** — represent learned, internal features that are *not* directly observed; they capture patterns/structure in the visible data.

**The "restriction":** connections are allowed *only between* visible and hidden units — there are **no** visible-to-visible or hidden-to-hidden connections. This creates a simple two-layer, "bipartite" structure:

```
   Hidden units:    (h1)     (h2)     (h3)
                      │ ╲   ╱  │  ╲   ╱ │
                      │   ╲╱   │    ╲╱  │
                      │   ╱╲   │    ╱╲  │
                      │ ╱   ╲  │  ╱   ╲ │
   Visible units:   (v1)     (v2)     (v3)

   (Every visible unit connects to every hidden unit,
    but NO visible-visible or hidden-hidden connections)
```

### 18.2 Why this restriction is useful

Because there are no connections *within* the visible layer or *within* the hidden layer, all the hidden units can be updated **at the same time** (simultaneously), given the visible units — and vice versa. This makes training dramatically faster than a fully-connected Boltzmann Machine, using a method called **Contrastive Divergence** (the details of this training algorithm are beyond this unit's scope — what matters here is the *intuition*: the RBM repeatedly compares "what the hidden units predict from real data" against "what the visible units predict when we run the network on its own imagination," and nudges the weights to make these two match more closely).

### 18.3 What can an RBM learn?

Once trained, the hidden units learn to represent useful, higher-level features of the data. For example, if trained on handwritten digit images, some hidden units might learn to detect "a loop in the top half" or "a straight vertical stroke" — building blocks that combine to represent complete digits.

### 18.4 Unit summary

- Feedback networks loop information back into themselves and settle into stable states, unlike feedforward networks which compute one instant output.
- A **Discrete Hopfield Network** stores patterns using a simple Hebbian (outer-product) weight rule, and retrieves them by repeatedly updating neurons until the network settles.
- The **energy function** always decreases (or stays the same) with every update, guaranteeing the network reaches a stable state — we verified this numerically.
- Storage capacity is limited (roughly 0.15×N patterns), and spurious states (reversed or mixture patterns) can appear.
- A **Boltzmann Machine** adds randomness (controlled by temperature) to the update rule, allowing it to escape poor local solutions — especially useful with simulated annealing.
- A **Restricted Boltzmann Machine (RBM)** splits neurons into visible and hidden layers with no within-layer connections, making training much faster and enabling it to learn useful hidden features from data.

### 18.5 Practice questions for self-study

1. Store the pattern ξ = (1, −1, 1, −1) in a 4-neuron Hopfield network. Compute the full weight matrix by hand.
2. Using the weight matrix from Q1, present the noisy input (1, −1, −1, −1) (one bit flipped) and show, step by step, how the network corrects it.
3. For a neuron with local field ΔEi = 3, compute the probability of it turning ON at T = 1, T = 3, and T = 10. Explain the trend you observe in your own words.
4. In your own words, explain why an RBM trains faster than a fully-connected Boltzmann Machine.

---

*End of Unit III — proceed to Unit IV: Dimensionality Reduction Techniques*
