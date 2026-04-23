# Deep Q-Network (DQN): A Hands-On Tutorial with CartPole

---

## The Essential Idea (Read This First)

In the Q-learning tutorial, we stored a lookup table: one row per state, one column per action. That works perfectly when states are discrete and countable (a 4×4 maze has 16 states). CartPole is different — the state is a vector of four continuous numbers (cart position, cart velocity, pole angle, pole angular velocity), which means the state space is **infinite**. No table can store infinite rows.

**The DQN insight in one sentence:**
> Instead of a lookup table, use a neural network that *approximates* the Q-function: it takes a state vector as input and outputs one Q-value per action.

The Q-function is now `Q(s, a; θ)` where `θ` are the neural network weights. Training means adjusting `θ` so the network's Q-value predictions satisfy the Bellman equation — just like Q-learning, but via gradient descent instead of a table update.

**The same Bellman target as Q-learning:**
```
Target = r + γ · max_a' Q(s', a'; θ)
Loss   = (Target − Q(s, a; θ))²
```

**But two problems arise immediately** when you naively train a neural network this way:

1. **Correlated samples** — consecutive environment steps are highly correlated (the cart is in similar positions second after second). Neural networks assume i.i.d. training data. Feeding correlated sequences causes the network to overfit locally and forget everything else.

2. **Moving target** — the same network `θ` is used both to *produce* the target and to *be updated* toward it. Every update changes the target, making training wildly unstable — like trying to catch a bus that moves every time you get close.

DQN fixes both problems with two specific mechanisms:
- **Experience Replay** → solves correlated samples
- **Target Network** → solves moving target

These are the two ideas that made DQN work. Everything else is standard deep learning.

---

## Part 1: Easy Concepts First

### 1.1 The CartPole Problem

CartPole is a classic control problem: a pole is balanced on a cart that slides on a frictionless track.

```
        |
       /|\
      / | \
     /  |  \
----[  cart ]----
```

**State (4 numbers):**
| Index | Variable | Typical Range |
|-------|----------|---------------|
| 0 | Cart position | [-4.8, 4.8] |
| 1 | Cart velocity | (-∞, +∞) |
| 2 | Pole angle (radians) | [-0.418, 0.418] |
| 3 | Pole angular velocity | (-∞, +∞) |

**Actions (2 choices):**
- `0` = Push cart LEFT
- `1` = Push cart RIGHT

**Reward:** +1 for every timestep the pole stays upright.

**Episode ends when:**
- Pole angle > 12° from vertical, OR
- Cart moves > 2.4 units from center, OR
- 500 timesteps reached (success!)

**Goal:** Keep the pole balanced for 500 steps consistently.

---

### 1.2 Why Not Q-Table?

With continuous state space, two states like `[0.01, 0.002, -0.003, 0.001]` and `[0.011, 0.002, -0.003, 0.001]` are practically identical, but a lookup table treats them as completely different entries with no relationship. A neural network can *generalize*: similar inputs produce similar outputs, so it interpolates between seen states to handle unseen ones.

---

### 1.3 The Q-Network Architecture

We use a simple 3-layer fully-connected network:

```
Input (4 values)
      ↓
[Dense 128, ReLU]
      ↓
[Dense 128, ReLU]
      ↓
Output (2 values: Q for LEFT, Q for RIGHT)
```

- Input = the state vector `[cart_pos, cart_vel, pole_angle, pole_vel]`
- Output = Q-values for each action, simultaneously
- We pick the action with the highest output Q-value

This is more efficient than one forward pass per action — we get all action Q-values in a single pass.

---

### 1.4 Experience Replay Buffer

The replay buffer is just a fixed-size queue (a deque) that stores past transitions:

```
Buffer = deque(maxlen=10000)
Each entry: (state, action, reward, next_state, done)
```

**Training:** Instead of learning from the most recent step, we randomly sample a **mini-batch** of 64 transitions from the buffer. This breaks temporal correlation — the batch now contains transitions from many different points in time, which is approximately i.i.d.

**Intuition:** Like a student who doesn't study by reading the textbook cover-to-cover in order, but by randomly reviewing flashcards from anywhere in the book.

---

### 1.5 Epsilon-Greedy (Same as Q-Learning)

Same concept as before: start with ε = 1.0 (fully random) and decay toward 0.01. The agent explores early and exploits later.

---

## Part 2: The Two Hard Innovations

> ⚠️ **FLAGGED — This section contains the conceptually difficult parts of DQN. Read carefully.**

### 2.1 Why Training a Q-Network Is Unstable by Default

Standard supervised learning has stable targets. If you're classifying images, the label for "cat" doesn't change during training. But in Q-learning, the target for state `s` is:

```
Target(s, a) = r + γ · max_a' Q(s', a'; θ)
```

This target depends on `Q(s', a'; θ)` — **the same weights θ that we are updating**. Every gradient step changes θ, which changes every target everywhere in the state space simultaneously. The network chases a moving signal and often diverges or oscillates.

> ⚠️ **FLAGGED — Target Network (Hard Concept):**

**The Target Network fix:** Keep two copies of the network:
- **Online network** `Q(s, a; θ)` — trained every step via gradient descent
- **Target network** `Q(s, a; θ⁻)` — frozen copy, only periodically updated (e.g., every 100 steps by copying `θ → θ⁻`)

The training loss uses the **target network** to compute targets, and the **online network** for predictions:

```
Target  = r + γ · max_a' Q(s', a'; θ⁻)    ← uses frozen target network
Loss    = (Target − Q(s, a; θ))²            ← online network is updated
```

This keeps the target stable for 100 steps at a time, giving training something consistent to aim for. The target network "catches up" periodically via hard copy.

> ⚠️ **FLAGGED — Subtle point:** You might wonder: if the target network is frozen, isn't it always outdated? Yes — deliberately. A slightly stale but stable target is much better than a perfectly current but wildly oscillating one.

---

### 2.2 Why Correlated Samples Hurt

Imagine the cart is tilting right for 50 consecutive steps. Feeding these 50 transitions in order to the network means 50 consecutive gradient updates all pushing the weights in the same direction. The network overfits to "rightward-tilt situations" and degrades for everything else. This is called **catastrophic forgetting**.

> ⚠️ **FLAGGED — Experience Replay (Hard Concept):**

**The Replay Buffer fix:** Store all transitions in a buffer. Sample randomly. The key property:

```
A mini-batch from random sampling spans many different situations
→ gradients are diverse
→ no systematic bias toward recent experience
→ each experience can be reused many times (sample efficiency)
```

**Minimum buffer fill:** We typically don't start training until the buffer has at least `batch_size` (e.g., 64) entries. Some implementations wait for 1000 entries to ensure sufficient diversity.

---

### 2.3 The Full DQN Loss Function

```
For a mini-batch of N transitions {(sᵢ, aᵢ, rᵢ, s'ᵢ, doneᵢ)}:

yᵢ = rᵢ                                   if doneᵢ = True
yᵢ = rᵢ + γ · max_a' Q(s'ᵢ, a'; θ⁻)     if doneᵢ = False

Loss = (1/N) · Σ (yᵢ − Q(sᵢ, aᵢ; θ))²
```

This is just Mean Squared Error between the Bellman target `yᵢ` and the network's prediction. We backpropagate through the online network only (the target `yᵢ` is treated as a constant — we **do not** backpropagate through `θ⁻`).

---

## Part 3: The Full DQN Algorithm

```
Initialize online network Q(s,a; θ) with random weights
Initialize target network Q(s,a; θ⁻) = copy of online network
Initialize replay buffer D (capacity N)
Set ε = 1.0

For each episode:
    s ← env.reset()
    
    For each timestep t:
        1. Select action:
           With prob ε:  a = random action
           Otherwise:    a = argmax_a Q(s, a; θ)
        
        2. Execute a in env → observe r, s', done
        
        3. Store (s, a, r, s', done) in D
        
        4. If |D| >= batch_size:
           - Sample random mini-batch from D
           - Compute targets yᵢ using θ⁻
           - Compute loss = MSE(yᵢ, Q(sᵢ, aᵢ; θ))
           - Update θ via gradient descent
        
        5. Every C steps: θ⁻ ← θ  (sync target network)
        
        6. s ← s'
        
        7. If done: break
    
    Decay ε
```

---

## Part 4: DQN vs Q-Learning — Side-by-Side

| Aspect | Q-Learning | DQN |
|--------|-----------|-----|
| State representation | Discrete index | Continuous vector |
| Q-function storage | Table (n_states × n_actions) | Neural network weights θ |
| Update method | Bellman table update | Gradient descent on MSE loss |
| Generalization | None (each state independent) | Yes (similar states → similar Q) |
| Correlated data | Not a problem (tabular) | Fixed by experience replay |
| Unstable targets | Not a problem (tabular) | Fixed by target network |
| Scalability | Fails beyond ~10K states | Scales to continuous/image states |

---

## Part 5: Key Hyperparameters

> ⚠️ **FLAGGED — DQN has more hyperparameters than Q-learning, and they interact.**

| Parameter | Typical Value | Role |
|-----------|--------------|------|
| Buffer size | 10,000 – 100,000 | More = more diversity; too large = stale data |
| Batch size | 32 – 128 | Stability vs. speed tradeoff |
| Target sync interval | 50 – 1000 steps | Smaller = more current but less stable |
| Learning rate | 1e-4 – 1e-3 | Adam optimizer; too high = divergence |
| γ (discount) | 0.99 | Same as Q-learning |
| ε decay | Episode-based or step-based | How fast to stop exploring |
| Network size | 64–256 neurons per layer | CartPole needs small; Atari needs CNN |

The most common failure mode is a **learning rate too high** combined with a **target sync interval too small** — the targets shift faster than the network can track them, causing oscillation.

---

## Part 6: What DQN Doesn't Solve

DQN still has known limitations, each addressed by subsequent algorithms:

- **Overestimates Q-values** because `max` is biased upward → fixed by **Double DQN** (DDQN)
- **All transitions equally sampled** → fixed by **Prioritized Experience Replay** (PER)
- **Single scalar Q-value** → **Dueling DQN** separates value V(s) and advantage A(s,a)
- **Still off-policy, discrete actions only** → Policy gradient methods (PPO, SAC) for continuous control

---

## Concept Map

```
                    ┌─────────────────────────┐
                    │        DQN Agent         │
                    └─────────┬───────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
      ε-greedy policy   Online Network   Target Network
      (exploration)     Q(s,a; θ)        Q(s,a; θ⁻)
              │               │               │
              │               │ gradient      │ frozen
              │               │ descent       │ (sync every C steps)
              ▼               ▼               ▼
         Environment     Bellman Loss ←── Stable Targets
              │
              ▼
      Experience Buffer
      (s, a, r, s', done)
      random mini-batch
      → breaks correlation
```

---

## ✅ Quiz: Check Your Understanding

**Q1.** Why can't we use a Q-table for CartPole?

*(a) The action space is too large (millions of actions)*
*(b) The state space is continuous — infinite possible states that can't be enumerated*
*(c) CartPole is too fast for tabular methods*
*(d) Q-tables don't support rewards above +1*

---

**Q2.** The target network is:

*(a) A second neural network trained simultaneously on different data*
*(b) A periodically-updated frozen copy of the online network, used to compute stable Bellman targets*
*(c) The final policy network deployed after training*
*(d) A network that predicts rewards directly*

---

**Q3.** Consider a DQN with γ = 0.99 and target network weights θ⁻. The agent is in state `s`, takes action LEFT, receives reward `r = 1`, and transitions to `s'` (not terminal). The target network outputs `Q(s', LEFT; θ⁻) = 20` and `Q(s', RIGHT; θ⁻) = 25`. What is the Bellman target `y`?

*(a) 1 + 0.99 × 20 = 20.8*
*(b) 1 + 0.99 × 25 = 25.75*
*(c) 1 + 0.99 × (20 + 25) / 2 = 23.275*
*(d) 0.99 × 25 = 24.75*

---

**Q4.** Why do we sample random mini-batches from the replay buffer instead of learning from the most recent transition?

*(a) Random sampling is faster computationally*
*(b) The most recent transition has the highest reward*
*(c) Random sampling breaks temporal correlation, preventing the network from overfitting to recent experience*
*(d) The most recent transitions haven't been assigned correct labels yet*

---

**Q5.** ⚠️ Harder: During DQN training, we compute the loss as `(y - Q(s, a; θ))²`. We backpropagate gradients through `Q(s, a; θ)` but **not** through `y = r + γ · max Q(s', a'; θ⁻)`. Why?

> Think: what would happen to training stability if we also differentiated through the target `y`?

---

**Q6.** ⚠️ Harder: After training converges, the TD error `(y - Q(s,a;θ))` should approach zero on average — same as Q-learning. However, in practice it never reaches exactly zero. Name two reasons why a trained DQN might still show non-zero TD error.

---

**Quiz Answers:**

Q1: **(b)** — Continuous states cannot be enumerated into table rows.

Q2: **(b)** — Frozen copy, periodically synced, used only to compute targets.

Q3: **(b)** — Target = `r + γ · max_a' Q(s', a'; θ⁻)` = `1 + 0.99 × max(20, 25)` = `1 + 0.99 × 25` = **25.75**

Q4: **(c)** — Breaking temporal correlation is the key motivation for experience replay.

Q5: If we differentiated through `y` as well, gradients from the target would update `θ` (via `θ⁻`) in the same step that we're updating `θ` for the loss — the target and the prediction would move together, eliminating the stabilizing effect. We treat `y` as a **stop-gradient constant** to keep the Bellman target fixed for one update step.

Q6: Two reasons — (1) **function approximation error**: a finite neural network cannot represent the true Q-function exactly, so some irreducible approximation error remains. (2) **Non-stationarity**: even a converged DQN keeps decaying ε slightly (if ε_min > 0), meaning the behavior policy keeps changing, continuously introducing new transitions that shift the data distribution.

---

*Next: See the companion notebook `dqn_cartpole.ipynb` for a full PyTorch implementation with visualizations.*
