# Proximal Policy Optimization (PPO): A Hands-On Tutorial with LunarLander-v3

---

## The Essential Idea (Read This First)

REINFORCE showed us how to directly optimize a policy using gradient ascent on expected return. But it has a fundamental instability: if you take too large a gradient step, you can land on a drastically worse policy — and since the next batch of data comes from *that* worse policy, training can collapse catastrophically and never recover.

**PPO solves this with one central insight:**

> Constrain each policy update so the new policy never strays too far from the old one. This keeps learning stable without requiring a small, conservative learning rate on every parameter.

**The mechanism:** PPO computes a ratio `r_t(θ) = π_new(a|s) / π_old(a|s)` that measures how much the new policy has changed. It then **clips** this ratio to stay within `[1−ε, 1+ε]` (typically ε = 0.2), preventing any single update from making a dramatic policy shift.

**In one sentence:** PPO is REINFORCE with a trust-region constraint implemented cheaply via clipping, plus an Actor-Critic architecture to reduce variance.

**Why PPO dominates in practice:**
- More stable than REINFORCE (clipping prevents catastrophic updates)
- More sample-efficient than REINFORCE (each collected batch used for K gradient steps)
- Simpler than TRPO (which enforces the constraint via constrained optimization)
- Works for both discrete (LunarLander) and continuous (robotics) action spaces
- Default choice for most RL practitioners today

---

## Part 1: Building Blocks — Easy Concepts First

### 1.1 The LunarLander-v3 Problem

The agent controls a lander descending toward a landing pad.

```
         ●  ← lander
        / \
       /   \
──────[     ]──────   ← landing pad (target)
```

**State (8 numbers):**
| Index | Variable | Meaning |
|-------|----------|---------|
| 0 | x position | Horizontal distance from pad centre |
| 1 | y position | Height above ground |
| 2 | x velocity | Horizontal speed |
| 3 | y velocity | Vertical speed (negative = falling) |
| 4 | angle | Tilt (0 = upright) |
| 5 | angular velocity | Rotation speed |
| 6 | left leg contact | 1.0 if left leg touching ground |
| 7 | right leg contact | 1.0 if right leg touching ground |

**Actions (4 discrete):**
- `0` = Do nothing
- `1` = Fire left orientation engine
- `2` = Fire main engine (thrust up)
- `3` = Fire right orientation engine

**Reward shaping:** +100 to +140 for landing on pad, −100 for crashing, +10 per leg contact, −0.3 per frame firing main engine, −0.03 per frame firing side engines. Episode ends at crash, landing, or 1000 steps.

**Solved threshold:** mean reward ≥ 200 over 100 consecutive episodes.

---

### 1.2 Actor-Critic Architecture (PPO's Backbone)

PPO uses an **Actor-Critic** architecture — two networks (or one shared network with two heads):

```
         State (8 values)
               ↓
    [Shared backbone: Dense 64, ReLU → Dense 64, ReLU]
         ↙                    ↘
  Actor head               Critic head
  [Dense → 4 logits]      [Dense → 1 value]
       ↓                         ↓
  π(a|s; θ)                  V(s; φ)
  (action probs)           (state value)
```

- **Actor**: same stochastic policy as REINFORCE — outputs action probabilities
- **Critic**: outputs a scalar `V(s)` estimating the expected return from state `s`
- The critic's job is to provide a **baseline** that dramatically reduces gradient variance

### 1.3 Advantage Estimation

The **advantage** `A_t` answers: *"Was this action better or worse than average?"*

```
A_t = G_t − V(s_t)
```

Where `G_t` is the actual return and `V(s_t)` is what the critic predicted. If the agent fires the main engine and lands safely (high `G_t`) but the critic already expected a good outcome (high `V(s_t)`), the advantage is small — the action wasn't surprisingly good. This centering effect drastically reduces variance compared to raw returns.

### 1.4 Generalized Advantage Estimation (GAE)

> ⚠️ **FLAGGED — GAE is somewhat technical but worth understanding as it's the standard in practice.**

Pure Monte Carlo returns `G_t = r_t + γ·r_{t+1} + ...` require the full episode to complete (high variance, unbiased). TD(0) uses only one step: `A_t = r_t + γ·V(s_{t+1}) − V(s_t)` (low variance, biased because `V` is imperfect).

GAE interpolates between these extremes using a parameter `λ`:

```
δ_t = r_t + γ·V(s_{t+1}) − V(s_t)        ← TD residual at step t

A_t^GAE = δ_t + (γλ)·δ_{t+1} + (γλ)²·δ_{t+2} + ...
        = Σ_{k=0}^{T-t} (γλ)^k · δ_{t+k}
```

- `λ = 0` → pure TD(0): `A_t = δ_t` (low variance, high bias)
- `λ = 1` → full Monte Carlo: `A_t = G_t − V(s_t)` (high variance, low bias)
- `λ = 0.95` → sweet spot used in practice (mostly low-variance with little bias)

**In code**, this is computed with a single backward pass through the episode.

---

## Part 2: The PPO Clipped Objective

> ⚠️ **FLAGGED — This is the mathematical heart of PPO. Read it twice.**

### 2.1 The Probability Ratio

After collecting a batch of trajectories with policy `π_old`, we want to update the policy `π_new`. Define the **probability ratio**:

```
r_t(θ) = π_new(a_t | s_t; θ) / π_old(a_t | s_t; θ_old)
```

- `r_t = 1.0` means the new policy assigns exactly the same probability to action `a_t` as the old policy did
- `r_t = 2.0` means the new policy is twice as likely to take that action
- `r_t = 0.5` means the new policy is half as likely

The **standard policy gradient** objective (from REINFORCE) can be rewritten as:

```
L^PG(θ) = E_t [ r_t(θ) · A_t ]
```

If `A_t > 0` (action was good), maximize `r_t` → make the action more probable. If `A_t < 0`, minimize `r_t` → make it less probable. Without any constraint, the optimizer will push `r_t` as large as possible for positive advantages, potentially making the policy wildly different from `π_old`.

### 2.2 The Clipped Surrogate Objective

PPO clips the ratio to prevent large updates:

```
L^CLIP(θ) = E_t [ min( r_t(θ) · A_t,
                        clip(r_t(θ), 1−ε, 1+ε) · A_t ) ]
```

**Unpacking the clip:**
- `clip(r_t, 1−ε, 1+ε)` restricts `r_t` to the range `[0.8, 1.2]` when `ε=0.2`
- The `min` of the clipped and unclipped terms creates a **pessimistic lower bound**:
  - When `A_t > 0`: clip prevents `r_t` from going above `1+ε` (can't exploit good actions too aggressively)
  - When `A_t < 0`: clip prevents `r_t` from going below `1−ε` (can't punish bad actions too aggressively)

**Visualizing the clip:**

```
A_t > 0 (good action):
  Unclipped:  ────────/           (keeps rising as r_t increases)
  Clipped:    ────────┐           (flat cap at r_t = 1+ε)
  min of both:────────┐           (we take the min → pessimistic)

A_t < 0 (bad action):
  Unclipped:  \────────           (keeps falling as r_t decreases)
  Clipped:    ┌────────           (floor at r_t = 1-ε)
  min of both:┌────────           (we take the min → pessimistic)
```

The net effect: gradients only flow when `r_t` is within `[1−ε, 1+ε]`. Outside this range, the gradient is zero — the optimizer gets no signal to push the policy further in that direction.

> ⚠️ **FLAGGED — Common confusion: why is the `min` used instead of just clipping?**
>
> The `min(unclipped, clipped)` formulation is more careful than just `clipped`. Consider `A_t > 0` and `r_t = 0.5` (new policy already reduced the probability of a good action). The unclipped term gives a smaller objective than the clipped term, so `min` takes the unclipped — providing a gradient signal to increase `r_t` back toward 1. Pure clipping wouldn't do this. The `min` ensures we never optimistically overestimate the objective.

---

## Part 3: The Full PPO Loss

PPO combines three terms:

```
L^PPO(θ, φ) = L^CLIP(θ)                    ← policy loss (maximize)
              − c₁ · L^VF(φ)               ← value function loss (minimize)
              + c₂ · H[π(·|s; θ)]          ← entropy bonus (maximize)
```

### 3.1 Value Function Loss

The critic is trained to minimize prediction error:

```
L^VF(φ) = E_t [ (V(s_t; φ) − G_t)² ]
```

Where `G_t` is the return target (either Monte Carlo or TD). We minimize this — better value estimates → better advantage estimates → better policy gradient direction.

### 3.2 Entropy Bonus

> ⚠️ **FLAGGED — The entropy bonus is easy to overlook but important for exploration.**

The entropy of the policy distribution is:

```
H[π(·|s)] = −Σ_a π(a|s) · log π(a|s)
```

High entropy = more uniform distribution = more exploration. Low entropy = deterministic policy.

PPO adds `c₂ · H` to the objective to prevent the policy from becoming too deterministic too early. If the policy collapses to always taking one action, entropy goes to zero and the agent stops exploring. The entropy bonus fights this.

Typical values: `c₁ = 0.5`, `c₂ = 0.01`.

---

## Part 4: The PPO Training Loop

> ⚠️ **FLAGGED — The two-phase loop (collect then update) is the key structural difference from DQN.**

```
Initialize Actor-Critic network with weights θ

For each iteration:
    ┌─────────────────────────────────────────────────────┐
    │  PHASE 1: COLLECT (N_STEPS steps across all envs)   │
    │                                                     │
    │  For t = 1 to N_STEPS:                              │
    │      a_t ~ π_old(·|s_t)                             │
    │      s_{t+1}, r_t ← env.step(a_t)                  │
    │      Store (s_t, a_t, r_t, V(s_t), log π_old(a_t)) │
    └─────────────────────────────────────────────────────┘
              ↓
    Compute advantages A_t via GAE
    Compute return targets G_t = A_t + V(s_t)
              ↓
    ┌─────────────────────────────────────────────────────┐
    │  PHASE 2: UPDATE (K epochs over the same batch)     │
    │                                                     │
    │  For epoch = 1 to K_EPOCHS:                         │
    │      For each mini-batch from collected data:        │
    │          Compute r_t(θ) = π_new/π_old               │
    │          Compute L^CLIP + L^VF + H                  │
    │          Gradient step on θ                         │
    └─────────────────────────────────────────────────────┘
    
    θ_old ← θ  (update old policy for next iteration)
```

**Critical insight:** PPO reuses each collected batch for `K_EPOCHS` gradient steps (typically 4–10). This is why it's more sample-efficient than REINFORCE (which discards each episode after one update). The clipping constraint ensures that doing K updates on the same data doesn't cause catastrophic policy changes — the clipping keeps each update conservative.

---

## Part 5: PPO vs REINFORCE vs DQN — Complete Picture

| Feature | REINFORCE | DQN | PPO |
|---------|-----------|-----|-----|
| Policy type | Stochastic | Deterministic | Stochastic |
| Architecture | Policy net only | Q-network + target | Actor + Critic |
| Update frequency | Once per episode | Every step | K epochs per N steps |
| Data reuse | None (1 update/ep) | High (replay buffer) | Moderate (K epochs) |
| Variance reduction | Baseline (optional) | N/A | GAE + critic |
| Stability mechanism | None | Target network | Clipping ratio ε |
| Sample efficiency | Low | Medium-high | Medium |
| Action spaces | Discrete/continuous | Discrete only | Discrete/continuous |
| LunarLander suitability | Possible but slow | Good | Excellent |

---

## Part 6: Key Hyperparameters

> ⚠️ **FLAGGED — PPO has more hyperparameters than REINFORCE. These are the important ones.**

| Parameter | Typical Value | Effect |
|-----------|--------------|--------|
| `clip_epsilon` (ε) | 0.1–0.2 | Core trust region. Too large = instability. Too small = slow. |
| `gamma` (γ) | 0.99 | Discount. LunarLander needs high γ for delayed landing reward. |
| `gae_lambda` (λ) | 0.95 | GAE smoothing. Lower = more bias but less variance. |
| `n_steps` | 512–2048 | Batch size per iteration. Larger = lower variance, slower. |
| `k_epochs` | 4–10 | Gradient passes per batch. Higher = more efficient but risks over-fitting old data. |
| `lr_actor` | 3e-4 | Actor learning rate. Adam optimizer. |
| `lr_critic` | 1e-3 | Critic usually learns faster than actor. |
| `entropy_coef` (c₂) | 0.01 | Exploration bonus. Higher = more exploration. |
| `value_coef` (c₁) | 0.5 | Weight on critic loss. |

---

## ✅ Quiz: Check Your Understanding

**Q1.** What problem does PPO's clipping solve compared to vanilla policy gradient (REINFORCE)?

*(a) It prevents the Q-table from overflowing*
*(b) It prevents any single gradient update from making the new policy too different from the old policy, avoiding catastrophic performance collapse*
*(c) It prevents the replay buffer from storing duplicate transitions*
*(d) It prevents the critic from learning faster than the actor*

---

**Q2.** The probability ratio `r_t(θ) = π_new(a_t|s_t) / π_old(a_t|s_t)` equals 1.5. This means:

*(a) The new policy is 50% more likely to take action `a_t` in state `s_t` than the old policy*
*(b) The reward increased by 50%*
*(c) The advantage `A_t` is positive*
*(d) The policy has been updated 1.5 times*

---

**Q3.** In the clipped objective `min(r_t · A_t, clip(r_t, 1−ε, 1+ε) · A_t)`, when does the gradient become zero?

*(a) When the advantage is exactly zero*
*(b) When `r_t` is outside `[1−ε, 1+ε]` and the gradient would push it further outside*
*(c) During the value function update step*
*(d) The gradient is never zero in PPO*

---

**Q4.** GAE with `λ = 0` gives pure TD(0) advantage `A_t = r_t + γV(s_{t+1}) − V(s_t)`. What is the trade-off vs `λ = 1` (full Monte Carlo)?

*(a) λ=0 is unbiased but high variance; λ=1 is biased but low variance*
*(b) λ=0 is low variance but biased (relies on imperfect V estimates); λ=1 is high variance but unbiased (uses actual returns)*
*(c) λ=0 trains faster but λ=1 converges to a better policy always*
*(d) λ has no effect when the critic V is well-trained*

---

**Q5.** Why does PPO update for K epochs on the same batch of data, while REINFORCE only does one gradient step per episode?

*(a) PPO needs more updates to converge to the same gradient direction*
*(b) The clipping constraint allows multiple updates without large policy changes; REINFORCE has no such safeguard so multiple updates on the same batch would cause divergence*
*(c) PPO has a larger replay buffer that requires more passes to empty*
*(d) REINFORCE discards old episodes to save memory*

---

**Q6.** ⚠️ Harder: The entropy bonus `c₂ · H[π(·|s)]` is added to the PPO objective. What would happen during LunarLander training if `c₂ = 0` (no entropy bonus), and the agent discovers that "fire main engine always" gives marginally positive reward early in training?

*(a) Nothing — the clipping constraint prevents the policy from becoming too deterministic*
*(b) The policy would converge faster to the optimal solution without the entropy distraction*
*(c) The policy could prematurely collapse to a near-deterministic "always fire main engine" behavior, missing the exploration needed to discover the precise landing sequence. It could get stuck in a local optimum.*
*(d) The critic would compensate by outputting higher value estimates*

---

**Q7.** ⚠️ Harder: PPO is described as an "on-policy" algorithm. Yet it reuses the same batch of data for K gradient update epochs. Doesn't this make it off-policy?

> Think: after the first gradient update, `θ` has changed — so the second update uses data collected by a *slightly different* policy. How does PPO justify this, and what keeps it from becoming fully off-policy?

---

**Quiz Answers:**

Q1: **(b)** — The core purpose of clipping is stability: prevent any single step from jumping too far in policy space.

Q2: **(a)** — `r_t = π_new/π_old = 1.5` means the new policy has increased the probability of that action by 50%.

Q3: **(b)** — When `r_t > 1+ε` and `A_t > 0` (trying to increase probability of an already-boosted good action), the clipped term becomes flat (gradient = 0), preventing further increase. Similarly when `r_t < 1−ε` and `A_t < 0`.

Q4: **(b)** — λ=0 uses the TD residual which depends on `V` (biased if V is wrong, but low variance). λ=1 uses actual sampled returns (unbiased but noisy/high variance). λ=0.95 balances both.

Q5: **(b)** — The clipping trust region is what makes multiple updates safe. After each gradient step, if `r_t` moves outside `[1−ε, 1+ε]`, the gradient from that sample becomes zero — it stops contributing to further drift. Without clipping, REINFORCE's standard policy gradient has no such brake.

Q6: **(c)** — Entropy collapse is a real failure mode in policy gradient methods. Without entropy regularization, the policy can become over-confident early, causing it to stop exploring. The agent might learn to hover (getting modest reward) but never learn the precise throttle control needed for a clean landing. The entropy bonus counteracts premature convergence.

Q7: After K updates, `π_new` is no longer exactly `π_old`. In strict on-policy terms, the data is "slightly stale." PPO justifies this by arguing that the clipping constraint limits how far `π_new` drifts from `π_old` during those K epochs — so the data remains approximately on-policy. When `r_t` hits the clip boundary (1±ε), the gradient zeroes out, preventing further drift. This makes PPO a **near-on-policy** algorithm: it exploits multiple passes for sample efficiency while the clip bounding prevents the bias from becoming too large.

---

*Next: See the companion notebook `ppo_lunarlander.ipynb` for a complete PyTorch implementation.*
