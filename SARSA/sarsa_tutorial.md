# SARSA: A Hands-On Tutorial with CliffWalking — and a Deep Comparison with Q-Learning

---

## The Essential Idea (Read This First)

SARSA is a **Temporal Difference (TD)** learning algorithm — like Q-learning — that learns the value of state-action pairs by bootstrapping from the next step. Both SARSA and Q-learning update the same Q-table using the Bellman equation. They differ in exactly **one word** of their update rule.

**Q-learning update:**
```
Q(s, a) ← Q(s, a) + α · [r + γ · max_a' Q(s', a') − Q(s, a)]
                                   ^^^^^^^^^^^
                         uses the BEST action at s' (regardless of what agent actually does)
```

**SARSA update:**
```
Q(s, a) ← Q(s, a) + α · [r + γ · Q(s', a') − Q(s, a)]
                                   ^^^^^^^^^^
                         uses the ACTUAL next action a' that the agent will take
```

The name **SARSA** comes from the five things it uses in each update:

```
(S, A, R, S', A')
 ↑  ↑  ↑  ↑   ↑
 |  |  |  |   └── next Action (actually selected from policy)
 |  |  |  └────── next State
 |  |  └───────── Reward received
 |  └──────────── Action taken
 └─────────────── current State
```

Q-learning only uses `(S, A, R, S')` and then **imagines** the best action at `S'`. SARSA insists on knowing `A'` — the action the agent **actually committed to** — before making the update.

**Why does this one difference matter so much?**

Because the agent is exploring — using ε-greedy — it will sometimes take random actions. Q-learning ignores this: it always assumes the agent will take the best action from the next state. SARSA respects this: it uses the actual next action, including the risky exploratory ones. This makes SARSA **on-policy** and Q-learning **off-policy**, and that difference produces dramatically different behaviour on CliffWalking.

---

## Part 1: Easy Concepts First

### 1.1 The CliffWalking Problem

CliffWalking is a 4×12 grid world:

```
 ┌────────────────────────────────────────────────┐
 │  0   1   2   3   4   5   6   7   8   9  10  11 │  ← row 0
 │ 12  13  14  15  16  17  18  19  20  21  22  23 │  ← row 1
 │ 24  25  26  27  28  29  30  31  32  33  34  35 │  ← row 2
 │ S  [C   C   C   C   C   C   C   C   C   C]  G  │  ← row 3
 │ 36  37  38  39  40  41  42  43  44  45  46  47 │
 └────────────────────────────────────────────────┘
       S = Start (36)    G = Goal (47)    C = Cliff (37–46)
```

**State:** a single integer 0–47 (grid cell index)

**Actions:** `0=Up`, `1=Right`, `2=Down`, `3=Left`

**Rewards:**
- `-1` per step (encourages efficiency)
- `-100` for stepping onto the cliff (and agent resets to Start — episode continues)
- Episode ends when the agent reaches Goal (state 47) or after max steps

**Two natural paths:**
- **Safe path**: Go up (row 3→2), traverse row 2 rightward, come down to Goal. Length 13+, but avoids cliff. Total reward ≈ −13
- **Risky path**: Go right along row 3 edge (one step from cliff). Length 12. Total reward ≈ −12 but **very dangerous with exploration**

The tension between these two paths is where SARSA and Q-learning diverge.

---

### 1.2 The Q-Table (Same as Before)

Both SARSA and Q-learning use the same Q-table structure:
- Shape: `(48 states, 4 actions)`
- Initialized to zeros (or small values)
- Updated incrementally as the agent explores

The policy is ε-greedy: with probability ε take a random action, otherwise take `argmax Q(s, ·)`.

---

### 1.3 TD Learning Refresher

Both algorithms are **Temporal Difference** methods. At each step, they observe `(s, a, r, s')` and compute:

```
TD error = [r + γ · (something about s')] − Q(s, a)
           └──────────────────────────────────────┘
               "target"              minus "current estimate"
```

The target tells us what `Q(s, a)` *should* be, based on what just happened. The TD error measures how wrong the current estimate is. We nudge `Q(s, a)` by a fraction `α` of the TD error.

The **only difference** is what goes into "something about s'":

| Algorithm | Target at s' | Name |
|-----------|-------------|------|
| Q-learning | `max_a' Q(s', a')` | optimistic bootstrap |
| SARSA | `Q(s', a')` where `a' ~ π(s')` | honest bootstrap |

---

## Part 2: On-Policy vs Off-Policy — The Core Conceptual Difference

> ⚠️ **FLAGGED — This is the single most important concept in this tutorial. The on/off-policy distinction determines which algorithm you should use in any given situation.**

### 2.1 What Does "Policy" Mean Here?

The agent has **two roles**:

1. **Behavior policy**: How the agent actually acts during training (ε-greedy — explores randomly sometimes)
2. **Target policy**: What the agent is trying to learn to do (the optimal greedy policy)

### 2.2 On-Policy: SARSA

SARSA is **on-policy** because it evaluates and improves the **same policy it uses to act**.

When SARSA updates `Q(s, a)` using `Q(s', a')`, the action `a'` is drawn from the **ε-greedy behavior policy** — the same policy the agent actually follows. The Q-values SARSA learns represent the value of the ε-greedy policy, including its occasional random mistakes.

**Consequence for CliffWalking:** SARSA knows that with probability ε, its policy will take a random action. When standing next to the cliff, a random action might send the agent off the cliff (reward −100). SARSA accounts for this risk in its Q-values. It will therefore **learn to prefer the safe path** — farther from the cliff — even though it's slightly longer.

### 2.3 Off-Policy: Q-Learning

Q-learning is **off-policy** because it learns the value of one policy (the greedy/optimal policy) while following a different policy (ε-greedy behavior).

When Q-learning updates `Q(s, a)` using `max_a' Q(s', a')`, it imagines that at `s'`, the agent will always take the greedy action — no exploration mistakes. The Q-values Q-learning learns represent the value of the **fully greedy** (optimal) policy.

**Consequence for CliffWalking:** Q-learning sees the risky path as slightly better (reward −12 vs −13) and learns to take it — because in its mind, a perfect greedy agent would never randomly fall off the cliff. But during training, with ε > 0, random actions DO happen. This means Q-learning will fall off the cliff repeatedly during training, suffering more cumulative training penalties than SARSA.

### 2.4 The Asymmetry Visualized

```
                        SARSA (on-policy)
                        ─────────────────
  Episode collection:   π_ε-greedy  →  actions including random moves
  Q-value update:       uses Q(s', a') from  π_ε-greedy
  Learning:             value of ε-greedy policy (safe, risk-aware)
  Converges to:         NEAR-optimal path (safe path with small ε)

                        Q-learning (off-policy)
                        ──────────────────────
  Episode collection:   π_ε-greedy  →  actions including random moves
  Q-value update:       uses max Q(s', ·) from  π_greedy (imagined)
  Learning:             value of optimal greedy policy
  Converges to:         OPTIMAL path (risky path — perfect agent)
```

**Key insight:** After training, if you set ε = 0 (pure greedy), Q-learning's policy becomes optimal (risky path). SARSA's policy also becomes greedy, and since it learned values under ε > 0, it may take the slightly longer safe path. If you reduce ε slowly to 0 during training, both converge to the same optimal policy — but their training behavior (and training rewards) differ substantially.

---

## Part 3: The Update Rule in Code

> ⚠️ **FLAGGED — The a' selection timing is the subtlest implementation detail.**

In SARSA, you must select `a'` **before** doing the update, and then **use that same `a'`** at the start of the next step. The loop structure is:

```python
# SARSA loop (on-policy)
state = env.reset()
action = policy(state)           # ← select initial action

while not done:
    next_state, reward, done = env.step(action)
    next_action = policy(next_state)   # ← select NEXT action BEFORE update
    
    # Update uses (s, a, r, s', a') — all 5 elements
    td_error = reward + γ · Q[next_state, next_action] - Q[state, action]
    Q[state, action] += α · td_error
    
    state  = next_state
    action = next_action           # ← carry a' forward as new action
```

Compare with Q-learning:

```python
# Q-learning loop (off-policy)
state = env.reset()

while not done:
    action = policy(state)         # ← select action from current state
    next_state, reward, done = env.step(action)
    
    # Update uses (s, a, r, s') — only 4 elements, then imagines best a'
    td_error = reward + γ · max(Q[next_state, :]) - Q[state, action]
    Q[state, action] += α · td_error
    
    state = next_state
```

> ⚠️ **FLAGGED — Common implementation mistake:** In SARSA, some beginners select `next_action` and then **re-select a new action** at the top of the loop without using `next_action`. This breaks the on-policy guarantee — the update would use one action, but the agent would execute a different one.

---

## Part 4: Expected SARSA (Bonus)

> ⚠️ **FLAGGED — Expected SARSA is conceptually clean but often overlooked.**

There is a third variant that sits between SARSA and Q-learning:

**Expected SARSA** uses the *expected value* over all actions under the current policy, rather than either the sampled action (SARSA) or the maximum (Q-learning):

```
Q(s, a) ← Q(s, a) + α · [r + γ · Σ_{a'} π(a'|s') · Q(s', a') − Q(s, a)]
```

- When `π` is greedy, `Σ π(a')·Q(s',a') = max Q(s', a')` → reduces to Q-learning
- When `π` is uniform random, takes average → very conservative
- With ε-greedy: weights best action by `(1−ε+ε/|A|)` and others by `ε/|A|`

Expected SARSA has lower variance than SARSA (no random sampling of `a'`) and is theoretically cleaner, but requires computing the full expectation at every step.

---

## Part 5: Convergence and Practical Behaviour

### 5.1 What Both Algorithms Converge To

Both SARSA and Q-learning converge to the **true Q-values of their respective target policies** given:
- All (state, action) pairs visited infinitely often
- Learning rate `α` satisfies the Robbins-Monro conditions (decreasing but not too fast)
- ε decays to 0

But on CliffWalking during training (fixed ε > 0):
- **Q-learning achieves higher reward per episode at convergence** (finds the risky path) but gets big penalties during training when exploration causes cliff falls
- **SARSA achieves lower reward per episode at convergence** (takes safe path) but gets fewer cliff falls during training → **higher cumulative training reward**

This is the classic **online performance vs. optimal convergence** tradeoff.

### 5.2 When to Prefer Which

| Situation | Prefer SARSA | Prefer Q-learning |
|-----------|-------------|-------------------|
| Training cost matters (real robot) | ✓ Safe training | |
| Simulation only, cost-free | | ✓ Finds optimal policy |
| Cliff / catastrophic failure risks | ✓ Avoids risk | |
| Standard benchmark, maximize final reward | | ✓ |
| ε decayed to near-zero | Both converge | Both converge |
| Continuous exploration needed | ✓ Adapts to behavior | |

---

## Quick Reference

```
SARSA:       Q(s,a) ← Q(s,a) + α[r + γ·Q(s',a') − Q(s,a)]       ON-POLICY
Q-learning:  Q(s,a) ← Q(s,a) + α[r + γ·max Q(s',·) − Q(s,a)]   OFF-POLICY

Both:  a' from ε-greedy for BEHAVIOR
SARSA: a' from ε-greedy for UPDATE  (same policy → on-policy)
Q-learning: imagines greedy a' for UPDATE  (different → off-policy)
```

---

## ✅ Quiz: Check Your Understanding

**Q1.** SARSA's name comes from the five quantities `(S, A, R, S', A')`. What does the final `A'` represent that makes SARSA different from Q-learning?

*(a) The action with the maximum Q-value at S'*
*(b) The average action across all choices at S'*
*(c) The action actually selected by the behavior policy at S' before the update is made*
*(d) The action taken at the previous step S*

---

**Q2.** On CliffWalking with ε = 0.1 throughout training, which statement best describes the policies each algorithm converges to?

*(a) Both algorithms converge to the same optimal risky path along the cliff edge*
*(b) SARSA tends to learn the safer path away from the cliff; Q-learning tends to learn the risky optimal path along the cliff edge*
*(c) SARSA learns the risky path; Q-learning learns the safer path*
*(d) Both algorithms avoid the cliff because ε-greedy is used for both*

---

**Q3.** Q-learning is called "off-policy" because:

*(a) It uses a different Q-table from SARSA*
*(b) It updates Q-values toward the greedy (optimal) policy's target, while the agent actually follows an ε-greedy behavior policy — two different policies*
*(c) It does not use a policy at all*
*(d) Its policy is turned off during the update step*

---

**Q4.** Which code fragment correctly implements the SARSA update?

*(a)*
```python
action = epsilon_greedy(Q, state)
next_state, reward, done = env.step(action)
next_action = epsilon_greedy(Q, next_state)
Q[state, action] += alpha * (reward + gamma * Q[next_state, next_action] - Q[state, action])
state, action = next_state, next_action   # carry next_action forward
```

*(b)*
```python
action = epsilon_greedy(Q, state)
next_state, reward, done = env.step(action)
Q[state, action] += alpha * (reward + gamma * max(Q[next_state]) - Q[state, action])
state = next_state
```

*(c)*
```python
action = epsilon_greedy(Q, state)
next_state, reward, done = env.step(action)
next_action = np.argmax(Q[next_state])    # greedy, not ε-greedy
Q[state, action] += alpha * (reward + gamma * Q[next_state, next_action] - Q[state, action])
state = next_state
action = epsilon_greedy(Q, next_state)   # re-samples — breaks on-policy
```

*(d) (b) and (c) are both valid SARSA implementations*

---

**Q5.** At the end of training, if ε is set to 0 for evaluation (pure greedy), SARSA's policy on CliffWalking will:

*(a) Still always take the safe path, because that is what its Q-values encode*
*(b) Crash into the cliff, because it was trained with exploration*
*(c) Take the risky path — identical to Q-learning's policy — because both are greedy at ε=0*
*(d) Refuse to act because it never trained without exploration*

---

**Q6.** ⚠️ Harder: On CliffWalking, Q-learning achieves a higher per-episode reward at convergence (≈−12) than SARSA (≈−13), yet SARSA typically achieves higher **cumulative reward across all training episodes**. Explain the mechanism behind this apparent paradox.

---

**Q7.** ⚠️ Harder: Expected SARSA replaces the sampled `Q(s', a')` with `Σ_{a'} π(a'|s')·Q(s', a')`. Under what exact condition does Expected SARSA become mathematically identical to Q-learning? Under what condition does it become identical to standard SARSA?

---

**Quiz Answers:**

Q1: **(c)** — The distinguishing feature of SARSA is that `A'` is the actual action selected by the ε-greedy behavior policy at S', committed to before the update, and executed at the start of the next step.

Q2: **(b)** — SARSA's Q-values reflect the cost of random exploration near the cliff (occasional −100 penalties), so it routes away from the cliff. Q-learning's Q-values assume perfect greedy behaviour and discover the shorter risky path.

Q3: **(b)** — Off-policy means the target policy (greedy) differs from the behavior policy (ε-greedy). Q-learning learns the value of perfect greedy behaviour while behaving ε-greedily.

Q4: **(a)** — Option (b) is Q-learning (uses max). Option (c) selects `next_action` greedily and then re-selects a new action at the top of the loop — breaking the SARSA guarantee that the action used in the update IS the action that will be taken.

Q5: **(a)** — SARSA's Q-values encode the value of each state-action pair under the ε-greedy policy used during training. With ε > 0 during training, cliff-adjacent states have low Q-values (discounted for risk). When ε=0 at evaluation, the greedy policy based on those Q-values still avoids the cliff. (If ε is slowly annealed to 0 during training, both algorithms converge to the same optimal path.)

Q6: During training, Q-learning repeatedly triggers cliff falls (−100 each) because its ε-greedy behavior still takes random actions near the cliff, but its Q-values learned to ignore this risk. SARSA avoids the cliff edge, so its exploration rarely causes falls. The cumulative penalty of frequent −100 cliff falls during Q-learning's training exceeds the small per-episode gain (−12 vs −13) over hundreds of episodes — SARSA accumulates more total reward during the training process itself.

Q7: Expected SARSA becomes identical to Q-learning when `π` is the **greedy policy** (probability 1 on the argmax action, 0 everywhere else) — the expectation collapses to `max Q(s', a')`. It becomes identical to standard SARSA when `π` takes a **single deterministic action** (the sampled one), which is always the case in standard SARSA — the expectation over a one-point distribution equals the single value.

---

*Next: See the companion notebook `sarsa_cliffwalking.ipynb` for a full from-scratch implementation with side-by-side SARSA vs Q-learning comparison.*
