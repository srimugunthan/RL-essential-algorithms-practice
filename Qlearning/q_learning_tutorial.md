# Q-Learning: A Hands-On Tutorial with the Robot-in-a-Maze Problem

---

## The Essential Idea (Read This First)

Imagine you are learning to navigate an unfamiliar city. You try different routes, get stuck in traffic or find shortcuts, and over time you build a **mental map** of which roads are good from which intersections. Q-learning is exactly this process, formalized for machines.

**The core idea in one sentence:**  
> A robot explores an environment, receives rewards or penalties for its actions, and gradually builds a table that tells it — *from any position, which action leads to the best long-term outcome*.

That table is called the **Q-table**. The "Q" stands for **Quality** — the quality of taking an action in a given state.

**The three key actors:**
- **Agent** — the robot making decisions
- **Environment** — the maze the robot lives in
- **Q-table** — the robot's learned knowledge, mapping (state, action) → expected future reward

Once the Q-table is fully learned, the robot simply looks it up: "I am at cell (2,3), what is my best action?" → look up the row for (2,3) and pick the action with the highest Q-value.

---

## Part 1: Easy Concepts First

### 1.1 The Maze Environment

We use a simple grid maze:

```
S . . #
. # . .
. . . G
```

- `S` = Start position
- `G` = Goal (reward = +100)
- `#` = Wall (blocked)
- `.` = Open cell (small penalty = -1 per step, so the robot learns to be efficient)

The robot can move: **Up, Down, Left, Right**.

### 1.2 States and Actions

- **State** = the robot's current cell, e.g., (row=0, col=0)
- **Action** = one of {Up, Down, Left, Right}
- **State space** = all valid cells in the grid
- **Action space** = {0: Up, 1: Down, 2: Left, 3: Right}

This is simple enough to enumerate completely, which is why maze problems are ideal for introducing Q-learning.

### 1.3 The Q-Table

The Q-table is just a 2D array:

```
         Up     Down   Left   Right
State 0  0.00   0.00   0.00   0.00
State 1  0.00   0.00   0.00   0.00
...
State N  0.00   0.00   0.00   0.00
```

- Rows = states (one per valid cell)
- Columns = actions (4 per state)
- Values = Q-values (all initialized to 0)

At the start, the robot knows nothing. By the end of training, each cell will contain a meaningful number reflecting the expected future reward.

### 1.4 Exploration vs. Exploitation (Epsilon-Greedy)

**Easy part:** The robot must sometimes try new things (explore) and sometimes use what it already knows (exploit).

We control this with **epsilon (ε)**:
- With probability ε → pick a **random** action (explore)
- With probability (1 - ε) → pick the **best known** action (exploit)

We start with ε = 1.0 (pure exploration) and decay it toward 0.01 over time (mostly exploitation). This is called **epsilon decay**.

---

## Part 2: The Q-Learning Update Rule

> ⚠️ **FLAGGED — This is the hardest part conceptually. Read slowly.**

The Q-table is updated after every step using this equation:

```
Q(s, a) ← Q(s, a) + α × [r + γ × max Q(s', a') − Q(s, a)]
```

Let's unpack every symbol:

| Symbol | Name | Meaning |
|--------|------|---------|
| `s` | Current state | Where the robot is now |
| `a` | Action taken | What the robot just did |
| `r` | Reward | Immediate feedback from the environment |
| `s'` | Next state | Where the robot ended up |
| `α` (alpha) | Learning rate | How much to update (0 to 1). Too high → unstable. Too low → slow. |
| `γ` (gamma) | Discount factor | How much future rewards matter (0 to 1). γ=0 → greedy. γ=1 → far-sighted. |
| `max Q(s', a')` | Best future Q | The best Q-value achievable from the next state |

### What the equation really means

The term in brackets `[r + γ × max Q(s', a') − Q(s, a)]` is called the **TD error** (Temporal Difference error). It answers: *"How wrong was my current estimate?"*

- `r + γ × max Q(s', a')` = the **target** (what we now think the value should be)
- `Q(s, a)` = the **current estimate** (what we previously thought)
- The difference = how much we were wrong
- We update Q by a fraction `α` of that error

**Intuition:** If the robot took action Right from state (0,0) and got a reward of -1, and from the next state the best future Q-value is 50, then the target is `-1 + 0.9 × 50 = 44`. If the current Q was 0, the error is 44, and we nudge Q upward.

> ⚠️ **FLAGGED — Why γ (discount factor) is subtle:**  
> γ makes future rewards worth *less* than immediate ones. With γ = 0.9, a reward 2 steps away is worth 0.9² = 0.81 of its face value. This reflects reality: a reward now is better than the same reward later. Without γ < 1, the algorithm can fail to converge in environments with infinite loops.

---

## Part 3: The Full Training Loop

```
Initialize Q-table to zeros
Set ε = 1.0

For each episode (one full run through the maze):
    Reset robot to Start position
    
    While goal not reached (and step limit not hit):
        Choose action using ε-greedy policy
        Take action → observe reward r and next state s'
        Update Q(s, a) using the Bellman equation
        s ← s'
    
    Decay ε (ε = ε × decay_rate)
```

Each "episode" is one attempt by the robot to reach the goal. Early episodes are chaotic (random exploration). Later episodes show the robot confidently taking near-optimal paths.

---

## Part 4: Key Hyperparameters and Their Effects

> ⚠️ **FLAGGED — Hyperparameter tuning is tricky in practice.**  
> The values below work for the maze. For real problems, these require careful tuning.

| Parameter | Typical Value | Effect if Too High | Effect if Too Low |
|-----------|--------------|-------------------|-------------------|
| α (learning rate) | 0.1 – 0.5 | Unstable updates, oscillation | Very slow convergence |
| γ (discount) | 0.9 – 0.99 | Risk of divergence in cyclic envs | Short-sighted, ignores goal |
| ε (initial) | 1.0 | N/A | Insufficient exploration |
| ε decay | 0.995 – 0.9995 | Explores too long | Stops exploring too early |
| Episodes | 1000 – 10000 | Diminishing returns | Undertrained |

---

## Part 5: What Q-Learning Guarantees (and What It Doesn't)

**Guarantees** (under ideal conditions):
- Q-learning **converges** to the optimal Q-values if every (state, action) pair is visited infinitely often
- It is **model-free**: the robot does not need to know the maze layout in advance

**Limitations:**
- Does not scale to large state spaces (a 100×100 maze has 10,000 states — Q-table gets huge)
- For large/continuous state spaces, Q-values are approximated using neural networks → **Deep Q-Network (DQN)**
- Assumes a **Markov** environment: next state depends only on current state and action, not history

---

## Part 6: From Q-Table to Policy

After training, extracting the policy is trivial:

```python
# For each state, find the action with the highest Q-value
policy[s] = argmax(Q[s, :])
```

You can then visualize this as arrows on the maze grid, showing the optimal direction to move from every cell.

---

## Quick Reference: Q-Learning Algorithm

```
1. Initialize Q(s, a) = 0 for all s, a
2. For each episode:
   a. s ← start state
   b. While s is not terminal:
      i.  Choose a using ε-greedy on Q
      ii. Take action a, observe r, s'
      iii. Q(s,a) ← Q(s,a) + α[r + γ·max_a'Q(s',a') - Q(s,a)]
      iv. s ← s'
   c. Decay ε
3. Return Q
```

---

## Concept Map

```
Agent
  │
  ├── observes ──► State (s)
  ├── takes ──────► Action (a)   [chosen via ε-greedy from Q-table]
  └── receives ──► Reward (r) + Next State (s')
                         │
                         ▼
              Q(s,a) updated via Bellman equation
                         │
                         ▼
              After many episodes → Optimal Q-table
                         │
                         ▼
              Policy = argmax over actions at each state
```

---

## ✅ Quiz: Check Your Understanding

Answer these before looking at the notebook code.

**Q1.** The Q-table has shape `(num_states, num_actions)`. For a 4×4 maze with 4 actions, what is the shape of the Q-table?  
*(a) (4, 4)*  
*(b) (16, 4)*  
*(c) (4, 16)*  
*(d) (16, 16)*

---

**Q2.** After one training step, the robot was at state `s=5`, took action `a=Right`, received reward `r=-1`, and landed in state `s'=6`. The current `Q(5, Right)=0`, `max Q(6, :)=10`, `α=0.5`, `γ=0.9`. What is the new `Q(5, Right)`?  
*(a) 4.0*  
*(b) 4.5*  
*(c) 9.0*  
*(d) -0.5*

> Hint: `Q(5, Right) = 0 + 0.5 × [-1 + 0.9×10 - 0] = ?`

---

**Q3.** Setting `γ = 0` means the robot:  
*(a) Only cares about the immediate reward, ignoring future rewards*  
*(b) Equally values immediate and future rewards*  
*(c) Ignores immediate rewards and only optimizes for the future*  
*(d) Explores randomly forever*

---

**Q4.** Why do we decay epsilon (ε) over time?  
*(a) To reduce memory usage*  
*(b) To shift the robot from random exploration toward using learned knowledge*  
*(c) To increase the learning rate*  
*(d) To make the maze harder*

---

**Q5.** ⚠️ Harder: What is the **TD error** in the update equation, and why does it go to zero when the Q-table has fully converged?

> Think: if the Q-table is perfectly accurate, what would `r + γ × max Q(s', a')` equal, and how does that relate to `Q(s, a)`?

---

**Quiz Answers:**  
Q1: **(b)** — 16 cells × 4 actions  
Q2: **(a)** — `0 + 0.5 × (-1 + 9 - 0) = 0.5 × 8 = 4.0`  
Q3: **(a)**  
Q4: **(b)**  
Q5: When fully converged, the Bellman equation is satisfied exactly: `Q(s,a) = r + γ·max Q(s',a')`. So the TD error `[r + γ·max Q(s',a') - Q(s,a)]` = 0, and there are no more updates.

---

*Next: See the companion notebook `q_learning_maze.ipynb` for a full from-scratch implementation with visualizations.*
