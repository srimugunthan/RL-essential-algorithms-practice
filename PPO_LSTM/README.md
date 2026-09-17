## Problem Definition: Maze Navigation with Only Local Information

**Type:** Partially Observable Markov Decision Process (POMDP)

**Environment**
A grid maze with open cells, walls, a fixed start, and a single goal cell — same grid encoding used elsewhere in the repo (`0`=open, `1`=wall, `2`=goal).

**True (hidden) state**
The agent's actual `(row, col)` position in the grid. This exists in the environment but is never handed to the agent directly — that's what makes it partially observable rather than a plain MDP like `CliffWalking` or the original `MazeEnv`.

**Observation** (what the agent actually receives each step)
An egocentric local window of size `(2R+1) × (2R+1)` centered on the agent (e.g. `R=1` → a 3×3 patch), where each cell reads as open/wall/goal — plus optionally the agent's previous action, since the raw window alone can't distinguish "just arrived here" from "been circling." Cells outside the grid boundary count as walls. Critically, **the agent never sees its coordinates or the full map.**

**Actions**
4 discrete moves: Up, Down, Left, Right.

**Transition dynamics**
Deterministic: a move into an open cell succeeds; a move into a wall or off the grid leaves the agent in place.

**Reward**
- Step into open cell: small negative (e.g. `-1`) — encourages shortest paths
- Bump into a wall/boundary: larger negative (e.g. `-5`) — discourages wasted moves
- Reach the goal: large positive (e.g. `+100`), episode ends
- Episode also truncates after a max step count to bound wandering

**Episode**
Starts at a fixed start cell, ends on goal reached or step limit hit.

**What makes this hard (the actual research question)**
Because the maze corridors are only 1 cell wide, most interior open cells produce an **identical local observation** — wall on the sides, open ahead, open behind. Two different physical locations can be observationally indistinguishable. A memoryless (reactive) policy that maps observation → action has no way to break that symmetry beyond blind exploration. This is the formal reason a **recurrent policy (GRU/LSTM)** is a meaningfully different architecture here, not just a bigger network — it can integrate the sequence of past observations/actions into an implicit belief about where it is, which a stateless MLP structurally cannot do.

**Objective**
Learn a policy π(action | observation history) that reaches the goal in as few steps as possible, using only this local, aliased observation stream — and compare how differently a stateful vs. stateless architecture solves that same underdetermined problem.
