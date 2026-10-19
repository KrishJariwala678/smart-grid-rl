# Three-Node Grid Balancer

A live, hour-by-hour simulation of a three-node smart grid balanced by deep
reinforcement learning controllers.

Nodes **A** (city), **B** (suburb) and **C** (industrial) each generate and consume
power on their own daily cycle. Every simulated hour a controller may route 20 MW
along one line to pull all three net imbalances toward zero.

**Live:** https://power-grid-rl.vercel.app

## What it shows

- A schematic with power flowing along the line in use, hour by hour
- Per-node load, generation, net imbalance and post-transfer imbalance
- A scrolling 48-hour chart: managed (solid) against unmanaged (dashed)
- Every one of the 7 discrete actions scored each hour, with the greedy optimum marked
- A process log narrating each stage: observe, regime, evaluate, select, settle, reward
- Measured results from the training run: forecaster skill, decision quality,
  physical grid performance, and a reported PPO negative result

## Controllers

| Option | Status |
|---|---|
| Greedy oracle | exact policy |
| Transformer-DQN | **behavioural emulation** — oracle scoring at the measured 68.3 % agreement |
| Rule-based proportional | exact policy |
| Always hold / Random | exact baselines |

The DQN option is an emulation because the trained Keras weights cannot run in a
browser. This is stated on the page itself, under Limitations.

## Fidelity

The simulator is a direct port of `generate_complex_data`, `FastEnv` and
`PettingZooGridEnv`: same diurnal cycles, same 3-state weather Markov chain, same
AR(1) noise at momentum 0.85, same rare-event probabilities, same action deltas and
same differential reward. The browser RNG is not NumPy's, so a live run will not
reproduce the training run's exact totals — the physics and parameters are
identical, the noise draw is not.

## Running it

One self-contained HTML file with no build step and no server. Open `index.html`
directly, or serve the folder:

    python3 -m http.server 8000

## Deploying

    npx vercel deploy --prod --yes
