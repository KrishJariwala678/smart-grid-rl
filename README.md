# Power Grid Management using AI

Deep reinforcement learning for multi-node smart-grid power balancing.

Three nodes — **A** (city), **B** (suburb), **C** (industrial) — each generate and
consume power on their own daily cycle. The goal is to keep the net imbalance at
every node as close to zero as possible by routing power between them.

**Live simulator → https://power-grid-rl.vercel.app**

## What's here

| Path | |
|---|---|
| `power-grid-rl/` | Browser simulator — runs the grid live, hour by hour. This is what's deployed. |
| `smart_grid_rl.ipynb` | Full training notebook: data generation, EDA, forecaster, both control paradigms, evaluation |
| `artifacts/` | Trained forecaster and the figures produced by the run |
| `deliverables/` | Report, presentation and architecture diagrams |

## Approach

Three stages, each building on the last:

1. **Grid simulator** — synthetic hourly data with node-specific diurnal cycles, a
   3-state weather Markov chain (normal / heatwave / storm), AR(1) noise, and rare
   events: solar outages, industrial demand spikes, suburb load collapse.
2. **Supervised pre-training** — an LSTM forecasts next-hour imbalance from a 6-hour
   window. Its feature extractor is reused later as a frozen backbone.
3. **Control**, two paradigms compared on the same environment:
   - **Centralized** — a Transformer-DQN over 7 discrete routing actions, plus a
     transfer-learning agent built on the frozen LSTM backbone
   - **Decentralized** — three PPO agents, one per substation, each with continuous
     control over a single transmission line

## Results

Evaluated over 240 hours of unseen data, with the window pinned so every controller
sees identical hours and identical weather.

**Forecaster** — MAE 5.00 MW, RMSE 7.54 MW, R² 0.911, +21.2 % skill over persistence.

**Centralized control**

| Controller | Reduction in total imbalance |
|---|---|
| Greedy oracle (upper bound) | 24.39 % |
| Transformer-DQN | **23.52 %** |
| TransferAgent (frozen backbone) | 22.04 % |
| Always hold | 0.00 % |
| Random | −14.91 % |

The Transformer-DQN recovers **96 %** of the control the oracle proves is achievable
for a single 20 MW transfer per hour.

**Decentralized control**

| Controller | Reduction |
|---|---|
| Rule-based proportional | 21.71 % |
| No control (sanity check) | 0.00 % |
| Decentralized PPO | **−14.93 %** |

The PPO agents made the grid *worse* than leaving it alone. The learned mean policy
drives each line to its capacity limit and holds it there, so the shunts stop
tracking the imbalance. Each agent is rewarded only for its own node while its
imbalance also depends on its neighbour's line, and a ±5 MW/hour delta cannot follow
a signal whose hourly swings are several times larger. Balancing the critic loss and
raising the learning rate were both measured and did not fix it. This is reported as
a genuine negative result rather than tuned away.

## Limitations

- The grid is synthetic — no real weather, demand or price data
- The transmission model has no losses and no line impedance
- The centralized action set allows one transfer per hour, so the oracle's figure is
  a ceiling for that action space, not for the problem
- Each figure is a single seeded run, not a mean over seeds

## Reproducing

    pip install tensorflow pandas numpy scipy scikit-learn matplotlib pettingzoo gymnasium
    jupyter notebook smart_grid_rl.ipynb

Seeded throughout (`SEED = 42`). Budgets are overridable by environment variable —
`PBL_MARL_EPISODES=20` and friends — for a quick smoke test.

## Note on references

The reference papers used for this project are not included in this repository, as
they are published works that aren't ours to redistribute.
