"""
run_training.py
================
Executes the GBWM + RL training pipeline extracted from Phase_one_fixed.ipynb.
Outputs: results/all_results.json
"""

# ── Cell 3: GBWM environment ─────────────────────────────────────────────────
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

MU_ANNUAL  = np.array([0.0493, 0.0770, 0.0886])
COV_ANNUAL = np.array([
    [ 0.0017, -0.0017, -0.0021],
    [-0.0017,  0.0396,  0.0309],
    [-0.0021,  0.0309,  0.0392],
])
MU_MIN, MU_MAX = 0.0526, 0.0886
N_PORTFOLIOS   = 15


def _build_frontier():
    Si   = np.linalg.inv(COV_ANNUAL)
    ones = np.ones(3)
    k = MU_ANNUAL @ Si @ ones
    l = MU_ANNUAL @ Si @ MU_ANNUAL
    p = ones       @ Si @ ones
    D = l*p - k**2
    g = (l * Si @ ones       - k * Si @ MU_ANNUAL) / D
    h = (p * Si @ MU_ANNUAL  - k * Si @ ones)      / D
    a = float(h @ COV_ANNUAL @ h)
    b = float(2*(g @ COV_ANNUAL @ h))
    c = float(g @ COV_ANNUAL @ g)
    mu_grid  = np.linspace(MU_MIN, MU_MAX, N_PORTFOLIOS)
    sig_grid = np.sqrt(np.maximum(a*mu_grid**2 + b*mu_grid + c, 0))
    return mu_grid, sig_grid

PORTFOLIO_MU, PORTFOLIO_SIGMA = _build_frontier()


@dataclass
class InvestorProfile:
    name:       str
    W0:         float
    G:          float
    T:          int
    cash_flows: np.ndarray = field(default_factory=lambda: np.zeros(0))

    def __post_init__(self):
        if len(self.cash_flows) == 0:
            self.cash_flows = np.zeros(self.T)


SCENARIOS = {
    "base": InvestorProfile(
        name="Base case", W0=100_000, G=200_000, T=10, cash_flows=np.zeros(10),
    ),
    "young": InvestorProfile(
        name="Young investor", W0=50_000, G=500_000, T=30, cash_flows=np.full(30, 5_000),
    ),
    "retirement": InvestorProfile(
        name="Retirement", W0=100_000, G=121_400, T=30,
        cash_flows=np.array([15_000]*15 + [-50_000*(1.03**t) for t in range(1,16)]),
    ),
    "stressed": InvestorProfile(
        name="Stressed", W0=100_000, G=200_000, T=10, cash_flows=np.full(10, -10_000),
    ),
}


class GBWMEnv:
    def __init__(self, profile: InvestorProfile, seed: Optional[int] = None):
        self.profile = profile
        self.rng     = np.random.default_rng(seed)
        self.W       = 0.0
        self.t       = 0

    def reset(self) -> np.ndarray:
        self.W = self.profile.W0
        self.t = 0
        return self._state()

    def step(self, action: int):
        mu  = PORTFOLIO_MU[action]
        sig = PORTFOLIO_SIGMA[action]
        cf  = self.profile.cash_flows[self.t]
        W_pre = self.W + cf
        if W_pre <= 0:
            self.W = 0.0
            self.t += 1
            done   = self.t >= self.profile.T
            reward = 1.0 if (done and self.W >= self.profile.G) else 0.0
            return self._state(), reward, done
        Z      = self.rng.standard_normal()
        self.W = W_pre * np.exp((mu - 0.5*sig**2) + sig*Z)
        self.t += 1
        done   = self.t >= self.profile.T
        reward = 1.0 if (done and self.W >= self.profile.G) else 0.0
        return self._state(), reward, done

    def _state(self) -> np.ndarray:
        return np.array([
            np.clip(self.W / self.profile.G, 0, 5),
            (self.profile.T - self.t) / self.profile.T,
        ], dtype=np.float32)


def simulate_policy(profile, policy_fn, n_episodes=10_000, seed=0):
    env      = GBWMEnv(profile, seed=seed)
    terminal = np.zeros(n_episodes)
    for ep in range(n_episodes):
        state = env.reset()
        env.rng = np.random.default_rng(seed + ep)
        done  = False
        while not done:
            action        = policy_fn(state)
            state, r, done = env.step(action)
        terminal[ep] = env.W
    success = terminal >= profile.G
    return {
        "pr_goal":        float(success.mean()),
        "pr_bankruptcy":  float((terminal <= 0).mean()),
        "mean_terminal":  float(terminal.mean()),
        "p10":            float(np.percentile(terminal, 10)),
        "median":         float(np.median(terminal)),
        "p90":            float(np.percentile(terminal, 90)),
        "terminal":       terminal,
    }


def tdf_policy(state):
    port = int(round(2 + state[1] * 10))
    return int(np.clip(port, 0, N_PORTFOLIOS-1))

def aggressive_policy(state):  return N_PORTFOLIOS - 1
def conservative_policy(state): return 0
def moderate_policy(state):     return N_PORTFOLIOS // 2

def greedy_policy(state):
    w = state[0]
    if w < 0.7:   return 12
    elif w < 1.0: return 7
    else:         return 2


# ── Cell 5: Gym wrapper (with reward shaping) ────────────────────────────────
import gymnasium as gym
from gymnasium import spaces

class GBWMGymEnv(gym.Env):
    def __init__(self, profile):
        super().__init__()
        self.profile = profile
        self.W = profile.W0
        self.t = 0
        self.rng = np.random.default_rng()
        self.observation_space = spaces.Box(
            low=np.array([0., 0.], dtype=np.float32),
            high=np.array([5., 1.], dtype=np.float32))
        self.action_space = spaces.Discrete(N_PORTFOLIOS)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.W = self.profile.W0
        self.t = 0
        return self._obs(), {}

    def step(self, action):
        mu  = PORTFOLIO_MU[int(action)]
        sig = PORTFOLIO_SIGMA[int(action)]
        cf  = (self.profile.cash_flows[self.t]
               if self.t < len(self.profile.cash_flows) else 0.)
        W_pre = self.W + cf
        if W_pre <= 0:
            self.W = 0.
            self.t += 1
            done = self.t >= self.profile.T
            return self._obs(), 0., done, False, {}
        Z = self.rng.standard_normal()
        self.W = W_pre * np.exp((mu - 0.5*sig**2) + sig*Z)
        self.t += 1
        done = self.t >= self.profile.T
        if done:
            reward = 1.0 if self.W >= self.profile.G else 0.0
        else:
            # Reward shaping: stronger intermediate signal for stressed scenario
            w_ratio = float(np.clip(self.W / self.profile.G, 0, 1))
            reward = 0.01 * w_ratio          # 10x stronger baseline
            if w_ratio >= 0.5:               # bonus for staying above half-goal
                reward += 0.005
        return self._obs(), reward, done, False, {}

    def _obs(self):
        return np.array([
            float(np.clip(self.W / self.profile.G, 0, 5)),
            float((self.profile.T - self.t) / self.profile.T)
        ], dtype=np.float32)


# ── Cell 9: Train & Evaluate ─────────────────────────────────────────────────
import time, json, os
from stable_baselines3 import PPO as SB3_PPO

np.random.seed(42)

RESULTS_DIR = 'results'
N_TEST      = 10_000
N_TRAJ      = 50_000
BATCH_SIZE  = 4_800

print('=' * 60)
print('GBWM PPO TRAINING & EVALUATION')
print('Paper: Das, Mittal, Ostrov et al. (2024)')
print('=' * 60)

all_results = {}

for scenario_key, profile in SCENARIOS.items():
    print(f"\n" + '-'*60)
    print(f'SCENARIO: {profile.name}')
    print(f'  W0=${profile.W0:,.0f}  G=${profile.G:,.0f}  T={profile.T}yr')
    print('-'*60)

    print('\n[1/3] Training PPO agent (Stable-Baselines3)...')
    t0  = time.time()
    env = GBWMGymEnv(profile)

    model = SB3_PPO(
        'MlpPolicy', env,
        learning_rate = 0.01,
        n_steps       = BATCH_SIZE,
        clip_range    = 0.50,
        policy_kwargs = dict(net_arch=[64, 64]),
        verbose       = 0,
        seed          = 42,
    )
    model.learn(total_timesteps=N_TRAJ)
    train_time = time.time() - t0
    print(f'  Training time: {train_time:.1f}s')

    def ppo_fn(state):
        obs = np.array(state, dtype=np.float32).reshape(1, -1)
        action, _ = model.predict(obs, deterministic=True)
        return int(action)

    print(f'\n[2/3] Evaluating strategies ({N_TEST:,} episodes each)...')
    strategies = {
        'PPO (RL)':     ppo_fn,
        'TDF':          tdf_policy,
        'Aggressive':   aggressive_policy,
        'Conservative': conservative_policy,
        'Moderate':     moderate_policy,
        'Greedy':       greedy_policy,
    }

    eval_results = {}
    for name, fn in strategies.items():
        res = simulate_policy(profile, fn, n_episodes=N_TEST, seed=999)
        eval_results[name] = res
        print(f'  {name:<18} Pr[goal]={res["pr_goal"]:.4f}  median=${res["median"]:>10,.0f}')

    dp_approx = {'base':0.669,'young':0.750,'retirement':0.586,'stressed':0.380}
    dp_pr  = dp_approx.get(scenario_key, 0.65)
    ppo_pr = eval_results['PPO (RL)']['pr_goal']
    rl_eff = ppo_pr / dp_pr if dp_pr > 0 else 0.0

    print(f'\n  DP benchmark (from paper): {dp_pr:.4f}')
    print(f'  PPO result:                {ppo_pr:.4f}')
    print(f'  RL Efficiency:             {rl_eff:.4f}  ({rl_eff*100:.1f}% of DP)')

    all_results[scenario_key] = {
        'profile':        {'name':profile.name,'W0':profile.W0,'G':profile.G,'T':profile.T},
        'strategies':     {k:{kk:vv for kk,vv in v.items() if kk!='terminal'} for k,v in eval_results.items()},
        'dp_pr_goal':     dp_pr,
        'ppo_pr_goal':    ppo_pr,
        'rl_efficiency':  rl_eff,
        'training_log':   [],
        'train_time_s':   train_time,
        'terminal_wealth':{k:v['terminal'].tolist() for k,v in eval_results.items()},
    }

print('\n[3/3] Saving results...')
os.makedirs(RESULTS_DIR, exist_ok=True)
with open(f'{RESULTS_DIR}/all_results.json','w') as f:
    json.dump(all_results, f, indent=2)

print('\n' + '='*65)
print(f'{"SCENARIO":<20} {"PPO":>8} {"DP":>8} {"TDF":>8} {"RL Eff":>10}')
print('='*65)
for sk, res in all_results.items():
    ppo=res['ppo_pr_goal']; dp=res['dp_pr_goal']
    tdf=res['strategies']['TDF']['pr_goal']; eff=res['rl_efficiency']
    flag='OK' if eff>=0.90 else ('~' if eff>=0.70 else 'X')
    print(f"{res['profile']['name']:<20} {ppo:>8.4f} {dp:>8.4f} {tdf:>8.4f} {eff:>9.1%}  {flag}")
print('='*65)
print(f'\nResults saved to {RESULTS_DIR}/all_results.json')
