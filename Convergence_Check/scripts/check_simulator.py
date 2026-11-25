"""Unit checks of the synthetic simulator against the documentation's model (run before trusting any training result).
Usage: ./run.sh scripts/check_simulator.py      -> results/eval/simulator_checks.json"""
import json
import math
import os

import numpy as np

from convcheck.baselines import policy_jsq
from convcheck.stages import closed_c_system
from convcheck.systems import stage_b_system, stage_c_system, with_load
from convcheck.world import EdgeEnv, ServerSpec, SystemSpec, alone_mean_time, size_time_factor

RES = {}


def check(name, ok, **detail):
    RES[name] = dict(passed=bool(ok), **{k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in detail.items()})
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")


def quiet_server(dev="gpu", speed=1.0, cap=4):
    return ServerSpec(dev, speed=speed, capacity=cap, outlier_p=(0.0, 0.0, 0.0), noise=0.0)


def spec(servers, n_req=1, net=20.0, closed=True, alpha=0.4):
    return SystemSpec(servers=servers, n_users=1, net_base=np.full((1, len(servers)), net), net_jitter=0.0,
                      alpha=alpha, n_requests=n_req, closed_loop=closed, load=0.0)


def run_fixed(env, edges_fn):
    while env.advance():
        win = env.window()
        env.apply(edges_fn(env, win), win)
    return env


# 1. latency of one copy alone: T = N + E (no waiting, no sharing, no noise)
env = EdgeEnv(spec([quiet_server("gpu", 1.2)]), np.random.default_rng(0))
run_fixed(env, lambda e, w: [(0, 0)])
s = int(env.req_service[0])
E = alone_mean_time(s, env.spec.servers[0], size_time_factor(s, env.req_size[0]))
check("latency T = network + execution (single copy)", abs(env.req_T[0] - (20.0 + E)) < 1e-6,
      T=env.req_T[0], expected=20.0 + E)

# 2. progress rule: two copies on one server share it: rate 1/(base (1 + alpha (n-1)))
sp = spec([quiet_server("cpu", 1.0)], n_req=2, closed=False)
sp.load = 1.0
env = EdgeEnv(sp, np.random.default_rng(1))
env.arrival = np.array([0.0, 0.0])
env.req_service[:] = 1
env.req_size[:] = np.array([1.0, 2.0])
run_fixed(env, lambda e, w: [(p, 0) for p in range(len(w))])
b = [alone_mean_time(1, sp.servers[0], size_time_factor(1, z)) for z in (1.0, 2.0)]
t1 = 10 + b[0] * 1.4 + 10
t2 = 10 + b[0] * 1.4 + (b[1] - b[0]) + 10
check("progress rule (two tasks share a server, alpha = 0.4)",
      abs(env.req_T[0] - t1) < 1e-6 and abs(env.req_T[1] - t2) < 1e-6, T=list(env.req_T), expected=[t1, t2])

# 3. redundancy: first finished copy wins, the other is cancelled and frees its slot
sp = spec([quiet_server("gpu", 1.0), quiet_server("gpu", 0.5)])
env = EdgeEnv(sp, np.random.default_rng(2))
run_fixed(env, lambda e, w: [(0, 0), (0, 1)])
s = int(env.req_service[0])
Ef = alone_mean_time(s, sp.servers[0], size_time_factor(s, env.req_size[0]))
loser = [h for h in env.history if h["won"] == 0]
check("first copy wins, slower copy cancelled and its slot freed",
      abs(env.req_T[0] - (20 + Ef)) < 1e-6 and all(len(c) == 0 for c in env.copies_on) and len(loser) == 1,
      T=env.req_T[0], expected=20 + Ef, live_copies_after=[len(c) for c in env.copies_on])

# 4. reward: P(T) and late penalty and copy cost (documentation Eq. 3 / 11)
r_expected = []
for h in env.history[:1]:
    pass
T, D1, D2 = env.req_T[0], env.req_D1[0], env.req_D2[0]
P = 1.0 if T <= D1 else ((D2 - T) / (D2 - D1) if T <= D2 else 0.0)
busy = sum((c.exec_end_t if c.exec_end_t is not None else env.t) - c.dispatch_t for c in env.req_copies[0])
busy = sum(min(c.exec_end_t or 1e18, env.arrival[0] + T) - c.dispatch_t for c in env.req_copies[0])
r = P - 1.0 * max(0.0, T - D2) / D2 - 0.1 * busy / D2
check("reward r = P(T) - w_late * lateness - w_cost * busy time", abs(env.req_reward[0] - r) < 1e-6,
      reward=env.req_reward[0], expected=r)
# P(T) shape on a grid
Ps = []
for T in [10, 60, 105, 150, 200]:
    D1, D2 = 60.0, 150.0
    Ps.append(1.0 if T <= D1 else ((D2 - T) / (D2 - D1) if T <= D2 else 0.0))
check("satisfaction P(T): 1 before D1, linear to 0 at D2, 0 after", Ps == [1.0, 1.0, 0.5, 0.0, 0.0], P=Ps)

# 5. noise is mean-preserving and outliers happen at the stated rate
srv = ServerSpec("gpu", 1.0, 4, outlier_p=(0.15, 0.0, 0.0), noise=0.15)
env = EdgeEnv(spec([srv], n_req=1), np.random.default_rng(3))
env.req_service[0] = 0
env.req_size[0] = 1.0
xs = np.array([env._sample_base(0, 0) for _ in range(40000)])
m = alone_mean_time(0, srv)
out_rate = float(np.mean(xs > 2.5 * m))
clean_mean = float(xs[xs <= 2.5 * m].mean() / m)
check("noise is mean-preserving, outlier rate as specified (15%)",
      abs(clean_mean - 1.0) < 0.01 and abs(out_rate - 0.15) < 0.01, clean_mean_ratio=clean_mean, outlier_rate=out_rate)

# 6. every request finishes, no copy left, capacity never exceeded (random policy, bursty high load)
rng = np.random.default_rng(4)
ok, maxocc = True, 0
for k in range(10):
    env = EdgeEnv(with_load(stage_b_system(), 0.9, bursty=True), np.random.default_rng(k))
    while env.advance():
        win = env.window()
        free = env.free_capacity()
        edges = []
        for p in range(len(win)):
            for i in rng.permutation(env.n)[:rng.integers(1, 4)]:
                if free[i] > 0:
                    edges.append((p, int(i)))
                    free[i] -= 1
        env.apply(edges, win)
        maxocc = max(maxocc, max(len(c) / env.spec.servers[i].capacity for i, c in enumerate(env.copies_on)))
    ok &= bool(env.req_done.all()) and all(len(c) == 0 for c in env.copies_on)
check("every request finishes, no copy left over, capacity never exceeded", ok and maxocc <= 1.0, max_occupancy=maxocc)

# 7. load calibration: under single-copy JSQ, load < 1 is stable (waiting stays bounded)
waits = {}
for load in [0.3, 0.6, 0.85]:
    W = []
    for k in range(6):
        env = EdgeEnv(with_load(stage_b_system(), load, n_requests=300), np.random.default_rng(100 + k))
        run_fixed(env, lambda e, w: policy_jsq(e, np.random.default_rng(0)))
        W.append(np.nanmean(env.req_dispatch_t - env.arrival))
    waits[load] = float(np.mean(W))
check("load knob behaves: waiting grows with load and stays bounded below 1",
      waits[0.3] <= waits[0.6] <= waits[0.85] < 1000, mean_wait_ms=waits)

# 8. closed loop: at every decision all servers are idle (premise of the history test)
idle_ok = True
for k in range(10):
    env = EdgeEnv(closed_c_system(np.random.default_rng(k)), np.random.default_rng(k), d_max=1)
    while env.advance():
        idle_ok &= all(len(c) == 0 for c in env.copies_on) and len(env.queue) == 1
        win = env.window()
        env.apply([(0, int(np.argmax(env.free_capacity())))], win)
check("closed loop: one waiting request and all servers idle at every decision", idle_ok)

# 9. slowed/normal twins are the same system except the secret slowdown
same = True
for k in range(20):
    a = stage_c_system(np.random.default_rng(k), slowed=True)
    b = stage_c_system(np.random.default_rng(k), slowed=False)
    diff = [i for i, (x, y) in enumerate(zip(a.servers, b.servers)) if x.__dict__ != y.__dict__]
    same &= len(diff) == 1 and a.servers[diff[0]].slow_factor == 3.0 and np.allclose(a.net_base, b.net_base)
check("slowed / normal twin systems differ only in the slowed server", same)

# 10. determinism: same seed -> identical episode
def episode(seed):
    env = EdgeEnv(with_load(stage_b_system(), 0.6), np.random.default_rng(seed))
    run_fixed(env, lambda e, w: policy_jsq(e, np.random.default_rng(0)))
    return env.req_T.copy()
check("same seed gives an identical episode", np.array_equal(episode(5), episode(5)))

# 11. the hard-rule audit (used on every evaluation decision) catches fabricated violations and passes legal decisions
env = EdgeEnv(with_load(stage_b_system(), 0.5), np.random.default_rng(7), theta_u=0.25)
env.arrival[:2] = 0.0                                           # two requests waiting, every server free
env.advance()
win = env.window()
env.audit = {k: 0 for k in env.audit}
env._audit([(0, 0)] * 9, win)                                   # 9 copies of one request on one server
bad1 = dict(env.audit)
env.audit = {k: 0 for k in env.audit}
env.arrival[win[0]] = env.t - 0.9 * env.req_D2[win[0]]          # make request 0 urgent, then send only request 1
env._audit([(1, 0)], win)
bad2 = dict(env.audit)
env.audit = {k: 0 for k in env.audit}
env._audit([(0, 0), (1, 1)], win)
good = dict(env.audit)
check("hard-rule audit catches fabricated violations and passes legal decisions",
      bad1["capacity"] == 1 and bad1["copy_limit"] == 1 and bad1["duplicate_edge"] == 1 and bad2["urgent_not_sent"] == 1
      and sum(v for k, v in good.items() if k != "decisions") == 0, fabricated=bad1, urgent_case=bad2, legal=good)

os.makedirs("results/eval", exist_ok=True)
json.dump(RES, open("results/eval/simulator_checks.json", "w"), indent=1, default=str)
print(f"{sum(v['passed'] for v in RES.values())} of {len(RES)} simulator checks passed")
