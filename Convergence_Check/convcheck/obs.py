"""Builds the model input (tokens, edge features, masks) from the simulator state.
Token contents follow Section 4.2 of the documentation, with one-hot input/output types."""
import math

import numpy as np

from .world import SERVICES, N_IN_TYPES, N_OUT_TYPES

REQ_DIM, SRV_DIM, RUN_DIM, HIST_DIM, GLOB_DIM = 11, 4, 3, 15, 3
EDGE_STATIC = 1      # estimated network time / D2 (the other edge features change during a decision)


def _type_vec(service):
    s = SERVICES[service]
    v = np.zeros(N_IN_TYPES + N_OUT_TYPES, dtype=np.float32)
    v[s["in_type"]] = 1.0
    v[N_IN_TYPES + s["out_type"]] = 1.0
    return v


def _size_vec(service, size):
    if SERVICES[service]["size_kind"] == "duration":
        return np.array([math.log(size), 0.0], dtype=np.float32)
    return np.array([0.0, math.log(size)], dtype=np.float32)


def build_obs(env):
    t = env.t
    win = env.window()
    k = len(win)
    n = env.n
    kmax_cap = max(s.capacity for s in env.spec.servers)
    # requests
    req = np.zeros((k, REQ_DIM), dtype=np.float32)
    req_tag = np.zeros(k, dtype=np.int64)
    urgent = np.zeros(k, dtype=bool)
    edge_net = np.zeros((k, n), dtype=np.float32)
    for p, j in enumerate(win):
        s = int(env.req_service[j])
        W = t - env.arrival[j]
        D1, D2 = env.req_D1[j], env.req_D2[j]
        hard_left = (D2 - W) / D2
        urgent[p] = hard_left <= env.theta_u
        req[p] = np.concatenate([_type_vec(s), _size_vec(s, env.req_size[j]),
                                 [W / D2, (D1 - W) / D1, hard_left, float(urgent[p])]])
        req_tag[p] = env.svc_tag[s]
        edge_net[p] = env.spec.net_base[env.req_user[j]] / D2
    # servers
    free = env.free_capacity()
    kappa = np.array([s.capacity for s in env.spec.servers], dtype=np.float32)
    srv = np.zeros((n, SRV_DIM), dtype=np.float32)
    for i, s in enumerate(env.spec.servers):
        srv[i] = [1.0 if s.device == "cpu" else 0.0, 1.0 if s.device == "gpu" else 0.0,
                  len(env.copies_on[i]) / s.capacity, s.capacity / kmax_cap]
    srv_tag = env.srv_tag.astype(np.int64)
    # privileged training target for the optional auxiliary loss (never an input): log effective speed
    srv_aux = np.array([math.log(s.speed / s.slow_factor) for s in env.spec.servers], dtype=np.float32)
    idle_any = bool(any(len(env.copies_on[i]) == 0 for i in range(n)))
    # running copies
    run_rows, run_svc, run_srv = [], [], []
    for i in range(n):
        for c in env.copies_on[i]:
            j = c.req
            D2 = env.req_D2[j]
            alive_copies = sum(1 for cc in env.req_copies[j] if cc.alive and cc.exec_end_t is None)
            run_rows.append([(t - c.dispatch_t) / D2, (D2 - (t - env.arrival[j])) / D2, alive_copies / env.d_max])
            run_svc.append(env.svc_tag[env.req_service[j]])
            run_srv.append(env.srv_tag[i])
    run = np.array(run_rows, dtype=np.float32).reshape(-1, RUN_DIM)
    # history (last M ended copies)
    hist_rows, hist_svc, hist_srv = [], [], []
    for h in env.history[-env.M:]:
        s = h["service"]
        cap = env.spec.servers[h["server"]].capacity
        hist_rows.append(np.concatenate([_type_vec(s), _size_vec(s, h["size"]),
                                         [h["n_start"] / cap, h["obs"], h["won"], h["met1"], h["met2"],
                                          h["P"], h["r"], (t - h["end_t"]) / 1000.0]]))
        hist_svc.append(env.svc_tag[s])
        hist_srv.append(env.srv_tag[h["server"]])
    hist = np.array(hist_rows, dtype=np.float32).reshape(-1, HIST_DIM)
    # global
    recent = np.sum((env.arrival[:env.next_arrival_idx] > t - 100.0))
    glob = np.array([len(env.queue) / env.K, recent / 10.0, sum(1 for i in range(n) if env.copies_on[i]) / n],
                    dtype=np.float32)
    return dict(window=win, req=req, req_tag=req_tag, urgent=urgent, edge_net=edge_net,
                srv=srv, srv_tag=srv_tag, srv_aux=srv_aux, free=free.astype(np.float32), kappa=kappa, idle_any=idle_any,
                run=run, run_svc=np.array(run_svc, dtype=np.int64), run_srv=np.array(run_srv, dtype=np.int64),
                hist=hist, hist_svc=np.array(hist_svc, dtype=np.int64), hist_srv=np.array(hist_srv, dtype=np.int64),
                glob=glob)
