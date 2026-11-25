"""Synthetic edge world for the convergence check.

Everything here is made up (no measured data is used). It follows the design document:
users -> controller -> heterogeneous servers, requests of three service types, redundant copies,
first finished copy wins and the others are cancelled, tasks that share a server slow each other down
(progress rule), rare slow outliers, per user-server network time, deadline satisfaction and the reward
of the document (Eq. 3 and Eq. 11 of the method documentation).
"""
import heapq
import math
from dataclasses import dataclass, field

import numpy as np

# ----------------------------------------------------------------------------------------------
# Services. Input/output types are one-hot (as in the current documentation).
# base_cpu: time of one input of size factor 1 alone on a reference CPU server (ms).
# gpu_speedup: how much faster a reference GPU server is for this service.
# ----------------------------------------------------------------------------------------------
SERVICES = [
    dict(name="speech-like", in_type=0, out_type=0, base_cpu=400.0, gpu_speedup=25.0, D1=60.0, D2=150.0,
         size_kind="duration"),
    dict(name="detect-like", in_type=1, out_type=1, base_cpu=60.0, gpu_speedup=12.0, D1=35.0, D2=150.0,
         size_kind="pixels"),
    dict(name="segment-like", in_type=1, out_type=2, base_cpu=80.0, gpu_speedup=12.0, D1=40.0, D2=150.0,
         size_kind="pixels"),
]
N_SERVICES = len(SERVICES)
N_IN_TYPES, N_OUT_TYPES = 2, 3


@dataclass
class ServerSpec:
    device: str                      # "gpu" or "cpu"
    speed: float = 1.0               # multiplies the speed of the reference device (hidden from the model)
    capacity: int = 4                # max copies at once (kappa_i)
    outlier_p: tuple = (0.02, 0.02, 0.02)   # probability of a slow outlier, per service
    outlier_factor: float = 5.0      # outlier makes one input this many times slower
    noise: float = 0.15              # log-normal noise (sigma)
    slow_factor: float = 1.0         # secret slowdown (Stage C); > 1 means slower than it looks


@dataclass
class SystemSpec:
    servers: list
    n_users: int = 20
    net_base: np.ndarray = None      # (users, servers) base network time in ms (upload + download)
    net_jitter: float = 1.5
    service_probs: tuple = (1 / 3, 1 / 3, 1 / 3)
    alpha: float = 0.4               # each extra running task adds 40% to everyone's time on that server
    load: float = 0.5                # arrival rate as a fraction of the single-copy capacity
    bursty: bool = False
    n_requests: int = 120
    closed_loop: bool = False        # Stage A: next request only after the previous one finished
    deadline_scale: float = 1.0


def alone_mean_time(service, server: ServerSpec, size_factor=1.0):
    """Expected time of one input alone on a server (without noise and outliers), in ms."""
    s = SERVICES[service]
    base = s["base_cpu"] / (s["gpu_speedup"] if server.device == "gpu" else 1.0)
    return base * size_factor / server.speed * server.slow_factor


def size_time_factor(service, size_value):
    """How the input size changes computation time: linear for audio duration, weak for image pixels."""
    if SERVICES[service]["size_kind"] == "duration":
        return size_value
    return 0.8 + 0.2 * size_value


def single_copy_capacity(spec: SystemSpec):
    """Requests per ms the system can serve with one copy per request at full occupancy (rough)."""
    rate = 0.0
    for srv in spec.servers:
        mean_t = sum(p * alone_mean_time(s, srv) for s, p in enumerate(spec.service_probs))
        k = srv.capacity
        rate += k / (mean_t * (1 + spec.alpha * (k - 1)))
    return rate


def sample_net(rng, n_users, n_servers):
    return rng.uniform(14.0, 26.0, size=(n_users, n_servers))


# ----------------------------------------------------------------------------------------------
# Simulator
# ----------------------------------------------------------------------------------------------
@dataclass
class Copy:
    req: int
    server: int
    dispatch_t: float
    net_up: float
    net_down: float
    base: float = 0.0          # time this copy needs alone (noise and outlier included), ms
    progress: float = 0.0      # fraction of work done
    running: bool = False      # False while the input is still being uploaded
    alive: bool = True
    exec_end_t: float = None
    n_on_server_at_start: int = 0
    copy_id: int = 0


class EdgeEnv:
    """Event-driven controller + servers. Call advance() until it returns True (a decision is needed),
    then observe(), then apply(edges). Episode ends when every request has finished."""

    def __init__(self, spec: SystemSpec, rng: np.random.Generator, K=8, M=32, d_max=3, theta_u=0.25,
                 w_late=1.0, w_cost=0.1, n_svc_tags=8, n_srv_tags=8):
        self.spec, self.rng = spec, rng
        self.K, self.M, self.d_max, self.theta_u = K, M, d_max, theta_u
        self.w_late, self.w_cost = w_late, w_cost
        self.n_svc_tags, self.n_srv_tags = n_svc_tags, n_srv_tags
        self.reset()

    # ---------------- setup ----------------
    def reset(self):
        sp, rng = self.spec, self.rng
        n = len(sp.servers)
        self.n = n
        if sp.net_base is None:
            sp.net_base = sample_net(rng, sp.n_users, n)
        # anonymous tags, reshuffled every episode (1..n_tags; 0 = no tag)
        self.svc_tag = rng.permutation(self.n_svc_tags)[:N_SERVICES] + 1
        self.srv_tag = rng.permutation(self.n_srv_tags)[:n] + 1
        R = sp.n_requests
        self.R = R
        self.req_service = rng.choice(N_SERVICES, size=R, p=np.asarray(sp.service_probs))
        self.req_user = rng.integers(0, sp.n_users, size=R)
        self.req_size = np.where(self.req_service == 0,
                                 np.exp(rng.uniform(np.log(0.5), np.log(2.0), size=R)),   # duration factor
                                 np.exp(rng.uniform(np.log(0.5), np.log(2.0), size=R)))   # pixel factor
        self.req_D1 = np.array([SERVICES[s]["D1"] for s in self.req_service]) * sp.deadline_scale
        self.req_D2 = np.array([SERVICES[s]["D2"] for s in self.req_service]) * sp.deadline_scale
        # arrival times
        if sp.closed_loop:
            self.arrival = np.full(R, np.inf)
            self.arrival[0] = 0.0
        else:
            lam = sp.load * single_copy_capacity(sp)          # requests per ms
            if sp.bursty:                                      # calm / burst periods, burst rate 3x calm
                times, t, i = [], 0.0, 0
                while i < R:
                    burst = (int(t // 200.0) % 2) == 1
                    rate = lam * (1.5 if burst else 0.5)
                    t += rng.exponential(1.0 / rate)
                    times.append(t)
                    i += 1
                self.arrival = np.array(times)
            else:
                self.arrival = np.cumsum(rng.exponential(1.0 / lam, size=R))
        self.t = 0.0
        self.next_arrival_idx = 0
        self.queue = []                       # waiting request ids
        self.copies_on = [[] for _ in range(n)]   # live copies per server
        self.req_copies = [[] for _ in range(R)]
        self.req_done = np.zeros(R, dtype=bool)
        self.req_T = np.full(R, np.nan)
        self.req_reward = np.zeros(R)
        self.req_P = np.zeros(R)
        self.req_ncopies = np.zeros(R, dtype=int)
        self.req_decision = np.full(R, -1)
        self.req_dispatch_t = np.full(R, np.nan)
        self.upload_heap, self.result_heap = [], []
        self.counter = 0
        self.history = []                     # list of dicts (ended copies)
        self.decision_times = []
        self.decision_rewards = []
        self.decision_c = []                  # requests sent per decision
        self.copies_per_server = np.zeros(n, dtype=int)
        self.copies_by_decision_server = []   # (decision index, server) for adaptation curves
        # audit of the hard rules at every decision (counts only; never changes behaviour)
        self.audit = dict(decisions=0, capacity=0, copy_limit=0, duplicate_edge=0, urgent_not_sent=0, idle_not_used=0)
        self.dirty = True
        self.n_done = 0
        self.last_update_t = np.zeros(n)

    # ---------------- helpers ----------------
    def free_capacity(self):
        return np.array([self.spec.servers[i].capacity - len(self.copies_on[i]) for i in range(self.n)])

    def n_running(self, i):
        return sum(1 for c in self.copies_on[i] if c.running)

    def _advance_servers(self, t_new):
        """Progress rule: running copies advance by dt / (base * (1 + alpha (n_running - 1)))."""
        a = self.spec.alpha
        for i in range(self.n):
            run = [c for c in self.copies_on[i] if c.running]
            if run:
                f = 1.0 + a * (len(run) - 1)
                dt = t_new - self.t
                for c in run:
                    c.progress += dt / (c.base * f)
        self.t = t_new

    def _next_exec_end(self):
        best_t, best_c = math.inf, None
        a = self.spec.alpha
        for i in range(self.n):
            run = [c for c in self.copies_on[i] if c.running]
            if run:
                f = 1.0 + a * (len(run) - 1)
                for c in run:
                    te = self.t + max(0.0, 1.0 - c.progress) * c.base * f
                    if te < best_t:
                        best_t, best_c = te, c
        return best_t, best_c

    def _sample_base(self, req, i):
        srv = self.spec.servers[i]
        s = self.req_service[req]
        mean = alone_mean_time(s, srv, size_time_factor(s, self.req_size[req]))
        x = mean * math.exp(self.rng.normal(0.0, srv.noise) - 0.5 * srv.noise ** 2)
        if self.rng.random() < srv.outlier_p[s]:
            x *= srv.outlier_factor
        return x

    # ---------------- main loop ----------------
    def finished(self):
        return self.n_done >= self.R

    def needs_decision(self):
        return self.dirty and len(self.queue) > 0 and (self.free_capacity() > 0).any()

    def advance(self):
        """Run events until a decision is needed (True) or the episode is over (False)."""
        while True:
            if self.needs_decision():
                return True
            if self.finished():
                return False
            # candidate next events
            t_arr = self.arrival[self.next_arrival_idx] if self.next_arrival_idx < self.R else math.inf
            t_up = self.upload_heap[0][0] if self.upload_heap else math.inf
            t_res = self.result_heap[0][0] if self.result_heap else math.inf
            t_end, c_end = self._next_exec_end()
            t_next = min(t_arr, t_up, t_res, t_end)
            if t_next == math.inf:
                raise RuntimeError("simulator stuck")
            self._advance_servers(t_next)
            if t_next == t_end:
                self._on_exec_end(c_end)
            elif t_next == t_res:
                _, _, c = heapq.heappop(self.result_heap)
                self._on_result(c)
            elif t_next == t_up:
                _, _, c = heapq.heappop(self.upload_heap)
                self._on_exec_start(c)
            else:
                self._on_arrival(self.next_arrival_idx)
                self.next_arrival_idx += 1

    def _on_arrival(self, j):
        self.queue.append(j)
        self.dirty = True

    def _on_exec_start(self, c):
        if not c.alive:
            return
        c.running = True
        c.progress = 0.0
        c.base = self._sample_base(c.req, c.server)
        self.dirty = True

    def _on_exec_end(self, c):
        c.running = False
        c.exec_end_t = self.t
        self.copies_on[c.server].remove(c)
        self.counter += 1
        heapq.heappush(self.result_heap, (self.t + c.net_down, self.counter, c))
        self.dirty = True

    def _on_result(self, c):
        if not c.alive:
            return
        j = c.req
        if self.req_done[j]:
            return
        self._complete(j, c)

    def _complete(self, j, winner):
        t = self.t
        T = t - self.arrival[j]
        D1, D2 = self.req_D1[j], self.req_D2[j]
        if T <= D1:
            P = 1.0
        elif T <= D2:
            P = (D2 - T) / (D2 - D1)
        else:
            P = 0.0
        busy = 0.0
        ended = []
        for c in self.req_copies[j]:
            if c is winner:
                b = c.exec_end_t - c.dispatch_t
                obs_t = t - c.dispatch_t
                won = 1
            else:
                if c.alive and c in self.copies_on[c.server]:
                    self.copies_on[c.server].remove(c)
                end = c.exec_end_t if c.exec_end_t is not None else t
                b = end - c.dispatch_t
                obs_t = t - c.dispatch_t
                won = 0
                c.alive = False
            busy += b
            ended.append((c, obs_t, won))
        r = P - self.w_late * max(0.0, T - D2) / D2 - self.w_cost * busy / D2
        self.req_done[j] = True
        self.n_done += 1
        self.req_T[j], self.req_P[j], self.req_reward[j] = T, P, r
        k = self.req_decision[j]
        if k >= 0:
            self.decision_rewards[k] += r
        for c, obs_t, won in ended:
            self.history.append(dict(service=int(self.req_service[j]), server=c.server, size=self.req_size[j],
                                     n_start=c.n_on_server_at_start, obs=obs_t / D2, won=won,
                                     met1=float(T <= D1), met2=float(T <= D2), P=P, r=r, end_t=t))
        if len(self.history) > 4 * self.M:
            self.history = self.history[-self.M:]
        self.dirty = True
        if self.spec.closed_loop and self.next_arrival_idx < self.R and not self.queue:
            self.arrival[self.next_arrival_idx] = t + 1.0

    # ---------------- decisions ----------------
    def window(self):
        """The K most urgent waiting requests (earliest absolute hard deadline first)."""
        q = sorted(self.queue, key=lambda j: self.arrival[j] + self.req_D2[j])
        return q[:self.K]

    def _audit(self, edges, window):
        a = self.audit
        a["decisions"] += 1
        free = self.free_capacity()
        per_srv = np.zeros(self.n, dtype=int)
        per_req = {}
        for pos, i in edges:
            per_srv[i] += 1
            per_req[pos] = per_req.get(pos, 0) + 1
        a["capacity"] += int((per_srv > free).any())
        a["copy_limit"] += int(any(v > self.d_max for v in per_req.values()))
        a["duplicate_edge"] += int(len(set(edges)) != len(edges))
        left = int(free.sum()) - len(edges)
        if left > 0:                      # capacity was still free after the decision
            for pos, j in enumerate(window):
                if pos not in per_req and (self.req_D2[j] - (self.t - self.arrival[j])) / self.req_D2[j] <= self.theta_u:
                    a["urgent_not_sent"] += 1
                    break
        if not edges and len(window) > 0 and any(len(c) == 0 for c in self.copies_on):
            a["idle_not_used"] += 1

    def apply(self, edges, window):
        """edges: list of (position in window, server index)."""
        k = len(self.decision_times)
        self._audit(edges, window)
        self.decision_times.append(self.t)
        self.decision_rewards.append(0.0)
        sent = set()
        for pos, i in edges:
            j = window[pos]
            N = self.spec.net_base[self.req_user[j], i] + self.rng.normal(0.0, self.spec.net_jitter)
            N = max(2.0, N)
            self.counter += 1
            c = Copy(req=j, server=i, dispatch_t=self.t, net_up=N / 2, net_down=N / 2,
                     n_on_server_at_start=len(self.copies_on[i]), copy_id=self.counter)
            self.copies_on[i].append(c)
            self.req_copies[j].append(c)
            heapq.heappush(self.upload_heap, (self.t + c.net_up, self.counter, c))
            self.copies_per_server[i] += 1
            self.copies_by_decision_server.append((k, i))
            sent.add(j)
        for j in sent:
            self.queue.remove(j)
            self.req_decision[j] = k
            self.req_ncopies[j] = len(self.req_copies[j])
            self.req_dispatch_t[j] = self.t
        self.decision_c.append(len(sent))
        self.dirty = False
        return k

    # ---------------- statistics ----------------
    def stats(self):
        done = self.req_done
        Tn = self.req_T[done] / self.req_D2[done]
        return dict(
            mean_reward=float(self.req_reward[done].mean()),
            mean_P=float(self.req_P[done].mean()),
            miss_hard=float((self.req_T[done] > self.req_D2[done]).mean()),
            p50_norm=float(np.percentile(Tn, 50)), p95_norm=float(np.percentile(Tn, 95)),
            p99_norm=float(np.percentile(Tn, 99)),
            copies_per_req=float(self.req_ncopies[done].mean()),
            mean_c=float(np.mean([c for c in self.decision_c if c > 0])) if any(self.decision_c) else 0.0,
            n_decisions=len(self.decision_times),
        )
