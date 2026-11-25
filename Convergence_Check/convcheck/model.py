"""The proposed scheduler model as described in the documentation (Section 4), sized down for a laptop.

Encoder: per-token-type linear embedding + service/server tag embeddings + Transformer encoder (no positions).
Decoder: picks (request, server) edges one at a time until STOP. Edge score = MatNet-style mixed score
         C*tanh(mix(q_j.k_i/sqrt(d), f_ji)); STOP score = C*tanh(small MLP on [g; f_stop]). Masks enforce
         the hard rules (capacity, d_max, no repeated edge, urgent requests, idle server).
Critic:  small MLP on the mean of the encoder outputs (training only).
"""
import math

import numpy as np
import torch
import torch.nn as nn

from .obs import REQ_DIM, SRV_DIM, RUN_DIM, HIST_DIM, GLOB_DIM

N_EDGE_FEAT, N_STOP_FEAT = 4, 4


def collate(obs_list, use_history=True):
    """Pad a list of observations into batch tensors."""
    B = len(obs_list)
    Kp = max(1, max(len(o["window"]) for o in obs_list))
    Np = max(len(o["free"]) for o in obs_list)
    Rp = max(1, max(len(o["run"]) for o in obs_list))
    Mp = max(1, max(len(o["hist"]) for o in obs_list))
    out = dict(
        req=np.zeros((B, Kp, REQ_DIM), np.float32), req_tag=np.zeros((B, Kp), np.int64),
        req_mask=np.zeros((B, Kp), bool), urgent=np.zeros((B, Kp), bool),
        edge_net=np.zeros((B, Kp, Np), np.float32),
        srv=np.zeros((B, Np, SRV_DIM), np.float32), srv_tag=np.zeros((B, Np), np.int64),
        srv_aux=np.zeros((B, Np), np.float32),
        srv_mask=np.zeros((B, Np), bool), free=np.zeros((B, Np), np.float32), kappa=np.ones((B, Np), np.float32),
        run=np.zeros((B, Rp, RUN_DIM), np.float32), run_svc=np.zeros((B, Rp), np.int64),
        run_srv=np.zeros((B, Rp), np.int64), run_mask=np.zeros((B, Rp), bool),
        hist=np.zeros((B, Mp, HIST_DIM), np.float32), hist_svc=np.zeros((B, Mp), np.int64),
        hist_srv=np.zeros((B, Mp), np.int64), hist_mask=np.zeros((B, Mp), bool),
        glob=np.zeros((B, GLOB_DIM), np.float32), idle_any=np.zeros(B, bool))
    for b, o in enumerate(obs_list):
        k, n, r, m = len(o["window"]), len(o["free"]), len(o["run"]), len(o["hist"])
        out["req"][b, :k] = o["req"]; out["req_tag"][b, :k] = o["req_tag"]; out["req_mask"][b, :k] = True
        out["urgent"][b, :k] = o["urgent"]; out["edge_net"][b, :k, :n] = o["edge_net"]
        out["srv"][b, :n] = o["srv"]; out["srv_tag"][b, :n] = o["srv_tag"]; out["srv_aux"][b, :n] = o["srv_aux"]; out["srv_mask"][b, :n] = True
        out["free"][b, :n] = o["free"]; out["kappa"][b, :n] = o["kappa"]
        if r:
            out["run"][b, :r] = o["run"]; out["run_svc"][b, :r] = o["run_svc"]
            out["run_srv"][b, :r] = o["run_srv"]; out["run_mask"][b, :r] = True
        if m and use_history:
            out["hist"][b, :m] = o["hist"]; out["hist_svc"][b, :m] = o["hist_svc"]
            out["hist_srv"][b, :m] = o["hist_srv"]; out["hist_mask"][b, :m] = True
        out["glob"][b] = o["glob"]; out["idle_any"][b] = o["idle_any"]
    return {k: torch.from_numpy(v) for k, v in out.items()}, (Kp, Np)


class SchedulerModel(nn.Module):
    def __init__(self, d=64, layers=2, heads=4, ffn=128, n_tags=8, mix_hidden=16, C=10.0, d_max=3, K=8,
                 use_history=True, stop_bias=0.0, aux=False):
        super().__init__()
        self.d, self.C, self.d_max, self.K, self.use_history = d, C, d_max, K, use_history
        self.emb_req = nn.Linear(REQ_DIM, d)
        self.emb_srv = nn.Linear(SRV_DIM, d)
        self.emb_run = nn.Linear(RUN_DIM, d)
        self.emb_hist = nn.Linear(HIST_DIM, d)
        self.emb_glob = nn.Linear(GLOB_DIM, d)
        self.svc_tag = nn.Embedding(n_tags + 1, d, padding_idx=0)
        self.srv_tag = nn.Embedding(n_tags + 1, d, padding_idx=0)
        layer = nn.TransformerEncoderLayer(d, heads, ffn, dropout=0.0, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.W_q = nn.Linear(2 * d, d)
        self.W_k = nn.Linear(d, d)
        self.mix = nn.Sequential(nn.Linear(1 + N_EDGE_FEAT, mix_hidden), nn.ReLU(), nn.Linear(mix_hidden, 1))
        self.stop = nn.Sequential(nn.Linear(d + N_STOP_FEAT, d), nn.ReLU(), nn.Linear(d, 1))
        self.critic = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, 1))
        # optional auxiliary head (training only): predict each server's hidden log effective speed from its
        # encoded token, so the encoder learns to read speed evidence out of the history tokens
        self.aux = nn.Linear(d, 1) if aux else None
        self.last_aux_loss = None
        # initial bias of the STOP score: with stop_bias=0.5 an untrained model sends about one copy per request
        # (after the first edge STOP scores ~C*tanh(0.5)=4.6 against ~0 for edges); it must learn when more copies help
        with torch.no_grad():
            self.stop[-1].bias.fill_(stop_bias)

    # ---------------- encoder ----------------
    def encode(self, x):
        req = self.emb_req(x["req"]) + self.svc_tag(x["req_tag"])
        srv = self.emb_srv(x["srv"]) + self.srv_tag(x["srv_tag"])
        run = self.emb_run(x["run"]) + self.svc_tag(x["run_svc"]) + self.srv_tag(x["run_srv"])
        hist = self.emb_hist(x["hist"]) + self.svc_tag(x["hist_svc"]) + self.srv_tag(x["hist_srv"])
        glob = self.emb_glob(x["glob"]).unsqueeze(1)
        tokens = torch.cat([req, srv, run, hist, glob], 1)
        valid = torch.cat([x["req_mask"], x["srv_mask"], x["run_mask"], x["hist_mask"],
                           torch.ones_like(x["idle_any"]).unsqueeze(1)], 1)
        h = self.encoder(tokens, src_key_padding_mask=~valid)
        Kp, Np = req.shape[1], srv.shape[1]
        vf = valid.float().unsqueeze(-1)
        g = (h * vf).sum(1) / vf.sum(1)
        return h[:, :Kp], h[:, Kp:Kp + Np], g

    def match_scores(self, h_req, h_srv, g):
        q = self.W_q(torch.cat([h_req, g.unsqueeze(1).expand_as(h_req)], -1))
        k = self.W_k(h_srv)
        return torch.einsum("bkd,bnd->bkn", q, k) / math.sqrt(self.d)

    def edge_logits(self, s, f):
        """s: (..., K, N) match scores, f: (..., K, N, 4) edge features -> mixed scores."""
        z = self.mix(torch.cat([s.unsqueeze(-1), f], -1)).squeeze(-1)
        return self.C * torch.tanh(z)

    def stop_logits(self, g, fs):
        return self.C * torch.tanh(self.stop(torch.cat([g, fs], -1)).squeeze(-1))

    def value(self, g):
        return self.critic(g).squeeze(-1)

    # ---------------- decoding (rollout) ----------------
    @torch.no_grad()
    def decide(self, obs_list, greedy=False, max_steps=12, generator=None):
        x, (Kp, Np) = collate(obs_list, self.use_history)
        B = len(obs_list)
        h_req, h_srv, g = self.encode(x)
        s = self.match_scores(h_req, h_srv, g)
        v = self.value(g)
        free, kappa, d_max = x["free"], x["kappa"], self.d_max
        copies = torch.zeros(B, Kp)
        added = torch.zeros(B, Np)
        chosen = torch.zeros(B, Kp, Np, dtype=torch.bool)
        n_edges = torch.zeros(B)
        done = torch.zeros(B, dtype=torch.bool)
        logp = torch.zeros(B)
        ent = torch.zeros(B)
        nsteps = torch.zeros(B)
        steps = [[] for _ in range(B)]
        edges = [[] for _ in range(B)]
        base_valid = x["req_mask"].unsqueeze(2) & x["srv_mask"].unsqueeze(1)
        for l in range(max_steps):
            left = free - added
            valid = base_valid & ~chosen & (left > 0).unsqueeze(1) & (copies < d_max).unsqueeze(2)
            covered = copies > 0
            f = torch.stack([x["edge_net"], (copies / d_max).unsqueeze(2).expand(B, Kp, Np),
                             (added / kappa).unsqueeze(1).expand(B, Kp, Np),
                             (left / kappa).unsqueeze(1).expand(B, Kp, Np)], -1)
            fs = torch.stack([n_edges / self.K, covered.float().sum(1) / self.K,
                              (x["urgent"] & ~covered & x["req_mask"]).float().sum(1) / self.K,
                              left.clamp(min=0).sum(1) / kappa.sum(1)], -1)
            any_valid = valid.flatten(1).any(1)
            urgent_block = (x["urgent"] & ~covered & valid.any(2)).any(1)
            idle_block = x["idle_any"] & (n_edges == 0) & any_valid
            stop_ok = ~(urgent_block | idle_block) | ~any_valid
            if l == max_steps - 1:
                valid = torch.zeros_like(valid)
                stop_ok = torch.ones_like(stop_ok)
            le = self.edge_logits(s, f).masked_fill(~valid, -1e9)
            ls = self.stop_logits(g, fs).masked_fill(~stop_ok, -1e9)
            logits = torch.cat([le.flatten(1), ls.unsqueeze(1)], 1)
            logprobs = torch.log_softmax(logits, 1)
            if greedy:
                a = logits.argmax(1)
            else:
                a = torch.multinomial(logprobs.exp(), 1, generator=generator).squeeze(1)
            p = logprobs.exp()
            step_ent = -(p * logprobs).masked_fill(p == 0, 0).sum(1)
            for b in range(B):
                if done[b]:
                    continue
                k, n = len(obs_list[b]["window"]), len(obs_list[b]["free"])
                ab = int(a[b])
                is_stop = ab == Kp * Np
                pos, srv = (-1, -1) if is_stop else divmod(ab, Np)
                steps[b].append(dict(f=f[b, :k, :n].numpy().copy(), fs=fs[b].numpy().copy(),
                                     mask=valid[b, :k, :n].numpy().copy(), stop_ok=bool(stop_ok[b]),
                                     pos=pos, srv=srv))
                logp[b] += logprobs[b, ab]
                ent[b] += step_ent[b]
                nsteps[b] += 1
                if is_stop:
                    done[b] = True
                else:
                    edges[b].append((pos, srv))
                    chosen[b, pos, srv] = True
                    copies[b, pos] += 1
                    added[b, srv] += 1
                    n_edges[b] += 1
            if bool(done.all()):
                break
        return edges, steps, logp.numpy(), (ent / nsteps.clamp(min=1)).numpy(), v.numpy()

    # ---------------- PPO evaluation ----------------
    def evaluate(self, obs_list, steps_list, max_steps=12):
        """Recompute log-probabilities, entropies and values of stored decisions (with gradients)."""
        x, (Kp, Np) = collate(obs_list, self.use_history)
        dev = self.W_k.weight.device                 # the PPO update may run on a GPU; rollouts stay on the CPU
        x = {k: v.to(dev) for k, v in x.items()}
        B = len(obs_list)
        S = max(len(s) for s in steps_list)
        F = np.zeros((B, S, Kp, Np, N_EDGE_FEAT), np.float32)
        Me = np.zeros((B, S, Kp, Np), bool)
        Fs = np.zeros((B, S, N_STOP_FEAT), np.float32)
        Ms = np.zeros((B, S), bool)
        A = np.zeros((B, S), np.int64)
        V = np.zeros((B, S), bool)
        for b, st in enumerate(steps_list):
            for l, r in enumerate(st):
                k, n = r["mask"].shape
                F[b, l, :k, :n] = r["f"]; Me[b, l, :k, :n] = r["mask"]; Fs[b, l] = r["fs"]; Ms[b, l] = r["stop_ok"]
                A[b, l] = Kp * Np if r["pos"] < 0 else r["pos"] * Np + r["srv"]
                V[b, l] = True
        F, Me, Fs, Ms, A, V = (torch.from_numpy(a).to(dev) for a in (F, Me, Fs, Ms, A, V))
        h_req, h_srv, g = self.encode(x)
        s = self.match_scores(h_req, h_srv, g)
        le = self.edge_logits(s.unsqueeze(1).expand(B, S, Kp, Np), F).masked_fill(~Me, -1e9)
        ls = self.stop_logits(g.unsqueeze(1).expand(B, S, g.shape[-1]), Fs).masked_fill(~Ms, -1e9)
        logits = torch.cat([le.flatten(2), ls.unsqueeze(2)], 2)
        logprobs = torch.log_softmax(logits, 2)
        lp = logprobs.gather(2, A.unsqueeze(2)).squeeze(2)
        logp = (lp * V).sum(1)
        p = logprobs.exp()
        ent_step = -(p * logprobs).masked_fill(p == 0, 0).sum(2)
        ent = (ent_step * V).sum(1) / V.sum(1).clamp(min=1)
        if self.aux is not None:
            pred = self.aux(h_srv).squeeze(-1)
            m = x["srv_mask"].float()
            self.last_aux_loss = (((pred - x["srv_aux"]) ** 2) * m).sum() / m.sum()
        return logp, ent, self.value(g)
