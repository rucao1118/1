"""
v4 model.

The v3 architecture was right about the big thing -- the decoder needs to see
edges and DP state, not just node embeddings -- so v4 does not rearrange it.
It changes three things and leaves everything else alone so the two are
comparable and so a v3 checkpoint can be warm-started into v4 EXACTLY.

1.  CAND_DIM 10 -> 13.  The three new channels are the exact DP lookahead
    (fstsp_dp_v4.lookahead_feats): the true marginal dp[t+1] - dp[t] of
    appending each candidate, whether the operation that would close there
    flies somebody, and the gap to the best legal candidate.  v3's channels
    5-7 were a hand-rolled approximation of one term of this; the DP can just
    be asked.

2.  Optional pre-norm encoder (`prenorm=1`).  Post-norm is fine at 4 layers
    and gets progressively worse past 6, and the n=20 logs say the model is
    UNDER-trained, not over-capacity, so depth is the cheap knob to turn.

3.  `inflate_v3_state_dict`.  Loading a v3 checkpoint pads the candidate MLP's
    first layer with ZERO columns for the three new inputs.  The v4 model is
    therefore not merely "initialised near" the v3 model, it computes the
    identical function on the first step, and every gradient it takes from
    there is a strict improvement search rather than a re-learning of what
    the n=20 run already paid for.  verify_v4.py check 3 asserts the greedy
    orders match bit-for-bit.

n_dec > 1 (decoder population, best-of-population loss) is carried over
unchanged, including clone_population.
"""

import math

import torch
import torch.nn as nn

from .dp import DP_DIM

NEG_INF = -1.0e9
CAND_DIM_V3 = 10
CAND_DIM = 13           # = v3's 10 + exact-DP marginal, flies flag, regret
N_NEW_CAND = CAND_DIM - CAND_DIM_V3


class EdgeMHALayer(nn.Module):
    """Post-norm by default (v3-identical); pre-norm when prenorm=True."""

    def __init__(self, dim=128, heads=8, ff=512, edge_dim=5,
                 use_edge_bias=True, prenorm=False):
        super().__init__()
        assert dim % heads == 0
        self.dim, self.heads, self.dh = dim, heads, dim // heads
        self.use_edge_bias = use_edge_bias
        self.prenorm = bool(prenorm)

        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out = nn.Linear(dim, dim)
        self.edge_proj = nn.Linear(edge_dim, heads, bias=False) if use_edge_bias else None
        self.n1 = nn.LayerNorm(dim)
        self.n2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, ff), nn.ReLU(), nn.Linear(ff, dim))

    def _attn(self, x, edge_feat):
        B, N, D = x.shape
        qkv = self.qkv(x).view(B, N, 3, self.heads, self.dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.dh)
        if self.use_edge_bias and edge_feat is not None:
            scores = scores + self.edge_proj(edge_feat).permute(0, 3, 1, 2)
        h = torch.matmul(torch.softmax(scores, dim=-1), v)
        return self.out(h.transpose(1, 2).contiguous().view(B, N, D))

    def forward(self, x, edge_feat=None):
        if self.prenorm:
            x = x + self._attn(self.n1(x), edge_feat)
            return x + self.ff(self.n2(x))
        x = self.n1(x + self._attn(x, edge_feat))
        return self.n2(x + self.ff(x))


class Encoder(nn.Module):
    def __init__(self, in_dim=12, edge_dim=5, dim=128, heads=8, layers=4,
                 ff=512, use_edge_bias=True, prenorm=False):
        super().__init__()
        self.inp = nn.Linear(in_dim, dim)
        self.elig_emb = nn.Embedding(2, dim)
        self.prenorm = bool(prenorm)
        self.layers = nn.ModuleList([
            EdgeMHALayer(dim, heads, ff, edge_dim, use_edge_bias, prenorm)
            for _ in range(layers)])
        self.nf = nn.LayerNorm(dim) if prenorm else nn.Identity()

    def forward(self, feat, edge_feat=None, elig=None):
        h = self.inp(feat)
        if elig is not None:
            h = h + self.elig_emb(elig.long())
        for lay in self.layers:
            h = lay(h, edge_feat=edge_feat)
        return self.nf(h)


class AMDecoder(nn.Module):
    """Context -> MHA glimpse -> dot-product compatibility + candidate bias."""

    def __init__(self, dim=128, heads=8, cand_dim=CAND_DIM, dp_dim=DP_DIM,
                 cand_hidden=32, clip=10.0, use_cand_bias=True):
        super().__init__()
        self.dim, self.heads, self.dh = dim, heads, dim // heads
        self.clip = clip

        self.ctx = nn.Linear(dim * 4, dim, bias=False)
        self.dp_proj = nn.Sequential(
            nn.Linear(dp_dim, dim // 2), nn.ReLU(), nn.Linear(dim // 2, dim))

        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)
        self.wk2 = nn.Linear(dim, dim, bias=False)

        self.cand = nn.Sequential(
            nn.Linear(cand_dim, cand_hidden), nn.ReLU(),
            nn.Linear(cand_hidden, 1)) if use_cand_bias else None

    def precompute(self, node_emb):
        M, V, D = node_emb.shape
        H, dh = self.heads, self.dh
        kh = self.wk(node_emb).view(M, V, H, dh).permute(0, 2, 1, 3).contiguous()
        vh = self.wv(node_emb).view(M, V, H, dh).permute(0, 2, 1, 3).contiguous()
        return kh, vh, self.wk2(node_emb)

    def forward(self, node_emb, graph_emb, last_emb, first_emb, remaining_emb,
                dp_feats, cand_feats, selectable_mask, temperature=1.0,
                cache=None):
        M, V, D = node_emb.shape
        H, dh = self.heads, self.dh
        kh, vh, k2 = self.precompute(node_emb) if cache is None else cache

        q = self.ctx(torch.cat([graph_emb, last_emb, first_emb, remaining_emb], -1))
        q = q + self.dp_proj(dp_feats)
        qh = self.wq(q).view(M, H, 1, dh)

        sc = torch.matmul(qh, kh.transpose(-1, -2)) / math.sqrt(dh)
        sc = sc.masked_fill(~selectable_mask[:, None, None, :], NEG_INF)
        g = torch.matmul(torch.softmax(sc, dim=-1), vh)
        g = self.wo(g.permute(0, 2, 1, 3).reshape(M, D))

        logits = (k2 * g[:, None, :]).sum(-1) / math.sqrt(D)
        if self.cand is not None and cand_feats is not None:
            logits = logits + self.cand(cand_feats).squeeze(-1)

        logits = self.clip * torch.tanh(logits)
        logits = logits / max(float(temperature), 1e-6)
        logits = logits.masked_fill(~selectable_mask, NEG_INF)
        return torch.log_softmax(logits, dim=-1)


class FSTSPv4(nn.Module):
    def __init__(self, in_dim=12, edge_dim=5, dim=128, heads=8, layers=4,
                 ff=512, use_edge_bias=True, use_cand_bias=True,
                 use_dp_state=True, n_dec=1, cand_dim=CAND_DIM, dp_dim=DP_DIM,
                 prenorm=False):
        super().__init__()
        self.dim = dim
        self.use_edge_bias = use_edge_bias
        self.use_cand_bias = use_cand_bias
        self.use_dp_state = use_dp_state
        self.n_dec = int(n_dec)
        self.dp_dim = dp_dim
        self.cand_dim = int(cand_dim)
        self.prenorm = bool(prenorm)

        self.encoder = Encoder(in_dim, edge_dim, dim, heads, layers, ff,
                               use_edge_bias, prenorm)
        self.decoders = nn.ModuleList([
            AMDecoder(dim, heads, cand_dim, dp_dim, use_cand_bias=use_cand_bias)
            for _ in range(self.n_dec)])
        self.drone_head = nn.Sequential(
            nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, 1))

    # ---- population helpers ----------------------------------------------
    @property
    def decoder(self):
        return self.decoders[0]

    def clone_population(self, noise=0.02, generator=None):
        src = self.decoders[0].state_dict()
        for d in range(1, self.n_dec):
            self.decoders[d].load_state_dict(src)
            with torch.no_grad():
                for p in self.decoders[d].parameters():
                    p.mul_(1.0 + noise * torch.randn_like(p))

    # ---- forward ----------------------------------------------------------
    def encode(self, feat, edge_feat=None, elig=None):
        node_emb = self.encoder(
            feat, edge_feat=edge_feat if self.use_edge_bias else None, elig=elig)
        return node_emb, node_emb.mean(dim=1)

    def precompute_decoder(self, node_emb, dec=0):
        return self.decoders[dec].precompute(node_emb)

    def decode_step(self, node_emb, graph_emb, last_emb, first_emb,
                    remaining_emb, dp_feats, cand_feats, selectable_mask,
                    temperature=1.0, cache=None, dec=0):
        if not self.use_dp_state:
            dp_feats = torch.zeros_like(dp_feats)
        if not self.use_cand_bias:
            cand_feats = None
        return self.decoders[dec](node_emb, graph_emb, last_emb, first_emb,
                                  remaining_emb, dp_feats, cand_feats,
                                  selectable_mask, temperature, cache=cache)

    def drone_logits(self, node_emb):
        return self.drone_head(node_emb).squeeze(-1)


# ---------------------------------------------------------------------------
# warm start
# ---------------------------------------------------------------------------
def inflate_v3_state_dict(sd, cand_dim=CAND_DIM, n_dec=1, noise=0.0):
    """
    Turn a v3 (or v4-with-fewer-channels) state dict into one this model can
    load, by ZERO-padding the candidate MLP's input layer.

    `cand.0.weight` is [cand_hidden, old_cand_dim].  The new channels are
    appended at the END of the feature vector (see
    fstsp_rollout_v4.candidate_feats_v4), so the pad goes on the right and the
    inflated network computes exactly the old function until training moves
    those columns off zero.

    Also replicates `decoders.0.*` into decoders 1..n_dec-1 when the source is
    a single-decoder checkpoint, so --init_from can go straight to a
    population.
    """
    out = {}
    for k, v in sd.items():
        if k.endswith("cand.0.weight") and v.dim() == 2 and v.shape[1] < cand_dim:
            pad = torch.zeros(v.shape[0], cand_dim - v.shape[1],
                              dtype=v.dtype, device=v.device)
            if noise:
                pad.normal_(0.0, float(noise))
            v = torch.cat([v, pad], dim=1)
        out[k] = v

    have = {int(k.split(".")[1]) for k in out if k.startswith("decoders.")}
    if have and max(have) == 0 and n_dec > 1:
        base = {k: v for k, v in out.items() if k.startswith("decoders.0.")}
        for d in range(1, int(n_dec)):
            for k, v in base.items():
                out[k.replace("decoders.0.", f"decoders.{d}.", 1)] = v.clone()
    return out


def load_v3_checkpoint(model, path, map_location="cpu", noise=0.0, strict=False):
    """Load runs/n20/best.pt into an FSTSPv4.  Returns the load report."""
    ck = torch.load(path, map_location=map_location, weights_only=False)
    sd = inflate_v3_state_dict(ck["model"], cand_dim=model.cand_dim,
                               n_dec=model.n_dec, noise=noise)
    return model.load_state_dict(sd, strict=strict), ck
