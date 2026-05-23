"""Multi-scale BATCH estimator for temporal RUS.

Trains a single set of lag-conditioned neural network backbones that predict
temporal RUS at every lag tau in {0, 1, ..., K-1} simultaneously, instead of
re-training the per-lag BATCH estimator K times. This follows Algorithms 1-3
in Appendix C of the MERGE paper.

Key components:
  - Shared encoders g_{1, theta}, g_{2, theta}, g_{12, theta} mapping inputs to
    a hidden representation that is reused across every lag.
  - A learnable lag embedding e(tau) fused with the encoded features.
  - Lag-conditioned discriminators D_{1, theta}, D_{2, theta}, D_{12, theta}
    that estimate P(Y | X_1, tau), P(Y | X_2, tau), P(Y | X_1, X_2, tau).
  - A lag-conditioned alignment module producing the alignment tensor that is
    Sinkhorn-normalized into the optimal coupling Q*_tau.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from estimators.ce_alignment_information import mlp, sinkhorn_probs


class LagEmbedding(nn.Module):
    """Learnable embedding table over discrete lag indices."""

    def __init__(self, num_lags, embed_dim):
        super().__init__()
        self.embed = nn.Embedding(num_lags, embed_dim)

    def forward(self, tau):
        return self.embed(tau)


def _fuse(features, lag_emb):
    """Fusion operator phi: concatenate encoded features with lag embedding."""
    return torch.cat([features, lag_emb], dim=-1)


class MultiScaleDiscrim(nn.Module):
    """Lag-conditioned unimodal discriminator with a shared encoder backbone."""

    def __init__(self, x_dim, hidden_dim, num_labels, layers, activation,
                 lag_embed_dim, num_lags):
        super().__init__()
        self.encoder = mlp(x_dim, hidden_dim, hidden_dim, layers, activation)
        self.lag_embed = LagEmbedding(num_lags, lag_embed_dim)
        self.head = mlp(hidden_dim + lag_embed_dim, hidden_dim, num_labels,
                        layers, activation)

    def forward(self, x, tau):
        h = self.encoder(x)
        e = self.lag_embed(tau)
        return self.head(_fuse(h, e))


class MultiScaleDiscrim12(nn.Module):
    """Lag-conditioned bimodal discriminator with a shared joint encoder."""

    def __init__(self, x1_dim, x2_dim, hidden_dim, num_labels, layers,
                 activation, lag_embed_dim, num_lags):
        super().__init__()
        self.encoder = mlp(x1_dim + x2_dim, hidden_dim, hidden_dim, layers,
                           activation)
        self.lag_embed = LagEmbedding(num_lags, lag_embed_dim)
        self.head = mlp(hidden_dim + lag_embed_dim, hidden_dim, num_labels,
                        layers, activation)

    def forward(self, x1, x2, tau):
        h = self.encoder(torch.cat([x1, x2], dim=-1))
        e = self.lag_embed(tau)
        return self.head(_fuse(h, e))


class MultiScaleCEAlignment(nn.Module):
    """Lag-conditioned alignment module producing the Sinkhorn-normalized Q.

    Implements Eq. (8)-(9) in the paper: per-class embeddings q_{X_m}^(i,k,tau)
    are produced by a shared encoder followed by a head fed the fused lag
    embedding. The alignment tensor align_tau[i, j, k] is built via an einsum
    and normalized in parallel across the batch dimensions using Sinkhorn-Knopp
    to enforce marginal-matching constraints.
    """

    def __init__(self, x1_dim, x2_dim, hidden_dim, embed_dim, num_labels,
                 layers, activation, lag_embed_dim, num_lags):
        super().__init__()
        self.num_labels = num_labels
        self.embed_dim = embed_dim
        self.encoder1 = mlp(x1_dim, hidden_dim, hidden_dim, layers, activation)
        self.encoder2 = mlp(x2_dim, hidden_dim, hidden_dim, layers, activation)
        self.lag_embed = LagEmbedding(num_lags, lag_embed_dim)
        self.head1 = mlp(hidden_dim + lag_embed_dim, hidden_dim,
                         embed_dim * num_labels, layers, activation)
        self.head2 = mlp(hidden_dim + lag_embed_dim, hidden_dim,
                         embed_dim * num_labels, layers, activation)

    def _per_class_embeddings(self, x, tau, encoder, head):
        h = encoder(x)
        e = self.lag_embed(tau)
        q = head(_fuse(h, e)).unflatten(1, (self.num_labels, -1))
        # Per-class layer norm (matches CEAlignment in ce_alignment_information).
        q = (q - torch.mean(q, dim=2, keepdim=True)) / torch.sqrt(
            torch.var(q, dim=2, keepdim=True) + 1e-8)
        return q

    def forward(self, x1, x2, tau, x1_probs, x2_probs):
        q_x1 = self._per_class_embeddings(x1, tau, self.encoder1, self.head1)
        q_x2 = self._per_class_embeddings(x2, tau, self.encoder2, self.head2)

        align = torch.einsum('ahx, bhx -> abh', q_x1, q_x2) / math.sqrt(
            q_x1.size(-1))
        # Clamp before exp to prevent float32 overflow (overflows at >~88).
        align = torch.exp(torch.clamp(align, max=20.0))

        normalized = []
        for k in range(align.size(-1)):
            current = align[..., k]
            # Balance marginal masses so Sinkhorn can converge: a minibatch has
            # sum_i P(Y=k|X1_i) ≠ sum_j P(Y=k|X2_j) by chance; rescale x2 to
            # match x1's total mass before running the algorithm.
            m1 = x1_probs[:, k].sum()
            m2 = x2_probs[:, k].sum()
            x2_k = x2_probs[:, k] * (m1 / (m2 + 1e-8))
            for _ in range(500):
                current, stop = sinkhorn_probs(current, x1_probs[:, k], x2_k)
                if stop:
                    break
            normalized.append(current)
        return torch.stack(normalized, dim=-1)


class MultiLagDataset:
    """Pre-aligned per-lag tensor storage with cheap random batch sampling.

    For each lag tau, holds (X1[:-tau], X2[:-tau], Y[tau:]) so that sampling a
    minibatch for any lag is a tensor index into a contiguous buffer.
    """

    def __init__(self, X1, X2, Y, max_lag, device):
        if not torch.is_tensor(X1):
            X1 = torch.as_tensor(X1, dtype=torch.float32)
        if not torch.is_tensor(X2):
            X2 = torch.as_tensor(X2, dtype=torch.float32)
        if not torch.is_tensor(Y):
            Y = torch.as_tensor(Y, dtype=torch.long)

        self.max_lag = max_lag
        self.device = device
        self.datasets = {}
        for tau in range(max_lag + 1):
            if tau == 0:
                x1, x2, y = X1, X2, Y
            else:
                x1, x2, y = X1[:-tau], X2[:-tau], Y[tau:]
            self.datasets[tau] = (
                x1.to(device), x2.to(device), y.to(device)
            )

    def size(self, tau):
        return len(self.datasets[tau][2])

    def sample(self, tau, batch_size, generator=None):
        x1, x2, y = self.datasets[tau]
        n = len(y)
        idx = torch.randint(0, n, (min(batch_size, n),), device=self.device,
                            generator=generator)
        return x1[idx], x2[idx], y[idx]

    def iter_batches(self, tau, batch_size):
        x1, x2, y = self.datasets[tau]
        n = len(y)
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            yield x1[start:end], x2[start:end], y[start:end]


def _make_tau_tensor(tau, batch_len, device):
    return torch.full((batch_len,), tau, dtype=torch.long, device=device)


def train_multiscale_discriminators(discrim1, discrim2, discrim12, data,
                                    num_epochs, steps_per_epoch, batch_size,
                                    lr, device, lag_weights=None):
    """Phase 1: jointly train the three lag-conditioned discriminators.

    Implements Algorithm 1: at each step a lag is sampled from a categorical
    distribution over {0, ..., K-1} (uniform by default), a minibatch is drawn
    from D_tau, and all three discriminators are updated via a single backward
    pass on the summed cross-entropy loss.
    """
    discrim1.train(); discrim2.train(); discrim12.train()
    params = list(discrim1.parameters()) + list(discrim2.parameters()) + \
        list(discrim12.parameters())
    optimizer = optim.Adam(params, lr=lr)
    criterion = nn.CrossEntropyLoss()

    max_lag = data.max_lag
    num_lags = max_lag + 1
    if lag_weights is None:
        lag_weights = torch.ones(num_lags) / num_lags
    else:
        lag_weights = torch.as_tensor(lag_weights, dtype=torch.float32)
        lag_weights = lag_weights / lag_weights.sum()

    for _ in tqdm(range(num_epochs), desc="Phase 1: Discriminators"):
        for _step in range(steps_per_epoch):
            tau = int(torch.multinomial(lag_weights, 1).item())
            x1, x2, y = data.sample(tau, batch_size)
            tau_t = _make_tau_tensor(tau, len(y), device)

            optimizer.zero_grad()
            loss = (criterion(discrim1(x1, tau_t), y)
                    + criterion(discrim2(x2, tau_t), y)
                    + criterion(discrim12(x1, x2, tau_t), y))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()


def _alignment_loss(align, p_y_x1):
    """Cross-entropy alignment loss used to learn Q minimizing I(X1; X2 | Y)."""
    q_x2_x1y = align / (torch.sum(align, dim=1, keepdim=True) + 1e-8)
    log_term = (torch.log(q_x2_x1y + 1e-8)
                - torch.log(torch.einsum('aby, ay -> ab', q_x2_x1y, p_y_x1)
                            + 1e-8)[:, :, None])
    return torch.mean(torch.sum(torch.sum(
        p_y_x1[:, None, :] * q_x2_x1y * log_term, dim=-1), dim=-1)), q_x2_x1y, log_term


def train_multiscale_alignment(align_model, discrim1, discrim2, data,
                               num_epochs, steps_per_epoch, batch_size, lr,
                               device, lag_weights=None):
    """Phase 2: train the shared alignment module across all lags.

    Implements Algorithm 2: discriminators are frozen, the alignment module is
    optimized so that the Sinkhorn-normalized coupling Q*_tau eliminates the
    synergistic information between X_1 and X_2 conditional on Y for every
    sampled lag.
    """
    discrim1.eval(); discrim2.eval()
    align_model.train()
    optimizer = optim.Adam(align_model.parameters(), lr=lr)

    max_lag = data.max_lag
    num_lags = max_lag + 1
    if lag_weights is None:
        lag_weights = torch.ones(num_lags) / num_lags
    else:
        lag_weights = torch.as_tensor(lag_weights, dtype=torch.float32)
        lag_weights = lag_weights / lag_weights.sum()

    for _ in tqdm(range(num_epochs), desc="Phase 2: Alignment"):
        for _step in range(steps_per_epoch):
            tau = int(torch.multinomial(lag_weights, 1).item())
            x1, x2, y = data.sample(tau, batch_size)
            tau_t = _make_tau_tensor(tau, len(y), device)

            with torch.no_grad():
                p_y_x1 = F.softmax(discrim1(x1, tau_t), dim=-1)
                p_y_x2 = F.softmax(discrim2(x2, tau_t), dim=-1)

            optimizer.zero_grad()
            align = align_model(x1, x2, tau_t, p_y_x1, p_y_x2)
            loss, _, _ = _alignment_loss(align, p_y_x1)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(align_model.parameters(), max_norm=1.0)
            optimizer.step()


@torch.no_grad()
def _estimate_mi_unimodal(discrim, x_data, tau, p_y, device, batch_size,
                          x_partner=None):
    """E_x KL(P(Y|X) || P(Y)) -- mutual information in nats."""
    n = len(x_data)
    total = 0.0
    count = 0
    log_p_y = torch.log(p_y + 1e-8)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        x = x_data[start:end].to(device)
        tau_t = _make_tau_tensor(tau, end - start, device)
        if x_partner is None:
            logits = discrim(x, tau_t)
        else:
            xp = x_partner[start:end].to(device)
            logits = discrim(x, xp, tau_t)
        p_y_x = F.softmax(logits, dim=-1)
        mi = torch.sum(
            p_y_x * (torch.log(p_y_x + 1e-8) - log_p_y[None]), dim=-1)
        total += float(mi.sum().item())
        count += int(mi.numel())
    return total / max(count, 1)


@torch.no_grad()
def _estimate_mi_q(align_model, discrim1, discrim2, data, tau, p_y, device,
                   batch_size):
    """E_{x1, x2} sum_y P(y|x1) Q(x2|x1, y) [log Q(x2|x1, y) - log Q(x2|x1) +
    log P(y|x1) - log P(y)] -- I_{Q*_tau}(Y; X_1, X_2) in nats."""
    x1_all, x2_all, _ = data.datasets[tau]
    n = len(x1_all)
    total = 0.0
    count = 0
    log_p_y = torch.log(p_y + 1e-8)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        x1 = x1_all[start:end]
        x2 = x2_all[start:end]
        tau_t = _make_tau_tensor(tau, end - start, device)
        p_y_x1 = F.softmax(discrim1(x1, tau_t), dim=-1)
        p_y_x2 = F.softmax(discrim2(x2, tau_t), dim=-1)
        align = align_model(x1, x2, tau_t, p_y_x1, p_y_x2)
        _, q_x2_x1y, log_term = _alignment_loss(align, p_y_x1)
        mi_q = p_y_x1[:, None, :] * q_x2_x1y * (
            log_term + torch.log(p_y_x1 + 1e-8)[:, None, :] - log_p_y[None, None, :]
        )
        mi_q = torch.sum(torch.sum(mi_q, dim=-1), dim=-1)
        total += float(mi_q.sum().item())
        count += int(mi_q.numel())
    return total / max(count, 1)


def decompose_multilag_rus(align_model, discrim1, discrim2, discrim12, data,
                           p_y, device, batch_size):
    """Phase 3: read out R, U1, U2, S for every lag from the trained model.

    Implements Algorithm 3: for each lag tau the three discriminator-based
    mutual informations and the alignment-based I_{Q*_tau}(Y; X_1, X_2) are
    estimated on D_tau and combined into the PID components.
    """
    discrim1.eval(); discrim2.eval(); discrim12.eval(); align_model.eval()
    results = []
    log2 = math.log(2)
    for tau in tqdm(range(data.max_lag + 1), desc="Phase 3: RUS readout"):
        x1, x2, _ = data.datasets[tau]
        mi_y_x1 = _estimate_mi_unimodal(discrim1, x1, tau, p_y, device,
                                        batch_size)
        mi_y_x2 = _estimate_mi_unimodal(discrim2, x2, tau, p_y, device,
                                        batch_size)
        mi_y_x1x2 = _estimate_mi_unimodal(discrim12, x1, tau, p_y, device,
                                          batch_size, x_partner=x2)
        mi_q = _estimate_mi_q(align_model, discrim1, discrim2, data, tau, p_y,
                              device, batch_size)
        # Convert nats -> bits.
        mi_y_x1 /= log2
        mi_y_x2 /= log2
        mi_y_x1x2 /= log2
        mi_q /= log2

        redundancy = max(0.0, mi_y_x1 + mi_y_x2 - mi_q)
        unique_x1 = max(0.0, mi_q - mi_y_x2)
        unique_x2 = max(0.0, mi_q - mi_y_x1)
        synergy = max(0.0, mi_y_x1x2 - mi_q)
        results.append({
            'lag': tau,
            'redundancy': redundancy,
            'unique_x1': unique_x1,
            'unique_x2': unique_x2,
            'synergy': synergy,
            'total_di': mi_y_x1x2,
            'sum_components': redundancy + unique_x1 + unique_x2 + synergy,
        })
    return results
