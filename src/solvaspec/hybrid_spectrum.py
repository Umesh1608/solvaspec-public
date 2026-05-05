"""HybridSpectrum: Chemprop D-MPNN solute+solvent + PaiNN-SCA per-state Δε/Δf corrections.

Architecture:
    Solute  → Chemprop BondMessagePassing → solute_embed (d_solute)
    Solvent → Chemprop BondMessagePassing → solvent_embed (d_solvent)

    For each state k in 0..49:
        state_token_k = concat([E_k, f_k, log(f_k + 1e-3), positional_k,
                                solute_embed, solvent_embed])
        → MLP → (Δε_k, Δf_k)

    Outputs:
        E_k_sol = E_k + Δε_k                        (50 solvent-corrected energies)
        f_k_sol = softplus(f_k + Δf_k)              (50 solvent-corrected oscillators ≥ 0)
        spectrum_sol(λ) = Σ_k f_k_sol · Gauss(λ - 1240/E_k_sol, σ)
        λmax_sol = soft-argmax(spectrum_sol)

Trained with MAE(λmax_sol, λmax_true) + 0.01·||Δε||² + 0.01·||Δf||².
"""
import torch
import torch.nn as nn

HC_EV_NM = 1239.84


class HybridSpectrum(nn.Module):
    def __init__(self, solute_encoder, solvent_encoder, d_solute, d_solvent,
                 n_states=50, d_state=64, d_fusion=128, dropout=0.1,
                 sigma_nm=15.0, soft_beta=30.0,
                 lambda_min=200.0, lambda_max=500.0, n_grid=301):
        super().__init__()
        self.solute_encoder = solute_encoder
        self.solvent_encoder = solvent_encoder
        self.n_states = n_states
        self.sigma_nm = sigma_nm
        self.soft_beta = soft_beta
        self.register_buffer("grid", torch.linspace(lambda_min, lambda_max, n_grid))

        # Per-state positional embedding
        self.state_positional = nn.Embedding(n_states, d_state)
        # Per-state feature projection: (E, f, log(f+eps)) → d_state
        self.state_proj = nn.Linear(3, d_state)

        # Fusion MLP per state
        fuse_dim = d_state + d_solute + d_solvent
        self.fuse = nn.Sequential(
            nn.Linear(fuse_dim, d_fusion),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_fusion, d_fusion),
            nn.SiLU(),
        )
        self.delta_e_head = nn.Linear(d_fusion, 1)
        self.delta_f_head = nn.Linear(d_fusion, 1)

    def forward(self, E_gas, f_gas, solute_batch, solvent_batch):
        """
        Args:
            E_gas: [B, S] gas-phase excitation energies (eV)
            f_gas: [B, S] gas-phase oscillator strengths
            solute_batch, solvent_batch: pre-encoded graph batches for Chemprop encoders
        Returns:
            dict with λmax, spectrum, E_sol, f_sol, delta_e, delta_f
        """
        B, S = E_gas.shape
        device = E_gas.device

        # Encode solute + solvent via Chemprop D-MPNN (already aggregated per molecule)
        solute_embed = self.solute_encoder(solute_batch)    # [B, d_solute]
        solvent_embed = self.solvent_encoder(solvent_batch) # [B, d_solvent]

        # Build per-state tokens
        E_clip = torch.clamp(E_gas, min=0.3, max=12.0)
        f_clip = torch.clamp(f_gas, min=0.0, max=10.0)
        log_f = torch.log(f_clip + 1e-3)
        state_feats = torch.stack([E_clip, f_clip, log_f], dim=-1)  # [B, S, 3]
        state_tokens = self.state_proj(state_feats)
        pos_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)
        state_tokens = state_tokens + self.state_positional(pos_ids)   # [B, S, d_state]

        # Broadcast solute/solvent embeddings to each state
        sol_rep = solute_embed.unsqueeze(1).expand(-1, S, -1)   # [B, S, d_solute]
        slv_rep = solvent_embed.unsqueeze(1).expand(-1, S, -1)  # [B, S, d_solvent]
        fused = torch.cat([state_tokens, sol_rep, slv_rep], dim=-1)  # [B, S, fuse_dim]
        h = self.fuse(fused)

        delta_e = self.delta_e_head(h).squeeze(-1)  # [B, S] — correction in eV
        delta_f = self.delta_f_head(h).squeeze(-1)  # [B, S] — correction in f

        E_sol = E_clip + delta_e
        f_sol = torch.nn.functional.softplus(f_clip + delta_f)  # ≥0

        # Build broadened spectrum
        E_sol_clip = torch.clamp(E_sol, min=0.3, max=12.0)
        sticks = HC_EV_NM / E_sol_clip                          # [B, S]
        grid = self.grid.to(device)                              # [L]
        dev = grid.unsqueeze(0).unsqueeze(0) - sticks.unsqueeze(-1)
        gauss = torch.exp(-0.5 * (dev / self.sigma_nm) ** 2)
        spectrum = (f_sol.unsqueeze(-1) * gauss).sum(dim=1)     # [B, L]

        # Soft-argmax λmax
        w = torch.softmax(self.soft_beta * spectrum, dim=-1)
        lam_max = (w * grid.unsqueeze(0)).sum(dim=-1)

        return {
            "lambda_max": lam_max,
            "spectrum": spectrum,
            "E_sol": E_sol,
            "f_sol": f_sol,
            "delta_e": delta_e,
            "delta_f": delta_f,
            "solute_embed": solute_embed,
            "solvent_embed": solvent_embed,
        }
