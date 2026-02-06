import numpy as np
import matplotlib.pyplot as plt

def simulate(
    n_lineages=200, T=2000,
    sigma=1.0,
    p_shock=0.01, shock_scale=8.0, base_scale=0.2,
    corr_noise=0.0,         # 0に近いほど相関が強い
    lam=0.5, a=1.0,
    C_mean=200.0, C_sd=50.0,
    w_thresh=0.05, L=20,
    seed=0
):
    rng = np.random.default_rng(seed)

    # Lineage state
    x = rng.normal(0, 1, size=n_lineages)
    C = np.clip(rng.normal(C_mean, C_sd, size=n_lineages), 1e-6, None)
    B = C.copy()

    e = 0.0
    alive = np.ones(n_lineages, dtype=bool)
    low_w_run = np.zeros(n_lineages, dtype=int)

    extinct_counts = np.zeros(T, dtype=int)

    for t in range(T):
        # Environment step (mixture: small vs shock)
        if rng.random() < p_shock:
            eps = rng.normal(0, shock_scale)
            shock_flag = 1
        else:
            eps = rng.normal(0, base_scale)
            shock_flag = 0
        e = e + eps

        # lineage-specific perceived environment (tunes correlation)
        e_i = e + rng.normal(0, corr_noise, size=n_lineages)

        # mismatch
        delta = np.abs(x - e_i)

        # survival (fitness as probability)
        w = np.exp(-(delta**2) / (2 * sigma**2))

        # extinction bookkeeping (only for alive)
        low = (w < w_thresh) & alive
        low_w_run[low] += 1
        low_w_run[(~low)] = 0

        # adaptation step only if budget remains and alive
        # step proportional to mismatch, but cannot exceed what budget allows
        u_desired = lam * delta
        u_budget  = np.where(B > 0, B / max(a, 1e-12), 0.0)  # max step allowed by remaining budget
        u = np.minimum(u_desired, u_budget)

        # apply only to alive
        u = u * alive

        # cost and budget update
        D = a * u
        B = B - D

        # update x toward environment
        direction = np.sign(e_i - x)
        x = x + direction * u

        # extinction rule: (A) prolonged low fitness OR (B) zero budget and still poor fitness
        newly_extinct = alive & ((low_w_run >= L) | ((B <= 0) & (w < w_thresh)))
        alive[newly_extinct] = False
        extinct_counts[t] = newly_extinct.sum()

    return {
        "extinct_counts": extinct_counts,
        "alive_final": alive,
        "survival_fraction": alive.mean(),
    }

def run_grid():
    # Example: vary correlation and shock magnitude
    results = []
    for corr_noise in [0.0, 0.5, 1.0, 2.0]:
        out = simulate(corr_noise=corr_noise, seed=1)
        results.append((corr_noise, out["survival_fraction"]))
    return results

if __name__ == "__main__":
    print(run_grid())
