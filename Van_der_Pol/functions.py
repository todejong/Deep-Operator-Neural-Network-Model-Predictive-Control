import numpy as np  # Add this import statement
import casadi as ca

# Crest factor calculation
crest_factor = lambda uk: np.max(np.abs(uk)) / np.sqrt(np.mean(uk**2))

# Duplicate function
duplicate = lambda uk, n: np.concatenate([uk] * n)


# Multisine signal generator
def multisine(
    N_points_per_period,
    N_periods=1,
    pmin=1,
    pmax=21,
    prule=lambda p: p % 2 == 1 and p % 6 != 1,
    par=None,
    n_crest_factor_optim=1,
    seed=None,
):
    """A multi-sine generator with only odd frequencies and random phases."""

    if isinstance(seed, int) or seed is None:
        rng = np.random.RandomState(seed)
    else:
        rng = seed
    assert isinstance(rng, np.random.mtrand.RandomState)

    assert pmax < N_points_per_period // 2

    # Crest factor optimization
    if n_crest_factor_optim > 1:
        ybest = None
        crest_best = float("inf")
        for i in range(n_crest_factor_optim):
            seedi = None if seed is None else seed + i
            uk = multisine(
                N_points_per_period,
                N_periods=1,
                pmax=pmax,
                pmin=pmin,
                prule=prule,
                n_crest_factor_optim=1,
                seed=seedi,
            )
            crest = crest_factor(uk)
            if crest < crest_best:
                ybest = uk
                crest_best = crest
        return duplicate(ybest, N_periods)

    N = N_points_per_period
    uf = np.zeros((N,), dtype=complex)
    for p in range(pmin, pmax) if par is None else par:
        if par is None and not prule(p):
            continue
        uf[p] = np.exp(1j * rng.uniform(0, np.pi * 2))
        uf[N - p] = np.conjugate(uf[p])

    uk = np.real(np.fft.ifft(uf / 2) * N)
    uk /= np.std(uk)

    return duplicate(uk, N_periods)


# Define the Van der Pol oscillator
def van_der_pol(t, y, mu, u):
    x, dx = y
    ddx = mu * (1 - x**2) * dx - x + u
    return [dx, ddx]


# Define feedforward Neural Netowrk
def NN(u, Network, sigma):
    """Evaluate the neural network with input u."""
    y = u
    L = len(Network.linears)
    for i in range(L - 1):
        W = ca.DM(Network.linears[i].weight.detach().cpu().numpy())
        b = ca.DM(Network.linears[i].bias.detach().cpu().numpy())
        y = sigma(W @ y + b)  # @ is matrix multiplication

    W = ca.DM(Network.linears[-1].weight.detach().cpu().numpy())
    b = ca.DM(Network.linears[-1].bias.detach().cpu().numpy())
    y = W @ y + b
    return y
