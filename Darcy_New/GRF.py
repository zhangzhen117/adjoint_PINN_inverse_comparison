import numpy as np


class GaussianRF:
    """
    Gaussian random field sampler via truncated KL expansion (NumPy version).

    Domain assumed: [0,1]^d
    Basis: cosine functions like in your code:
      - d=1:  sqrt(2)*cos(pi*k*x), k=1,2,...
      - d=2:  combinations of cos(pi*kx*x)cos(pi*ky*y) with special handling for zeros

    Parameters
    ----------
    tau : float
        Controls correlation length (appears as tau^2 in eigenvalues)
    alpha : float
        Smoothness parameter (eigenvalues decay rate)
    sigma : float or None
        If None, uses sigma = tau^(0.5*(2*alpha - d)) like your code
    """

    def __init__(self, tau: float, alpha: float, sigma: float | None = None):
        self.tau = float(tau)
        self.alpha = float(alpha)
        self.sigma = None if sigma is None else float(sigma)

    # -------------------------
    # public API
    # -------------------------
    def sample(self, space_mesh, rand_samples, num_kl=None):
        """
        Sample GRF on given points.

        Parameters
        ----------
        space_mesh : ndarray, shape (N, d)
            Points where the field is evaluated. d=1 or d=2.
        rand_samples : ndarray, shape (r,) or (n_samples, r)
            Standard normal coefficients (or any coefficients you want).
        num_kl : int or None
            Truncation level r. If None, inferred from rand_samples last dim.

        Returns
        -------
        field : ndarray, shape (n_samples, N)
            Sampled field values at the input points.
            If rand_samples was shape (r,), returns shape (1, N).
        """
        X = np.asarray(space_mesh, dtype=float)
        if X.ndim == 1:
            # allow "a column" as (N,)
            X = X.reshape(-1, 1)

        N, d = X.shape
        if d not in (1, 2):
            raise ValueError(f"Only d=1 or d=2 supported, got d={d}")

        coeff = np.asarray(rand_samples, dtype=float)
        if coeff.ndim == 1:
            coeff = coeff.reshape(1, -1)  # (1, r)
        if coeff.ndim != 2:
            raise ValueError("rand_samples must be 1D or 2D array")

        if num_kl is None:
            r = coeff.shape[1]
        else:
            r = int(num_kl)
            if coeff.shape[1] != r:
                raise ValueError(f"rand_samples has r={coeff.shape[1]}, but num_kl={r}")

        # sigma default follows your code
        sigma = self.sigma
        if sigma is None:
            sigma = self.tau ** (0.5 * (2 * self.alpha - d))

        # build candidate KL points and eigenvalues, then take top-r
        sqrt_eigs, kl_points = self._eigen_val_and_points(r, d, sigma)

        # compute basis matrix Phi: shape (r, N)
        Phi = self._eigen_funcs(kl_points, X)  # (r, N)

        # field = coeff @ diag(sqrt_eigs) @ Phi
        # (n_samples, r) * (r,) * (r, N) -> (n_samples, N)
        field = (coeff * sqrt_eigs.reshape(1, -1)) @ Phi
        return field

    # -------------------------
    # internal helpers
    # -------------------------
    def _eigen_val_and_points(self, num_kl, d, sigma):
        """
        Returns top num_kl eigenvalues and their corresponding KL integer wave vectors.

        For d=2, we generate a square grid of kx,ky indices large enough,
        then rank by eigenvalue magnitude and select top num_kl.
        """
        if d == 1:
            # k = 1..K
            kl_points = np.arange(1, num_kl + 1, dtype=int).reshape(-1, 1)
        else:
            # mimic your heuristic: kl_dim ~ sqrt(2*num_kl)+1
            kl_dim = int(np.sqrt(2 * num_kl)) + 1
            kx, ky = np.meshgrid(np.arange(kl_dim), np.arange(kl_dim), indexing="ij")
            kl_points = np.stack([kx.ravel(), ky.ravel()], axis=1)
            # drop (0,0)
            kl_points = kl_points[1:, :]

        kl_norm2 = np.sum(kl_points.astype(float) ** 2, axis=1)  # ||k||^2
        sqrt_eigs_all = sigma * (np.pi**2 * kl_norm2 + self.tau**2) ** (-self.alpha / 2)

        # select top num_kl by eigenvalue (descending)
        idx = np.argsort(sqrt_eigs_all)[::-1][:num_kl]
        return sqrt_eigs_all[idx], kl_points[idx]

    def _eigen_funcs(self, kl_points, X):
        """
        Build eigenfunctions evaluated on points X.

        Parameters
        ----------
        kl_points : ndarray, shape (r, d)
        X : ndarray, shape (N, d)

        Returns
        -------
        Phi : ndarray, shape (r, N)
        """
        r, d = kl_points.shape
        N = X.shape[0]
        Phi = np.empty((r, N), dtype=float)

        if d == 1:
            k = kl_points[:, 0].reshape(-1, 1)  # (r,1)
            x = X[:, 0].reshape(1, -1)          # (1,N)
            # common 1D cosine basis: sqrt(2)*cos(pi*k*x)
            Phi[:] = np.sqrt(2.0) * np.cos(np.pi * k * x)
            return Phi

        # d == 2
        x = X[:, 0].reshape(1, -1)  # (1,N)
        y = X[:, 1].reshape(1, -1)  # (1,N)
        kx = kl_points[:, 0].reshape(-1, 1)  # (r,1)
        ky = kl_points[:, 1].reshape(-1, 1)  # (r,1)

        # handle cases like your code
        is00 = (kx[:, 0] == 0) & (ky[:, 0] == 0)
        is0y = (kx[:, 0] == 0) & (ky[:, 0] != 0)
        isx0 = (kx[:, 0] != 0) & (ky[:, 0] == 0)
        isxy = (kx[:, 0] != 0) & (ky[:, 0] != 0)

        # default fill
        Phi[:] = 0.0

        # (0,0): constant 1
        if np.any(is00):
            Phi[is00, :] = 1.0

        # (0,ky): sqrt(2)*cos(pi*ky*y)
        if np.any(is0y):
            Phi[is0y, :] = np.sqrt(2.0) * np.cos(np.pi * ky[is0y, :] * y)

        # (kx,0): sqrt(2)*cos(pi*kx*x)
        if np.any(isx0):
            Phi[isx0, :] = np.sqrt(2.0) * np.cos(np.pi * kx[isx0, :] * x)

        # (kx,ky): 2*cos(pi*kx*x)*cos(pi*ky*y)
        if np.any(isxy):
            Phi[isxy, :] = 2.0 * np.cos(np.pi * kx[isxy, :] * x) * np.cos(np.pi * ky[isxy, :] * y)

        return Phi
