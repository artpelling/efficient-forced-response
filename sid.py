from numpy import (
    arange,
    asarray,
    atleast_2d,
    complex128,
    concatenate,
    conj,
    diag,
    exp,
    eye,
    flip,
    imag,
    max,
    ndarray,
    pi,
    real,
    roll,
    squeeze,
    zeros,
)
from numpy.fft import fft, ifft
from numpy.random import randn
from numpy.linalg import cond
from scipy.linalg import qr, svd, eig, inv, cdf2rdf, logm
from scipy.linalg.blas import cgbmv, dgbmv


class BandedMatrix(ndarray):
    def __new__(cls, A, kl, ku, m=None):
        A = atleast_2d(A)
        if A.shape[0] == A.shape[1]:
            bands = zeros([ku + 1 + kl, A.shape[1]])
            for i in range(ku):
                bands[i] = concatenate([zeros(ku - i), diag(A, ku - i)])

            bands[ku] = diag(A)

            for i in range(kl):
                i += 1
                bands[ku + i] = concatenate([diag(A, -i), zeros(i)])

        else:
            bands = A
        obj = asarray(bands).view(cls)
        obj.kl = kl
        obj.ku = ku
        obj.n = bands.shape[1]
        obj.m = bands.shape[1]
        if m:
            obj.m = m

        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return

    def __matmul__(self, x):
        if self.dtype == complex128:
            return cgbmv(self.m, self.n, self.kl, self.ku, 1, self, x)
        else:
            return dgbmv(self.m, self.n, self.kl, self.ku, 1, self, x)

    def __rmatmul__(self, x):
        if self.dtype == complex128:
            return cgbmv(self.m, self.n, self.kl, self.ku, 1, self, x.T, trans=1)
        else:
            return dgbmv(self.m, self.n, self.kl, self.ku, 1, self, x.T, trans=1)

    @property
    def T(self):
        return BandedMatrix(self.view(ndarray).transpose(), self.ku, self.kl, m=self.n)

    @property
    def shape(self):
        return (self.m, self.n)


class BlockHankelMatrix(ndarray):
    def __new__(cls, hankel_blocks):
        p, m, s = hankel_blocks.shape
        if s % 2:
            hankel_blocks = concatenate([hankel_blocks, zeros([p, m, 1])], axis=-1)
            s += 1

        obj = asarray(hankel_blocks).view(cls)
        obj._circ = fft(
            roll(
                concatenate([zeros([p, m, 1]), hankel_blocks[..., :-1]], axis=-1),
                s // 2,
                axis=-1,
            )
        )
        obj.p = p
        obj.m = m
        obj.s = s // 2
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return

        self._circ = getattr(obj, "_circ", None)

    def __matmul__(self, x):
        if x.ndim == 1:
            x = x.reshape(-1, 1)

        n = x.shape[1]
        y = zeros([self.p * self.s, n], dtype="float")
        for i in range(self.p):
            for j in range(self.m):
                y[i :: self.p] += real(
                    ifft(
                        fft(
                            concatenate(
                                [flip(x[j :: self.m], axis=0), zeros([self.s, n])]
                            ),
                            axis=0,
                        )
                        * self._circ[i, j].reshape(-1, 1),
                        axis=0,
                    )
                )[: self.s]

        return y

    def __rmatmul__(self, x):
        if x.ndim == 1:
            x = x.reshape(1, -1)

        x = x.T
        n = x.shape[1]
        y = zeros([self.m * self.s, n], dtype="float")
        for i in range(self.m):
            for j in range(self.p):
                y[i :: self.m] += real(
                    ifft(
                        fft(
                            concatenate(
                                [flip(x[j :: self.p], axis=0), zeros([self.s, n])]
                            ),
                            axis=0,
                        )
                        * self._circ[j, i].reshape(-1, 1),
                        axis=0,
                    )
                )[: self.s]

        return y.T

    @property
    def T(self):
        return BlockHankelMatrix(self.view(ndarray).transpose(1, 0, 2))

    @property
    def shape(self):
        return (self.p * self.s, self.m * self.s)


def rsvd(A, rank, n_oversamples=None, n_subspace_iters=None):
    if n_oversamples is None:
        n_samples = 2 * rank
    else:
        n_samples = rank + n_oversamples

    P = randn(A.shape[-1], n_samples)
    Z = A @ P

    if n_subspace_iters:
        for i in range(n_subspace_iters):
            Z = A.T @ Z
            Z = A @ Z

    Q, _ = qr(Z, mode="economic")

    Y = Q.T @ A
    U_tilde, S, Vt = svd(Y, full_matrices=False)
    U = Q @ U_tilde

    U, S, Vt = U[:, :rank], S[:rank], Vt[:rank, :]

    return U, S, Vt


def _normalize_frequency(flim, ts):
    return 2 * pi * flim * ts


def _compute_F(A, omega):
    omega = atleast_2d(omega)
    n = A.shape[0]
    F = zeros([n, n])
    for w in omega:
        F += eye(n) * (w[1] - w[0]) / 2 + imag(
            logm(eye(n) - A * exp(-1j * w[1])) - logm(eye(n) - A * exp(-1j * w[0]))
        )

    return F / pi


def era(U, S, Vt, p, m, dt, r=None, flim=None, return_svs=False):
    if r is None:
        r = len(S)

    if flim is None:
        n = r
    else:
        n = len(S)

    sqS = diag(S[:n] ** 0.5)
    invsqS = diag(S[:n] ** -0.5)
    A = invsqS @ U[:-p, :n].T @ U[p:, :n] @ sqS
    B = sqS @ Vt[:n, :m]
    C = U[:p, :n] @ sqS
    S = diag(S[:n])

    if flim is not None:
        w = _normalize_frequency(flim, dt)
        F = _compute_F(A, w)

        Wc = F @ S + S @ conj(F.T)
        Wo = conj(F.T) @ S + S @ F

        lam, T = eig(Wc @ Wo)
        Ti = inv(T)

        if max(imag(lam[:r])) > 0:
            print("Warning, complex eigenvalues of Gramians")

        T = real(T[:, :r])
        Ti = real(Ti[:r])
        if return_svs:
            return Ti @ A @ T, Ti @ B, C @ T, lam
        else:
            return Ti @ A @ T, Ti @ B, C @ T

    return A, B, C


def forced_response(A, B, C, u, x0=None, n=None):
    u = atleast_2d(u)
    m, k = u.shape
    if n is None:
        n = k
    elif n <= k:
        u = u[:, :n]
    else:
        u = concatenate([u, zeros([m, n - k])], axis=-1)

    y = zeros((C.shape[0], n))

    if x0 is None:
        x = zeros((A.shape[1],))
    else:
        x = x0

    return _dlsim_full(y, A, B, C, u, x)


def _dlsim_full(y, A, B, C, u, x):
    x = A @ x + B @ u[:, 0]
    for i in range(0, u.shape[-1] - 1):
        y[:, i] = real(C @ x)
        x = A @ x + B @ u[:, i + 1]

    y[:, -1] = real(C @ x)

    return y


def impulse_response(A, B, C, n, m=None, p=None):
    if m is None:
        nm = B.shape[1]
        m = arange(m)
    else:
        nm = 1
        m = [m]
    if p is None:
        np = C.shape[0]
        p = arange(p)
    else:
        np = 1
        p = [p]

    if nm == 1 and np == 1:
        h = forced_response(A, B[:, m], C[p], 1, n=n)
    else:
        h = zeros([np, nm, n])
        An = eye(A.shape[0])
        for i in range(n):
            h[..., i] = C[p] @ An @ B[:, m]
            An = An @ A

    return squeeze(h)


def canonical_diagonal(A, B, C, real=True):
    lam, V = eig(A)
    if real:
        lam, V = cdf2rdf(lam, V)
        return BandedMatrix(lam, 1, 1), inv(V) @ B, C @ V, cond(V)
    else:
        return BandedMatrix(lam, 0, 0), inv(V) @ B, C @ V, cond(V)
