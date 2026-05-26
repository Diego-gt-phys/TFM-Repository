# -*- coding: utf-8 -*-
"""
Created on Tue May 26 17:55:26 2026

@author: Usuario
"""

"""
Abelian Higgs Model for CosmoTransitions — OS-scheme counterterms.
"""

import numpy as np
from cosmoTransitions import generic_potential

class Abelian_Model(generic_potential.generic_potential):

    def init(self, lam, mu, g):
        """
        Parameters (all in GeV):
          lam : quartic scalar coupling
          mu  : scalar mass parameter  (mu^2 > 0 drives SSB)
          g   : U(1) gauge coupling

        Tree-level relations:
          v  = mu / sqrt(lam)
          mh = sqrt(2) * mu
          mA = g * v
        """
        self.lam = lam
        self.mu  = mu
        self.g   = g

        self.Ndim          = 1
        self.x_eps         = 0.001
        self.T_eps         = 0.001
        self.renormScaleSq = (self.mu / np.sqrt(self.lam))**2  # mu_R^2 = v^2

        # Compute and cache OS counterterm coefficients at T = 0
        self._compute_OS_counterterms()

    # ------------------------------------------------------------------ #
    #  OS counterterms                                                     #
    # ------------------------------------------------------------------ #
    def _VCW_scalar(self, phi_val):
        """V_CW evaluated at a single field value phi_val, T = 0."""
        X       = np.array([[phi_val]])
        bosons  = self.boson_massSq(X, 0.0)
        fermions = self.fermion_massSq(X)
        return self.V1(bosons, fermions).item()

    def _compute_OS_counterterms(self):
        """
        Compute the two coefficients of delta_V once and cache them.

        OS conditions imposed at phi = v, T = 0:
            (V_CW + delta_V)'  (v) = 0
            (V_CW + delta_V)'' (v) = 0

        Solution (second-order Taylor cancellation):
            delta_V(phi) = -V_CW'(v) * (phi - v)
                         - (1/2) * V_CW''(v) * (phi - v)^2

        We expand this into a linear + quadratic form in phi:
            delta_V(phi) = ct_A * phi + ct_B * phi^2
        dropping the phi-independent constant (irrelevant for dynamics).
        """
        v   = self.mu / np.sqrt(self.lam)
        eps = v * 1e-3   # step size ~0.1% of v

        # Numerical first and second derivatives via central differences
        dVCW  = (self._VCW_scalar(v + eps) - self._VCW_scalar(v - eps)) \
                / (2.0 * eps)
        d2VCW = (self._VCW_scalar(v + eps)
                 - 2.0 * self._VCW_scalar(v)
                 + self._VCW_scalar(v - eps)) / eps**2

        # delta_V = -dVCW*(phi-v) - 0.5*d2VCW*(phi-v)^2
        #         = [-dVCW + d2VCW*v]*phi + [-0.5*d2VCW]*phi^2  + const
        self._ct_A = -dVCW  + d2VCW * v   # coefficient of phi
        self._ct_B = -0.5 * d2VCW         # coefficient of phi^2

    def counterterm(self, X):
        """
        Finite OS counterterm potential delta_V(phi).
        Enforces V_CW'(v) = V_CW''(v) = 0 at T = 0.
        """
        X   = np.asarray(X, dtype=float)
        phi = X[..., 0]
        return self._ct_A * phi + self._ct_B * phi**2

    # ------------------------------------------------------------------ #
    #  Full effective potential (override)                                 #
    # ------------------------------------------------------------------ #
    def Vtot(self, X, T, include_radiation=True):
        """V0 + V_CW + delta_V (OS counterterms) + V_thermal."""
        return (super().Vtot(X, T, include_radiation)
                + self.counterterm(X))

    # ------------------------------------------------------------------ #
    #  Phase-tracker helpers                                               #
    # ------------------------------------------------------------------ #
    def forbidPhaseCrit(self, X):
        return (np.array([X])[..., 0] < -5.0).any()

    def approxZeroTMin(self):
        v = self.mu / np.sqrt(self.lam)
        return [np.array([v])]

    def approxFiniteTMin(self):
        return [np.array([0.0])]

    # ------------------------------------------------------------------ #
    #  Tree-level potential                                                #
    # ------------------------------------------------------------------ #
    def V0(self, X):
        X   = np.asarray(X, dtype=float)
        phi = X[..., 0]
        return -0.5 * self.mu**2 * phi**2 + 0.25 * self.lam * phi**4

    # ------------------------------------------------------------------ #
    #  Field-dependent boson masses  (Parwani daisy resummation)          #
    # ------------------------------------------------------------------ #
    def boson_massSq(self, X, T):
        X   = np.asarray(X, dtype=float)
        phi = X[..., 0]
        T   = np.asarray(T, dtype=float)

        mHsq  = -self.mu**2 + 3.0 * self.lam * phi**2
        mGsq  = -self.mu**2 +       self.lam * phi**2
        mATsq =  self.g**2  * phi**2
        mALsq =  self.g**2  * phi**2

        Pi_s  = (8.0 * self.lam + 3.0 * self.g**2) * T**2 / 24.0
        Pi_A  =  self.g**2 * T**2 / 3.0

        mHsq  = mHsq  + Pi_s
        mGsq  = mGsq  + Pi_s
        mALsq = mALsq + Pi_A
        # mATsq: transverse — no Debye shift

        masses = np.stack([mHsq, mGsq, mATsq, mALsq], axis=-1)
        dofs   = np.array([1.,   1.,   2.,    1.  ])
        cs     = np.array([1.5,  1.5,  5./6., 1.5 ])

        return masses, dofs, cs

    # ------------------------------------------------------------------ #
    #  Fermion masses — none                                               #
    # ------------------------------------------------------------------ #
    def fermion_massSq(self, X):
        X   = np.asarray(X, dtype=float)
        phi = X[..., 0]
        masses = np.stack([np.zeros_like(phi)], axis=-1)
        dofs   = np.array([0.0])
        return masses, dofs

"""
m = Abelian_Model(lam=0.13, mu=88.4, g=0.35)

v    = m.mu / np.sqrt(m.lam)
eps  = v * 1e-4

# Numerical minimum of Vtot at T = 0
from scipy.optimize import minimize_scalar
res = minimize_scalar(lambda phi: m.Vtot(np.array([[phi]]), 0.0).item(),
                      bounds=(v * 0.5, v * 1.5), method='bounded')

print(f"Tree-level vev:        v     = {v:.4f} GeV")
print(f"One-loop minimum:      phi*  = {res.x:.4f} GeV")
print(f"Residual shift:        |v - phi*| = {abs(v - res.x):.2e} GeV")

# Check derivatives of (V_CW + delta_V) vanish at v
dVeff  = (m.Vtot(np.array([[v + eps]]), 0.) - m.Vtot(np.array([[v - eps]]), 0.)) / (2*eps)
d2Veff = (m.Vtot(np.array([[v + eps]]), 0.) - 2*m.Vtot(np.array([[v]]), 0.)
          + m.Vtot(np.array([[v - eps]]), 0.)) / eps**2
print(f"Vtot'(v)  = {dVeff.item():.2e}  (should be ~0)")
print(f"Vtot''(v) = {d2Veff.item():.4f} vs tree-level mh^2 = {2*m.mu**2:.4f}")
"""