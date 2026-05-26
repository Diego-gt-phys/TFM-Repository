# -*- coding: utf-8 -*-
"""
Abelian Higgs Model for CosmoTransitions.

Created on Fri May 22 14:38:44 2026

@author: Diego garcía Tejada
"""

import numpy as np
from cosmoTransitions import generic_potential

class Abelian_Model(generic_potential.generic_potential):

    def init(self, lam, mu, g):
        """
        Parameters (all in GeV):
          lam : quartic scalar coupling  (lambda)
          mu  : scalar mass parameter    (mu^2 > 0 drives SSB)
          g   : U(1) gauge coupling

        Tree-level relations:
          vev     : v   = mu / sqrt(lam)
          Higgs   : mh  = sqrt(2) * mu
          Gauge   : mA  = g * v
        """
        self.lam = lam
        self.mu  = mu
        self.g   = g

        # CosmoTransitions bookkeeping
        self.Ndim          = 1
        self.x_eps         = 0.001
        self.T_eps         = 0.001
        self.renormScaleSq = (self.mu / np.sqrt(self.lam))**2  # v^2

    def forbidPhaseCrit(self, X):
        """Discard the Z2-mirror phase at phi < 0 to avoid double-counting."""
        return (np.array([X])[..., 0] < -5.0).any()

    def approxZeroTMin(self):
        """
        Seed the phase tracker at the T=0 broken-phase minimum.
        Tree-level vev: v = mu / sqrt(lam).
        """
        v = self.mu / np.sqrt(self.lam)
        return [np.array([v])]

    def approxFiniteTMin(self):
        """
        Seed the phase tracker at the high-T symmetric phase.
        phi = 0 is a maximum at T=0 but becomes the global minimum
        for T > Tc. Without this, the phase tracker never finds it
        since it only starts from T=0 minima and evolves upward.
        """
        return [np.array([0.0])]

    # ------------------------------------------------------------------ #
    #  Tree-level potential                                                #
    # ------------------------------------------------------------------ #
    def V0(self, X):
        X   = np.asarray(X, dtype=float)
        phi = X[..., 0]

        return - 0.5 * self.mu**2 * phi**2 \
               + 0.25 * self.lam * phi**4

    # ------------------------------------------------------------------ #
    #  Field-dependent boson masses  (Parwani daisy resummation)          #
    # ------------------------------------------------------------------ #
    def boson_massSq(self, X, T):
        X   = np.asarray(X, dtype=float)
        phi = X[..., 0]
        T   = np.asarray(T, dtype=float)   # FIX: vectorized, no if T > 0

        # --- Scalar masses ---
        mHsq  = -self.mu**2 + 3.0 * self.lam * phi**2   # Higgs (radial)
        mGsq  = -self.mu**2 +       self.lam * phi**2   # Goldstone

        # --- Gauge boson masses ---
        mATsq =  self.g**2 * phi**2                     # transverse (2 dof)
        mALsq =  self.g**2 * phi**2                     # longitudinal (1 dof), Debye-shifted below

        # --- Thermal (Debye) corrections — always added, zero at T=0 ---
        # Scalar self-energy:    Pi_s = (8*lam + 3*g^2) * T^2 / 24
        Pi_s  = (8.0 * self.lam + 3.0 * self.g**2) * T**2 / 24.0
        # Longitudinal gauge:    Pi_A = g^2 * T^2 / 3
        Pi_A  =  self.g**2 * T**2 / 3.0

        mHsq  = mHsq  + Pi_s    # Higgs gets thermal correction
        mGsq  = mGsq  + Pi_s    # Goldstone gets thermal correction
        # mATsq: transverse gauge has NO Debye shift
        mALsq = mALsq + Pi_A    # longitudinal gauge gets Debye mass

        #                        H      G      AT     AL
        masses = np.stack([mHsq, mGsq, mATsq, mALsq], axis=-1)
        dofs   = np.array([1.,   1.,   2.,    1.  ])
        cs     = np.array([1.5,  1.5,  5/6,   3/2 ])
        #  ^ 5/6 for transverse gauge (tensor structure in CW integral)
        #  ^ 3/2 for longitudinal gauge (same as scalars)

        return masses, dofs, cs

    # ------------------------------------------------------------------ #
    #  Fermion masses — none in the Abelian Higgs model                   #
    # ------------------------------------------------------------------ #
    def fermion_massSq(self, X):
        """
        No fermions in the Abelian Higgs model.
        Return properly-shaped empty arrays.
        """
        X = np.asarray(X)
        phi = X[..., 0]
    
        # Dummy fermion with zero physical contribution
        dummy_mass = np.zeros_like(phi)
    
        masses = np.stack([dummy_mass], axis=-1)
        dofs = np.array([0.0])
    
        return masses, dofs