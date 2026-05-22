# -*- coding: utf-8 -*-
"""
Define the class containig the model, in this case an Abelian Higgs Model,
for the later inplementation into cosmoTransitions notebook.

Created on Fri May 22 14:38:44 2026

@author: Usuario
"""

import numpy as np
from cosmoTransitions import generic_potential

class Abelian_Model(generic_potential.generic_potential):
    
    def init(self, lam, mu, g):
        """
        Parameters (all in GeV or GeV^2 as appropriate):
          lam : Scalar quartic coupling (\lambda)
          mu  : Scalar mass parameter (\mu)
          g   : U(1) gauge coupling
        """
        self.lam = lam
        self.mu = mu
        self.g = g

        # CosmoTransitions bookkeeping
        self.Ndim = 1                  # 1D field space: (phi)
        self.x_eps = 0.001             # step size for numerical derivatives
        self.T_eps = 0.001
        self.renormScaleSq = (246.0)**2  # MS-bar scale squared (GeV^2)

    # ------------------------------------------------------------------ #
    #  Tree-level potential                                              #
    # ------------------------------------------------------------------ #
    def V0(self, X):
        X = np.asarray(X)
        phi = X[..., 0]
        
        return -0.5 * self.mu**2 * phi**2 + 0.25 * self.lam * phi**4

    # ------------------------------------------------------------------ #
    #  Field-dependent boson masses                                      #
    # ------------------------------------------------------------------ #
    def boson_massSq(self, X, T):
        X = np.asarray(X)
        phi = X[..., 0]

        # --- Scalar mass elements ---
        mHsq = -self.mu**2 + 3 * self.lam * phi**2  # Higgs mode
        mGsq = -self.mu**2 + self.lam * phi**2      # Goldstone mode

        # --- Gauge boson (split into Transverse and Longitudinal) ---
        mATsq = self.g**2 * phi**2
        mALsq = self.g**2 * phi**2

        # --- Debye (thermal) masses for ring resummation ---
        if T > 0:
            # Scalar thermal mass contribution (Pi)
            # For Abelian Higgs: Pi = (8*lam + 3*e^2) * T^2 / 24
            Pi_scalar = (8 * self.lam + 3 * self.g**2) * T**2 / 24.0
            mHsq = mHsq + Pi_scalar
            mGsq = mGsq + Pi_scalar

            # Gauge boson longitudinal Debye mass
            # Pi_A = e^2 * T^2 / 3
            Pi_A = (self.g**2 * T**2) / 3.0
            mALsq = mALsq + Pi_A

        # Return: (mass_sq_array, dof_array, c_array)
        # dof: 1 Higgs, 1 Goldstone, 2 Transverse Gauge, 1 Longitudinal Gauge
        masses = np.stack([mHsq, mGsq, mATsq, mALsq], axis=-1)
        dofs   = np.array([1,    1,    2,     1])
        cs     = np.array([1.5,  1.5,  5/6,   5/6])

        return masses, dofs, cs

    # ------------------------------------------------------------------ #
    #  Field-dependent fermion masses                                    #
    # ------------------------------------------------------------------ #
    def fermion_massSq(self, X):
        X = np.asarray(X)
        phi = X[..., 0]

        # Create a dummy mass array matching the shape of the scalar fields.
        # We multiply by 0.0 degrees of freedom so it has absolutely ZERO physical effect.
        dummy_mass = np.zeros_like(phi)
        
        # Stack it along the last axis to respect CosmoTransitions' broadcasting
        masses = np.stack([dummy_mass], axis=-1)
        dofs = np.array([0.0])  # 0 degrees of freedom means it is physically absent

        return masses, dofs
    
    