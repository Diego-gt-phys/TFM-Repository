# -*- coding: utf-8 -*-
"""
Created on Thu May 14 21:59:29 2026

@author: Diego García Tejada
"""

import numpy as np
from cosmoTransitions import generic_potential

class xSM(generic_potential.generic_potential):
    
    def init(self, lH, muH, lHS, muHS, lS, mu3, muS):
        """
        Parameters (all in GeV or GeV^2 as appropriate):
          lH   : lambda_H  (Higgs quartic)
          muH  : mu_H      (Higgs mass parameter, muH^2 appears in potential)
          lHS  : lambda_HS (quartic portal)
          muHS : mu_HS     (Z2-breaking cubic portal)
          lS   : lambda_S  (singlet quartic)
          mu3  : mu_3      (singlet cubic, Z2-breaking)
          muS  : mu_S      (singlet mass parameter)
        """
        self.lH   = lH
        self.muH  = muH
        self.lHS  = lHS
        self.muHS = muHS
        self.lS   = lS
        self.mu3  = mu3
        self.muS  = muS

        # SM gauge couplings at mZ
        self.g  = 0.6483   # SU(2)
        self.gp = 0.3587   # U(1)_Y
        self.yt = 0.9946   # top Yukawa
        self.yb = 0.0243   # bottom Yukawa

        # CosmoTransitions bookkeeping
        self.Ndim = 2          # 2D field space: (h, s)
        self.x_eps = 0.001     # step size for numerical derivatives
        self.T_eps = 0.001
        self.renormScaleSq = (246.0)**2  # MS-bar scale squared (GeV^2)

    # ------------------------------------------------------------------ #
    #  Tree-level potential                                                #
    # ------------------------------------------------------------------ #
    def V0(self, X):
        X = np.asarray(X)
        h = X[..., 0]
        s = X[..., 1]
        
        return (  self.lH/4  * h**4
                - self.muH**2/2 * h**2
                + self.lHS/4 * h**2 * s**2
                + self.muHS/2 * h**2 * s
                + self.lS/4  * s**4
                - self.mu3/3 * s**3
                - self.muS**2/2 * s**2  )

    # ------------------------------------------------------------------ #
    #  Field-dependent boson masses                                        #
    # ------------------------------------------------------------------ #
    def boson_massSq(self, X, T):
        X = np.asarray(X)
        h = X[..., 0]
        s = X[..., 1]

        # --- Scalar 2x2 mass matrix elements ---
        # M^2_hh = d^2V/dh^2
        Mhh = 3*self.lH*h**2 - self.muH**2 + self.lHS/2*s**2 + self.muHS*s
        # M^2_ss = d^2V/ds^2
        Mss = self.lHS/2*h**2 + 3*self.lS*s**2 - 2*self.mu3*s - self.muS**2
        # M^2_hs = d^2V/dh ds
        Mhs = self.lHS*h*s + self.muHS*h

        # Eigenvalues of the 2x2 matrix analytically
        avg   = (Mhh + Mss) / 2.0
        delta = np.sqrt(np.maximum(((Mhh - Mss)/2.0)**2 + Mhs**2, 0.0))
        m1sq  = avg + delta   # heavier scalar
        m2sq  = avg - delta   # lighter scalar

        # --- Goldstone bosons (3 components) ---
        mGsq = self.lH*h**2 - self.muH**2 + self.lHS/2*s**2 + self.muHS*s

        # --- Gauge bosons ---
        mWsq = self.g**2/4  * h**2
        mZsq = (self.g**2 + self.gp**2)/4 * h**2

        # --- Debye (thermal) masses for ring resummation ---
        # These add to the longitudinal modes only; CosmoTransitions uses
        # the full list here and handles the ring improvement via findApproxTransitions
        if T > 0:
            # Scalar thermal masses (leading order Daisy)
            PiH = T**2/48 * (9*self.g**2 + 3*self.gp**2 + 24*self.lH
                 + 2*self.lHS + 12*self.yt**2 + 12*self.yb**2)
            PiS = T**2/12 * (2*self.lHS + 3*self.lS)
            # Add to diagonal (approximate — full matrix would mix again)
            Mhh_T = Mhh + PiH
            Mss_T = Mss + PiS
            avg_T   = (Mhh_T + Mss_T) / 2.0
            delta_T = np.sqrt(np.maximum(((Mhh_T - Mss_T)/2.0)**2 + Mhs**2, 0.0))
            m1sq = avg_T + delta_T
            m2sq = avg_T - delta_T
            mGsq = mGsq + PiH
        
        # --- Gauge boson longitudinal Debye masses ---
        PiW = self.g**2 * T**2 * 11/6  if T > 0  else 0.0
        PiZ = (self.g**2 + self.gp**2) * T**2 * 11/6  if T > 0  else 0.0
        mWLsq = mWsq + PiW
        mZLsq = mZsq + PiZ

        # Return: (mass_sq_array, dof_array, c_array)
        # c = 3/2 for scalars, 5/6 for gauge bosons (in MS-bar CW)
        # dof: scalars=1 each, W=6 (2 charged x 3), Z=3, W_L=2, Z_L=1
        masses = np.stack([m1sq, m2sq, mGsq,
                   mWsq, mZsq,
                   mWLsq, mZLsq], axis=-1)
        dofs   = np.array([1,    1,    3,
                           4,    2,       # W transverse (x2 charged x2 transverse), Z transverse
                           2,    1])      # W longitudinal, Z longitudinal
        cs     = np.array([1.5,  1.5, 1.5,
                           5/6,  5/6,
                           5/6,  5/6])

        return masses, dofs, cs

    # ------------------------------------------------------------------ #
    #  Field-dependent fermion masses (top quark dominates)               #
    # ------------------------------------------------------------------ #
    def fermion_massSq(self, X):
        X = np.asarray(X)
        h = X[..., 0]
        mtsq = self.yt**2 / 2.0 * h**2
        mbsq = self.yb**2 / 2.0 * h**2
        masses = np.stack([mtsq, mbsq], axis=-1)
        return masses, np.array([12, 12])
    
