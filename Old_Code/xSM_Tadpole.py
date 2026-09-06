# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 09:06 2026

xSM - Real Singlet Extension of the Standard Model (no Z_2 symmetry), 
Now with a Tadpole Term and no singlet vev.

@author: Diego García Tejada
"""

import numpy as np
from cosmoTransitions import generic_potential

# ================================================================== #
#  PDG 2025 defaults                                                 #
# ================================================================== #
_v_EW    = 246.22   # GeV
_mt_phys = 172.4    # GeV
_mb_phys = 4.183  # GeV
_mW_phys = 80.369  # GeV
_mZ_phys = 91.188  # GeV

class xSMt(generic_potential.generic_potential):

    def init(self,
             lH   = 0.1292,
             lHS  = 0.0,
             muHS = 0.0,
             lS   = 0.0,
             mu3  = 0.0,
             muS2 = 0.0,
             v    = _v_EW,
             mt   = _mt_phys,
             mb   = _mb_phys,
             mW   = _mW_phys,
             mZ   = _mZ_phys):
        """
        Lagrangian Parameters for the model.

        Parameters
        ----------
        lH   : lambda_H,  Higgs quartic coupling
        lHS  : lambda_HS, quartic portal coupling
        muHS : mu_HS,     Z2-breaking cubic portal   (GeV)
        lS   : lambda_S,  singlet quartic
        mu3  : mu_3,      Z2-breaking singlet cubic  (GeV)
        v    : v,         Higgs vev                  (GeV)
        muS2 : mu_S^2,    Singlet tachyonic mass squared   (GeV^2)
        mt, mb, mW, mZ : SM inputs (GeV), 
        
        Defaults correspond to the SM (PDG 2025)
        """

        # --- Store the Lagrangian Couplings ---
        self.lH   = lH
        self.lHS  = lHS
        self.muHS = muHS
        self.lS   = lS
        self.mu3  = mu3
        self.muS2 = muS2

        # --- Store the EW vev ---
        self.v_os = v

        # --- SM gauge / Yukawa couplings ---
        self.g2 = 2.0 * mW / v
        self.g1 = 2.0 * np.sqrt(abs(mZ**2 - mW**2)) / v
        self.yt = np.sqrt(2.0) * mt / v
        self.yb = np.sqrt(2.0) * mb / v

        # --- Tadpole Removed Lagrange Parameters ---
        self.muH2 = lH * v**2            # Higgs tachyonic mass squared
        self.mu13 = - 0.5 * v**2 * muHS  # Singlet tadpole term

        # --- Compute Symmetric vev ---
        coeffs = [lS,-mu3,-muS2, self.mu13]
        roots = np.roots(coeffs)
        real_roots = roots[np.abs(roots.imag) < 1e-8].real
        d2V = -self.muS2 - 2.0 * self.mu3 * real_roots + 3.0 * self.lS * real_roots**2
        minima_roots = real_roots[d2V > 0]
        candidates = np.stack([np.zeros_like(minima_roots), minima_roots], axis=-1)
        v_vals = self.V0(candidates)
        idx = np.argmin(v_vals)
        self.w_os = minima_roots[idx]

        # --- CosmoTransitions bookkeeping ---
        self.Ndim          = 2
        self.x_eps         = 0.001
        self.T_eps         = 0.001
        self.renormScaleSq = v**2

        # --- Precompute OS counterterms at T = 0 ---
        self._compute_OS_counterterms()

    # ================================================================== #
    #  Tree-level potential                                              #
    # ================================================================== #

    def V0(self, X):
        X = np.asarray(X, dtype=float)
        h = X[..., 0]
        s = X[..., 1]
        return (
            self.lH   / 4.0 * h**4
            - self.muH2 / 2.0 * h**2
            + self.lHS  / 4.0 * h**2 * s**2
            + self.muHS / 2.0 * h**2 * s
            + self.lS   / 4.0 * s**4
            - self.mu3  / 3.0 * s**3
            - self.muS2 / 2.0 * s**2
            + self.mu13 * s
        )
    
    # ================================================================== #
    #  Field-dependent boson masses (Parwani Ressumation)                #
    # ================================================================== #

    def boson_massSq(self, X, T, Goldstone=True):
        """
        Field- and T-dependent squared boson masses.

        Goldstone : bool, optional (default True)
        If False, the Goldstone dof are set to zero, removing their
        contribution from V1

        Particle content and conventions
        ---------------------------------
        idx  particle          dof    c
         0   h1 scalar          1    3/2
         1   h2 scalar          1    3/2
         2   Goldstones (x3)    3    3/2
         3   W transverse       4    5/6   (2 charged * 2 transverse)
         4   Z transverse       2    5/6
         5   W longitudinal     2    3/2   (2 charged)
         6   Z longitudinal     1    3/2
         """
        X = np.asarray(X, dtype=float)
        h = X[..., 0]
        s = X[..., 1]
        T = np.asarray(T, dtype=float)

        # --- Zero-T Scalar Mass Matrix --- (Entries are squared but I clean notation)
        Mhh  = 3.0*self.lH*h**2 - self.muH2 + 0.5*self.lHS*s**2 + self.muHS*s
        Mss  = 0.5*self.lHS*h**2 + 3.0*self.lS*s**2 - 2.0*self.mu3*s - self.muS2
        Mhs  = self.lHS*h*s + self.muHS*h
        mGsq = self.lH*h**2 - self.muH2 + 0.5*self.lHS*s**2 + self.muHS*s

        # --- Zero-T Gauge Bosons ---
        mWsq = 0.25 * self.g2**2 * h**2
        mZsq = 0.25 * (self.g1**2 + self.g2**2) * h**2

        # --- Thermal Masses --- (Entries are squared but I clean notation)
        # Scalars
        PiH = T**2 / 48.0 * (9.0*self.g2**2 + 3.0*self.g1**2
                              + 24.0*self.lH + 4.0*self.lHS
                              + 12.0*self.yt**2 + 12.0*self.yb**2)
        PiS = T**2 / 12.0 * (2.0*self.lHS + 3.0*self.lS)
        # Gauge Bosons, Longitudinal
        PiWL = (11.0/6.0) * self.g2**2 * T**2
        PiZL = (11.0/6.0) * (self.g1**2 + self.g2**2) * T**2

        # --- Diagonalize the Scalar Mass Matrix ---
        Mhh_T = Mhh + PiH
        Mss_T = Mss + PiS
        avg   = 0.5 * (Mhh_T + Mss_T)
        delta = np.sqrt(np.maximum(0.25*(Mhh_T - Mss_T)**2 + Mhs**2, 0.0))
        m1sq  = avg + delta
        m2sq  = avg - delta
        # Rest of the thermally corrected Masses
        mGsq_T = mGsq + PiH
        mWLsq  = mWsq + PiWL
        mZLsq  = mZsq + PiZL

        # --- Cosmo Transitions Output ---
        masses = np.stack([m1sq, m2sq, mGsq_T,
                           mWsq,  mZsq,
                           mWLsq, mZLsq], axis=-1)
        dofs   = np.array([1.,  1.,  3.,
                           4.,  2.,
                           2.,  1.])
        cs     = np.array([3/2, 3/2, 3/2,
                           5/6, 5/6,
                           3/2, 3/2])
        
        # --- Optional: Remove the Goldstone ---
        if not Goldstone:
            dofs = dofs.copy()
            dofs[2] = 0.0

        return masses, dofs, cs
    
    # ================================================================== #
    #  Fermion masses                                                    #
    # ================================================================== #

    def fermion_massSq(self, X):
        X = np.asarray(X, dtype=float)
        h = X[..., 0]
        mtsq = 0.5 * self.yt**2 * h**2
        mbsq = 0.5 * self.yb**2 * h**2
        masses = np.stack([mtsq, mbsq], axis=-1)
        dofs   = np.array([12., 12.])
        return masses, dofs
    
    # ================================================================== #
    #  On-Shell counterterms                                             #
    # ================================================================== #
    """
    Our Counter-Term Potential is

    -A/2*h^2 + B/4*h^4 - C/2*s^2 + D/4*s^4 + E/2*h^2*s + F/4*h^2*s^2 + G*s - H/3*s^3 + I

    Our choice of renormalization is as follows:
    1) The EW vacuum and the symetric vacuum have the same values as in V0.
    -> Vh(v,0) + B v^3 - A v = 0
    -> Vs(v,0) + G + E/2 v^2 = 0
    -> Vs(0,w) + G - C w - H w^2 + D w^3 = 0
    2) Preserve the Mass Matrix.
    -> Vhh(v,0) - A + 3 B v^2 = 0
    -> Vss(v,0) - C + (F v^2)/2 = 0
    -> Vhs(v,0) + E v = 0
    3) Preserve the singlet cubic coupling
    -> Vsss(v,0) -2 H = 0
    4) Ensure the Minimas have the same value
    -> V(v,0) + I - (A v^2)/2 + (B v^4)/4
    -> V(0,w) + I + G w - (C w^2)/2 - (H w^3)/3 + (D w^4)/4
    """
    def _V_CW_at(self, h_val, s_val):
        """V_CW at a single point, T=0."""
        X        = np.array([[h_val, s_val]])
        bosons   = self.boson_massSq(X, 0.0, Goldstone=True)
        fermions = self.fermion_massSq(X)
        return self.V1(bosons, fermions).item()
    
    def _cw_derivs(self, h0, s0):
        """
        V_CW and its derivatives at (h0, s0) via 5-point stencils.
        """
        epsH = max(1e-3 * abs(h0), 0.1)
        epsS = max(1e-3 * abs(s0), 0.1)

        def f(dh, ds):
            return self._V_CW_at(h0 + dh, s0 + ds)

        f00 = f(0, 0)

        Vh  = (-f(2*epsH,0) + 8*f(epsH,0) - 8*f(-epsH,0) + f(-2*epsH,0)) / (12.0*epsH)
        Vs  = (-f(0,2*epsS) + 8*f(0,epsS) - 8*f(0,-epsS) + f(0,-2*epsS)) / (12.0*epsS)

        Vhh = (-f(2*epsH,0) + 16*f(epsH,0) - 30*f00
               + 16*f(-epsH,0) - f(-2*epsH,0)) / (12.0*epsH**2)
        Vss = (-f(0,2*epsS) + 16*f(0,epsS) - 30*f00
               + 16*f(0,-epsS) - f(0,-2*epsS)) / (12.0*epsS**2)

        Vhs = (  f( epsH, epsS) - f( epsH,-epsS)
               - f(-epsH, epsS) + f(-epsH,-epsS)) / (4.0*epsH*epsS)

        Vsss = (-f(0,2*epsS) + 2*f(0,epsS)
                - 2*f(0,-epsS) + f(0,-2*epsS)) / (2.0*epsS**3)

        return dict(V=f00, Vh=Vh, Vs=Vs, Vhh=Vhh, Vss=Vss, Vhs=Vhs, Vsss=Vsss)
    
    def _compute_OS_counterterms(self):
        """
        Our Counter-Term Potential is

          -A/2 h^2 + B/4 h^4 - C/2 s^2 + D/4 s^4
          + E/2 h^2 s + F/4 h^2 s^2 + G s - H/3 s^3 + I

        Renormalization conditions (9 eqs, 9 unknowns A..I):
          1) EW and symmetric vacua stay stationary points of V_tot
             at (v,0) and (0,w).
          2) Tree-level mass matrix at (v,0) preserved.
          3) Tree-level singlet cubic self-coupling preserved.
          4) V_tot(v,0) = V_tot(0,w)  [degenerate minima].

        w (symmetric-vacuum VEV) is prviusly given.
        """
        v = self.v_os
        w = self.w_os

        ew = self._cw_derivs(v, 0.0)
        VEW, VEWh, VEWs   = ew['V'], ew['Vh'], ew['Vs']
        VEWhh, VEWss      = ew['Vhh'], ew['Vss']
        VEWhs, VEWsss     = ew['Vhs'], ew['Vsss']

        # --- Carefull with the Z2 symmetry ---

        if w==0:
            A = (3.0*VEWh - v*VEWhh) / (2.0*v)
            B = (VEWh - v*VEWhh) / (2.0*v**3)
            C = VEWss
            D = 0.0
            F = 0.0
            I = -VEW + 0.5*A*v**2 - 0.25*B*v**4

            self._ct_A, self._ct_B, self._ct_C = A, B, C
            self._ct_D, self._ct_F, self._ct_I = D, F, I
            self._ct_E = self._ct_G = self._ct_H = 0.0
        
        else:
            sy = self._cw_derivs(0.0, w)
            VSY, VSYs = sy['V'], sy['Vs']

            A = (3.0*VEWh - v*VEWhh) / (2.0*v)
            B = (VEWh - v*VEWhh) / (2.0*v**3)

            bracket_C = (24.0*VEW - 15.0*v*VEWh + 3.0*v**2*VEWhh - 24.0*VSY
                        - 9.0*v*VEWhs*w + 18.0*VEWs*w + 6.0*VSYs*w + VEWsss*w**3)
            C = -bracket_C / (6.0*w**2)

            bracket_D = (24.0*VEW - 15.0*v*VEWh + 3.0*v**2*VEWhh - 24.0*VSY
                        - 6.0*v*VEWhs*w + 12.0*VEWs*w + 12.0*VSYs*w - 2.0*VEWsss*w**3)
            D = -bracket_D / (6.0*w**4)

            E = -VEWhs / v

            bracket_F = (24.0*VEW - 15.0*v*VEWh + 3.0*v**2*VEWhh - 24.0*VSY
                        - 9.0*v*VEWhs*w + 18.0*VEWs*w + 6.0*VSYs*w
                        + 6.0*VEWss*w**2 + VEWsss*w**3)
            F = -bracket_F / (3.0*v**2*w**2)

            G = 0.5*(v*VEWhs - 2.0*VEWs)
            H = 0.5*VEWsss
            I = 0.125*(-8.0*VEW + 5.0*v*VEWh - v**2*VEWhh)

            self._ct_A, self._ct_B, self._ct_C = A, B, C
            self._ct_D, self._ct_E, self._ct_F = D, E, F
            self._ct_G, self._ct_H, self._ct_I = G, H, I

    def counterterm(self, X):
        """
        delta_V = -A/2 h^2 + B/4 h^4 - C/2 s^2 + D/4 s^4
                  + E/2 h^2 s + F/4 h^2 s^2 + G s - H/3 s^3 + I
        """
        X = np.asarray(X, dtype=float)
        h = X[..., 0]
        s = X[..., 1]
        return ( -0.5*self._ct_A * h**2
                 + 0.25*self._ct_B * h**4
                 - 0.5*self._ct_C * s**2
                 + 0.25*self._ct_D * s**4
                 + 0.5*self._ct_E * h**2 * s
                 + 0.25*self._ct_F * h**2 * s**2
                 + self._ct_G * s
                 - (1.0/3.0)*self._ct_H * s**3
                 + self._ct_I )
    
    # ================================================================== #
    #  Full effective potential                                          #
    # ================================================================== #

    def Vtot(self, X, T, include_radiation=True):
        """V_eff = V0 + V_CW + delta_V_OS + V_thermal."""
        return super().Vtot(X, T, include_radiation) + self.counterterm(X)
    
    # ================================================================== #
    #  Phase-tracker helpers                                              #
    # ================================================================== #

    def approxZeroTMin(self):
        return [np.array([self.v_os, 0.0])]

    def approxFiniteTMin(self):
        return [np.array([0.0, 0.0])]

    def forbidPhaseCrit(self, X):
        return (np.array([X])[..., 0] < -5.0).any()
    
    # ================================================================== #
    #  Diagnostics                                                       #
    # ================================================================== #
    def print_params(self):
        """Print all Lagrangian and derived parameters."""
        print("="*52)
        print("  xSM Lagrangian parameters")
        print("="*52)
        print(f"  muH2  = {self.muH2:.4f}  GeV^2")
        print(f"  lH    = {self.lH:.4f}")
        print(f"  lHS   = {self.lHS:.4f}")
        print(f"  muHS  = {self.muHS:.4f}  GeV")
        print(f"  lS    = {self.lS:.4f}")
        print(f"  muS2  = {self.muS2:.4f}  GeV^2")
        print(f"  mu13   = {self.mu13:.4f}  GeV")
        print(f"  mu3   = {self.mu3:.4f}  GeV")
        print("="*52)

    