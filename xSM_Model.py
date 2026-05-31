# -*- coding: utf-8 -*-
"""
xSM - Real Singlet Extension of the Standard Model (no Z_2 symmetry)
======================================================================
Author : Diego Garcia Tejada
Created: May 2026

Lagrangian parametrization
---------------------------
Free inputs : lH, lHS, muHS, lS, mu3   (quartic and cubic couplings)
              v, u                       (EW and singlet vevs)
              mt, mb, mW, mZ            (SM inputs, PDG 2025 defaults)

Derived     : muH2, muS2  from tadpole conditions at (v, u)
                muH2 = lH*v^2 + lHS/2*u^2 + muHS*u
                muS2 = lHS/2*v^2 + muHS*v^2/(2u) + lS*u^2 - mu3*u

SM limit    : u=0, lHS=0, muHS=0, lS=free, mu3=0
              lH = mh^2 / (2 v^2),  singlet decouples completely.

Tree-level potential
--------------------
V0 = lH/4 * h^4  - muH2/2 * h^2
   + lHS/4 * h^2*s^2  + muHS/2 * h^2*s
   + lS/4 * s^4  - mu3/3 * s^3  - muS2/2 * s^2

OS counterterms (2D)
---------------------
Ansatz (mirrors full Lagrangian monomial structure):
  dV = A*h^2 + B*s^2 + C*h^4 + D*s^4 + E*h^2*s + F*s^3

6 conditions imposed at (v, u):
  [I]   dV_eff/dh   = 0         h-tadpole
  [II]  dV_eff/ds   = 0         s-tadpole
  [III] d^2V_eff/dh^2 = M_hh    h-mass
  [IV]  d^2V_eff/ds^2 = M_ss    s-mass
  [V]   d^2V_eff/dhds = M_hs    off-diagonal mass
  [VI]  d^3V_eff/ds^3 = V0'''   s-cubic (pins delta mu3)

Analytic solution (Cramer's rule on three decoupled subsystems):
  E  = -V_hs / (2v)
  A  = (-3*V_h  + V_hh*v  + 2*V_hs*u) / (4v)
  C  = ( V_h  - V_hh*v) / (8v^3)
  B  = (3*V_hs*v - 6*V_s + u*(4*V_ss - V_sss*u)) / (4u)
  D  = (V_hs*v - 2*V_s + 2*V_ss*u - V_sss*u^2) / (8u^3)
  F  = (-V_hs*v/2 + V_s - V_ss*u + V_sss*u^2/3) / u^2

where V_h, V_hh, ... are derivatives of V_CW evaluated at (v, u).

Daisy resummation
-----------------
Arnold-Espinosa scheme. Thermal Debye masses added directly to the
entries in boson_massSq; parent class Vtot handles the rest.
"""

import numpy as np
from cosmoTransitions import generic_potential

# ================================================================== #
#  PDG 2025 defaults                                                 #
# ================================================================== #
_v_EW    = 246.22   # GeV
_mt_phys = 172.4    # GeV
_mb_phys =   4.183  # GeV
_mW_phys =  80.369  # GeV
_mZ_phys =  91.188  # GeV


class xSM(generic_potential.generic_potential):

    def init(self,
             lH   = 0.1292,   # Higgs quartic  (SM value: mh^2/2v^2 ~ 0.129)
             lHS  = 0.0,      # quartic portal
             muHS = 0.0,      # GeV    Z2-breaking cubic portal
             lS   = 0.0,      # singlet quartic
             mu3  = 0.0,      # GeV    singlet cubic  (Z2-breaking)
             v    = _v_EW,    # GeV    Higgs vev
             u    = 0.0,      # GeV    singlet vev  (0 = Z2-symmetric / SM limit)
             mt   = _mt_phys,
             mb   = _mb_phys,
             mW   = _mW_phys,
             mZ   = _mZ_phys):
        """
        Lagrangian parameters for the xSM without Z_2.

        Parameters
        ----------
        lH   : lambda_H,  Higgs quartic coupling
        lHS  : lambda_HS, quartic portal coupling
        muHS : mu_HS,     Z2-breaking cubic portal  (GeV)
        lS   : lambda_S,  singlet quartic
        mu3  : mu_3,      Z2-breaking singlet cubic  (GeV)
        v    : Higgs vev (GeV)
        u    : singlet vev (GeV); set to 0 for SM / Z2-symmetric limit
        mt, mb, mW, mZ : SM inputs (GeV), PDG 2025 defaults
        """

        # --- Store Lagrangian couplings ---
        self.lH   = lH
        self.lHS  = lHS
        self.muHS = muHS
        self.lS   = lS
        self.mu3  = mu3

        # --- Store vevs ---
        self.v_os = v
        self.u_os = u

        # --- SM gauge / Yukawa couplings ---
        self.g2 = 2.0 * mW / v
        self.g1 = 2.0 * np.sqrt(abs(mZ**2 - mW**2)) / v
        self.yt = np.sqrt(2.0) * mt / v
        self.yb = np.sqrt(2.0) * mb / v

        # --- Tadpole conditions: fix muH2 and muS2 so that (v, u) is
        #     an exact tree-level minimum by construction. ---
        #
        #   dV0/dh|_{v,u} = 0  =>  muH2 = lH*v^2 + lHS/2*u^2 + muHS*u
        self.muH2 = lH * v**2  +  0.5*lHS * u**2  +  muHS * u

        #   dV0/ds|_{v,u} = 0  =>  muS2 = lHS/2*v^2 + muHS*v^2/(2u) + lS*u^2 - mu3*u
        #   (singular at u=0; handle separately below)
        if abs(u) > 1e-6:
            self.muS2 = (0.5*lHS * v**2
                         + muHS * v**2 / (2.0*u)
                         + lS * u**2
                         - mu3 * u)
        else:
            # u -> 0: singlet tadpole reduces to  -muS2*s = 0  trivially;
            # muS2 is a free parameter - set to a positive mass^2 by default
            # so the singlet does not condense.
            self.muS2 = abs(lS) * v**2   # safe positive default

        # --- CosmoTransitions bookkeeping ---
        self.Ndim          = 2
        self.x_eps         = 0.001
        self.T_eps         = 0.001
        self.renormScaleSq = v**2

        # --- Precompute OS counterterms at T = 0 ---
        self._compute_OS_counterterms()

    # ================================================================== #
    #  Tree-level potential                                               #
    # ================================================================== #

    def V0(self, X):
        X = np.asarray(X, dtype=float)
        h = X[..., 0]
        s = X[..., 1]
        return (  self.lH   / 4.0 * h**4
                - self.muH2 / 2.0 * h**2
                + self.lHS  / 4.0 * h**2 * s**2
                + self.muHS / 2.0 * h**2 * s
                + self.lS   / 4.0 * s**4
                - self.mu3  / 3.0 * s**3
                - self.muS2 / 2.0 * s**2  )

    # ================================================================== #
    #  Field-dependent boson masses (Arnold-Espinosa daisy built in)     #
    # ================================================================== #

    def boson_massSq(self, X, T):
        """
        Field- and T-dependent squared boson masses.

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

        Daisy (Arnold-Espinosa)
        -----------------------
        Scalar Debye masses Pi_H, Pi_S added to the diagonal of the 2x2
        scalar matrix before diagonalization (correct AE prescription).
        Goldstone gets Pi_H.  Longitudinal gauge bosons get Pi_WL, Pi_ZL.
        Transverse gauge bosons: no thermal correction.
        """
        X = np.asarray(X, dtype=float)
        h = X[..., 0]
        s = X[..., 1]
        T = np.asarray(T, dtype=float)

        # ---- Zero-T scalar mass matrix  d^2V0/dfield_i dfield_j ----
        Mhh = 3.0*self.lH*h**2 - self.muH2 + 0.5*self.lHS*s**2 + self.muHS*s
        Mss = 0.5*self.lHS*h**2 + 3.0*self.lS*s**2 - 2.0*self.mu3*s - self.muS2
        Mhs = self.lHS*h*s + self.muHS*h

        # ---- Goldstone (neutral + charged, all degenerate at tree level) ----
        mGsq = self.lH*h**2 - self.muH2 + 0.5*self.lHS*s**2 + self.muHS*s

        # ---- Gauge bosons ----
        mWsq = 0.25 * self.g2**2 * h**2
        mZsq = 0.25 * (self.g1**2 + self.g2**2) * h**2

        # ---- Thermal Debye masses (leading order) ----
        # Scalar sector
        PiH = T**2 / 48.0 * (9.0*self.g2**2 + 3.0*self.g1**2
                              + 24.0*self.lH + 4.0*self.lHS
                              + 12.0*self.yt**2 + 12.0*self.yb**2)
        PiS = T**2 / 12.0 * (2.0*self.lHS + 3.0*self.lS)

        # Gauge longitudinal
        PiWL = (11.0/6.0) * self.g2**2 * T**2
        PiZL = (11.0/6.0) * (self.g1**2 + self.g2**2) * T**2

        # ---- Scalar eigenvalues with daisy-improved diagonal ----
        Mhh_T = Mhh + PiH
        Mss_T = Mss + PiS
        avg   = 0.5 * (Mhh_T + Mss_T)
        delta = np.sqrt(np.maximum(0.25*(Mhh_T - Mss_T)**2 + Mhs**2, 0.0))
        m1sq  = avg + delta
        m2sq  = avg - delta

        mGsq_T = mGsq + PiH
        mWLsq  = mWsq + PiWL
        mZLsq  = mZsq + PiZL

        masses = np.stack([m1sq, m2sq, mGsq_T,
                           mWsq,  mZsq,
                           mWLsq, mZLsq], axis=-1)
        dofs   = np.array([1.,  1.,  3.,
                           4.,  2.,
                           2.,  1.])
        cs     = np.array([3/2, 3/2, 3/2,
                           5/6, 5/6,
                           3/2, 3/2])
        return masses, dofs, cs

    # ================================================================== #
    #  Fermion masses                                                     #
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
    #  On-Shell counterterms - analytic                                  #
    # ================================================================== #

    def _V_CW_at(self, h_val, s_val):
        """V_CW (= V1 in CosmoTransitions) at a single point, T=0."""
        X        = np.array([[h_val, s_val]])
        bosons   = self.boson_massSq(X, 0.0)
        fermions = self.fermion_massSq(X)
        return self.V1(bosons, fermions).item()

    def _compute_OS_counterterms(self):
        """
        Compute OS counterterm coefficients analytically.

        All six coefficients are closed-form rational functions of the
        V_CW derivatives at (v, u), derived by Cramer's rule on the
        three decoupled linear systems:

          E  = -V_hs / (2v)                                 [1x1]
          A,C from [[2v,4v^3],[2,12v^2]]   det = 16v^3      [2x2]
          B,D,F from [[2u,4u^3,3u^2],
                      [2,12u^2,6u],
                      [0,24u,6]]            det = -48u^3     [3x3]

        Numerical derivatives use 5-point stencils for O(eps^4) accuracy.
        """
        v    = self.v_os
        u    = self.u_os
        epsH = max(1e-3 * v, 0.1)
        epsS = max(1e-3 * abs(u), 0.1) if abs(u) > 1e-6 else 1.0

        def f(dh, ds):
            return self._V_CW_at(v + dh, u + ds)

        # ---- 5-point stencil derivatives of V_CW at (v, u) ----

        # First derivatives
        Vh  = (-f(2*epsH,0) + 8*f(epsH,0) - 8*f(-epsH,0) + f(-2*epsH,0)) / (12.0*epsH)
        Vs  = (-f(0,2*epsS) + 8*f(0,epsS) - 8*f(0,-epsS) + f(0,-2*epsS)) / (12.0*epsS)

        # Second derivatives (diagonal)
        Vhh = (-f(2*epsH,0) + 16*f(epsH,0) - 30*f(0,0)
               + 16*f(-epsH,0) - f(-2*epsH,0)) / (12.0*epsH**2)
        Vss = (-f(0,2*epsS) + 16*f(0,epsS) - 30*f(0,0)
               + 16*f(0,-epsS) - f(0,-2*epsS)) / (12.0*epsS**2)

        # Mixed second derivative
        Vhs = (  f( epsH, epsS) - f( epsH,-epsS)
               - f(-epsH, epsS) + f(-epsH,-epsS)) / (4.0*epsH*epsS)

        # Third derivative in s
        Vsss = (-f(0,2*epsS) + 2*f(0,epsS)
                - 2*f(0,-epsS) + f(0,-2*epsS)) / (2.0*epsS**3)

        # ---- Analytic Cramer solutions ----

        # [V]  off-diagonal mass  ->  E
        E = -Vhs / (2.0 * v)

        # [I,III]  h-sector  ->  A, C      (det = 16v^3)
        A = (-3.0*Vh  + Vhh*v  + 2.0*Vhs*u) / (4.0*v)
        C = (     Vh  - Vhh*v             ) / (8.0*v**3)

        # [II,IV,VI]  s-sector  ->  B, D, F      (det = -48u^3)
        if abs(u) > 1e-6:
            B = (3.0*Vhs*v - 6.0*Vs + u*(4.0*Vss - Vsss*u)) / (4.0*u)
            D = (    Vhs*v - 2.0*Vs + 2.0*Vss*u - Vsss*u**2) / (8.0*u**3)
            F = (-0.5*Vhs*v + Vs - Vss*u + Vsss*u**2/3.0)    / u**2
        else:
            # u = 0: singlet sector decoupled, only muS2 shift needed
            # B*s^2 counterterm fixes the singlet mass; D, F vanish.
            B = -Vss / 2.0
            D = 0.0
            F = 0.0

        # Store coefficients with physical labels for transparency
        self._ct_A = A   # delta(muH2)   h^2  term
        self._ct_B = B   # delta(muS2)   s^2  term
        self._ct_C = C   # delta(lH)     h^4  term
        self._ct_D = D   # delta(lS)     s^4  term
        self._ct_E = E   # delta(muHS)   h^2*s  term  [Z2-breaking]
        self._ct_F = F   # delta(mu3)    s^3  term    [Z2-breaking]

    def counterterm(self, X):
        """
        delta_V = A*h^2 + B*s^2 + C*h^4 + D*s^4 + E*h^2*s + F*s^3

        Maps onto shifts of all six Lagrangian mass/coupling parameters.
        """
        X = np.asarray(X, dtype=float)
        h = X[..., 0]
        s = X[..., 1]
        return (  self._ct_A * h**2
                + self._ct_B * s**2
                + self._ct_C * h**4
                + self._ct_D * s**4
                + self._ct_E * h**2 * s
                + self._ct_F * s**3  )

    # ================================================================== #
    #  Full effective potential                                           #
    # ================================================================== #

    def Vtot(self, X, T, include_radiation=True):
        """V_eff = V0 + V_CW + delta_V_OS + V_thermal."""
        return super().Vtot(X, T, include_radiation) + self.counterterm(X)

    # ================================================================== #
    #  Phase-tracker helpers                                              #
    # ================================================================== #

    def approxZeroTMin(self):
        return [np.array([self.v_os, self.u_os])]

    def approxFiniteTMin(self):
        return [np.array([0.0, 0.0])]

    def forbidPhaseCrit(self, X):
        return (np.array([X])[..., 0] < -5.0).any()

    # ================================================================== #
    #  Diagnostics                                                        #
    # ================================================================== #

    def print_params(self):
        """Print all Lagrangian and derived parameters."""
        print("="*52)
        print("  xSM Lagrangian parameters")
        print("="*52)
        print(f"  lH    = {self.lH:.6f}")
        print(f"  lHS   = {self.lHS:.6f}")
        print(f"  muHS  = {self.muHS:.4f}  GeV")
        print(f"  lS    = {self.lS:.6f}")
        print(f"  mu3   = {self.mu3:.4f}  GeV")
        print(f"  v     = {self.v_os:.4f}  GeV  (input)")
        print(f"  u     = {self.u_os:.4f}  GeV  (input)")
        print("  --- derived from tadpoles ---")
        print(f"  muH2  = {self.muH2:.4f}  GeV^2")
        print(f"  muS2  = {self.muS2:.4f}  GeV^2")
        print("  --- OS counterterm coefficients ---")
        print(f"  A (d muH2) = {self._ct_A:+.4f}  GeV^2")
        print(f"  B (d muS2) = {self._ct_B:+.4f}  GeV^2")
        print(f"  C (d lH)   = {self._ct_C:+.6f}")
        print(f"  D (d lS)   = {self._ct_D:+.6f}")
        print(f"  E (d muHS) = {self._ct_E:+.4f}  GeV")
        print(f"  F (d mu3)  = {self._ct_F:+.4f}  GeV")
        print("="*52)

    def check_vacuum(self, verbose=True):
        """
        Verify OS conditions at (v, u) for V_eff at T=0.
        Tadpoles should be ~0; masses should match tree-level M^2 matrix.
        """
        v, u  = self.v_os, self.u_os
        epsH  = max(1e-4 * v, 0.01)
        epsS  = max(1e-4 * abs(u), 0.01) if abs(u) > 1e-6 else 1.0

        def Vf(h, s): return self.Vtot(np.array([[h, s]]), 0.0).item()
        def f(dh, ds): return Vf(v+dh, u+ds)

        dVh  = (-f(2*epsH,0)+8*f(epsH,0)-8*f(-epsH,0)+f(-2*epsH,0))  / (12.0*epsH)
        dVs  = (-f(0,2*epsS)+8*f(0,epsS)-8*f(0,-epsS)+f(0,-2*epsS))  / (12.0*epsS)
        d2Vh = (-f(2*epsH,0)+16*f(epsH,0)-30*f(0,0)+16*f(-epsH,0)-f(-2*epsH,0)) / (12.0*epsH**2)
        d2Vs = (-f(0,2*epsS)+16*f(0,epsS)-30*f(0,0)+16*f(0,-epsS)-f(0,-2*epsS)) / (12.0*epsS**2)
        d2Vhs= (f(epsH,epsS)-f(epsH,-epsS)-f(-epsH,epsS)+f(-epsH,-epsS)) / (4.0*epsH*epsS)

        # Tree-level mass matrix elements at (v, u)
        Mhh = 2.0*self.lH*v**2
        Mss = (0.5*self.lHS*v**2 + 3.0*self.lS*u**2
               - 2.0*self.mu3*u - self.muS2)
        Mhs = (self.lHS*v*u + self.muHS*v)

        if verbose:
            print("="*56)
            print(f"  OS check at (v,u) = ({v:.2f}, {u:.2f}) GeV")
            print("="*56)
            print(f"  dV/dh    = {dVh:+.3e}  (target: 0)")
            print(f"  dV/ds    = {dVs:+.3e}  (target: 0)")
            print(f"  d2V/dh2  = {d2Vh:+.5g}  (tree M_hh = {Mhh:+.5g})")
            print(f"  d2V/ds2  = {d2Vs:+.5g}  (tree M_ss = {Mss:+.5g})")
            print(f"  d2V/dhds = {d2Vhs:+.5g}  (tree M_hs = {Mhs:+.5g})")
            print("="*56)

        return dict(dVh=dVh, dVs=dVs, d2Vh=d2Vh, d2Vs=d2Vs, d2Vhs=d2Vhs)


# ================================================================== #
#  Tests                                                             #
# ================================================================== #
if __name__ == "__main__":

    print("\n" + "="*56)
    print("  TEST 1: SM limit  (u=0, singlet decoupled)")
    print("="*56)
    mh = 125.20;  v = 246.22
    sm = xSM(lH   = mh**2 / (2.0*v**2),
             lHS  = 0.0,
             muHS = 0.0,
             lS   = 0.0,
             mu3  = 0.0,
             v    = v,
             u    = 0.0)
    sm.check_vacuum()
    sm.print_params()

    print("\n" + "="*56)
    print("  TEST 2: xSM benchmark  (u=40 GeV, Z2-breaking)")
    print("="*56)
    bsm = xSM(lH   = 0.13376,
              lHS  = 0.25,
              muHS = -24.6,
              lS   = 0.5,
              mu3  = -480.4,
              v    = 246.22,
              u    = 40.0)
    bsm.check_vacuum()
    bsm.print_params()
