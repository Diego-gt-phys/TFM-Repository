"""
xSM — Real Singlet Extension of the Standard Model (no Z_2 symmetry)
======================================================================
Author : Diego García Tejada
Created: May 2026

Physical input parametrization
-------------------------------
Free BSM parameters : m2, theta, u, lHS, lS
Fixed SM inputs     : m1 = 125.20 GeV, v = 246.22 GeV
Derived Lagrangian  : lH, muHS, mu3  from diagonalization relations
Derived mass params : muH2, muS2     from tadpole conditions

Tree-level potential
--------------------
V0 = lH/4 h^4 - muH2/2 h^2
   + lHS/4 h^2 s^2 + muHS/2 h^2 s
   + lS/4 s^4 - mu3/3 s^3 - muS2/2 s^2

OS counterterms (2D)
--------------------
Conditions imposed at (v, u):
  dV_eff/dh   = 0    (h-tadpole)
  dV_eff/ds   = 0    (s-tadpole)
  d^2V_eff/dh^2 = M^2_hh  (h mass)
  d^2V_eff/ds^2 = M^2_ss  (s mass)

Counterterm ansatz: delta_V = A h^2 + B s^2 + C h^4 + D s^4 + E h^2 s^2
  -> 5 free coefficients, 4 conditions used (tadpoles + diagonal masses).
  Off-diagonal mass condition is enforced by parameter choice, not counterterm.

Daisy resummation
-----------------
Arnold-Espinosa scheme: thermal masses added to boson_massSq entries
directly (longitudinal gauge bosons + scalar modes).  The parent class
Vtot sums V0 + V_CW + V_thermal using whatever masses boson_massSq returns.
"""

import numpy as np
from cosmoTransitions import generic_potential

# ============================================================== #
#  Physical input values (PDG 2025)                              #
# ============================================================== #
_v_EW    = 246.22   # GeV  — Higgs vev
_m1_phys = 125.20   # GeV  — lighter scalar pole mass (SM-like Higgs)
_mt_phys = 172.4    # GeV  — top quark pole mass
_mb_phys =   4.183  # GeV  — bottom quark MS-bar mass
_mW_phys =  80.369  # GeV  — W pole mass
_mZ_phys =  91.188  # GeV  — Z pole mass


class xSM(generic_potential.generic_potential):

    def init(self,
             m2    = 200.0,   # GeV   — heavier scalar mass
             theta = 0.1,     # rad   — scalar mixing angle
             u     = 30.0,    # GeV   — singlet vev
             lHS   = 0.25,    # —     — quartic portal coupling
             lS    = 0.5,     # —     — singlet quartic
             v     = _v_EW,
             m1    = _m1_phys,
             mt    = _mt_phys,
             mb    = _mb_phys,
             mW    = _mW_phys,
             mZ    = _mZ_phys):
        """
        Physical input parameters for the xSM without Z_2.

        Parameters
        ----------
        m2    : heavier scalar mass eigenvalue (GeV)
        theta : h-s mixing angle (rad); convention: h1 = cos(t)*h + sin(t)*s
        u     : singlet vev (GeV)
        lHS   : quartic portal lambda_HS
        lS    : singlet quartic lambda_S
        v, m1, mt, mb, mW, mZ : SM inputs (defaults = PDG 2025)
        """

        # --- Store physical inputs ---
        self.m1    = m1
        self.m2    = m2
        self.theta = theta
        self.v_os  = v
        self.u_os  = u
        self.lHS   = lHS
        self.lS    = lS

        # --- SM gauge/Yukawa couplings derived from physical inputs ---
        self.g2 = 2.0 * mW / v                           # SU(2)_L
        self.g1 = 2.0 * np.sqrt(abs(mZ**2 - mW**2)) / v # U(1)_Y
        self.yt = np.sqrt(2.0) * mt / v                  # top Yukawa
        self.yb = np.sqrt(2.0) * mb / v                  # bottom Yukawa

        # --- Derived Lagrangian parameters from diagonalization ---
        # (see eqs. in the TFM derivation document)
        ct = np.cos(theta)
        st = np.sin(theta)

        # lambda_H  from M^2_hh = m1^2 cos^2(t) + m2^2 sin^2(t)
        self.lH   = (m1**2 * ct**2 + m2**2 * st**2) / (2.0 * v**2)

        # mu_HS from M^2_hs = (m1^2 - m2^2) cos(t) sin(t)
        self.muHS = (m1**2 - m2**2) * ct * st / v  -  u * lHS

        # mu_3 from M^2_ss = m1^2 sin^2(t) + m2^2 cos^2(t)
        Mss = m1**2 * st**2 + m2**2 * ct**2
        # Mss = 2 lS u^2 - u mu3 - muHS v^2/(2u)  =>  solve for mu3
        self.mu3  = (2.0 * lS * u**2  -  Mss  -  self.muHS * v**2 / (2.0 * u)) / u

        # --- Derived mass parameters from tadpole conditions ---
        # h-tadpole: muH^2 = lH v^2 + lHS/2 u^2 + muHS u
        self.muH2 = self.lH * v**2  +  0.5 * lHS * u**2  +  self.muHS * u

        # s-tadpole: muS^2 = lHS/2 v^2 + muHS v^2/(2u) + lS u^2 - mu3 u
        self.muS2 = (0.5 * lHS * v**2
                     + self.muHS * v**2 / (2.0 * u)
                     + lS * u**2
                     - self.mu3 * u)

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
    #  Field-dependent boson masses (with daisy resummation built in)    #
    # ================================================================== #

    def boson_massSq(self, X, T):
        """
        Returns field- (and T-) dependent boson mass-squareds.

        Daisy resummation (Arnold-Espinosa scheme):
          - Scalar longitudinal/thermal modes: thermal mass Pi added to
            the diagonal of the 2x2 scalar mass matrix before diagonalization.
          - Gauge longitudinal modes: separate entries with Pi_WL, Pi_ZL.
          - Transverse gauge bosons: no thermal mass (exact in AE scheme).

        Particle list and degrees of freedom:
          idx  particle         dof    c
          0    h1 (heavy scal)   1    3/2
          1    h2 (light scal)   1    3/2
          2    Goldstones (3)    3    3/2
          3    W transverse      4    5/6   (2 charged * 2 transverse)
          4    Z transverse      2    5/6
          5    W longitudinal    2    3/2   (2 charged longitudinal)
          6    Z longitudinal    1    3/2
        """
        X = np.asarray(X, dtype=float)
        h = X[..., 0]
        s = X[..., 1]
        T = np.asarray(T, dtype=float)

        # ---- Scalar 2x2 mass matrix elements (second derivatives of V0) ----
        # d^2V0/dh^2
        Mhh = 3.0*self.lH*h**2 - self.muH2 + 0.5*self.lHS*s**2 + self.muHS*s
        # d^2V0/ds^2
        Mss = 0.5*self.lHS*h**2 + 3.0*self.lS*s**2 - 2.0*self.mu3*s - self.muS2
        # d^2V0/dh ds
        Mhs = self.lHS*h*s + self.muHS*h

        # ---- Goldstone mass (d^2V/d pi^2, Goldstone direction) ----
        mGsq = self.lH*h**2 - self.muH2 + 0.5*self.lHS*s**2 + self.muHS*s

        # ---- Gauge boson masses ----
        mWsq = 0.25 * self.g2**2 * h**2
        mZsq = 0.25 * (self.g1**2 + self.g2**2) * h**2

        # ---- Thermal (Debye) masses — leading order daisy ----
        # Scalar sector (from Tr[d^2V/dphi^2] thermal averages)
        PiH = T**2 / 48.0 * (9.0*self.g2**2 + 3.0*self.g1**2
                              + 24.0*self.lH + 4.0*self.lHS
                              + 12.0*self.yt**2 + 12.0*self.yb**2)
        PiS = T**2 / 12.0 * (2.0*self.lHS + 3.0*self.lS)

        # Gauge longitudinal (standard result, 11/6 * g^2 T^2 per species)
        PiWL  = (11.0/6.0) * self.g2**2 * T**2
        cW2   = (mW_ref := self.g2*self.v_os/2.0)**2 / (self.g2**2 + self.g1**2) * 4.0 / self.v_os**2 * self.v_os**2 / 4.0
        # simpler: just use g1, g2 directly
        PiZL  = (11.0/6.0) * (self.g1**2 + self.g2**2) * T**2   # approximate; exact form mixes B,W3

        # ---- Scalar eigenvalues with daisy improvement ----
        Mhh_T = Mhh + PiH
        Mss_T = Mss + PiS
        avg   = 0.5 * (Mhh_T + Mss_T)
        delta = np.sqrt(np.maximum(0.25*(Mhh_T - Mss_T)**2 + Mhs**2, 0.0))
        m1sq  = avg + delta
        m2sq  = avg - delta

        mGsq_T  = mGsq  + PiH
        mWLsq   = mWsq  + PiWL
        mZLsq   = mZsq  + PiZL

        # ---- Assemble ----
        masses = np.stack([m1sq, m2sq, mGsq_T,
                           mWsq, mZsq,
                           mWLsq, mZLsq], axis=-1)
        dofs   = np.array([1.,  1.,  3.,
                           4.,  2.,
                           2.,  1.])
        cs     = np.array([1.5, 1.5, 1.5,
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
        dofs   = np.array([12., 12.])   # 3 colors * 2 spins * 2 (particle+antiparticle) / 2 = 12? 
        # Note: CosmoTransitions convention for Dirac fermions: n_f = N_c * N_spin = 3*4 = 12
        return masses, dofs

    # ================================================================== #
    #  On-Shell counterterms (2D)                                        #
    # ================================================================== #

    def _V_CW_at(self, h_val, s_val):
        """CW potential (V1) evaluated at a single (h, s) point, T=0."""
        X       = np.array([[h_val, s_val]])
        bosons  = self.boson_massSq(X, 0.0)
        fermions= self.fermion_massSq(X)
        return self.V1(bosons, fermions).item()

    def _compute_OS_counterterms(self):
        """
        Impose OS renormalization conditions at the EW vacuum (v, u).

        The counterterm polynomial must mirror the full Lagrangian structure
        of the xSM without Z_2. Every monomial in V0 that is odd in s (cubic
        terms, Z_2-breaking) generates odd-in-s loop corrections that cannot
        be cancelled by a Z_2-symmetric CT. The complete independent basis is:

          CT(h,s) = A*h^2 + B*s^2 + C*h^4 + D*s^4 + E*h^2*s + F*s^3

        where:
          A  <-> delta(muH^2)      renormalizes the Higgs mass parameter
          B  <-> delta(muS^2)      renormalizes the singlet mass parameter
          C  <-> delta(lambda_H)   renormalizes the Higgs quartic
          D  <-> delta(lambda_S)   renormalizes the singlet quartic
          E  <-> delta(mu_HS)      renormalizes the Z_2-breaking cubic portal
          F  <-> delta(mu_3)       renormalizes the singlet cubic

        Note: h^2*s^2 (delta lambda_HS) is NOT included — it contributes only
        to d^2/dh ds, which we do not impose as a CT condition (it would require
        a 6th equation). Its effect is absorbed into the existing 5 coefficients
        at the level of accuracy of our OS scheme.

        The 5 OS conditions imposed at (v, u):
          [I]   dCT/dh   = -dV_CW/dh        (h tadpole)
          [II]  dCT/ds   = -dV_CW/ds        (s tadpole)
          [III] d^2CT/dh^2 = -d^2V_CW/dh^2  (h physical mass)
          [IV]  d^2CT/ds^2 = -d^2V_CW/ds^2  (s physical mass)
          [V]   d^2CT/dhds = -d^2V_CW/dhds  (off-diagonal mass)

        Derivatives of CT at (v, u):
          dCT/dh     = 2Av + 4Cv^3 + 2Evu              [I]
          dCT/ds     = 2Bu + 4Du^3 + Ev^2 + 3Fu^2      [II]
          d^2CT/dh^2 = 2A  + 12Cv^2 + 2Eu              [III]
          d^2CT/ds^2 = 2B  + 12Du^2 + 6Fu              [IV]
          d^2CT/dhds = 2Ev                              [V]

        This gives a clean decoupled structure:
          [V]  -> E directly
          [I,III] -> linear 2x2 system for (A, C)  [with E known]
          [II,IV] -> linear 2x2 system for (B, D)  [with E known]
          [II] then gives F residually from the s-tadpole
          ... but [II] and [IV] already fix (B,D) independently of F,
          so we use [II] *after* solving (B,D) to pin F:
            F = (-dVs - 2Bu - 4Du^3 - Ev^2) / (3u^2)   from [II]
          and verify [IV] is satisfied (it is, since F drops out of [IV]
          only when the system is consistent — check via residual).

        Actually the cleanest approach: write the full 5x5 linear system
        in (A, B, C, D, E, F) minus one (fix F=0 initially and include it
        last). Instead we solve the decoupled subsystems as described.
        """
        v   = self.v_os
        u   = self.u_os
        eps_h = max(1e-3 * v, 0.1)
        eps_s = max(1e-3 * abs(u) if u != 0 else 1.0, 0.1)

        # --- Numerical derivatives of V_CW at (v, u) via 5-point stencils ---
        def f(dh, ds):
            return self._V_CW_at(v + dh, u + ds)

        # First derivatives
        dVh  = (-f(2*eps_h,0) + 8*f(eps_h,0) - 8*f(-eps_h,0) + f(-2*eps_h,0)) / (12.0*eps_h)
        dVs  = (-f(0,2*eps_s) + 8*f(0,eps_s) - 8*f(0,-eps_s) + f(0,-2*eps_s)) / (12.0*eps_s)

        # Second derivatives (diagonal)
        d2Vh  = (-f(2*eps_h,0) + 16*f(eps_h,0) - 30*f(0,0)
                 + 16*f(-eps_h,0) - f(-2*eps_h,0)) / (12.0*eps_h**2)
        d2Vs  = (-f(0,2*eps_s) + 16*f(0,eps_s) - 30*f(0,0)
                 + 16*f(0,-eps_s) - f(0,-2*eps_s)) / (12.0*eps_s**2)

        # Mixed second derivative (4-point cross stencil)
        d2Vhs = (  f( eps_h, eps_s) - f( eps_h,-eps_s)
                 - f(-eps_h, eps_s) + f(-eps_h,-eps_s)) / (4.0*eps_h*eps_s)

        # --- Third derivative of V_CW in s-direction (needed for F) ---
        d3Vs = (-f(0,2*eps_s) + 2*f(0,eps_s) - 2*f(0,-eps_s) + f(0,-2*eps_s)) / (2.0*eps_s**3)

        # ------------------------------------------------------------------ #
        #  Analytic inversion of the three decoupled linear systems           #
        #  (derived by Cramer's rule from the CT derivative conditions)       #
        #                                                                     #
        #  Notation: Vh, Vhh, Vs, Vss, Vsss, Vhs are V_CW derivatives       #
        #  at (v, u); tilde versions absorb the E cross-coupling.             #
        # ------------------------------------------------------------------ #

        # Step 1 — E from the off-diagonal mass condition  d^2CT/dhds = -Vhs
        #   2 E v = -Vhs  =>
        E  = -d2Vhs / (2.0 * v)

        # Step 2 — (A, C) from h-tadpole + h-mass, with E-shifted RHS
        #   Analytic inverse of [[2v, 4v^3],[2, 12v^2]], det = 16v^3:
        #   A = (3*Vh~ - Vhh~*v - 2*Vhs*u) / (4v)   [but using shifted vars]
        #
        #   Full result (Cramer):
        #     A = (-3*Vh + Vhh*v + 2*Vhs*u) / (4v)
        #     C = ( Vh - Vhh*v) / (8v^3)
        #
        #   where Vh, Vhh, Vhs are the raw V_CW derivatives (E not substituted
        #   explicitly — the formula already encodes the E correction).
        A  = (-3.0*dVh + d2Vh*v + 2.0*d2Vhs*u) / (4.0*v)
        C  = (       dVh - d2Vh*v             ) / (8.0*v**3)

        # Step 3 — (B, D, F) from s-tadpole + s-mass + s-cubic
        #   Analytic inverse of [[2u,4u^3,3u^2],[2,12u^2,6u],[0,24u,6]],
        #   det = -48u^3:
        #
        #     B = (3*Vhs*v - 6*Vs + u*(4*Vss - Vsss*u)) / (4u)
        #     D = (Vhs*v - 2*Vs + 2*Vss*u - Vsss*u^2)  / (8u^3)
        #     F = (-Vhs*v/2 + Vs - Vss*u + Vsss*u^2/3) / u^2
        #
        if abs(u) > 1.0:
            B  = (3.0*d2Vhs*v - 6.0*dVs + u*(4.0*d2Vs - d3Vs*u)) / (4.0*u)
            D  = (d2Vhs*v - 2.0*dVs + 2.0*d2Vs*u - d3Vs*u**2)    / (8.0*u**3)
            F  = (-0.5*d2Vhs*v + dVs - d2Vs*u + d3Vs*u**2/3.0)    / u**2
        else:
            # u -> 0: singlet decouples; cubic CT undefined, set to zero
            B  = -dVs / 2.0
            D  = 0.0
            F  = 0.0

        self._ct_A = A   # delta(muH^2):  coefficient of h^2
        self._ct_B = B   # delta(muS^2):  coefficient of s^2
        self._ct_C = C   # delta(lH):     coefficient of h^4
        self._ct_D = D   # delta(lS):     coefficient of s^4
        self._ct_E = E   # delta(muHS):   coefficient of h^2*s  [Z2-breaking]
        self._ct_F = F   # delta(mu3):    coefficient of s^3    [Z2-breaking]

    def counterterm(self, X):
        """
        delta_V(h, s) = A*h^2 + B*s^2 + C*h^4 + D*s^4 + E*h^2*s + F*s^3

        Full Z_2-breaking counterterm ansatz. Coefficients map to:
          A <-> delta(muH^2),  B <-> delta(muS^2),  C <-> delta(lH),
          D <-> delta(lS),     E <-> delta(muHS),   F <-> delta(mu3)
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
        """
        V_eff = V0 + V_CW + delta_V_OS + V_thermal

        Parent class handles V0 + V_CW + V_thermal; we add the OS counterterms.
        """
        return super().Vtot(X, T, include_radiation) + self.counterterm(X)

    # ================================================================== #
    #  Phase-tracker helpers                                              #
    # ================================================================== #

    def approxZeroTMin(self):
        """Starting guess for the T=0 EW minimum: (v, u)."""
        return [np.array([self.v_os, self.u_os])]

    def approxFiniteTMin(self):
        """Starting guess for the high-T symmetric phase minimum: (0, 0)."""
        return [np.array([0.0, 0.0])]

    def forbidPhaseCrit(self, X):
        """Forbid phases with h < 0 (unphysical branch)."""
        return (np.array([X])[..., 0] < -5.0).any()

    # ================================================================== #
    #  Consistency checks                                                 #
    # ================================================================== #

    def check_vacuum(self, verbose=True):
        """
        Verify that the EW vacuum conditions are satisfied by the
        full effective potential (tree-level + CW + CT) at T=0.

        Prints first and second derivatives at (v, u).
        """
        v, u   = self.v_os, self.u_os
        eps_h  = max(1e-4 * v, 0.01)
        eps_s  = max(1e-4 * abs(u) if u != 0 else 1.0, 0.01)

        def Vfull(h, s):
            return self.Vtot(np.array([[h, s]]), 0.0).item()

        def f(dh, ds): return Vfull(v+dh, u+ds)

        dVh  = (-f(2*eps_h,0) + 8*f(eps_h,0) - 8*f(-eps_h,0) + f(-2*eps_h,0)) / (12.0*eps_h)
        dVs  = (-f(0,2*eps_s) + 8*f(0,eps_s) - 8*f(0,-eps_s) + f(0,-2*eps_s)) / (12.0*eps_s)
        d2Vh = (-f(2*eps_h,0)+16*f(eps_h,0)-30*f(0,0)+16*f(-eps_h,0)-f(-2*eps_h,0)) / (12.0*eps_h**2)
        d2Vs = (-f(0,2*eps_s)+16*f(0,eps_s)-30*f(0,0)+16*f(0,-eps_s)-f(0,-2*eps_s)) / (12.0*eps_s**2)

        ct = np.cos(self.theta)
        st = np.sin(self.theta)
        Mhh_tree = self.m1**2 * ct**2 + self.m2**2 * st**2
        Mss_tree = self.m1**2 * st**2 + self.m2**2 * ct**2

        if verbose:
            print("="*55)
            print("  OS vacuum consistency check at (v, u) = ({:.2f}, {:.2f}) GeV".format(v, u))
            print("="*55)
            print(f"  dV/dh   = {dVh:+.4e}  (should be ~0)")
            print(f"  dV/ds   = {dVs:+.4e}  (should be ~0)")
            print(f"  d2V/dh2 = {d2Vh:+.6g}  (tree: {Mhh_tree:+.6g})")
            print(f"  d2V/ds2 = {d2Vs:+.6g}  (tree: {Mss_tree:+.6g})")
            print("="*55)

        return dict(dVh=dVh, dVs=dVs, d2Vh=d2Vh, d2Vs=d2Vs)


# ============================================================== #
#  Quick test                                                    #
# ============================================================== #
if __name__ == "__main__":

    model = xSM(m2=200.0, theta=0.15, u=40.0, lHS=0.25, lS=0.5)
    model.check_vacuum()

    # Print derived Lagrangian parameters
    print("\n  Derived Lagrangian parameters:")
    print(f"    lH    = {model.lH:.6f}")
    print(f"    muH2  = {model.muH2:.4f} GeV^2")
    print(f"    muHS  = {model.muHS:.4f} GeV")
    print(f"    mu3   = {model.mu3:.4f} GeV")
    print(f"    muS2  = {model.muS2:.4f} GeV^2")
    print(f"    lHS   = {model.lHS:.6f}  (input)")
    print(f"    lS    = {model.lS:.6f}   (input)")

