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
            PiH = (3*self.lH + self.lHS/2 + self.g**2*(3/16 + 1/16)
                   + self.yt**2/4) * T**2 / 3.0   # rough Higgs Debye
            PiS = (self.lS + self.lHS/6) * T**2   # singlet Debye
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
        mtsq = np.stack([mtsq], axis=-1)   # shape (..., 1)
        return mtsq, np.array([12])
    
# ---- Quick test ----
"""
import matplotlib.pyplot as plt

# Benchmark point (not physical yet, just to test)
model = xSM(lH=0.13, muH=88.4, lHS=0.3, muHS=10.0,
            lS=0.5,  mu3=30.0, muS=50.0)

# Evaluate V0 along the h axis with s=0
h_vals = np.linspace(0, 300, 500)
s_vals = np.zeros_like(h_vals)
X = np.column_stack([h_vals, s_vals])

V_vals = model.V0(X)

plt.figure()
plt.plot(h_vals, V_vals)
plt.xlabel("h (GeV)")
plt.ylabel("V0 (GeV⁴)")
plt.title("Tree-level potential along h axis (s=0)")
plt.axhline(0, color='k', linewidth=0.5)
plt.show()
"""
# ---- Phase tracing ----
model = xSM(lH=0.13, muH=88.4, lHS=0.25, muHS=5.0,
            lS=0.3,  mu3=20.0, muS=60.0)

model.findAllTransitions()

# ---- Extract T_c and xi_c ----
for i, trans in enumerate(model.TnTrans):
    Tc = trans['Tnuc']
    high_vev = trans['high_vev']
    low_vev  = trans['low_vev']
    
    # xi_c = |delta phi_h| / T_c
    delta_h = abs(low_vev[0] - high_vev[0])
    xi_c = delta_h / Tc
    
    print(f"Transition {i}:")
    print(f"  T_c     = {Tc:.2f} GeV")
    print(f"  high phase vev = {high_vev}")
    print(f"  low  phase vev = {low_vev}")
    print(f"  Δφ_h    = {delta_h:.2f} GeV")
    print(f"  ξ_c     = {xi_c:.4f}")
    print()
    
import matplotlib.pyplot as plt

trans = model.TnTrans[1]
Tc    = trans['Tnuc']
high  = trans['high_vev']
low   = trans['low_vev']
xi_c  = abs(low[0] - high[0]) / Tc

# --- Plot the potential at T=Tc along the tunneling path ---
T_vals  = np.linspace(80, 160, 200)
h_high  = []
h_low   = []

for T in T_vals:
    phases = model.calcTcTrans()
    break  # just need the vevs at each T — use phase tracer instead

# Plot the two minima as function of T
fig, ax = plt.subplots(figsize=(8, 6))

# Grid in (h, s)
h_vals = np.linspace(-220, 220, 300)
s_vals = np.linspace(-220, 220, 300)
H, S   = np.meshgrid(h_vals, s_vals)
X_grid = np.column_stack([H.ravel(), S.ravel()])
V_grid = model.Vtot(X_grid, Tc).reshape(H.shape)

# Clip for better contrast
V_plot = np.clip(V_grid, np.percentile(V_grid, 2), np.percentile(V_grid, 95))

cp = ax.contourf(H, S, V_plot, levels=60, cmap='RdBu_r')
ax.contour(H, S, V_plot, levels=20, colors='k', linewidths=0.3, alpha=0.4)
plt.colorbar(cp, ax=ax, label=r'$V_{\rm eff}$ (GeV$^4$)')

# Mark the two minima
from scipy.optimize import minimize

def Vtot_scalar(X, T):
    return model.Vtot(np.array([X]), T)[0]

# Polish the minima at T_c
res_high = minimize(Vtot_scalar, high, args=(Tc,), method='Nelder-Mead',
                    options={'xatol':1e-6, 'fatol':1e-6})
res_low  = minimize(Vtot_scalar, low,  args=(Tc,), method='Nelder-Mead',
                    options={'xatol':1e-6, 'fatol':1e-6})

high_min = res_high.x
low_min  = res_low.x

ax.plot(high_min[0], high_min[1], 'r*', markersize=15,
        label=f'High phase (φ_h={high_min[0]:.1f}, φ_s={high_min[1]:.1f})', zorder=5)
ax.plot(low_min[0],  low_min[1],  'b*', markersize=15,
        label=f'Low phase (φ_h={low_min[0]:.1f}, φ_s={low_min[1]:.1f})',  zorder=5)

# Update xi_c with polished values
xi_c = abs(low_min[0] - high_min[0]) / Tc
ax.text(0.02, 0.05, rf'$\xi_c = {xi_c:.3f}$', transform=ax.transAxes,
        fontsize=13, color='black',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

ax.set_xlabel(r'$\varphi_h$ (GeV)', fontsize=13)
ax.set_ylabel(r'$\varphi_s$ (GeV)', fontsize=13)
ax.set_title(rf'$V_{{\rm eff}}(\varphi_h, \varphi_s)$ at $T_c = {Tc:.1f}$ GeV', fontsize=14)
ax.legend(fontsize=11)
ax.text(0.02, 0.05, rf'$\xi_c = {xi_c:.3f}$', transform=ax.transAxes,
        fontsize=13, color='black',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig("EWPT_result.png", dpi=150)
plt.show()