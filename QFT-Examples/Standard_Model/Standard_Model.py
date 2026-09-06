"""
Standard Model class for CosmoTransitions.
It includes all the heavy DoF until the b quark.
It uses the MSbar CW and ads constant counter terms to impose the OS conditions.
The main use of this class is to compare it to the limiting case of xSM_model.py

Created on 16/08/2026 at 12:50

@author: Diego García Tejada
"""
import numpy as np
from cosmoTransitions import generic_potential

# ============================================================== #
#  Physical input values (PDG 2026)                              #
# ============================================================== #
v_EW    = 246.22 # GeV  — Higgs vev
mh_phys = 125    # GeV  — Higgs mass
mt_phys = 172    # GeV  — top quark mass
mb_phys = 4.18   # GeV  — bottom quark mass
mW_phys = 80.36  # GeV  — W mass
mZ_phys = 91.18  # GeV  — Z mass

class Standard_Model(generic_potential.generic_potential):
    
    def init(self,
             v   = v_EW,
             mh  = mh_phys,
             mt  = mt_phys,
             mb  = mb_phys,
             mW  = mW_phys,
             mZ  = mZ_phys):
        """
        Input parameters for the model. All in GeV

        Parameters
        ----------
        v : Higgs VEV
        mh : Higgs pole mass
        mt : top quark pole mass
        mb : bottom quark MS-bar mass
        mW : W pole mass
        mZ : Z pole mass
        """
        
        self.v_os  = v
        self.mh_os = mh
        self.mt_os = mt
        self.mb_os = mb
        self.mW_os = mW
        self.mZ_os = mZ
        
        # ----- derive Lagrangian parameters from physical inputs ------
        self.mu2  = 0.5 * mh**2
        self.lam  = mh**2 / (2.0 * v**2)
        self.yt   = np.sqrt(2.0) * mt / v
        self.yb   = np.sqrt(2.0) * mb / v
        self.g2   = 2.0 * mW / v
        self.g1   = 2.0 * np.sqrt(abs(mZ**2 - mW**2)) / v
        
        # ----- CosmoTransitions bookkeeping --------------------------
        self.Ndim          = 1
        self.x_eps         = 0.001
        self.T_eps         = 0.001
        self.renormScaleSq = v**2

        # ----- precompute OS counterterms at T = 0 -------------------
        self._compute_OS_counterterms()
        
    # ==================================================================
    #  Tree-level potential
    # ==================================================================

    def V0(self, X):
        X   = np.asarray(X, dtype=float)
        phi = X[..., 0]
        return -0.5 * self.mu2 * phi**2 + 0.25 * self.lam * phi**4
    
    # ==================================================================
    #  Field-dependent boson masses
    # ==================================================================
    
    def boson_massSq(self, X, T):
        X   = np.asarray(X, dtype=float)
        phi = X[..., 0]
        T   = np.asarray(T, dtype=float)
        
        g1, g2, lam, mu2 = self.g1, self.g2, self.lam, self.mu2
        
        # ---- zero-T field-dependent masses ---------------------------
        mHsq  = -mu2 + 3.0 * lam * phi**2           # Higgs
        mG0sq = -mu2 +       lam * phi**2            # neutral Goldstone
        mGcsq = -mu2 +       lam * phi**2            # charged Goldstone

        mWsq  = 0.25 * g2**2 * phi**2                # W  (both ±)
        mZsq  = 0.25 * (g1**2 + g2**2) * phi**2      # Z
        
        # ---- Thermal masses ---------------------------
        Pi_phi = T**2 * (3.0*g2**2/16.0 + g1**2/16.0 + 0.5*lam + 0.25*self.yt**2 + 0.25*self.yb**2)
        Pi_WL  = (11.0/6.0) * g2**2 * T**2
        Pi_B   = (11.0/6.0) * g1**2 * T**2

        a = 0.25*self.g2**2*phi**2 + Pi_WL
        d = 0.25*self.g1**2*phi**2 + Pi_B
        b = 0.25*self.g1*self.g2*phi**2
        disc = np.sqrt((0.5*(a-d))**2 + b**2)

        
        mHTsq  = mHsq  + Pi_phi
        mG0Tsq = mG0sq + Pi_phi
        mGcTsq = mGcsq + Pi_phi
        mWLTsq = mWsq  + Pi_WL
        mZLTsq = 0.5*(a+d) + disc
        mgLTsq = 0.5*(a+d) - disc
        
        # ---- Assemble arrays -----------------------------------------
        #                       h       G0      G+/-  WT   ZT     WL     ZL     gamL
        masses = np.stack([mHTsq, mG0Tsq, mGcTsq, mWsq,  mZsq, mWLTsq, mZLTsq, mgLTsq], axis=-1)
        dofs = np.array([1., 1., 2., 4., 2., 2., 1., 1.])
        cs   = np.array([1.5, 1.5, 1.5, 0.5, 0.5, 1.5, 1.5, 1.5])

        return masses, dofs, cs

    # ==================================================================
    #  Fermion masses
    # ==================================================================
    
    def fermion_massSq(self, X):
        X   = np.asarray(X, dtype=float)
        phi = X[..., 0]
    
        # Yukawa-generated masses
        mtsq = 0.5 * self.yt**2 * phi**2
        mbsq = 0.5 * self.yb**2 * phi**2
    
        masses = np.stack([mtsq, mbsq], axis=-1)
        dofs   = np.array([6., 6.])   # 3 colours * 2 (Dirac)
    
        return masses, dofs

    # ==================================================================
    #  On-Shell counterterms
    # ==================================================================
    
    def V_CW_scalar(self, phi_val):
        """Coleman-Weinberg potential evaluated at a scalar field value."""
        X        = np.array([[phi_val]])
        bosons   = self.boson_massSq(X, 0.0)
        fermions = self.fermion_massSq(X)
        return self.V1(bosons, fermions).item()
    
    def _compute_OS_counterterms(self):
        v   = self.v_os
        eps = 1e-4 * v   # safe step; avoids IR issues near Goldstone zero crossing

        dVCW  = (self.V_CW_scalar(v + eps) - self.V_CW_scalar(v - eps)) / (2.0 * eps)
        d2VCW = (self.V_CW_scalar(v + eps)
                 - 2.0 * self.V_CW_scalar(v)
                 + self.V_CW_scalar(v - eps)) / eps**2

        # Analytically solved OS counterterms
        self.dlam = (dVCW / v - d2VCW) / (2.0 * v**2)
        self.dm2  = 0.5 * d2VCW - 1.5 * dVCW / v
        self.dVac = (5 * v * dVCW - v**2 * d2VCW - 8*self.V_CW_scalar(v))/(8)
        
    def counterterm(self, X):
        """
        delta_V(phi) = -1/2 * dm2 * phi^2 + 1/4 * dlam * phi^4

        Note the sign convention: we ADD delta_V to V_CW so that
            V_CW(phi) + delta_V(phi)  has  (d/dphi) = 0  and  (d^2/dphi^2) = 0  at phi = v.
        """
        X   = np.asarray(X, dtype=float)
        phi = X[..., 0]
        return 0.5 * self.dm2 * phi**2 + 0.25 * self.dlam * phi**4 + self.dVac

    # ==================================================================
    #  Full effective potential (override)
    # ==================================================================

    def Vtot(self, X, T, include_radiation=True):
        """
        V_eff = V0 + V_CW + delta_V (OS c.t.) + V_thermal [+ V_daisy (AE only)]

        The parent class Vtot already computes V0 + V_CW + V_thermal.
        We add the OS counterterms and, if daisy_scheme == 'arnold_espinosa',
        the ring-improvement term.
        """
        V = super().Vtot(X, T, include_radiation) + self.counterterm(X)
        return V
    
    # ==================================================================
    #  Phase-tracker helpers
    # ==================================================================

    def forbidPhaseCrit(self, X):
        return (np.array([X])[..., 0] < -5.0).any()

    def approxZeroTMin(self):
        return [np.array([self.v_os])]

    def approxFiniteTMin(self):
        return [np.array([0.0])]