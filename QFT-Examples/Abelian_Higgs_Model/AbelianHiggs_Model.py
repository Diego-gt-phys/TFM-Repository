"""
Abelian Higgs Model for CosmoTransitions, now with On-Shell Counterterms.

Created on 13/08/2026 at 20:25

@author: Diego García Tejada
"""
import numpy as np
from cosmoTransitions import generic_potential

class Abelian_Model(generic_potential.generic_potential):

    def init(self, lam, mu, g):
            """
            Parameters (all in GeV):
              lam : quartic scalar coupling
              mu  : scalar mass parameter  (mu^2 > 0)
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

    # ================================================================== #
    #  Tree-level potential                                              #
    # ================================================================== #
    def V0(self, X):
        X   = np.asarray(X, dtype=float)
        phi = X[..., 0]
        return -0.5 * (self.mu)**2 * phi**2 + 0.25 * (self.lam) * phi**4

    # ================================================================== #
    #  Field-dependent boson masses (Parwani Ressumation)                #
    # ================================================================== #
    def boson_massSq(self, X, T):
        X   = np.asarray(X, dtype=float)
        phi = X[..., 0]
        T   = np.asarray(T, dtype=float)

        # --- Scalar masses ---
        mHsq  = -self.mu**2 + 3.0 * self.lam * phi**2   # Higgs (radial)
        mGsq  = -self.mu**2 +       self.lam * phi**2   # Goldstone

        # --- Gauge boson masses ---
        mATsq =  self.g**2 * phi**2                     # transverse (2 dof)
        mALsq =  self.g**2 * phi**2                     # longitudinal (1 dof), Debye-shifted below

        # --- Thermal (Debye) corrections — always added, zero at T=0 ---
        # Scalar self-energy:    Pi_s = (4*lam + 3*g^2) * T^2 / 12
        Pi_s  = (4.0 * self.lam + 3.0 * self.g**2) * T**2 / 12.0
        # Longitudinal gauge:    Pi_A = g^2 * T^2 / 3
        Pi_A  =  self.g**2 * T**2 / 3.0

        mHsq  = mHsq  + Pi_s    # Higgs gets thermal correction
        mGsq  = mGsq  + Pi_s    # Goldstone gets thermal correction
        mALsq = mALsq + Pi_A    # longitudinal gauge gets Debye mass

        #                        H      G      AT     AL
        masses = np.stack([mHsq, mGsq, mATsq, mALsq], axis=-1)
        dofs   = np.array([1.,   1.,   2.,    1.  ])
        cs     = np.array([1.5,  1.5,  0.5,   1.5 ])

        return masses, dofs, cs

    # ================================================================== #
    #  Fermion Masses - None in this model                               #
    # ================================================================== #
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

    # ================================================================== #
    #  On-Shell counterterms                                             #
    # ================================================================== #
    def V_CW(self, phi_val):
        """V_CW evaluated at a single field value phi_val, T = 0."""
        X       = np.array([[phi_val]])
        bosons  = self.boson_massSq(X, 0.0)
        fermions = self.fermion_massSq(X)
        return self.V1(bosons, fermions).item()

    def _compute_OS_counterterms(self):
        v = self.mu / np.sqrt(self.lam)
        eps = 1e-4 * v  # Step size acts as a safe IR regulator for the Goldstone mode
        
        # Numerical first and second derivatives via central differences
        dVCW  = (self.V_CW(v + eps) - self.V_CW(v - eps)) \
                    / (2.0 * eps)
        d2VCW = (self.V_CW(v + eps)
                     - 2.0 * self.V_CW(v)
                     + self.V_CW(v - eps)) / eps**2
        
        # Analytically solved OS counterterms
        self.dlam = (dVCW / v - d2VCW) / (2.0 * v**2)
        self.dm2  = 0.5 * d2VCW - 1.5 * dVCW / v
        self.dVac = (5 * v * dVCW - v**2 * d2VCW - 8*self.V_CW(v))/(8)

    def counterterm(self, X):
        """
        Finite OS counterterm potential delta_V(phi).
        Enforces V_CW'(v) = V_CW''(v) = 0 at T = 0.
        """
        X   = np.asarray(X, dtype=float)
        phi = X[..., 0]
        return 0.5 * self.dm2 * phi**2 + 0.25 * self.dlam * phi**4 + self.dVac

    # ================================================================== #
    #  Full effective potential                                          #
    # ================================================================== #

    def Vtot_no_CT(self, X, T, include_radiation=True):
        """V0 + V_CW (+ thermal if requested), without the finite counterterm."""
        return super().Vtot(X, T, include_radiation)
    
    def Vtot(self, X, T, include_radiation=True):
        """V_eff = V0 + V_CW + delta_V_OS + V_thermal."""
        return super().Vtot(X, T, include_radiation) + self.counterterm(X)
        
    # ================================================================== #
    #  Phase-tracker helpers                                              #
    # ================================================================== #
    
    def forbidPhaseCrit(self, X):
        return (np.array([X])[..., 0] < -5.0).any()
    
    def approxZeroTMin(self):
        v = self.mu / np.sqrt(self.lam)
        return [np.array([v])]
    
    def approxFiniteTMin(self):
        return [np.array([0.0])]
