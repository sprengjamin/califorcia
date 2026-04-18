from math import sqrt, exp
from scipy.constants import c


def _combine_reflection_coefficients(r_left, r_right, kappa_layer, thickness):
    phase = exp(-2 * kappa_layer * thickness)
    rTM_left, rTE_left = r_left
    rTM_right, rTE_right = r_right
    rTM = (rTM_left + rTM_right * phase) / (1 + rTM_left * rTM_right * phase)
    rTE = (rTE_left + rTE_right * phase) / (1 + rTE_left * rTE_right * phase)
    return rTM, rTE

def def_reflection_coeff(medium, materials, thicknesses):
    """
    Define reflection coefficients of the plane by specifying medium and materials of the plane and thickness of the
    coating layers.

    Parameters
    ----------
    medium : object
    materials : list of objects
    thicknesses : list of floats

    Returns
    -------
    function(float, float)->(float, float)
        reflection coefficients taking vacuum wavenumber k0 and in-plane wave number k and returning the values of the
        reflection coefficients for TM and TE polarization, respectively

    """
    Nlayers = len(materials)
    interface_coefficients = [def_fresnel_coefficients(medium, materials[0])]
    interface_coefficients.extend(
        def_fresnel_coefficients(materials[idx], materials[idx + 1])
        for idx in range(Nlayers - 1)
    )

    if Nlayers == 1:
        return interface_coefficients[0]

    def reflection_coeff(k0, k):
        effective_reflection = interface_coefficients[-1](k0, k)

        for idx in range(Nlayers - 2, 0, -1):
            effective_reflection = _combine_reflection_coefficients(
                interface_coefficients[idx](k0, k),
                effective_reflection,
                kappa(materials[idx], k0, k),
                thicknesses[idx],
            )

        return _combine_reflection_coefficients(
            interface_coefficients[0](k0, k),
            effective_reflection,
            kappa(materials[0], k0, k),
            thicknesses[0],
        )

    return reflection_coeff

def def_longitudinal_reflection_coeff(medium, materials, thicknesses):
    """
    Define longitudinal reflection coefficients of the plane.
    Currently only implemented for zero frequency.
    """
    Nlayers = len(materials)
    interface_coefficients = [def_longitudinal_fresnel_coefficients(medium, materials[0])]
    interface_coefficients.extend(
        def_longitudinal_fresnel_coefficients(materials[idx], materials[idx + 1])
        for idx in range(Nlayers - 1)
    )

    if Nlayers == 1:
        return interface_coefficients[0]

    # For longitudinal waves, we might need a recursive relation similar to transverse.
    # However, Ref. 2 says for gold slab at n=0, r_ll = -1 independent of d.
    # We will implement the recursive relation for generality.
    
    def reflection_coeff(k0, k):
        if k0 != 0.:
            return 0. # Longitudinal channel only relevant at k0=0
            
        effective_reflection = interface_coefficients[-1](k0, k)

        for idx in range(Nlayers - 2, 0, -1):
            kappa_l = sqrt(k**2 + materials[idx].kappa_D**2) if materials[idx].materialclass == "electrolites" else k
            effective_reflection = _combine_longitudinal_reflection_coefficients(
                interface_coefficients[idx](k0, k),
                effective_reflection,
                kappa_l,
                thicknesses[idx],
            )

        kappa_l_0 = sqrt(k**2 + materials[0].kappa_D**2) if materials[0].materialclass == "electrolites" else k
        return _combine_longitudinal_reflection_coefficients(
            interface_coefficients[0](k0, k),
            effective_reflection,
            kappa_l_0,
            thicknesses[0],
        )

    return reflection_coeff

def _combine_longitudinal_reflection_coefficients(r_left, r_right, kappa_l, thickness):
    # Same form as transverse for a single mode
    phase = exp(-2 * kappa_l * thickness)
    return (r_left + r_right * phase) / (1 + r_left * r_right * phase)

def def_longitudinal_fresnel_coefficients(mat1, mat2):
    """
    Defines longitudinal Fresnel reflection coefficients for a halfspace at k0=0.
    Eq. (A5) of Phys. Rev. A 111, 012816 (2025).
    """
    def fresnel_coefficients(k0, k):
        if k0 != 0.:
            return 0.
            
        # PEC limit
        if mat1.materialclass == "pec":
            return -1.
        if mat2.materialclass == "pec":
            return 1.
            
        # Ref 2 says for gold-water interface r_ll = -1.
        # Let's use the general formula from Eq. (A5)
        # r_ll = (eps2*k3 + eps3*k2 + (k*kl/epsb)*eps2*(eps3-epsb)) / (eps2*k3 + eps3*k2 - (k*kl/epsb)*eps2*(eps3-epsb))
        # At k0=0, k3 = k2 = k.
        # r_ll = (eps2 + eps3 + (kl/epsb)*eps2*(eps3-epsb)) / (eps2 + eps3 - (kl/epsb)*eps2*(eps3-epsb))
        
        # If mat1 is electrolyte (nonlocal) and mat2 is local (dielectric/drude/plasma)
        if mat1.materialclass == "electrolites":
            eps1 = mat1.epsilon(0.)
            # Access solvent_model and kappa_D whether mat1 is a module or an object
            solvent_model = getattr(mat1, "solvent_model", getattr(getattr(mat1, "epsilon", None), "solvent_model", None))
            kappa_D = getattr(mat1, "kappa_D", getattr(getattr(mat1, "epsilon", None), "kappa_D", None))
            
            eps_b = solvent_model.epsilon(0.)
            eps_s = mat2.epsilon(0.)
            kappa_l = sqrt(k**2 + kappa_D**2)
            
            # For metal (drude/plasma), eps_s -> inf
            if mat2.materialclass in ["drude", "plasma"]:
                return -1.
                
            # If mat2 is dielectric
            num = eps_s + eps1 + (kappa_l / eps_b) * eps_s * (eps1 - eps_b)
            den = eps_s + eps1 - (kappa_l / eps_b) * eps_s * (eps1 - eps_b)
            return num / den

        # If mat1 is local and mat2 is electrolyte
        if mat2.materialclass == "electrolites":
            eps_s = mat1.epsilon(0.)
            solvent_model = getattr(mat2, "solvent_model", getattr(getattr(mat2, "epsilon", None), "solvent_model", None))
            kappa_D = getattr(mat2, "kappa_D", getattr(getattr(mat2, "epsilon", None), "kappa_D", None))
            
            eps_b = solvent_model.epsilon(0.)
            kappa_l = sqrt(k**2 + kappa_D**2)
            return (1 - (kappa_l/k)*(eps_s/eps_b)) / (1 + (kappa_l/k)*(eps_s/eps_b))

        return 0. # No longitudinal waves in local-local interface
        
    return fresnel_coefficients

def kappa(mat, k0, k):
    if k0 == 0.:
        if mat.materialclass in ["dielectric", "drude", "electrolites"]:
            return k
        elif mat.materialclass == "plasma":
            Kp = mat.wp/c
            return sqrt(Kp**2 + k**2)
    else: # k0 > 0
        return sqrt(mat.epsilon(k0*c)*k0**2 + k**2)


def def_fresnel_coefficients(mat1, mat2):
    """
    Defines Fresnel reflection coefficients for a halfspace.

    Parameters
    ----------
    mat1 : object
        material representing the medium from which the planewave is incident
    mat2 : object
        material of the halfspace

    Returns
    -------
    function(float, float)->(float, float)
        reflection coefficients taking vacuum wavenumber k0 and in-plane wave number k and returning the values of the
        reflection coefficients for TM and TE polarization, respectively
    """
    def fresnel_coefficients(k0, k):
        rTM, rTE = 0., 0.
        if mat1.materialclass == "pec":
            return -1., 1.
        if mat2.materialclass == "pec":
            return 1., -1.
        if k0 == 0.:
            # Special case for electrolytes: screening at zero frequency
            if mat1.materialclass == "electrolites" or mat2.materialclass == "electrolites":
                return -1., 0.

            if mat1.materialclass == "dielectric":
                if mat2.materialclass == "dielectric":
                    eps1 = mat1.epsilon(0.)
                    eps2 = mat2.epsilon(0.)
                    rTM = (eps2 - eps1)/(eps2 + eps1)
                    rTE = 0.
                elif mat2.materialclass == "drude":
                    rTM = 1.
                    rTE = 0.
                elif mat2.materialclass == "plasma":
                    Kp = mat2.wp/c
                    rTM = 1.
                    rTE = (k - sqrt(Kp**2 + k**2))/(k + sqrt(Kp**2 + k**2))
            elif mat1.materialclass == "drude":
                if mat2.materialclass == "dielectric":
                    rTM = -1.
                    rTE = 0.
                elif mat2.materialclass == "drude":
                    wp1 = mat1.wp
                    gamma1 = mat1.gamma
                    wp2 = mat2.wp
                    gamma2 = mat2.gamma
                    rTM = (gamma1*wp2**2 - gamma2*wp1**2)/(gamma1*wp2**2 + gamma2*wp1**2)
                    rTE = 0.
            elif mat1.materialclass == "plasma":
                if mat2.materialclass == "dielectric":
                    Kp = mat1.wp/c
                    rTM = -1.
                    rTE = -(k - sqrt(Kp**2 + k**2))/(k + sqrt(Kp**2 + k**2))
                elif mat2.materialclass == "plasma":
                    Kp1 = mat1.wp/c
                    Kp2 = mat2.wp/c
                    q1 = sqrt(Kp1**2 + k**2)
                    q2 = sqrt(Kp2**2 + k**2)
                    rTM = (Kp2**2*q1 - Kp1**2*q2)/(Kp2**2*q1 + Kp1**2*q2)
                    rTE = (q1 - q2)/(q1 + q2)
            
        else: # k0 > 0
            eps1 = mat1.epsilon(k0*c)
            eps2 = mat2.epsilon(k0*c)
            q1 = sqrt(eps1*k0**2 + k**2)
            q2 = sqrt(eps2*k0**2 + k**2)
            rTM = (eps2*q1 - eps1*q2)/(eps1*q2 + eps2*q1)
            rTE = (q1 - q2)/(q2 + q1)

        return rTM, rTE
    return fresnel_coefficients
