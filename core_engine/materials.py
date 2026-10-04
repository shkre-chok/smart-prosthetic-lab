"""
Materials definitions for the smart prosthetic core engine.

This module models the crystalline structure and piezoelectric behaviour of
Barium Titanate (BaTiO3), the active sensing material used in the prosthetic
interface alongside Liquid Crystal Elastomers (LCE).

Dependencies:
    - pymatgen : crystal structure representation
    - numpy    : numerical calculations
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from pymatgen.core import Lattice, Structure

# --- Physical constants (literature values for tetragonal BaTiO3, P4mm) ---

#: Lattice parameters of tetragonal BaTiO3 at room temperature (Angstrom).
BATIO3_A = 3.992
BATIO3_C = 4.036

#: Spontaneous polarization of tetragonal BaTiO3 (C/m^2).
SPONTANEOUS_POLARIZATION = 0.26

#: Effective piezoelectric charge coefficient d33 (C/N) for poled BaTiO3.
#: Placeholder of typical order for BaTiO3 ceramics (~200 pC/N).
D33_C_PER_N = 2e-10

#: Electrode area over which pressure acts (m^2); ~1 cm^2 sensing pad.
ELECTRODE_AREA_M2 = 1e-4

#: Relative permittivity of BaTiO3 ceramic at room temperature.
RELATIVE_PERMITTIVITY = 1500.0

#: Vacuum permittivity (F/m).
EPSILON_0 = 8.8541878128e-12

#: Elemental composition of BaTiO3.
BATIO3_SPECIES = ("Ba", "Ti", "O")


def build_batio3_structure() -> Structure:
    """Construct the tetragonal (P4mm) BaTiO3 unit cell.

    Uses experimentally reported fractional coordinates for the tetragonal
    phase stable at room temperature, including the Ti off-centring along
    the c-axis that gives rise to the spontaneous dipole moment.

    Returns:
        A pymatgen ``Structure`` of the BaTiO3 unit cell.
    """
    lattice = Lattice.tetragonal(BATIO3_A, BATIO3_C)

    species = ["Ba", "Ti", "O", "O", "O"]
    coords = [
        [0.0, 0.0, 0.0],    # Ba at corner
        [0.5, 0.5, 0.522],  # Ti displaced along +c (origin of the dipole)
        [0.5, 0.5, 0.0],    # O apical (below Ti)
        [0.5, 0.0, 0.512],  # O equatorial
        [0.0, 0.5, 0.512],  # O equatorial
    ]

    return Structure(
        lattice=lattice,
        species=species,
        coords=coords,
        coords_are_cartesian=False,
    )


@dataclass
class PiezoResponse:
    """Result of a piezoelectric simulation step.

    Attributes:
        force_n: Applied mechanical force in Newtons.
        polarization_cm2: Resulting polarization magnitude (C/m^2).
        voltage_v: Open-circuit voltage generated across the element (V).
    """

    force_n: float
    polarization_cm2: float
    voltage_v: float
    metadata: dict = field(default_factory=dict)


def simulate_voltage_from_pressure(
    force_n: float,
    thickness_m: float = 1e-3,
) -> PiezoResponse:
    """Placeholder simulation of dipole/voltage response to mechanical force.

    Models the direct piezoelectric effect in BaTiO3: applied stress shifts
    the Ti off-centring, changing the unit-cell dipole moment and hence the
    macroscopic polarization. Charge accumulates on the electrodes, and the
    open-circuit voltage across the thickness is approximated as

        Q = d33 * F            (charge generated on the electrodes)
        V = Q * t / (eps * A)  (voltage across the element)

    Args:
        force_n: Compressive force applied along the polar axis (N).
        thickness_m: Element thickness between electrodes (m).

    Returns:
        A ``PiezoResponse`` with force, polarization and generated voltage.

    Note:
        This is a first-order linear model for prototyping only. Replace
        with DFT-derived coefficients or experimental calibration data.
    """
    if force_n < 0:
        raise ValueError("Compressive force must be non-negative.")

    permittivity = RELATIVE_PERMITTIVITY * EPSILON_0

    charge_c = D33_C_PER_N * force_n
    polarization = charge_c / ELECTRODE_AREA_M2
    voltage = (charge_c * thickness_m) / (permittivity * ELECTRODE_AREA_M2)

    return PiezoResponse(
        force_n=force_n,
        polarization_cm2=float(polarization),
        voltage_v=float(voltage),
        metadata={"model": "linear_direct_piezo", "spontaneous_P": SPONTANEOUS_POLARIZATION},
    )

# --- LCE Muscle Constants (literature values for thermo-responsive liquid crystal elastomers) ---

#: Baseline transition temperature where nematic-to-isotropic phase change occurs (Celsius).
LCE_TRANSITION_TEMP_C = 42.0

#: Baseline mechanical stiffness of the relaxed LCE polymer matrix (N/m).
LCE_BASE_STIFFNESS_N_M = 500.0


@dataclass
class MuscleState:
    """Result of a liquid crystal elastomer muscle simulation step.

    Attributes:
        internal_temperature_c: Core temperature of the elastomer matrix (C).
        contraction_ratio: Contraction percentage compared to relaxed state (0.0 to 0.5).
        dynamic_stiffness_n_m: Current mechanical stiffness under operational load (N/m).
    """

    internal_temperature_c: float
    contraction_ratio: float
    dynamic_stiffness_n_m: float
    metadata: dict = field(default_factory=dict)


def simulate_lce_muscle_state(
    body_temperature_c: float,
    nir_light_intensity: float,
) -> MuscleState:
    """Simulate LCE contraction and dynamic stiffness adaptation via photothermal effect.

    Models the phase transition from a highly ordered nematic state (long/thin) 
    to a disordered isotropic state (short/thick). Near-infrared (NIR) light 
    acts via photothermal conversion to raise internal polymer temperature.

    Args:
        body_temperature_c: The patient's baseline tissue temperature (C).
        nir_light_intensity: Normalised intensity of control NIR light (0.0 to 1.0).

    Returns:
        A ``MuscleState`` mapping actuation parameters and biomechanical stiffness.
    """
    if not (0.0 <= nir_light_intensity <= 1.0):
        raise ValueError("NIR light intensity must be normalised between 0.0 and 1.0.")

    # NIR absorption generates a localized temperature delta (up to +12C at max intensity)
    internal_temp = body_temperature_c + (nir_light_intensity * 12.0)

    # Sigmoidal activation model representing the molecular melting of liquid crystal order
    # LCEs typically shrink up to ~40% of their original length (max ratio ~0.4)
    max_shrinkage = 0.4
    activation = 1.0 / (1.0 + np.exp(-(internal_temp - LCE_TRANSITION_TEMP_C) / 1.5))
    contraction = max_shrinkage * activation

    # Dynamic Stiffness (orthopaedic property): as the polymer chain packs tightly,
    # structural modulus scales up dynamically, locking the joint under tension.
    dynamic_stiffness = LCE_BASE_STIFFNESS_N_M * (1.0 + (activation * 2.5))

    return MuscleState(
        internal_temperature_c=float(internal_temp),
        contraction_ratio=float(contraction),
        dynamic_stiffness_n_m=float(dynamic_stiffness),
        metadata={"model": "sigmoidal_nematic_isotropic_transition", "max_shrinkage": max_shrinkage}
    )

# if __name__ == "__main__":
#     structure = build_batio3_structure()
#     print(structure)

#     # Example sweep: 0-10 N fingertip-scale forces.
#     for force in np.linspace(0, 10, num=6):
#         response = simulate_voltage_from_pressure(force)
#         print(
#             f"F = {response.force_n:5.2f} N | "
#             f"P = {response.polarization_cm2:.4e} C/m^2 | "
#             f"V = {response.voltage_v:8.3f} V"
#         )

if __name__ == "__main__":
    structure = build_batio3_structure()
    print("--- Tetrogonal BaTiO3 Crystal Structure successfully built in PyMatgen ---")
    print(structure)
    print("\n--- Running Multiphysics Simulation Sweeps ---")

    # 1. Example sweep: 0-10 N fingertip-scale forces (Sensing)
    print("\n[PIEZOELECTRIC SENSOR SWEEP]")
    for force in np.linspace(0, 10, num=4):
        response = simulate_voltage_from_pressure(force)
        print(
            f"Force = {response.force_n:5.2f} N | "
            f"P = {response.polarization_cm2:.4e} C/m^2 | "
            f"V = {response.voltage_v:6.2f} V"
        )

    # 2. Example sweep: 0% to 100% NIR Light Intensity on baseline body temp (Actuation/Stiffness)
    print("\n[LCE ARTIFICIAL MUSCLE SWEEP]")
    for nir in np.linspace(0, 100, num=4) / 100:
        muscle_resp = simulate_lce_muscle_state(body_temperature_c=36.5, nir_light_intensity=nir)
        print(
            f"NIR Light = {nir*100:3.0f}% | "
            f"Internal Temp = {muscle_resp.internal_temperature_c:5.2f} C | "
            f"Contraction = {muscle_resp.contraction_ratio*100:4.1f}% | "
            f"Stiffness = {muscle_resp.dynamic_stiffness_n_m:7.2f} N/m"
        )
