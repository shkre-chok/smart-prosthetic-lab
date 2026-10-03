"""Database schema for prosthetic telemetry timesteps.

Defines the uniform data object produced for every millisecond of motion,
which feeds the rehabilitation Dashboard. Values originate in the patient
(EMG from the stump), the prosthetic muscle (LCE actuation), and the
sensory feedback channel (BaTiO3 piezo crystal — see core_engine/materials).
"""

from datetime import UTC, datetime
from typing import Any

#: Conversion factor from piezo output voltage (mV) to neural stimulation
#: frequency (Hz). Placeholder — calibrate against patient comfort/sensation.
MV_TO_STIM_HZ = 2.5


def create_prosthetic_timestep(
    emg_microvolts: float,
    contraction_ratio: float,
    voltage_mv: float,
    stiffness: float,
) -> dict[str, Any]:
    """
    מייצר אובייקט דאטה אחיד עבור כל מילישנייה של תנועה בפרוסטזה.
    משמש כבסיס הנתונים שיזין את ה-Dashboard.

    Creates a uniform data object for each millisecond of prosthetic motion.
    Serves as the data basis that feeds the Dashboard.
    """
    return {
        "timestamp": datetime.now(UTC).isoformat(),
        "patient_input": {
            "emg_signal_microvolts": emg_microvolts  # אות השריר מהגדם
        },
        "prosthetic_muscle_state": {
            "lce_contraction_ratio": contraction_ratio,  # התכווצות השריר המולקולרי
            "dynamic_stiffness_n_m": stiffness,          # קשיחות משתנה (פילאטיס שיקומי)
        },
        "patient_feedback_output": {
            "piezo_generated_voltage_mv": voltage_mv,    # המתח שהגביש מייצר (תחושת המגע)
            "sensory_stimulation_hz": voltage_mv * MV_TO_STIM_HZ,  # תדר הגירוי העצבי שיוחזר למטופל
        },
    }
