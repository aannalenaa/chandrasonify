"""
Agentic Pipeline Configuration — LLM Fallback Defaults.

Default parameter values used by agentic agents when LLM inference fails
or returns invalid output. These serve as safe fallbacks to keep the
pipeline running even when the language model is unavailable.

These defaults are independent from the manual pipeline configuration.
"""

# — Detection ————————————————————————————————————————————————————
WAVDETECT_SCALES = "1 2 4 8"
SIGNIFICANCE_THRESHOLD = 1e-6

# — Sonification ———————————————————————————————————————————————
DURATION = 60.0
PEAK_VOLUME_RANGE = (0.3, 1.0)
STEREO_SPREAD = (0.3, 0.7)
NOTE_LEN = 1.0
SF_PRESET = 1
MASTER_VOLUME = 0.6

# — Band Mode Selection ———————————————————————————————————————
# If the full-band source count exceeds this threshold the pipeline
# will suggest dual-band processing (soft + hard).
DUAL_BAND_THRESHOLD = 25

# — Energy Band Definitions ————————————————————————————————————
ENERGY_BANDS: dict[str, str] = {
    "full": "500:7000",
    "soft": "500:2000",
    "hard": "2000:7000",
}

BAND_LABELS: dict[str, str] = {
    "full": "Full Band (500-7000 eV)",
    "soft": "Soft Band (500-2000 eV)",
    "hard": "Hard Band (2000-7000 eV)",
}
