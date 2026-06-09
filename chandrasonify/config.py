"""
System configuration for X-ray sonification pipeline.

This file contains paths and settings that may vary across different systems.
Update these values based on your local installation.
"""

import platform
from pathlib import Path

# region Configuration
# — System Paths ————————————————————————————————————————————————
CIAO_BIN = "/Applications/ciao-4.18/bin"

# — Soundfont ———————————————————————————————————————————————————
# Path to the FluidR3 soundfont file used for sonification
SF2 = "FluidR3_GM.sf2"

# — Video Codec —————————————————————————————————————————————————
VIDEO_CODEC = "h264_videotoolbox"

# — Broken Presets ————————————————————————————————————————————
# These presets cause IndexErrors with FluidR3
# They are automatically skipped during sonification
BROKEN_PRESETS = {1, 8, 11, 12, 13, 14, 15, 23, 33, 34, 51, 53, 78, 101, 121, 122, 128}
# if another sf2 file is used run the code for broken detection first
# endregion


# region Validation
# — Path Validation ————————————————————————————————————————————
def validate_paths() -> dict[str, bool]:
    """Validate that required paths exist.

    Returns:
        dict[str, bool]: Path existence status.
    """
    checks = {
        "CIAO_BIN": Path(CIAO_BIN).exists(),
        "SOUNDFONT": Path(SF2).exists(),
    }
    return checks


def print_config() -> None:
    """Print system configuration."""
    print("\n" + "=" * 60)
    print("SYSTEM CONFIGURATION")
    print("=" * 60)
    print(f"OS:               {platform.system()} {platform.release()}")
    print(f"CIAO_BIN:         {CIAO_BIN}")
    print(f"SOUNDFONT:        {SF2}")
    print(f"VIDEO_CODEC:      {VIDEO_CODEC}")
    print()

    checks = validate_paths()
    for name, exists in checks.items():
        status = "OK" if exists else "MISSING"
        print(f"  {name:<20} {status}")
    print("=" * 60 + "\n")


# endregion
