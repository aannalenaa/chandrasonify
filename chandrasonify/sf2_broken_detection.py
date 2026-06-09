"""
Test which sf_preset values pass the Strauss preset-selection step,
WITHOUT loading all samples (which is slow and not what breaks).

The crash happens in get_sfpreset_samples(), called before load_samples().
We monkey-patch load_samples to a no-op so each preset test is instant.

Usage:
    python test_soundfont_presets.py
    python test_soundfont_presets.py --sf2 path/to/your.sf2 --max-presets 128
"""

import argparse
from unittest.mock import patch
from strauss.generator import Sampler

# — Defaults —————————————————————————————————————————————————————
SF2_DEFAULT = 'FluidR3_GM.sf2'


# region Preset Checks
# — Preset Validation ———————————————————————————————————————————
def test_preset_fast(sf2_path: str, preset: int) -> tuple[bool, str]:
    """Test soundfont preset compatibility (fast method).

    Tests only the preset-selection step (get_sfpreset_samples) without
    loading all samples, which is slow and not where crashes occur.
    Uses monkey-patch to replace load_samples with a no-op.

    Args:
        sf2_path (str): Path to the .sf2 soundfont file.
        preset (int): Preset index to test.

    Returns:
        tuple[bool, str]: (success, message) where success is True if
            preset loads without error, False otherwise. Message is
            "OK" on success or error description on failure.
    """
    try:
        with patch.object(Sampler, 'load_samples', lambda self: None):
            Sampler(sampfiles=sf2_path, sf_preset=preset)
        return True, "OK"
    except IndexError as e:
        return False, f"IndexError: {e}"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"
# endregion


# region Entrypoint
# — CLI Entrypoint —————————————————————————————————————————————
def main() -> None:
    """Command-line entrypoint for soundfont preset testing.

    Parses command-line arguments and tests a range of soundfont presets
    for Strauss compatibility. Prints results showing safe and broken
    presets.
    """
    parser = argparse.ArgumentParser(
        description="Test soundfont presets for Strauss compatibility (fast)")
    parser.add_argument(
        '--sf2', default=SF2_DEFAULT, help='Path to .sf2 soundfont file')
    parser.add_argument(
        '--max-presets', type=int,
        default=128, help='How many presets to test (default: 128)')
    args = parser.parse_args()

    print(f"Soundfont: {args.sf2}")
    print(
        f"Testing presets 1 — {args.max_presets}  "
        "(load_samples skipped for speed)")
    print("="*60)

    safe = []
    broken = []

    for preset in range(1, args.max_presets + 1):
        ok, msg = test_preset_fast(args.sf2, preset)
        status = "PASS" if ok else "FAIL"
        detail = "" if ok else f"  ← {msg}"
        print(f"  Preset {preset:>3}: {status}{detail}")
        (safe if ok else broken).append(preset)

    print("\n" + "="*60)
    print(f"Safe   ({len(safe):>3}): {safe}")
    print(f"Broken ({len(broken):>3}): {broken}")

    if safe:
        print(f"\nRecommended default: sf_preset={safe[0]}")
    else:
        print("\nNo safe presets found — check your .sf2 path.")


if __name__ == '__main__':
    main()
# endregion
