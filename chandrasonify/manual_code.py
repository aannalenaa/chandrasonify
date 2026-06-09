# region Imports
import argparse
import os
import sys
import subprocess
import traceback
from datetime import datetime
from enum import Enum
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FuncAnimation  # noqa: E402
import numpy as np  # noqa: E402
import pexpect  # noqa: E402

from astropy.io import fits  # noqa: E402
from astropy.coordinates import SkyCoord  # noqa: E402
import astropy.units as u  # noqa: E402
from astroquery.simbad import Simbad  # noqa: E402
from pydantic import BaseModel  # noqa: E402
from strauss import score  # noqa: E402
from strauss.sources import Objects  # noqa: E402
from strauss.generator import Sampler  # noqa: E402
from strauss.sonification import Sonification  # noqa: E402

from chandrasonify.run_config import RunConfig  # noqa: E402
from chandrasonify.config import (  # noqa: E402
    CIAO_BIN,
    SF2,
    VIDEO_CODEC,
    BROKEN_PRESETS,
    print_config,
)
from chandrasonify.manual_config import (  # noqa: E402
    WAVDETECT_SCALES,
    SIGNIFICANCE_THRESHOLD,
    DURATION,
    PEAK_VOLUME_RANGE,
    STEREO_SPREAD,
    NOTE_LEN,
    SF_PRESET,
    MASTER_VOLUME,
    DUAL_BAND_THRESHOLD,
    ENERGY_BANDS,
    BAND_LABELS,
)

from typing import Any, cast  # noqa: E402

# — Environment Setup ————————————————————————————————————————————
print_config()
os.environ["PATH"] = f"{CIAO_BIN}:{os.environ['PATH']}"
# endregion


"""
Manual (non-agentic) X-ray data processing pipeline.

1. activate ciao_venv (inside the macos distribution)
2. install all required packages
3. source /Applications/ciao-4.18/bin/ciao.sh
4. source /ciao_venv/bin/activate python [filename].py

All tuneable parameters live in manual_config.py.
"""

# — SIMBAD Object Type Mapping ——————————————————————————————————


def _load_otype_map() -> dict[str, str]:
    """Load SIMBAD object type mappings from CSV.

    Returns:
        dict[str, str]: Mapping of otype codes to descriptions.
    """
    try:
        import csv

        csv_path = Path(__file__).resolve().parent.parent / "simbad_otypes.csv"
        otype_map = {}
        if csv_path.exists():
            with open(csv_path, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    otype_id = row["id"].strip() if row["id"] else None
                    otype_desc = (
                        row["description"].strip() if row["description"] else None
                    )
                    if otype_id and otype_desc:
                        otype_map[otype_id] = otype_desc
            print(f"Loaded {len(otype_map)} SIMBAD object types from CSV")
            return otype_map
        else:
            print(f"SIMBAD otypes CSV not found at {csv_path}")
    except Exception as e:
        print(f"Could not load SIMBAD otypes CSV: {e}")

    return {
        "*": "Star",
        "**": "Double or Multiple Star",
        "X": "X-ray Source",
        "G": "Galaxy",
        "AGN": "Active Galaxy Nucleus",
        "QSO": "Quasar",
    }


OTYPE_MAP = _load_otype_map()


def run_validity_checks(
    src_data,
    significance_threshold: float,
    exposure: float,
    chip_size: int = 1024,
    edge_margin: int = 50,
) -> dict[str, Any]:
    """Run validity checks on wavdetect results.

    Args:
        src_data: Source data from wavdetect.
        significance_threshold (float): Significance threshold used.
        exposure (float): Exposure time in seconds.
        chip_size (int): CCD chip size in pixels (default 1024).
        edge_margin (int): Edge margin in pixels (default 50).

    Returns:
        dict[str, Any]: Validity check results and flags.
    """
    n = len(src_data)
    passed, failed = [], []

    # 1. Source count
    if n == 0:
        failed.append("zero_sources")
    elif n < 3:
        failed.append("too_few_sources")
    elif n > 500:
        failed.append("too_many_sources")
    else:
        passed.append("source_count")

    # 2. Expected false positives
    expected_fp = significance_threshold * chip_size**2
    fp_rate = expected_fp / max(n, 1)
    if fp_rate > 0.2:
        failed.append("high_false_positive_rate")
    else:
        passed.append("false_positive_rate")

    # 3. Edge fraction
    x = np.array(src_data["X"], dtype=float)
    y = np.array(src_data["Y"], dtype=float)
    if x.max() < 360:  # RA/Dec — normalise to pixel coords
        x = (x - x.min()) / max(x.max() - x.min(), 1) * chip_size
        y = (y - y.min()) / max(y.max() - y.min(), 1) * chip_size
    edge_mask = (
        (x < edge_margin)
        | (x > chip_size - edge_margin)
        | (y < edge_margin)
        | (y > chip_size - edge_margin)
    )
    edge_frac = float(edge_mask.mean()) if n > 0 else 0.0
    if edge_frac > 0.3:
        failed.append("high_edge_fraction")
    else:
        passed.append("edge_fraction")

    # 4. Negative counts fraction
    neg_cts_frac = median_cts = None
    if "NET_COUNTS" in src_data.dtype.names:
        cts = np.array(src_data["NET_COUNTS"], dtype=float)
        neg_cts_frac = float((cts < 0).mean()) if n > 0 else 0.0
        pos_cts = cts[cts > 0]
        median_cts = float(np.median(pos_cts)) if len(pos_cts) > 0 else 0.0
        if neg_cts_frac > 0.2:
            failed.append("high_negative_counts")
        else:
            passed.append("negative_counts")

    # 5. OAA
    median_oaa = None
    if "OAA" in src_data.dtype.names:
        oaa = np.array(src_data["OAA"], dtype=float)
        median_oaa = float(np.median(oaa))
        if float(np.max(oaa)) == 0:
            failed.append("oaa_not_computed")
        else:
            passed.append("oaa")

    should_rerun = (
        "zero_sources" in failed
        or "too_many_sources" in failed
        or ("high_false_positive_rate" in failed and fp_rate > 0.5)
    )

    return {
        "n_sources": n,
        "expected_fp": expected_fp,
        "fp_rate": fp_rate,
        "edge_frac": edge_frac,
        "neg_cts_frac": neg_cts_frac,
        "median_cts": median_cts,
        "median_oaa": median_oaa,
        "exposure": exposure,
        "checks_passed": passed,
        "checks_failed": failed,
        "n_passed": len(passed),
        "n_total": len(passed) + len(failed),
        "should_rerun_detection": should_rerun,
    }


# region Models
# — Pydantic Models ————————————————————————————————————————————
class ObservationMetadata(BaseModel):
    """Observation metadata from FITS header.

    Attributes:
        obs_id (str): Observation ID.
        instrument (str): Instrument name.
        exposure_time (float): Exposure in seconds.
        target_name (str): Target name.
        ra (float): Right Ascension (degrees).
        dec (float): Declination (degrees).
        num_sources (int): Source count (default 0).
        object_type (str | None): SIMBAD object type.
        object_info (str | None): SIMBAD info.
        redshift (float | None): Redshift.
    """

    obs_id: str
    instrument: str
    exposure_time: float
    target_name: str
    ra: float
    dec: float
    num_sources: int = 0
    object_type: str | None = None
    object_info: str | None = None
    redshift: float | None = None


class ProcessingParameters(BaseModel):
    """Wavdetect source-detection parameters.

    Attributes:
        wavdetect_scales (str): Scales for wavdetect.
        significance_threshold (float): Significance threshold.
        reasoning (str): Parameter reasoning.
    """

    wavdetect_scales: str = WAVDETECT_SCALES
    significance_threshold: float = SIGNIFICANCE_THRESHOLD
    reasoning: str = "manual_config defaults"


class SonificationConfig(BaseModel):
    """Sonification rendering configuration.

    Attributes:
        duration (float): Duration in seconds.
        peak_volume_range (tuple[float, float]): Volume range (min, max).
        stereo_spread (tuple[float, float]): Stereo pan range (min, max).
    """

    duration: float = DURATION
    peak_volume_range: tuple[float, float] = PEAK_VOLUME_RANGE
    stereo_spread: tuple[float, float] = STEREO_SPREAD


class ProcessingMode(str, Enum):
    """Energy-band processing strategy.

    Attributes:
        FULL_BAND (str): All energies together.
        DUAL_BAND (str): Soft + hard bands.
        TRIPLE_BAND (str): Soft + medium + hard bands.
        CUSTOM (str): Custom energy ranges.
    """

    FULL_BAND = "full_band"
    DUAL_BAND = "dual_band"
    TRIPLE_BAND = "triple_band"
    CUSTOM = "custom"


# endregion


# region Pipeline Manager
# — Pipeline Manager ———————————————————————————————————————————
class PipelineManager:
    """
    Pipeline utilities: file handling, workspace inspection,
    resume logic, and deterministic step sequencing.
    """

    def __init__(self, workspace: str):
        """Initialize pipeline manager with workspace directory.

        Args:
            workspace: Absolute path to the workspace root directory.
        """
        self.path = Path(workspace).resolve()
        self.workspace = None

    @staticmethod
    def validate_detection_params(
        params: ProcessingParameters, exposure_time: float
    ) -> ProcessingParameters:
        """Validate and adjust detection parameters if they seem too strict.

        Applies adaptive rules based on exposure time:
        - For exposures > 5000s, threshold relaxed to 1e-6 if stricter.
        - For exposures > 10000s, scale range expanded to include 16.

        Args:
            params: Detection parameters to validate.
            exposure_time: Observation exposure time in seconds.

        Returns:
            Potentially modified ProcessingParameters.
        """
        original_threshold = params.significance_threshold
        original_scales = params.wavdetect_scales

        if exposure_time > 5000 and params.significance_threshold < 1e-6:
            print(
                f"Detection threshold {params.significance_threshold} "
                f"is too strict for {exposure_time:.0f}s exposure."
            )
            print("    Relaxing to 1e-6 for better source recovery...")
            params.significance_threshold = 1e-6

        scales = [int(s) for s in params.wavdetect_scales.split()]
        if exposure_time > 10000 and max(scales) < 16:
            print(
                f"Scale range '{params.wavdetect_scales}' might be "
                f"too narrow for {exposure_time:.0f}s exposure."
            )
            print("    Using '1 2 4 8 16' for better multi-scale detection...")
            params.wavdetect_scales = "1 2 4 8 16"

        if (
            original_threshold != params.significance_threshold
            or original_scales != params.wavdetect_scales
        ):
            print("Adjusted detection parameters:")
            if original_threshold != params.significance_threshold:
                print(
                    f"  threshold: {original_threshold} -> "
                    f"{params.significance_threshold}"
                )
            if original_scales != params.wavdetect_scales:
                print(
                    f"  scales: '{original_scales}' -> " f"'{params.wavdetect_scales}'"
                )
        else:
            print("Detection parameters valid (no adjustments needed)")

        return params

    def enrich_metadata_with_object_info(
        self, metadata: ObservationMetadata
    ) -> ObservationMetadata:
        """Look up astronomical object information via SIMBAD.

        Enriches metadata with object type and redshift from SIMBAD
        database using cone search at observation coordinates.

        Args:
            metadata: Observation metadata to enrich.

        Returns:
            Updated metadata with object_type, redshift, and object_info
            fields populated (or set to None/messages if lookup fails).
        """
        try:
            print(f"Looking up object information for " f"'{metadata.target_name}'...")

            simbad = Simbad()
            simbad.add_votable_fields("otype", "rvz_redshift", "sp")

            coord = SkyCoord(ra=metadata.ra, dec=metadata.dec, unit="deg")
            result = simbad.query_region(coord, radius=5 * u.arcsec)
            if result is not None:
                types = list(set(str(row["otype"]) for row in result if row["otype"]))
                sp_types = list(
                    set(str(row["sp_type"]) for row in result if row["sp_type"])
                )
                redshifts = [
                    float(row["rvz_redshift"])
                    for row in result
                    if row["rvz_redshift"] and str(row["rvz_redshift"]) != "--"
                ]

                metadata.object_type = types[0] if types else None
                metadata.redshift = redshifts[0] if redshifts else None

                if metadata.object_type:
                    otype_label = OTYPE_MAP.get(metadata.object_type, "Unknown")
                    if otype_label == "Unknown":
                        print(
                            f"    otype code '{metadata.object_type}' not in OTYPE_MAP"
                        )
                        print(f"    Available codes: {list(OTYPE_MAP.keys())[:10]}...")

                summary = f"Field contains {len(result)} sources. "
                if types:
                    summary += f"Object types: {', '.join(types)}. "
                if sp_types:
                    summary += f"Spectral types: {', '.join(sp_types)}. "
                if redshifts:
                    summary += f"Redshifts: {', '.join(f'{r:.4f}'
                                                       for r in redshifts)}."

                metadata.object_info = summary
                print(f"  Found: {metadata.object_info}")
            else:
                print(f"  No SIMBAD entry found for '{metadata.target_name}'")
                metadata.object_info = "Object not found in SIMBAD"

        except ImportError:
            print("  Warning: astroquery not installed, skipping object lookup")
            metadata.object_info = "Object lookup skipped" "(astroquery not available)"
        except Exception as e:
            print(f"  Error looking up object: {e}")
            metadata.object_info = f"Lookup failed: {str(e)}"

        return metadata

    def check_observation_exists(self, obs_id: str) -> dict[str, Any]:
        """Check if an observation has been previously processed.

        Scans for existing output directory and run_config to determine
        processing state and resumability.

        Args:
            obs_id: Observation ID to check.

        Returns:
            Dictionary with keys: 'exists', 'output_dir', 'run_config',
            'last_step', 'can_resume', 'files_present', 'log_entries'.
        """
        output_dir = self.path / f"output_{obs_id}"

        result = {
            "exists": output_dir.exists(),
            "output_dir": output_dir,
            "run_config": None,
            "last_step": "none",
            "can_resume": False,
            "files_present": {},
            "log_entries": [],
        }

        if not output_dir.exists():
            print(f"No previous processing found for observation {obs_id}")
            return result

        try:
            run_config = RunConfig.load_or_create(obs_id)
            result["run_config"] = run_config

            if run_config.has("metadata"):
                result["files_present"]["metadata"] = True
                result["log_entries"].append("metadata_extracted")

            if run_config.has("full_band_detection"):
                result["files_present"]["full_band_detection"] = True
                result["log_entries"].append("full_band_detection_complete")

            if run_config.has("processing_mode"):
                result["files_present"]["processing_mode"] = True
                result["log_entries"].append("processing_mode_selected")

            if run_config.has("sonification_config_soft"):
                result["files_present"]["sonification_soft_config"] = True
                result["log_entries"].append("soft_band_sonification_config")

            if run_config.has("sonification_config_hard"):
                result["files_present"]["sonification_hard_config"] = True
                result["log_entries"].append("hard_band_sonification_config")

            if result["log_entries"]:
                result["can_resume"] = True
                if (
                    "sonification_soft_config" in result["log_entries"]
                    or "sonification_hard_config" in result["log_entries"]
                ):
                    result["last_step"] = "sonification_generated"
                elif "processing_mode_selected" in result["log_entries"]:
                    result["last_step"] = "processing_mode_selected"
                elif "full_band_detection_complete" in result["log_entries"]:
                    result["last_step"] = "full_band_detection_complete"
                elif "metadata_extracted" in result["log_entries"]:
                    result["last_step"] = "metadata_extracted"

            print(f"\nFound previous processing for observation {obs_id}")
            print(f"  Last completed step: {result['last_step']}")

        except Exception as e:
            print(f"Warning: Could not load run config: {e}")
            result["exists"] = True
            result["can_resume"] = False

        return result

    def should_resume_observation(self, obs_id: str) -> tuple[bool, str]:
        """Ask user if they want to resume previous observation or restart.

        Presents interactive menu if previous work is resumable. Returns
        user's choice and reason.

        Args:
            obs_id: Observation ID to check.

        Returns:
            Tuple of (should_resume: bool, reason: str).

        Raises:
            RuntimeError: If user cancels or config is corrupted and user
                chooses not to restart.
        """
        obs_state = self.check_observation_exists(obs_id)

        if not obs_state["exists"]:
            return False, "no_previous_work"

        if not obs_state["can_resume"]:
            print("\nPrevious work found but config is corrupted.")
            choice = input("Restart from scratch? (y/n): ").strip().lower()
            if choice == "y":
                return False, "config_corrupted_restart_requested"
            else:
                raise RuntimeError("Cannot resume corrupted state.")

        print(
            (
                f"\n{'='*60}\n"
                "RESUMABLE STATE DETECTED\n"
                f"{'='*60}\n"
                f"Observation {obs_id} has been partially processed."
                f"Last completed step: {obs_state['last_step']}\n\n"
                "Options:\n"
                "1. Resume from last completed step\n"
                "2. Restart from scratch (delete previous work)\n"
                "3. Cancel"
            )
        )

        while True:
            choice = input("\nEnter choice (1, 2, or 3): ").strip()
            if choice == "1":
                return True, obs_state["last_step"]
            elif choice == "2":
                confirm = (
                    input(f"Delete all work in {obs_state['output_dir']}? " "(y/n): ")
                    .strip()
                    .lower()
                )
                if confirm == "y":
                    import shutil

                    shutil.rmtree(obs_state["output_dir"])
                    print(f"Deleted {obs_state['output_dir']}")
                    return False, "restart_after_delete"
                else:
                    print("Not deleting. Please choose again.")
            elif choice == "3":
                raise RuntimeError("User cancelled observation processing")
            else:
                print("Please enter 1, 2, or 3.")

    def decide_next_step(
        self, obs_id: str, current_step: str | None = None
    ) -> dict[str, Any]:
        """Determine the next pipeline step based on current step.

        Implements pure state-machine logic for deterministic step sequencing.

        Args:
            obs_id: Observation ID (for potential logging; not used).
            current_step: Current pipeline step identifier or None for start.

        Returns:
            Dictionary with keys 'next_step', 'reasoning', 'can_skip'.
        """
        if current_step is None or current_step == "start":
            return {
                "next_step": "metadata",
                "reasoning": "Starting fresh pipeline",
                "can_skip": False,
            }
        elif current_step == "metadata_extracted":
            return {
                "next_step": "full_band_detection",
                "reasoning": "Metadata extracted; run full-band detection",
                "can_skip": False,
            }
        elif current_step == "full_band_detection_complete":
            return {
                "next_step": "band_selection",
                "reasoning": "Full band detection done; " "select processing mode",
                "can_skip": False,
            }
        elif current_step == "processing_mode_selected":
            return {
                "next_step": "band_processing",
                "reasoning": "Processing mode chosen; create band images",
                "can_skip": False,
            }
        elif current_step == "band_processing_complete":
            return {
                "next_step": "sonification",
                "reasoning": "Band processing complete; " "generate sonifications",
                "can_skip": False,
            }
        elif current_step == "sonification_generated":
            return {
                "next_step": "finished",
                "reasoning": "All sonifications complete",
                "can_skip": False,
            }
        else:
            return {
                "next_step": "unknown",
                "reasoning": f"Unknown state: {current_step}",
                "can_skip": False,
            }

    def find_latest_evt2(self) -> Path:
        """Find the most recent EVT2 FITS file in the workspace.

        Recursively searches workspace for EVT2 files and returns the
        one with the most recent modification time.

        Returns:
            Path to the most recent EVT2 file.

        Raises:
            FileNotFoundError: If no EVT2 files found in workspace.
        """
        evt2_files: list[Path] = []

        for p in self.path.rglob("*.fits*"):
            if "evt2" in p.name.lower():
                evt2_files.append(p)

        if not evt2_files:
            raise FileNotFoundError("No evt2 FITS file found in workspace")

        return max(evt2_files, key=lambda p: p.stat().st_mtime)


# endregion


# region Catalog
# — Sonification Catalog —————————————————————————————————————————
class SonificationCatalog:
    """Browse and manage existing sonification outputs in the workspace."""

    def __init__(self, workspace: str = "."):
        """Initialize the sonification catalog.

        Args:
            workspace: Path to the workspace directory (default: current dir).
        """
        self.workspace = Path(workspace).resolve()
        self.sonifications: list[dict[str, Any]] = []

    def scan_workspace(self) -> list[dict[str, Any]]:
        """Scan workspace for existing sonifications.

        Searches for output_* directories and catalogs all sonification
        artifacts (audio, video, reports, images).

        Returns:
            List of sonification entry dictionaries with metadata.
        """
        self.sonifications = []

        output_dirs = sorted(self.workspace.glob("output_*"))

        if not output_dirs:
            print("\n[WARNING] No sonifications found in workspace.")
            return []

        print(
            (
                f"\n{'='*60}\n"
                "SCANNING WORKSPACE FOR SONIFICATIONS\n"
                f"{'='*60}\n"
                f"Workspace: {self.workspace}\n"
            )
        )

        for output_dir in output_dirs:
            if not output_dir.is_dir():
                continue

            obs_id = output_dir.name.replace("output_", "")

            audio_files = {
                "soft": list(output_dir.glob("*sonification*soft*.wav")),
                "hard": list(output_dir.glob("*sonification*hard*.wav")),
                "full": list(output_dir.glob("*sonification*full*.wav")),
            }

            video_files = list(output_dir.glob("*sonification*.mp4"))
            report_files = list(output_dir.glob("REPORT_*.txt"))
            image_files = list(output_dir.glob("*source_distribution*.png"))

            mod_time = output_dir.stat().st_mtime
            created_date = datetime.fromtimestamp(mod_time).strftime(
                "%Y-%m-%d %H:%M:%S"
            )

            sonification_entry = {
                "obs_id": obs_id,
                "output_dir": output_dir,
                "has_audio_soft": bool(audio_files["soft"]),
                "has_audio_hard": bool(audio_files["hard"]),
                "has_audio_full": bool(audio_files["full"]),
                "has_video": bool(video_files),
                "has_report": bool(report_files),
                "has_images": bool(image_files),
                "created_date": created_date,
                "files": {
                    "audio_soft": audio_files["soft"],
                    "audio_hard": audio_files["hard"],
                    "audio_full": audio_files["full"],
                    "video": video_files,
                    "report": report_files,
                    "images": image_files,
                },
            }

            self.sonifications.append(sonification_entry)

        return self.sonifications

    def display_catalog(self) -> None:
        """Display formatted catalog of existing sonifications.

        Prints a table of all scanned sonifications with file presence
        indicators.
        """
        if not self.sonifications:
            print("No sonifications found.")
            return

        print(
            f"\n{'OBSERVATION ID':<15} {'CREATED':<20} {'AUDIO':<15} "
            f"{'VIDEO':<8} {'REPORT':<8} {'IMAGES':<8}"
        )
        print("=" * 60)

        for i, entry in enumerate(self.sonifications, 1):
            obs_id = entry["obs_id"]
            created = entry["created_date"]

            audio_status = []
            if entry["has_audio_full"]:
                audio_status.append("Full")
            if entry["has_audio_soft"]:
                audio_status.append("Soft")
            if entry["has_audio_hard"]:
                audio_status.append("Hard")
            audio_str = "+".join(audio_status) if audio_status else "[NONE]"

            video_str = "[YES]" if entry["has_video"] else "[NO]"
            report_str = "[YES]" if entry["has_report"] else "[NO]"
            images_str = "[YES]" if entry["has_images"] else "[NO]"

            print(
                f"{obs_id:<15} {created:<20} {audio_str:<15} "
                f"{video_str:<8} {report_str:<8} {images_str:<8}"
            )

        print(f"\n{len(self.sonifications)} sonification(s) found.\n")

    def get_sonification_by_id(self, obs_id: str) -> dict[str, Any] | None:
        """Get sonification entry by observation ID.

        Args:
            obs_id: Observation ID to search for.

        Returns:
            Sonification entry dictionary, or None if not found.
        """
        obs_id = obs_id.replace("output_", "")

        for entry in self.sonifications:
            if entry["obs_id"] == obs_id:
                return entry

        return None

    def show_sonification_details(self, obs_id: str) -> bool:
        """Display detailed information about a specific sonification.

        Args:
            obs_id: Observation ID to display.

        Returns:
            True if details displayed successfully, False if not found.
        """
        entry = self.get_sonification_by_id(obs_id)

        if not entry:
            print(f"\n[ERROR] Observation {obs_id} not found.")
            return False

        print(f"\n{'='*60}")
        print(f"SONIFICATION DETAILS: Observation {entry['obs_id']}")
        print(f"{'='*60}")
        print(f"Output Directory: {entry['output_dir']}")
        print(f"Created:          {entry['created_date']}")
        print()

        print("AUDIO FILES:")
        if entry["files"]["audio_full"]:
            for f in entry["files"]["audio_full"]:
                size_mb = f.stat().st_size / (1024**2)
                print(f"  - Full Band:  {f.name} ({size_mb:.1f} MB)")
        if entry["files"]["audio_soft"]:
            for f in entry["files"]["audio_soft"]:
                size_mb = f.stat().st_size / (1024**2)
                print(f"  - Soft Band:  {f.name} ({size_mb:.1f} MB)")
        if entry["files"]["audio_hard"]:
            for f in entry["files"]["audio_hard"]:
                size_mb = f.stat().st_size / (1024**2)
                print(f"  - Hard Band:  {f.name} ({size_mb:.1f} MB)")

        if not any(
            [
                entry["files"]["audio_full"],
                entry["files"]["audio_soft"],
                entry["files"]["audio_hard"],
            ]
        ):
            print("  [NONE] No audio files found")

        print("\nVIDEO FILES:")
        if entry["files"]["video"]:
            for f in entry["files"]["video"]:
                size_mb = f.stat().st_size / (1024**2)
                print(f"  - {f.name} ({size_mb:.1f} MB)")
        else:
            print("  [NONE] No video files found")

        print("\nREPORT FILES:")
        if entry["files"]["report"]:
            for f in entry["files"]["report"]:
                size_kb = f.stat().st_size / 1024
                print(f"  - {f.name} ({size_kb:.1f} KB)")
        else:
            print("  [NONE] No report files found")

        print("\nVISUALIZATIONS:")
        if entry["files"]["images"]:
            for f in entry["files"]["images"]:
                size_kb = f.stat().st_size / 1024
                print(f"  - {f.name} ({size_kb:.1f} KB)")
        else:
            print("  [NONE] No visualization files found")

        print()
        return True

    def open_sonification(self, obs_id: str) -> bool:
        """Open/play a sonification using the system default player.

        Attempts to open the best available file (prefers video over audio)
        using platform-specific mechanisms.

        Args:
            obs_id: Observation ID to open.

        Returns:
            True if opened successfully, False otherwise.
        """
        entry = self.get_sonification_by_id(obs_id)

        if not entry:
            print(f"\n[ERROR] Observation {obs_id} not found.")
            return False

        print(f"\n{'='*60}")
        print(f"OPENING SONIFICATION: Observation {entry['obs_id']}")
        print(f"{'='*60}\n")

        file_to_open = None
        file_type = None

        if entry["files"]["video"]:
            file_to_open = entry["files"]["video"][0]
            file_type = "video"
        elif entry["files"]["audio_full"]:
            file_to_open = entry["files"]["audio_full"][0]
            file_type = "audio (full band)"
        elif entry["files"]["audio_soft"]:
            file_to_open = entry["files"]["audio_soft"][0]
            file_type = "audio (soft band)"
        elif entry["files"]["audio_hard"]:
            file_to_open = entry["files"]["audio_hard"][0]
            file_type = "audio (hard band)"

        if not file_to_open:
            print("[ERROR] No playable files found for this sonification.")
            return False

        print(f"Opening {file_type}: {file_to_open.name}")

        try:
            import platform

            if platform.system() == "Darwin":
                subprocess.run(["open", str(file_to_open)], check=True)
            elif platform.system() == "Linux":
                subprocess.run(["xdg-open", str(file_to_open)], check=True)
            elif platform.system() == "Windows":
                os.startfile(str(file_to_open))  # type: ignore[attr-defined]

            print(f"[SUCCESS] Opened {file_type}")
            return True

        except Exception as e:
            print(f"[WARNING] Could not open file: {e}")
            print(f"File location: {file_to_open}")
            return False

    def interactive_browser(self) -> str | None:
        """Interactive menu to browse and open sonifications.

        Returns:
            Selected observation ID or None if user quit.
        """
        while True:
            print(f"\n{'='*60}")
            print("SONIFICATION BROWSER")
            print(f"{'='*60}\n")

            self.display_catalog()

            print("OPTIONS:")
            print("  Enter observation ID to view details and play")
            print("  'q' to quit\n")

            choice = input("Enter choice: ").strip().lower()

            if choice == "q":
                return None

            if not choice:
                print("Please enter a valid observation ID or 'q'.")
                continue

            if self.show_sonification_details(choice):
                play = input("\nPlay this sonification? (yes/no): ").strip().lower()
                if play in ["yes", "y"]:
                    self.open_sonification(choice)
                    break
                elif play in ["no", "n"]:
                    continue
            else:
                try_again = (
                    input("Try another observation ID? (yes/no): ").strip().lower()
                )
                if try_again not in ["yes", "y"]:
                    return None


# endregion


# region Deterministic Pipeline
# — Deterministic Pipeline Components ——————————————————————————
class FileSelection:
    """Prompt user to select desired file/observation for processing."""

    def __init__(self):
        """Initialize file selector with workspace context."""
        self.cwd = os.getcwd()
        self.pipeline_manager = PipelineManager(self.cwd)
        self.obs_id: str | None = None

    def extract_obs_id(self, file_path: Path) -> str | None:
        """Extract observation ID from directory structure.

        Assumes structure: .../obs_id/secondary/filename.fits
        Searches parent directories for numeric observation ID.

        Args:
            file_path: Path to a FITS file.

        Returns:
            Observation ID string, or None if extraction fails.
        """
        parents = file_path.parents

        if file_path.parent.name == "secondary":
            obs_id = file_path.parent.parent.name
            if obs_id.isdigit():
                print(f"Extracted observation ID: {obs_id}")
                return obs_id
            else:
                print(f"Warning: Expected numeric observation ID, got: {obs_id}")
                return obs_id

        for parent in parents:
            if parent.name.isdigit():
                print(f"Found observation ID in path: {parent.name}")
                return parent.name

        print("Warning: Could not extract observation ID from path")
        return None

    def select_evt2_file_by_obs_id(
        self, obs_id: str
    ) -> tuple[Path, str] | tuple[None, None]:
        """Try to find EVT2 file for a specific observation ID.

        Args:
            obs_id: Observation ID to search for.

        Returns:
            Tuple of (file_path, obs_id) if found, else (None, None).
        """
        for p in self.pipeline_manager.path.rglob("*.fits*"):
            if "evt2" not in p.name.lower():
                continue
            extracted = self.extract_obs_id(p)
            if extracted == obs_id:
                print(
                    (
                        f"\nFound evt2 file for observation {obs_id}:\n"
                        f"Path: {p}\n"
                        f"Modified: {datetime.fromtimestamp(p.stat().st_mtime)}\n"
                    )
                )
                return p, obs_id

        return None, None

    def select_evt2_file_with_obs_id_option(
        self, requested_obs_id: str | None = None
    ) -> tuple[Path, str] | tuple[None, None]:
        """Select EVT2 file, with optional observation ID preference.

        If obs_id provided but not found, shows menu with fallback options.

        Args:
            requested_obs_id: Optional preferred observation ID.

        Returns:
            Tuple of (file_path, obs_id) if selected, else (None, None).
        """
        if requested_obs_id:
            evt2_file, obs_id = self.select_evt2_file_by_obs_id(requested_obs_id)
            if evt2_file:
                assert obs_id is not None
                return evt2_file, obs_id

            print(f"\n{'='*60}")
            print(f"WARNING: Observation {requested_obs_id} not found")
            print(f"{'='*60}")
            print(f"Could not locate observation {requested_obs_id} " "in workspace.")
            print("\nOptions:")
            print("  1. Enter evt2 file path manually")
            print("  2. Process the latest evt2 file")
            print("  3. Cancel")

            while True:
                choice = input("\nChoose option (1-3): ").strip()
                if choice == "1":
                    return self._manual_file_selection()
                elif choice == "2":
                    return self.select_evt2_file()
                elif choice == "3":
                    return None, None
                else:
                    print("Invalid choice. Please enter 1, 2, or 3.")

        return self.select_evt2_file()

    def select_evt2_file(self) -> tuple[Path, str] | tuple[None, None]:
        """Prompt user to select EVT2 file for processing.

        Returns:
            Tuple of (file_path, obs_id) if selected, else (None, None).
        """
        try:
            latest = self.pipeline_manager.find_latest_evt2()
            print("\nNewest evt2 file found:")
            print(f"  Path: {latest}")
            print("  Modified: " f"{datetime.fromtimestamp(latest.stat().st_mtime)}")

            obs_id = self.extract_obs_id(latest)
            if obs_id is not None:
                print(f"  Observation ID: {obs_id}")
            else:
                print(
                    "Could not extract observation ID from path. "
                    "Please ensure it is in the path or filename."
                )

            print("\nProcess this file? (y/n): ", end="", flush=True)
            response = sys.stdin.readline()

            if not response:
                raise RuntimeError("stdin closed or not interactive")

            response = response.strip().lower()

            if response == "y":
                print(f"Selected: {latest}")
                self.obs_id = obs_id
                if obs_id is not None:
                    return latest, obs_id
                else:
                    print(
                        "Warning: Could not extract observation ID from file "
                        "path. Proceeding without obs_id."
                    )
                    return None, None
            else:
                print("\nManual file selection:")
                return self._manual_file_selection()

        except FileNotFoundError as e:
            print(f"Error: {e}")
            print(
                "\nNo evt2 files found automatically. " "Please provide path manually."
            )
            return self._manual_file_selection()

    def _manual_file_selection(self) -> tuple[Path, str] | tuple[None, None]:
        """Allow user to manually enter file path.

        Returns:
            Tuple of (file_path, obs_id) if valid file entered, else (None, None).
        """
        while True:
            file_path = input(
                "\nEnter path to evt2 file (or 'cancel' to quit): "
            ).strip()

            if file_path.lower() == "cancel":
                print("File selection cancelled.")
                return None, None

            path = Path(file_path)
            if not path.is_absolute():
                path = Path(self.cwd) / path

            if not path.exists():
                print(f"Error: File not found: {path}")
                continue

            if not path.is_file():
                print(f"Error: Path is not a file: {path}")
                continue

            if not path.suffix.startswith(".fits"):
                print(f"Warning: File does not have .fits extension: {path}")
                confirm = input("Continue anyway? (y/n): ").strip().lower()
                if confirm != "y":
                    continue

            if "evt2" not in path.name.lower():
                print(f"Warning: Filename does not contain 'evt2': {path.name}")
                confirm = input("Continue anyway? (y/n): ").strip().lower()
                if confirm != "y":
                    continue

            obs_id = self.extract_obs_id(path)

            print(f"Selected: {path}")

            self.obs_id = obs_id
            if obs_id is not None:
                print(f"Observation ID: {obs_id}")
                return path, obs_id
            else:
                print(
                    "Warning: Could not extract observation ID from file "
                    "path. Proceeding without obs_id."
                )
                return None, None


class DataIngestion:
    """Extract and manage data from FITS files."""

    def __init__(self, evt2_path: str, src_path: str):
        """Initialize data ingestion.

        Args:
            evt2_path: Path to the EVT2 FITS file.
            src_path: Path to the wavdetect source FITS file.
        """
        self.evt2_path = evt2_path
        self.src_path = src_path
        self.metadata: ObservationMetadata | None = None
        self.src: np.ndarray | None = None

    def load_metadata(self) -> ObservationMetadata:
        """Extract metadata from EVT2 FITS header.

        Returns:
            ObservationMetadata instance with OBS_ID, instrument, exposure,
            target name, and coordinates.
        """
        with fits.open(self.evt2_path) as hdul:
            header = cast(Any, hdul[1]).header
            self.metadata = ObservationMetadata(
                obs_id=header.get("OBS_ID", "unknown"),
                instrument=header.get("INSTRUME", "unknown"),
                exposure_time=header.get("EXPOSURE", 0.0),
                target_name=header.get("OBJECT", "unknown"),
                ra=header.get("RA_PNT", 0.0),
                dec=header.get("DEC_PNT", 0.0),
            )
        return self.metadata

    def load_source_list(self) -> Any:
        """Load source list from wavdetect SRCLIST extension.

        Returns:
            Numpy array of source data from SRCLIST HDU.
        """
        with fits.open(self.src_path) as hdul:
            self.src = cast(Any, hdul["SRCLIST"]).data
        source_count = len(self.src) if self.src is not None else 0
        print(f"Number of sources: {source_count}")
        return self.src

    @staticmethod
    def print_source_coordinates(
        src_data, band_label: str = "", save_path: str | None = None
    ) -> None:
        """Print and optionally save raw source coordinates.

        Args:
            src_data: Source array with coordinate and count fields.
            band_label: Optional header for printed output.
            save_path: Optional file path to save coordinate table.
        """
        header = (
            f"{'#':>4}  {'X':>10}  {'Y':>10}  "
            f"{'RA':>12}  {'DEC':>12}  {'NET_COUNTS':>12}"
        )
        if band_label:
            print(f"\nSource coordinates — {band_label}")
        print(header)
        print("=" * len(header))

        lines = [header]
        for i, src in enumerate(src_data):
            line = (
                f"{i+1:>4}  "
                f"{src['X']:>10.2f}  "
                f"{src['Y']:>10.2f}  "
                f"{src['RA']:>12.6f}  "
                f"{src['DEC']:>12.6f}  "
                f"{src['NET_COUNTS']:>12.2f}"
            )
            print(line)
            lines.append(line)

        if save_path:
            Path(save_path).write_text("\n".join(lines))
            print(f"\nCoordinates saved to {save_path}")


class CIAOPreprocessing:
    """Preprocess data using CIAO tools — stateful pipeline."""

    def __init__(self, evt2_path: str, out_dir: str = "ciao_pipeline_out"):
        """Initialize CIAO preprocessing pipeline.

        Args:
            evt2_path: Path to EVT2 FITS file.
            out_dir: Output directory for generated files (default:
                "ciao_pipeline_out").
        """
        self.evt2_path = evt2_path
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.band_images: dict[str, str] = {}
        self.src_data: np.ndarray | None = None
        self.figures: list[Any] = []

    def create_image(
        self,
        src: np.ndarray,
        params: ProcessingParameters,
        save_name: str | None = None,
        band_label: str = "",
    ) -> Any:
        """Create visualization from source data.

        Args:
            src: Source array with X, Y, NET_COUNTS fields.
            params: Processing parameters (for info logging).
            save_name: Optional filename to save visualization.
            band_label: Optional label for plot title.

        Returns:
            Matplotlib Figure object.
        """
        print(f"Creating image with parameters: {params}")

        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(
            src["X"],
            src["Y"],
            s=src["NET_COUNTS"] / src["NET_COUNTS"].max() * 200,
            c=src["Y"],
            cmap="viridis",
            alpha=0.6,
            edgecolors="black",
            linewidth=0.5,
        )

        title_suffix = f" — {band_label}" if band_label else ""
        plt.colorbar(scatter, label="Y position (pitch: low to high)")
        plt.xlabel("X position (scan direction left to right)")
        plt.ylabel("Y position (pitch)")
        plt.title(f"X-ray Source Distribution ({len(src)} sources){title_suffix}")
        plt.tight_layout()

        fig = plt.gcf()
        self.figures.append(fig)

        if save_name:
            save_path = self.out_dir / save_name
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Saved figure to {save_path}")

        plt.close()
        return fig

    def create_band_image(
        self, energy_spec: str = "500:7000", band_name: str = "soft"
    ) -> str:
        """Create FITS image for specific energy band via dmcopy.

        Args:
            energy_spec: Energy range filter string (default: "500:7000").
            band_name: Descriptive band name (default: "soft").

        Returns:
            Path to the created FITS image.
        """
        output_path = self.out_dir / f"image_{band_name}.fits"

        if output_path.exists():
            print(f"Band image already exists: {output_path}")
            self.band_images[band_name] = str(output_path)
            return str(output_path)

        subprocess.run(
            [
                "dmcopy",
                f"{self.evt2_path}[energy={energy_spec}][bin x=::1,y=::1]",
                str(output_path),
                "clobber=yes",
            ],
            check=True,
        )

        self.band_images[band_name] = str(output_path)
        print(f"Created {band_name} band image: {output_path}")
        return str(output_path)

    def create_multi_band_images(self, bands: dict[str, str] | None = None):
        """Create multiple energy band images.

        Args:
            bands: Dictionary of band_name -> energy_spec. If None, uses
                default soft/medium/hard bands.

        Returns:
            Dictionary of band_name -> image_path.
        """
        if bands is None:
            bands = {"soft": "500:2000", "medium": "2000:4000", "hard": "4000:7000"}

        for band_name, energy_spec in bands.items():
            self.create_band_image(energy_spec, band_name)

        return self.band_images

    def get_output_summary(self) -> dict[Any, Any]:
        """Get summary of all generated outputs.

        Returns:
            Dictionary with 'output_directory', 'band_images', 'figures_created'.
        """
        return {
            "output_directory": str(self.out_dir),
            "band_images": self.band_images,
            "figures_created": len(self.figures),
        }


class SourceDetection:
    """Run wavdetect to find X-ray sources."""

    def __init__(self, image_path: str, out_dir: str = "ciao_pipeline_out_02"):
        """Initialize source detection.

        Args:
            image_path: Path to FITS image for wavdetect input.
            out_dir: Output directory for wavdetect products (default:
                "ciao_pipeline_out_02").
        """
        self.image_path = image_path
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.psf_map_path: str | None = None
        self.nbkg_path: str | None = None
        self.src_path: str | None = None
        self.scell_path: str | None = None
        self.image_out_path: str | None = None

    def run_wavdetect(self, params: ProcessingParameters):
        """Run wavdetect with the given parameters.

        Executes PSF map, background map, and wavdetect sequentially using
        pexpect for interactive prompts. Creates source list and derived maps.

        Args:
            params: ProcessingParameters with scale and threshold settings.

        Returns:
            Dictionary of output file paths.
        """

        self.psf_map_path = f"{self.out_dir}/wavdetect_psf.fits"
        self.nbkg_path = f"{self.out_dir}/wavdetect_nbkg.fits"
        self.src_path = f"{self.out_dir}/wavdetect_src.fits"
        self.scell_path = f"{self.out_dir}/wavdetect_scell.fits"
        self.image_out_path = f"{self.out_dir}/wavdetect_image.fits"

        if (
            Path(self.src_path).exists()
            and Path(self.psf_map_path).exists()
            and Path(self.nbkg_path).exists()
        ):
            print(f"Wavdetect outputs already exist in {self.out_dir}")
            print(f"  Source list: {self.src_path}")
            print(f"  PSF map: {self.psf_map_path}")
            print(f"  Background: {self.nbkg_path}")
            return {
                "src_path": self.src_path,
                "scell_path": self.scell_path,
                "image_path": self.image_out_path,
                "psf_map_path": self.psf_map_path,
                "nbkg_path": self.nbkg_path,
            }

        print("running wavdetect")

        log_path = self.out_dir / "wavdetect.log"
        log_file = open(log_path, "wb")

        print("creating PSF map, this could take a bit")
        child = pexpect.spawn(
            f"mkpsfmap infile={self.image_path} outfile={self.psf_map_path} "
            f"energy=1.5 ecf=0.9"
        )
        child.logfile_read = log_file
        child.expect(pexpect.EOF, timeout=3600)
        print(f"Exit status: {child.exitstatus}\n")

        print("creating background map")
        child = pexpect.spawn(
            f'dmimgcalc "{self.image_path}" none {self.nbkg_path} '
            f'op="imgout=1.0" clob+'
        )
        child.logfile_read = log_file
        child.expect(pexpect.EOF, timeout=600)
        print(f"Exit status: {child.exitstatus}\n")

        print("running wavdetect")
        print("scell and image are being created, this could take a bit")
        child = pexpect.spawn(
            f"wavdetect infile={self.image_path} outfile={self.src_path} "
            f'scales="{params.wavdetect_scales}" '
            f"sigthresh={params.significance_threshold} clob+"
        )
        child.logfile_read = log_file
        child.expect("Output source cell image file name")
        child.sendline(self.scell_path)
        child.expect("Output reconstructed image file name")
        child.sendline(self.image_out_path)
        child.expect("Output normalized background file name")
        child.sendline(self.nbkg_path)
        child.expect("Image of the size of the PSF")
        child.sendline(self.psf_map_path)

        child.expect(pexpect.EOF, timeout=3600)
        log_file.close()
        print(f"\nWavdetect finished with exit status: {child.exitstatus}")
        print(f"Full log: {log_path}")
        # Run validity checks
        validity = run_validity_checks(
            src_data=src_data,
            significance_threshold=getattr(params, "significance_threshold", 1e-6),
            exposure=getattr(params, "exposure_time", 0.0),
        )
        print(
            f"  Validity: {validity['n_passed']}/{validity['n_total']} "
            "checks passed | failed: "
            f"{validity['checks_failed'] or 'none'}"
        )
        return {
            "src_path": self.src_path,
            "scell_path": self.scell_path,
            "image_path": self.image_out_path,
            "psf_path": self.psf_map_path,
            "nbkg_path": self.nbkg_path,
        }


class StraussSonification:
    """
    Sonification of X-ray source data using Strauss.

    Defaults are loaded from manual_config.py.
    """

    def __init__(
        self,
        src_data: np.ndarray,
        out_dir: str | None = None,
        duration: float = DURATION,
        peak_volume_range: tuple[float, float] = PEAK_VOLUME_RANGE,
        stereo_spread: tuple[float, float] = STEREO_SPREAD,
        note_len: float = NOTE_LEN,
        sf_preset: int = SF_PRESET,
        x_bounds: tuple[float, float] | None = None,
    ):
        self.src_data = src_data
        self.duration = duration
        self.peak_volume_range = peak_volume_range
        self.stereo_spread = stereo_spread
        self.note_len = note_len
        self.sf_preset = sf_preset
        self.x_bounds = x_bounds
        self.out_dir = Path(out_dir) if out_dir else Path(".")
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.objects: Objects | None = None
        self.score: score.Score | None = None
        self.sampler: Sampler | None = None
        self.soni: Sonification | None = None

    def visualize_sources(self):
        """Quick scatter plot of sources."""
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(
            self.src_data["X"],
            self.src_data["Y"],
            s=(self.src_data["NET_COUNTS"] / self.src_data["NET_COUNTS"].max() * 200),
            c=self.src_data["Y"],
            cmap="viridis",
            alpha=0.6,
            edgecolors="black",
            linewidth=0.5,
        )
        plt.colorbar(scatter, label="Y position (pitch: low -> high)")
        plt.xlabel("X position (scan direction ->)")
        plt.ylabel("Y position (pitch)")
        plt.title(
            f"X-ray Source Distribution ({len(self.src_data)} "
            f"sources)\nY=pitch, Size=intensity, Bar scans left->right"
        )
        plt.tight_layout()
        plt.close()

    def prepare_objects(self):
        """Normalize and convert src data to Strauss Objects."""
        if self.x_bounds is not None:
            x_min, x_max = self.x_bounds
        else:
            x_min, x_max = self.src_data["X"].min(), self.src_data["X"].max()

        x_range = x_max - x_min
        if x_range == 0:
            x_range = 1

        x_norm = (self.src_data["X"] - x_min) / x_range

        y_norm = (self.src_data["Y"] - self.src_data["Y"].min()) / max(
            self.src_data["Y"].max() - self.src_data["Y"].min(), 1
        )
        net_counts = self.src_data["NET_COUNTS"]

        counts_norm = (net_counts - net_counts.min()) / max(
            net_counts.max() - net_counts.min(),
            1,
        )

        self.objects = Objects(["azimuth", "pitch", "volume", "time"])
        self.objects.n_sources = len(self.src_data)

        x_safe = x_norm * ((self.duration - self.note_len) / self.duration)

        self.objects.mapping = {
            "azimuth": x_norm.astype(float),
            "pitch": y_norm.astype(float),
            "volume": (0.3 + counts_norm * 0.7).astype(float),
            "time": x_safe.astype(float),
            "time_evo": [],
        }

        for i in range(self.objects.n_sources):
            t0 = float(x_safe[i] * self.duration)
            t1 = t0 + self.note_len
            self.objects.mapping["time_evo"].append(np.array([t0, t0, t1, t1]))

        setattr(
            self.objects, "mapped_parameters", ["azimuth", "pitch", "volume", "time"]
        )

        print(
            f"  Created {self.objects.n_sources} objects "
            f"(x range: {x_min:.1f}-{x_max:.1f} px, "
            f"first note at {float(x_safe.min() * self.duration):.2f}s)"
        )

    def create_score(self):
        """Create a simple chromatic scale score for the duration."""
        self.score = score.Score(
            chord_sequence=[
                [
                    "C1",
                    "C#1",
                    "D1",
                    "D#1",
                    "E1",
                    "F1",
                    "F#1",
                    "G1",
                    "G#1",
                    "A1",
                    "A#1",
                    "B1",
                    "C2",
                    "C#2",
                    "D2",
                    "D#2",
                    "E2",
                    "F2",
                    "F#2",
                    "G2",
                    "G#2",
                    "A2",
                    "A#2",
                    "B2",
                    "C3",
                    "C#3",
                    "D3",
                    "D#3",
                    "E3",
                    "F3",
                    "F#3",
                    "G3",
                    "G#3",
                    "A3",
                    "A#3",
                    "B3",
                    "C4",
                    "C#4",
                    "D4",
                    "D#4",
                    "E4",
                    "F4",
                    "F#4",
                    "G4",
                    "G#4",
                    "A4",
                    "A#4",
                    "B4",
                ]
            ],
            length=self.duration,
        )

    def create_sample(self):
        """Generate Sample, skipping known-broken presets."""
        preferred = self.sf_preset if self.sf_preset not in BROKEN_PRESETS else 2

        self.sampler = Sampler(sampfiles=SF2, sf_preset=preferred)
        if preferred != self.sf_preset:
            print(
                f"  Note: sf_preset={self.sf_preset} is broken,"
                f" using {preferred} instead"
            )

        for attr in ["parameters", "settings", "presets"]:
            if hasattr(self.sampler, attr):
                getattr(self.sampler, attr)["note_length"] = self.note_len

    def render(self, save_path="xray_sonification.wav", master_volume=MASTER_VOLUME):
        """Render sonification and save stereo file."""
        full_path = self.out_dir / save_path

        if full_path.exists():
            print(f"Sonification file already exists: {full_path}")
            return str(full_path)

        self.soni = Sonification(
            self.score, self.objects, self.sampler, audio_setup="stereo"
        )
        print(f"\nRendering {self.duration}s sonification...")
        self.soni.render()
        print("Render complete!")
        self.soni.notebook_display(show_waveform=True)

        self.soni.save_stereo(str(full_path), master_volume=master_volume)
        print(f"Saved to {full_path}")
        return str(full_path)

    def run_full_pipeline(self, save_path="xray_sonification.wav"):
        """Execute all steps in order."""
        self.visualize_sources()
        self.prepare_objects()
        self.create_score()
        self.create_sample()
        self.render(save_path=save_path)


class SonificationVisualizer(StraussSonification):
    """
    Extends StraussSonification with animated visualization of scanning bar.
    """

    def __init__(
        self,
        src_data: np.ndarray,
        out_dir: str | None = None,
        duration: float = DURATION,
        peak_volume_range: tuple[float, float] = PEAK_VOLUME_RANGE,
        stereo_spread: tuple[float, float] = STEREO_SPREAD,
        note_len: float = NOTE_LEN,
        band_label: str = "",
        sf_preset: int = SF_PRESET,
        x_bounds: tuple[float, float] | None = None,
    ):
        super().__init__(
            src_data,
            out_dir,
            duration,
            peak_volume_range,
            stereo_spread,
            note_len,
            sf_preset,
            x_bounds,
        )
        self.band_label = band_label
        self.x_norm: np.ndarray | None = None
        self.y_norm: np.ndarray | None = None
        self.counts_norm: np.ndarray | None = None

    def prepare_objects(self):
        """Override to store normalized data for visualization."""
        super().prepare_objects()

        self.x_norm = (self.src_data["X"] - self.src_data["X"].min()) / (
            self.src_data["X"].max() - self.src_data["X"].min()
        )
        self.y_norm = (self.src_data["Y"] - self.src_data["Y"].min()) / (
            self.src_data["Y"].max() - self.src_data["Y"].min()
        )
        self.counts_norm = (
            self.src_data["NET_COUNTS"] - self.src_data["NET_COUNTS"].min()
        ) / (self.src_data["NET_COUNTS"].max() - self.src_data["NET_COUNTS"].min())

    def visualize_with_animation(
        self, num_frames: int | None = None, fps: int = 24, save_animation: bool = False
    ):
        """Create an animated visualization showing the scanning bar."""
        if self.x_norm is None:
            print("Must call prepare_objects() first")
            return

        scan_duration = self.duration - self.note_len
        total_duration = self.duration
        hold_time = self.note_len
        end_scan_frame = int(scan_duration * fps)

        if num_frames is None:
            num_frames = int(total_duration * fps)

        print(
            f"Animation timing: scan={scan_duration:.2f}s,"
            f"hold={hold_time:.2f}s, total={total_duration:.2f}s"
        )

        fig, ax = plt.subplots(figsize=(14, 8))

        scatter = ax.scatter(
            self.src_data["X"],
            self.src_data["Y"],
            s=(self.src_data["NET_COUNTS"] / self.src_data["NET_COUNTS"].max() * 200),
            c=self.src_data["Y"],
            cmap="viridis",
            alpha=0.6,
            edgecolors="black",
            linewidth=0.5,
            label="X-ray sources",
        )

        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("Y position (Normalized Pitch)", fontsize=12)

        data_x_min, data_x_max = (self.src_data["X"].min(), self.src_data["X"].max())
        data_x_range = data_x_max - data_x_min

        scan_line = ax.axvline(
            x=data_x_min, color="red", linewidth=3, alpha=0.8, label="Scanning bar"
        )
        ax.legend(loc="upper right", frameon=True)

        title_suffix = f" | {self.band_label}" if self.band_label else ""

        def animate(frame: int):
            if frame < end_scan_frame:
                progress = frame / end_scan_frame
            else:
                progress = 1.0

            x_pos = data_x_min + (progress * data_x_range)
            scan_line.set_xdata([x_pos, x_pos])

            elapsed_time = frame / fps
            ax.set_title(
                f"Chandra X-ray Sonification{title_suffix} |"
                f" Time: {elapsed_time:.2f}s / {total_duration:.2f}s",
                fontsize=14,
            )
            return (scan_line,)

        anim = FuncAnimation(
            fig, animate, frames=num_frames, interval=1000.0 / fps, blit=True
        )

        if save_animation:
            output_path = self.out_dir / "sonification_animation.mp4"
            anim.save(
                str(output_path),
                writer="ffmpeg",
                fps=fps,
                extra_args=["-vcodec", VIDEO_CODEC, "-pix_fmt", "yuv420p"],
            )
            print(f"Animation saved to {output_path}")

        plt.tight_layout()
        return fig, anim

    def run_full_pipeline_with_viz(
        self, save_path="xray_sonification.wav", show_animation=True
    ):
        """Execute sonification pipeline and show animated visualization."""
        self.visualize_sources()
        self.prepare_objects()
        self.create_score()
        self.create_sample()

        print("\nRendering sonification audio (WAV)...")
        wav_path = self.render(save_path=save_path)

        video_silent = None
        if show_animation:
            print("\nGenerating animated visualization (silent video)...")
            try:
                anim_out = self.out_dir / "sonification_animation.mp4"
                anim_result = self.visualize_with_animation(
                    num_frames=None, fps=24, save_animation=True
                )
                if anim_result is None:
                    raise RuntimeError("Animation generation returned no result")
                fig, _ = anim_result
                plt.close(fig)
                video_silent = str(anim_out)
            except Exception as e:
                print(f"Failed to create animation: {e}")

        final_video = None
        if video_silent and wav_path:
            final_video = str(self.out_dir / "sonification_with_audio.mp4")
            ffmpeg_cmd = [
                "ffmpeg",
                "-y",
                "-i",
                video_silent,
                "-i",
                wav_path,
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-b:a",
                "128k",
                "-pix_fmt",
                "yuv420p",
                final_video,
            ]
            try:
                print("\nMuxing audio into video...")
                subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
                print(f"Final video saved to {final_video}")
            except Exception as e:
                print(f"ffmpeg muxing failed: {e}")

        return {
            "wav": wav_path,
            "silent_video": video_silent,
            "final_video": final_video,
        }

    @staticmethod
    def create_overlay_animation(
        src_data_dict: dict[str, tuple[str, np.ndarray]],
        obs_id: str,
        duration: float = DURATION,
        note_len: float = NOTE_LEN,
        x_bounds: tuple[float, float] | None = None,
        fps: int = 24,
    ) -> str | None:
        """
        Create an overlay animation showing multiple bands' sources
        with different shapes.
        """
        if not src_data_dict:
            print("No source data to overlay")
            return None

        scan_duration = duration - note_len
        total_duration = duration
        hold_time = note_len
        end_scan_frame = int(scan_duration * fps)
        num_frames = int(total_duration * fps)

        print("\nCreating overlay animation...")
        print(
            f"Animation timing: scan={scan_duration:.2f}s, "
            f"hold={hold_time:.2f}s, total={total_duration:.2f}s"
        )

        marker_map = {"soft": "o", "hard": "s", "medium": "^", "full": "D"}

        fig, ax = plt.subplots(figsize=(14, 8))

        if x_bounds is None:
            all_x = np.concatenate([data[1]["X"] for data in src_data_dict.values()])
            x_min, x_max = all_x.min(), all_x.max()
        else:
            x_min, x_max = x_bounds

        x_range = x_max - x_min
        if x_range == 0:
            x_range = 1

        for band_name, (band_label, src_data) in src_data_dict.items():
            marker = marker_map.get(band_name, "o")
            ax.scatter(
                src_data["X"],
                src_data["Y"],
                s=src_data["NET_COUNTS"] / src_data["NET_COUNTS"].max() * 200,
                c=src_data["Y"],
                cmap="viridis",
                alpha=0.6,
                edgecolors="black",
                linewidth=0.5,
                marker=marker,
                label=f"{band_label} ({len(src_data)} sources)",
            )

        ax.set_xlabel("X position (scan direction left -> right)", fontsize=12)
        ax.set_ylabel("Y position (pitch)", fontsize=12)
        ax.legend(loc="upper right", frameon=True, fontsize=11)

        scan_line = ax.axvline(
            x=x_min, color="red", linewidth=3, alpha=0.8, label="Scanning bar"
        )

        def animate(frame: int):
            if frame < end_scan_frame:
                progress = frame / end_scan_frame
            else:
                progress = 1.0

            x_pos = x_min + (progress * x_range)
            scan_line.set_xdata([x_pos, x_pos])

            elapsed_time = frame / fps
            ax.set_title(
                f"Chandra X-ray Sonification (Overlay) | "
                f"Time: {elapsed_time:.2f}s / {total_duration:.2f}s",
                fontsize=14,
            )
            return (scan_line,)

        anim = FuncAnimation(
            fig, animate, frames=num_frames, interval=1000.0 / fps, blit=True
        )

        out_dir = Path(f"output_{obs_id}")
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = out_dir / "sonification_overlay_animation.mp4"

        try:
            anim.save(
                str(output_path),
                writer="ffmpeg",
                fps=fps,
                extra_args=["-vcodec", VIDEO_CODEC, "-pix_fmt", "yuv420p"],
            )
            print(f"Overlay animation saved to {output_path}")
            plt.close(fig)
            return str(output_path)
        except Exception as e:
            print(f"Failed to save overlay animation: {e}")
            plt.close(fig)
            return None

    @staticmethod
    def create_overlay_video_with_mixed_audio(
        animation_path: str,
        wav_paths: dict[str, str],
        obs_id: str,
        duration: float = DURATION,
    ) -> str | None:
        """Mix multiple WAV files and mux with overlay animation."""
        if not wav_paths or not animation_path:
            print("Missing animation or audio files for overlay video")
            return None

        wav_list = list(wav_paths.values())
        print(f"\nMixing {len(wav_list)} audio track(s) for overlay video...")

        out_dir = Path(f"output_{obs_id}")
        mixed_wav = str(out_dir / "sonification_overlay_mixed_audio.wav")

        if len(wav_list) == 1:
            import shutil

            shutil.copy(wav_list[0], mixed_wav)
            print(f"  Single audio track: {wav_list[0]}")
        else:
            inputs = []
            for wav_path in wav_list:
                inputs += ["-i", wav_path]

            filter_inputs = "".join(f"[{i}:a]" for i in range(len(wav_list)))
            filter_complex = (
                f"{filter_inputs}amix=" f"inputs={len(wav_list)}:normalize=0[a]"
            )

            mix_cmd = [
                "ffmpeg",
                "-y",
                *inputs,
                "-filter_complex",
                filter_complex,
                "-map",
                "[a]",
                mixed_wav,
            ]
            try:
                subprocess.run(mix_cmd, check=True, capture_output=True)
                print(f"  Mixed audio: {mixed_wav}")
            except subprocess.CalledProcessError as e:
                print(f"  Audio mixing failed: {e.stderr.decode()}")
                return None

        output_path = str(out_dir / "sonification_overlay_with_audio.mp4")
        mux_cmd = [
            "ffmpeg",
            "-y",
            "-i",
            animation_path,
            "-i",
            mixed_wav,
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-pix_fmt",
            "yuv420p",
            output_path,
        ]
        try:
            subprocess.run(mux_cmd, check=True, capture_output=True)
            print(f"  Final overlay video: {output_path}")
            Path(mixed_wav).unlink(missing_ok=True)
            return output_path
        except subprocess.CalledProcessError as e:
            print(f"  Video muxing failed: {e.stderr.decode()}")
            return None

    @staticmethod
    def suggest_processing_mode(num_sources: int) -> tuple[ProcessingMode, str]:
        """
        Suggest a processing mode based on source count (heuristic).
        """
        if num_sources == 0:
            return (
                ProcessingMode.FULL_BAND,
                "No sources detected — staying on full band.",
            )
        elif num_sources <= DUAL_BAND_THRESHOLD:
            return (
                ProcessingMode.FULL_BAND,
                f"{num_sources} sources — "
                "manageable in a single sonification. "
                "Full band recommended.",
            )
        else:
            return (
                ProcessingMode.DUAL_BAND,
                f"{num_sources} sources — splitting into "
                "soft (500-2000 keV) and hard (2000-7000 keV) "
                f"bands will make individual sources easier to hear.",
            )

    @staticmethod
    def select_processing_mode(num_sources: int) -> ProcessingMode:
        """
        Suggest a mode based on source count,
        then let the user confirm or override.
        """
        suggested_mode, explanation = SonificationVisualizer.suggest_processing_mode(
            num_sources
        )

        print("\n" + "=" * 60)
        print("BAND SELECTION")
        print("=" * 60)
        print(f"\n{explanation}")

        if suggested_mode == ProcessingMode.FULL_BAND:
            print("\nSuggested: Full Band (500-7000 keV)  ->  1 sonification")
        else:
            print("\nSuggested: Dual Band")
            print("  Soft (500-2000 keV)  ->  1 sonification")
            print("  Hard (2000-7000 keV) ->  1 sonification")

        print("\nOptions:")
        print("  1. Full Band (500-7000 keV)")
        print("  2. Dual Band (Soft + Hard)")

        suggested_num = "1" if suggested_mode == ProcessingMode.FULL_BAND else "2"

        while True:
            choice = input(
                "\nEnter choice (1 or 2) " f"[suggested: {suggested_num}]: "
            ).strip()
            if choice == "" or choice == suggested_num:
                print(f"Using suggested mode: {suggested_mode.value}")
                return suggested_mode
            elif choice == "1":
                print("Selected: Full Band")
                return ProcessingMode.FULL_BAND
            elif choice == "2":
                print("Selected: Dual Band")
                return ProcessingMode.DUAL_BAND
            else:
                print("Please enter 1 or 2 " "(or press Enter to accept suggestion).")

    @staticmethod
    def run_sonification_for_band(
        src_data,
        obs_id: str,
        band_name: str,
        band_label: str,
        duration: float = DURATION,
        note_len: float = NOTE_LEN,
        sf_preset: int = SF_PRESET,
        x_bounds: tuple[float, float] | None = None,
        show_animation: bool = True,
    ) -> dict:
        """Run the full sonification + visualization pipeline for one band."""
        print(f"\n  [{band_label}]  {len(src_data)} sources")
        print("=" * 60)

        visualizer = SonificationVisualizer(
            src_data=src_data,
            out_dir=f"output_{obs_id}/sonification_{band_name}",
            duration=duration,  # type: ignore
            peak_volume_range=PEAK_VOLUME_RANGE,
            stereo_spread=STEREO_SPREAD,
            note_len=note_len,
            band_label=band_label,
            sf_preset=sf_preset,
            x_bounds=x_bounds,
        )

        return visualizer.run_full_pipeline_with_viz(
            save_path=f"xray_sonification_{band_name}.wav",
            show_animation=show_animation,
        )

    def create_combined_video(
        self, animation_path: str, wav_paths: list[str], output_path: str
    ) -> str | None:
        """Mix multiple WAV files and mux with a video animation."""
        if not wav_paths:
            print("No WAV files to mix")
            return None

        print(
            f"\nMixing {len(wav_paths)} audio track(s) " "and muxing with animation..."
        )
        for p in wav_paths:
            print(f"  + {p}")

        mixed_wav = output_path.replace(".mp4", "_mixed_audio.wav")

        if len(wav_paths) == 1:
            import shutil

            shutil.copy(wav_paths[0], mixed_wav)
        else:
            inputs = []
            for p in wav_paths:
                inputs += ["-i", p]

            filter_inputs = "".join(f"[{i}:a]" for i in range(len(wav_paths)))
            filter_complex = (
                f"{filter_inputs}amix=inputs={len(wav_paths)}:normalize=0[a]"
            )

            mix_cmd = [
                "ffmpeg",
                "-y",
                *inputs,
                "-filter_complex",
                filter_complex,
                "-map",
                "[a]",
                mixed_wav,
            ]
            try:
                subprocess.run(mix_cmd, check=True, capture_output=True)
                print(f"  Audio mix: {mixed_wav}")
            except subprocess.CalledProcessError as e:
                print(f"  Audio mixing failed: {e.stderr.decode()}")
                return None

        mux_cmd = [
            "ffmpeg",
            "-y",
            "-i",
            animation_path,
            "-i",
            mixed_wav,
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-pix_fmt",
            "yuv420p",
            output_path,
        ]
        try:
            subprocess.run(mux_cmd, check=True, capture_output=True)
            print(f"  Final video: {output_path}")
            Path(mixed_wav).unlink(missing_ok=True)
            return output_path
        except subprocess.CalledProcessError as e:
            print(f"  Video muxing failed: {e.stderr.decode()}")
            return None


# endregion


# region Main
# — Main Entrypoint ————————————————————————————————————————————
if __name__ == "__main__":
    print("=" * 60)
    print("X-RAY DATA PROCESSING PIPELINE")
    print("=" * 60)

    parser = argparse.ArgumentParser(
        description="X-ray Data Processing Pipeline with optional flags",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python manual_code.py                         # Interactive mode
  python manual_code.py --check                 # Browse existing sonifications
  python manual_code.py --check --check-id 932  # Open obs_id 932
  python manual_code.py -o 932                  # Process obs_id 932
  python manual_code.py -o 932 --resume         # Force resume (no menu)
  python manual_code.py -o 932 --fresh          # Force fresh start
  python manual_code.py -o 932 --band-mode dual # Force dual band mode
  python manual_code.py -o 932 -s /custom/path  # Custom output dir
  python manual_code.py --no-animation          # Skip visualizations
        """,
    )

    parser.add_argument(
        "-o",
        "--obs-id",
        type=str,
        default=None,
        help="Observation ID to process (numeric, e.g., 932)",
    )
    parser.add_argument(
        "-s",
        "--save-dir",
        type=str,
        default=None,
        help="Custom output directory (default: output_[obs_id]/)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Force resume from previous work (skip menu, error if not found)",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Force fresh start (delete previous output_[obs_id]" "/ without asking)",
    )
    parser.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help="Interactive mode: show decision menus and confirm at each step",
    )
    parser.add_argument(
        "--no-animation",
        action="store_true",
        help="Skip animation/visualization rendering (faster, audio only)",
    )
    parser.add_argument(
        "--band-mode",
        type=str,
        choices=["full", "dual", "triple"],
        default=None,
        help="Force specific band mode (full/dual/triple) - skip suggestion",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=DURATION,
        help=f"Override sonification duration in seconds (default: {DURATION})",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enable debug output and verbose logging"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check for existing sonifications in workspace and browse them",
    )
    parser.add_argument(
        "--check-id",
        type=str,
        default=None,
        help="Check and open specific observation ID (use with --check)",
    )

    args = parser.parse_args()

    # ============================================================
    # HANDLE --check FLAG (Browse existing sonifications)
    # ============================================================
    if args.check:
        catalog = SonificationCatalog(workspace=os.getcwd())
        catalog.scan_workspace()

        if args.check_id:
            catalog.show_sonification_details(args.check_id)
            play = input("\nPlay this sonification? (yes/no): ").strip().lower()
            if play in ["yes", "y"]:
                catalog.open_sonification(args.check_id)
        else:
            selected_obs_id = catalog.interactive_browser()
            if selected_obs_id:
                print(f"\nSelected observation: {selected_obs_id}")

        sys.exit(0)

    # Store args
    requested_obs_id = args.obs_id
    custom_save_dir = args.save_dir
    force_resume = args.resume
    force_fresh = args.fresh
    interactive_mode = args.interactive
    skip_animation = args.no_animation
    forced_band_mode = args.band_mode
    custom_duration = args.duration
    debug_mode = args.debug

    if debug_mode:
        print("\n[DEBUG MODE ENABLED]")
    if requested_obs_id:
        print(f"\nRequested observation ID: {requested_obs_id}")
    if force_resume:
        print("Resume mode: FORCED (will error if no previous work)")
    if force_fresh:
        print("Fresh mode: FORCED (will delete previous work)")
    if forced_band_mode:
        print(f"Band mode: FORCED to {forced_band_mode}")
    if custom_duration != DURATION:
        print(f"Duration: {custom_duration}s (custom)")
    if skip_animation:
        print("Animation: DISABLED (audio only)")

    # ============================================================
    # Step 1: File Selection
    # ============================================================
    print("\n[1] File Selection")
    print("=" * 60)
    selector = FileSelection()
    evt2_file, obs_id = selector.select_evt2_file_with_obs_id_option(requested_obs_id)

    if not evt2_file:
        print("\nNo file selected. Exiting.")
        sys.exit(0)

    if obs_id is None:
        print("\nCould not determine observation ID for selected file.")
        sys.exit(1)

    print(f"\nProcessing observation {obs_id}")

    # ============================================================
    # Check if observation was previously started
    # ============================================================
    pipeline_manager = PipelineManager(os.getcwd())

    try:
        should_resume, resume_reason = pipeline_manager.should_resume_observation(
            obs_id
        )
    except RuntimeError as e:
        print(f"Error: {e}")
        sys.exit(1)

    if should_resume:
        print(f"\nResuming from step: {resume_reason}")
        run_config = RunConfig.load_or_create(obs_id)
        current_step = resume_reason
    else:
        print(f"\nStarting fresh: {resume_reason}")
        run_config = RunConfig.load_or_create(obs_id)
        current_step = "start"

    # ============================================================
    # Decide next step based on current state
    # ============================================================
    decision = pipeline_manager.decide_next_step(
        obs_id=obs_id, current_step=current_step
    )
    print(
        (
            "\nPipeline decision:\n"
            f"  Next step: {decision['next_step']}\n"
            f"  Reasoning: {decision['reasoning']}\n\n"
            f"File: {evt2_file}"
        )
    )

    # ============================================================
    # Step 2: Data Ingestion & Metadata Extraction
    # ============================================================
    if decision["next_step"] in [
        "metadata",
        "full_band_detection",
        "band_selection",
        "band_processing",
        "sonification",
        "finished",
    ]:
        print("\n[2] Loading Metadata")
        print("=" * 60)

        ingestion = DataIngestion(evt2_path=str(evt2_file), src_path="")

        try:
            metadata = ingestion.load_metadata()
            print(f"Observation ID: {metadata.obs_id}")
            print(f"Target: {metadata.target_name}")
            print(f"Instrument: {metadata.instrument}")
            print(f"Exposure Time: {metadata.exposure_time:.2f}s")
            print(f"Coordinates: RA={metadata.ra:.4f}, Dec={metadata.dec:.4f}")
            run_config.log(
                "metadata",
                {
                    "obs_id": metadata.obs_id,
                    "target_name": metadata.target_name,
                    "instrument": metadata.instrument,
                    "exposure_time": metadata.exposure_time,
                    "ra": metadata.ra,
                    "dec": metadata.dec,
                },
                source="fits_header",
            )
        except Exception as e:
            print(f"Error loading metadata: {e}")
            sys.exit(1)
    else:
        print("\n[2] Skipping metadata (already loaded)")
        metadata_dict = run_config.get("metadata")
        metadata = ObservationMetadata(
            obs_id=metadata_dict["obs_id"],
            target_name=metadata_dict["target_name"],
            instrument=metadata_dict["instrument"],
            exposure_time=metadata_dict["exposure_time"],
            ra=metadata_dict["ra"],
            dec=metadata_dict["dec"],
        )
        ingestion = DataIngestion(evt2_path=str(evt2_file), src_path="")

    # ============================================================
    # Step 2.5: Enrich Metadata with Object Information (SIMBAD)
    # ============================================================
    print("\n[2.5] Object Information Lookup")
    print("=" * 60)
    pipeline_manager = PipelineManager(os.getcwd())
    if run_config.replay and run_config.has("object_info"):
        saved = run_config.get("object_info")
        metadata.object_type = saved["object_type"]
        metadata.object_info = saved["object_info"]
        metadata.redshift = saved["redshift"]
    else:
        metadata = pipeline_manager.enrich_metadata_with_object_info(metadata)
        run_config.log(
            "object_info",
            {
                "object_type": metadata.object_type,
                "object_info": metadata.object_info,
                "redshift": metadata.redshift,
            },
            source="simbad",
        )
    if metadata.object_info:
        print(f"Object Info: {metadata.object_info}")

    # ============================================================
    # Step 3: Initial Full-Band Detection
    # ============================================================
    print("\n[3] Initial Source Detection (Full Band)")
    print("=" * 60)
    print(
        "Running full-band detection first to "
        "count sources and choose processing mode..."
    )

    preprocessing = CIAOPreprocessing(
        evt2_path=str(evt2_file), out_dir=f"output_{obs_id}/preprocessing"
    )

    # Load or build detection parameters from manual_config defaults
    if run_config.replay and run_config.has("detection_params"):
        saved = run_config.get("detection_params")
        detection_params = ProcessingParameters(
            wavdetect_scales=saved["wavdetect_scales"],
            significance_threshold=saved["significance_threshold"],
            reasoning=saved["reasoning"],
        )
    else:
        detection_params = ProcessingParameters()  # manual_config defaults
        detection_params = PipelineManager.validate_detection_params(
            detection_params, metadata.exposure_time
        )
        run_config.log(
            "detection_params",
            {
                "wavdetect_scales": detection_params.wavdetect_scales,
                "significance_threshold": detection_params.significance_threshold,
                "reasoning": detection_params.reasoning,
            },
            source="manual_config",
        )

    try:
        full_band_image = preprocessing.create_band_image(
            energy_spec=ENERGY_BANDS["full"], band_name="full"
        )
        detector_full = SourceDetection(
            image_path=full_band_image, out_dir=f"output_{obs_id}/detection_full"
        )

        full_detection = detector_full.run_wavdetect(detection_params)
        ingestion.src_path = full_detection["src_path"]
        full_src_data = ingestion.load_source_list()
        num_sources = len(full_src_data)
        print(f"\nFull band: {num_sources} sources detected")

        # Global x bounds used to sync all bands
        x_global_min = float(full_src_data["X"].min())  # type: ignore
        x_global_max = float(full_src_data["X"].max())  # type: ignore
        x_bounds = (x_global_min, x_global_max)
        print(f"Global x range: {x_global_min:.1f} - {x_global_max:.1f} px")

        run_config.log(
            "full_band_detection",
            {
                "num_sources": num_sources,
                "x_global_min": x_global_min,
                "x_global_max": x_global_max,
            },
            source="wavdetect",
        )

        DataIngestion.print_source_coordinates(
            full_src_data,
            band_label=BAND_LABELS["full"],
            save_path=f"output_{obs_id}/source_coords_full.txt",
        )

    except Exception as e:
        print(f"Error in initial detection: {e}")
        traceback.print_exc()
        sys.exit(1)

    if num_sources == 0:
        print("\nWarning: No sources detected!")
        proceed = input("Continue anyway? (y/n): ").strip().lower()
        if proceed != "y":
            sys.exit(0)
    elif num_sources < 5:
        print(f"\nOnly {num_sources} sources found — quite sparse.")
        print("Consider adjusting SIGNIFICANCE_THRESHOLD in manual_config.py")

    # ============================================================
    # Step 4: Band Mode Selection (heuristic)
    # ============================================================
    print("\n[4] Band Mode Selection")
    print("=" * 60)

    if run_config.replay and run_config.has("processing_mode"):
        mode_str = run_config.get("processing_mode")
        processing_mode = ProcessingMode(mode_str)
    else:
        if forced_band_mode:
            mode_map = {
                "full": ProcessingMode.FULL_BAND,
                "dual": ProcessingMode.DUAL_BAND,
                "triple": ProcessingMode.TRIPLE_BAND,
            }
            processing_mode = mode_map[forced_band_mode]
            print(f"Band mode forced via CLI: {processing_mode.value}")
        else:
            processing_mode = SonificationVisualizer.select_processing_mode(num_sources)
        run_config.log("processing_mode", processing_mode.value, source="manual")

    # ============================================================
    # Step 5: Process bands
    # ============================================================
    print("\n[5] Processing")
    print("=" * 60)

    sonification_results: dict[str, dict] = {}
    make_overlay = SonificationVisualizer.create_overlay_video_with_mixed_audio
    # Build sonification config from manual_config (or saved run config)
    soni_config = SonificationConfig(
        duration=custom_duration,
        peak_volume_range=PEAK_VOLUME_RANGE,
        stereo_spread=STEREO_SPREAD,
    )

    if processing_mode == ProcessingMode.FULL_BAND:
        print("\nUsing full band (500-7000 keV)")

        preprocessing.create_image(
            src=full_src_data,
            params=detection_params,
            save_name=f"source_distribution_full_{obs_id}.png",
            band_label=BAND_LABELS["full"],
        )

        soni_key = "sonification_config_full"
        if run_config.replay and run_config.has(soni_key):
            saved = run_config.get(soni_key)
            soni_config = SonificationConfig(
                duration=saved["duration"],
                peak_volume_range=tuple(saved["peak_volume_range"]),
                stereo_spread=tuple(saved["stereo_spread"]),
            )
        else:
            run_config.log(
                soni_key,
                {
                    "duration": soni_config.duration,
                    "peak_volume_range": list(soni_config.peak_volume_range),
                    "stereo_spread": list(soni_config.stereo_spread),
                },
                source="manual_config",
            )

        result = SonificationVisualizer.run_sonification_for_band(
            src_data=full_src_data,
            obs_id=obs_id,  # type: ignore
            band_name="full",
            band_label=BAND_LABELS["full"],
            duration=soni_config.duration,
            note_len=NOTE_LEN,
            x_bounds=x_bounds,
            show_animation=not skip_animation,
        )
        sonification_results["full"] = result

    elif processing_mode == ProcessingMode.DUAL_BAND:
        bands = {
            "soft": (ENERGY_BANDS["soft"], BAND_LABELS["soft"]),
            "hard": (ENERGY_BANDS["hard"], BAND_LABELS["hard"]),
        }

        for band_name, (energy_spec, band_label) in bands.items():
            print(f"\n--- {band_label} ---")

            try:
                band_image = preprocessing.create_band_image(
                    energy_spec=energy_spec, band_name=band_name
                )

                detector = SourceDetection(
                    image_path=band_image,
                    out_dir=f"output_{obs_id}/detection_{band_name}",
                )
                det_results = detector.run_wavdetect(detection_params)

                soni_key = f"sonification_config_{band_name}"
                if run_config.replay and run_config.has(soni_key):
                    saved = run_config.get(soni_key)
                    soni_config = SonificationConfig(
                        duration=saved["duration"],
                        peak_volume_range=tuple(saved["peak_volume_range"]),
                        stereo_spread=tuple(saved["stereo_spread"]),
                    )
                else:
                    soni_config = SonificationConfig(
                        duration=custom_duration,
                        peak_volume_range=PEAK_VOLUME_RANGE,
                        stereo_spread=STEREO_SPREAD,
                    )
                    run_config.log(
                        soni_key,
                        {
                            "duration": soni_config.duration,
                            "peak_volume_range": list(soni_config.peak_volume_range),
                            "stereo_spread": list(soni_config.stereo_spread),
                        },
                        source="manual_config",
                    )

                band_ingestion = DataIngestion(
                    evt2_path=str(evt2_file), src_path=det_results["src_path"]
                )
                src_data = band_ingestion.load_source_list()
                n = len(src_data)
                print(f"  {n} sources in {band_label}")

                if n == 0:
                    print("  Skipping sonification for " f"{band_name} (no sources)")
                    continue

                preprocessing.create_image(
                    src=src_data,
                    params=detection_params,
                    save_name=f"source_distribution_{band_name}_{obs_id}.png",
                    band_label=band_label,
                )

                result = SonificationVisualizer.run_sonification_for_band(
                    src_data=src_data,
                    obs_id=obs_id,  # type: ignore
                    band_name=band_name,
                    band_label=band_label,
                    duration=soni_config.duration,
                    note_len=NOTE_LEN,
                    x_bounds=x_bounds,
                    show_animation=not skip_animation,
                )
                sonification_results[band_name] = result

            except Exception as e:
                print(f"  Error processing {band_name} band: {e}")
                traceback.print_exc()

        # Optional stereo mix
        if "soft" in sonification_results and "hard" in sonification_results:
            print("\n" + "=" * 60)
            create_mix = (
                input("Create stereo mix (soft=left ear, hard=right ear)? (y/n): ")
                .strip()
                .lower()
            )
            if create_mix == "y":
                stereo_out = f"output_{obs_id}/sonification_stereo_mix.wav"
                ffmpeg_merge = [
                    "ffmpeg",
                    "-y",
                    "-i",
                    sonification_results["soft"]["wav"],
                    "-i",
                    sonification_results["hard"]["wav"],
                    "-filter_complex",
                    "[0:a][1:a]amerge=inputs=2[a]",
                    "-map",
                    "[a]",
                    "-ac",
                    "2",
                    stereo_out,
                ]
                try:
                    subprocess.run(ffmpeg_merge, check=True, capture_output=True)
                    print(f"  Stereo mix saved to: {stereo_out}")
                except Exception as e:
                    print(f"  Stereo mix failed: {e}")

            # Create overlay animation with mixed audio
            print("\n" + "=" * 60)
            create_overlay = (
                input(
                    "Create overlay visualization " "(both bands in one plot)? (y/n): "
                )
                .strip()
                .lower()
            )
            if create_overlay == "y":
                try:
                    soft_ingestion = DataIngestion(
                        evt2_path=str(evt2_file),
                        src_path=f"output_{obs_id}/"
                        "detection_soft/"
                        "wavdetect_src.fits",
                    )
                    soft_src = soft_ingestion.load_source_list()

                    hard_ingestion = DataIngestion(
                        evt2_path=str(evt2_file),
                        src_path=f"output_{obs_id}/"
                        "detection_hard/"
                        "wavdetect_src.fits",
                    )
                    hard_src = hard_ingestion.load_source_list()

                    src_data_dict = {
                        "soft": (BAND_LABELS["soft"], soft_src),
                        "hard": (BAND_LABELS["hard"], hard_src),
                    }

                    animation_path = SonificationVisualizer.create_overlay_animation(
                        src_data_dict=src_data_dict,  # type: ignore
                        obs_id=obs_id,  # type: ignore
                        duration=custom_duration,
                        note_len=NOTE_LEN,
                        x_bounds=(x_global_min, x_global_max),
                        fps=24,
                    )

                    if animation_path:
                        wav_dict = {
                            "soft": sonification_results["soft"]["wav"],
                            "hard": sonification_results["hard"]["wav"],
                        }

                        final_overlay_video = make_overlay(
                            animation_path=animation_path,
                            wav_paths=wav_dict,
                            obs_id=obs_id,  # type: ignore
                            duration=custom_duration,
                        )

                        if final_overlay_video:
                            sonification_results["overlay"] = {
                                "wav": "mixed (soft + hard)",
                                "silent_video": animation_path,
                                "final_video": final_overlay_video,
                            }

                except Exception as e:
                    print(f"  Error creating overlay: {e}")
                    traceback.print_exc()

    # ============================================================
    # Step 6: Summary
    # ============================================================
    print("\n[6] Pipeline Summary")
    print("=" * 60)
    print(f"Observation ID:    {obs_id}")
    print(f"Target:            {metadata.target_name}")
    print(f"Full band sources: {num_sources}")
    print(f"Processing mode:   {processing_mode.value}")
    print(f"\nOutputs in: output_{obs_id}/")

    for band_name, result in sonification_results.items():
        print(f"\n  [{band_name}]")
        print(f"    Audio: {result.get('wav', 'N/A')}")
        print(f"    Video: {result.get('final_video', 'N/A')}")

    print("\n" + "=" * 60)
    run_config.summary()
# endregion
