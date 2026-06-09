"""
Agentic Pipeline — Main Orchestrator.

Entry point for the truly agentic X-ray data sonification pipeline.
The PipelineDirector agent drives phases 1-3 via LLM reasoning;
phase 4 (per-band processing) is a structured loop whose band
strategy was chosen by an agent.

Usage:
    python -m chandrasonify.agentic_code_base [--obs-id ID] ...
    see 'python -m chandrasonify.agentic_code_base --help' for details.
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
import time
from pathlib import Path
from typing import Any, cast

import numpy as np
from astropy.io import fits

from chandrasonify.agentic_tools import (
    ProcessingMode,
    tool_find_evt2_files,
    tool_mux_audio_video,
    tool_print_source_coordinates,
)
from chandrasonify.agentic_agents import (
    AgentMemory,
    BandStrategist,
    DetectionOptimizer,
    ExecutionHistory,
    ObservationResearcher,
    OverlayComposer,
    PipelineDirector,
    QualityEvaluator,
    SonificationExpert,
)
from chandrasonify.run_config import RunConfig
from chandrasonify.config import print_config  # noqa: E402
from chandrasonify.agentic_config import ENERGY_BANDS, BAND_LABELS, DURATION

# region Pipeline Utilities
# — File Selection ——————————————————————————————————————————————


class FileSelection:
    """Interactive or argument-driven EVT2 file selector.

    Attributes:
        workspace (Path): Root directory to search for EVT2 files.
    """

    def __init__(self, workspace: str):
        self.workspace = Path(workspace)

    def find_all_evt2(self) -> list[Path]:
        """Find all EVT2 FITS files in workspace.

        Returns:
            list[Path]: List of EVT2 file paths.
        """
        result = tool_find_evt2_files(str(self.workspace))
        if result.success:
            return [Path(p) for p in result.data.get("files", [])]
        return []

    @staticmethod
    def present_choices(files: list[Path]) -> None:
        """Present numbered list of EVT2 files.

        Args:
            files (list[Path]): EVT2 files to display.
        """
        print(f"\n{'='*60}")
        print(f"  Found {len(files)} EVT2 file(s):")
        print(f"{'='*60}")
        for i, f in enumerate(files, 1):
            size_mb = f.stat().st_size / 1024 / 1024
            print(f"  [{i:>3}]  {f.name}  ({size_mb:.1f} MB)")
            print(f"        {f.parent}")
        print()

    def select_file(self, obs_id: str | None = None) -> Path | None:
        """Select an EVT2 file, optionally auto-matching by obs_id.

        Args:
            obs_id (str | None): Observation ID to auto-match files (default None).

        Returns:
            Path | None: Selected file path or None.
        """
        files = self.find_all_evt2()
        if not files:
            print("No EVT2 files found in workspace.")
            return None

        if obs_id:
            matches = [f for f in files if obs_id in f.name]
            if matches:
                print(f"  Auto-selected: {matches[0].name}")
                return matches[0]
            print(f"  No file matching obs_id '{obs_id}'")

        self.present_choices(files)
        while True:
            try:
                choice = input("Select file number (or 'q' to quit): ")
                if choice.strip().lower() == "q":
                    return None
                idx = int(choice) - 1
                if 0 <= idx < len(files):
                    return files[idx]
                print(f"  Enter 1–{len(files)}")
            except (ValueError, EOFError):
                return files[0] if files else None


# — Sonification Catalog —————————————————————————————————————————


class SonificationCatalog:
    """JSON catalog of processed observations.

    Attributes:
        workspace (Path): Root directory.
        catalog_path (Path): Path to catalog JSON file.
        entries (list[dict[str, Any]]): Catalog entries.
    """

    def __init__(
        self,
        workspace: str,
        catalog_name: str = "sonification_catalog.json",
    ):
        self.workspace = Path(workspace)
        self.catalog_path = self.workspace / catalog_name
        self.entries: list[dict[str, Any]] = []
        self._load()

    def _load(self):
        """
        Internal method to load the catalog entries from the JSON file
        if it exists.
        If the file is present and contains valid JSON, the entries are
        loaded into the `entries` attribute.
        If the file does not exist or contains invalid JSON, the 'entries'
        list is initialized as empty.
        This method is called during the initialization of
        the SonificationCatalog instance to ensure that any previously
        processed observations are available for browsing and selection.
        """
        if self.catalog_path.exists():
            try:
                self.entries = json.loads(self.catalog_path.read_text())
            except json.JSONDecodeError:
                self.entries = []

    def _save(self):
        """
        Internal method to save the current catalog entries to the JSON file.
        """
        self.catalog_path.write_text(json.dumps(self.entries, indent=2))

    def add_entry(
        self,
        obs_id: str,
        target_name: str = "",
        evt2_path: str = "",
        notes: str = "",
    ):
        """
        Add an entry to the sonification catalog.

        Args:
            obs_id (str): The observation ID associated with the entry, which serves as
                a unique identifier for the processed observation.
            target_name (str): The name of the target object observed, which helps users
                identify the entry in the catalog.
            evt2_path (str): The file path to the EVT2 FITS file that was used for
                processing this observation, providing a reference to the data source.
            notes (str): Any additional notes or comments about the processing.
        """
        entry = {
            "obs_id": obs_id,
            "target_name": target_name,
            "evt2_path": evt2_path,
            "notes": notes,
        }
        existing = [e for e in self.entries if e["obs_id"] == obs_id]
        if existing:
            existing[0].update(entry)
        else:
            self.entries.append(entry)
        self._save()

    def list_entries(self) -> list[dict[str, Any]]:
        """
        List entries in the catalog.
        This is a basic function right now, but in the future we might want to add
        filtering and sorting here.

        Returns:
            list[dict[str, Any]]: A list of dictionaries representing the entries in the
                catalog.
        """
        return list(self.entries)

    def get_entry(self, obs_id: str) -> dict[str, Any] | None:
        """
        Get entries matching an obs_id.

        Args:
            obs_id (str): The observation ID to search for in the catalog entries.

        Returns:
            dict[str, Any] | None: A dictionary representing the catalog entry that
                matches the provided obs_id,
                or None if no matching entry is found.
        """
        for e in self.entries:
            if e["obs_id"] == obs_id:
                return e
        return None

    def browse_interactive(self) -> str | None:
        """
        Interactive browser; returns selected obs_id or None.

        Returns:
            str | None: The observation ID of the entry selected by the user from the
                interactive catalog browsing interface, or None if the user chooses to
                skip selection or if the catalog is empty.
        """
        if not self.entries:
            print("  Catalog is empty.")
            return None
        print(f"\n{'='*60}")
        print("  Sonification Catalog")
        print(f"{'='*60}")
        for i, e in enumerate(self.entries, 1):
            print(f"  [{i}] {e['obs_id']:>8} — " f"{e.get('target_name', '?')}")
        print()
        try:
            choice = input("Select entry (or Enter to skip): ").strip()
            if not choice:
                return None
            idx = int(choice) - 1
            if 0 <= idx < len(self.entries):
                return self.entries[idx]["obs_id"]
        except (ValueError, EOFError):
            pass
        return None


def get_bands_for_mode(
    mode: ProcessingMode,
) -> list[tuple[str, str, str]]:
    """Get bands for processing mode.

    Args:
        mode (ProcessingMode): Processing mode.

    Returns:
        list[tuple[str, str, str]]: (band_name, energy_spec, band_label) tuples.
    """
    if mode == ProcessingMode.DUAL_BAND:
        return [
            ("soft", ENERGY_BANDS["soft"], BAND_LABELS["soft"]),
            ("hard", ENERGY_BANDS["hard"], BAND_LABELS["hard"]),
        ]
    if mode == ProcessingMode.TRIPLE_BAND:
        return [
            ("soft", ENERGY_BANDS["soft"], BAND_LABELS["soft"]),
            ("medium", ENERGY_BANDS["medium"], BAND_LABELS["medium"]),
            ("hard", ENERGY_BANDS["hard"], BAND_LABELS["hard"]),
        ]
    return [
        ("full", ENERGY_BANDS["full"], BAND_LABELS["full"]),
    ]


def mark_phase_complete(
    state: dict[str, Any],
    phase_name: str,
    step_number: int,
) -> None:
    """
    Record phase completion with timestamp and step number.

    Args:
        state (dict[str, Any]): The current state of the pipeline,
            which will be updated to include the completion status
            of the specified phase.
        phase_name (str): The name of the phase that has been completed,
            which will be used as a key in the state to track the completion status.
        step_number (int): The current step number in the pipeline execution,
            which will be recorded alongside the phase completion to provide context on
            when the phase was completed during the orchestration process.
    """
    state["phase_completions"][phase_name] = {
        "step": step_number,
        "timestamp": time.time(),
    }
    print(f"  → Phase '{phase_name}' marked complete at step {step_number}")


def is_phase_complete(
    state: dict[str, Any],
    phase_name: str,
) -> bool:
    """
    Check if a phase has been marked complete.

    Args:
        state (dict[str, Any]): The current state of the pipeline, which should include
            a "phase_completions" dictionary that tracks the completion
            status of various phases.
        phase_name (str): The name of the phase to check for completion status, which
            should correspond to a key in the "phase_completions" dictionary
            within the state.

    Returns:
        bool: True if the specified phase has been marked as complete in the state,
            False otherwise. The method checks if the phase_name exists as a key in the
                "phase_completions" dictionary of the state, which indicates that the
                phase has been completed and recorded with a timestamp and step number.
    """
    return phase_name in state.get("phase_completions", {})


def run_validity_checks(
    src_data,
    significance_threshold: float,
    exposure: float,
    chip_size: int = 1024,
    edge_margin: int = 50,
) -> dict[str, Any]:
    """
    Internal validity checks on wavdetect results.
    Returns metrics + flags the director uses to decide whether to rerun.

    Args:
        src_data: The source data output from wavdetect,
            which typically includes information about detected sources such as their
            coordinates, counts, and other relevant parameters that can be used to
            assess the quality and reliability of the source detection results.
        significance_threshold (float): The significance threshold used in the wavdetect
            source detection process, which can be used to estimate the expected number
            of false positives in the detected sources based on the area of the chip and
            the number of sources detected.
        exposure (float): The exposure time of the observation, which can be relevant
            for assessing the expected counts and the reliability of the detected
            sources.
        chip_size (int): The size of the chip, which can be used to calculate the area
            of the chip and assess the density of detected sources.
        edge_margin (int): The margin from the edges of the chip to consider when
            calculating the edge fraction of detected sources, which can be an indicator
            of potential issues with source detection near the edges of the chip.

    Returns:
        dict[str, Any]: A dictionary containing various metrics and flags related to the
            validity of the wavdetect results, including the number of sources detected,
            expected false positives, false positive rate, edge fraction, negative
            counts fraction, median counts, median OAA, and flags indicating which
            checks passed or failed.
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


def extract_obs_id(evt2_path: Path) -> str:
    """
    Extract observation ID from EVT2 filename.

    Args:
        evt2_path (Path): The file path to the EVT2 FITS file from which to extract the
            observation ID. The method attempts to read the OBS_ID from the FITS header,
            and if that fails, it falls back to parsing the filename to extract a
            plausible observation ID by removing common suffixes like "_evt2" and
            "_repro".

    Returns:
        str: The extracted observation ID as a string.
            If the OBS_ID is successfully read from the FITS header, it is returned;
            otherwise, a fallback method is used to derive the observation ID from the
            filename.
            If neither method yields a valid observation ID, "unknown" is returned as a
            default value.
    """
    try:
        with fits.open(evt2_path) as hdul:
            header = cast(Any, hdul[1]).header
            return str(header.get("OBS_ID", "unknown"))
    except Exception:
        stem = evt2_path.stem.replace("_evt2", "").replace("_repro", "")
        return stem


def print_summary(
    state: dict[str, Any],
    memory: AgentMemory,
) -> None:
    """
    Print pipeline summary.

    Args:
        state (dict[str, Any]): The current state of the pipeline, which may include
            information about the observation metadata, number of sources detected,
            processing mode, sonification results, and any other relevant data that
            provides an overview of the pipeline's execution and outcomes.
        memory (AgentMemory): The memory of the director agent, which may contain a
            summary of the execution history, decisions made, and any other relevant
            information that can be included in the pipeline summary to provide insights
            into the orchestration process and the reasoning behind the decisions taken
            by the director throughout the pipeline execution.
    """
    print(f"\n{'='*60}")
    print("  PIPELINE SUMMARY")
    print(f"{'='*60}")

    if state.get("metadata"):
        meta = state["metadata"]
        print(f"  Target:     {meta.target_name}")
        print(f"  OBS_ID:     {meta.obs_id}")
        print(f"  Instrument: {meta.instrument}")
        print(f"  Exposure:   {meta.exposure_time:.0f}s")
        if meta.object_type:
            print(f"  Type:       {meta.object_type}")

    print(f"  Sources:    {state.get('num_sources', 0)}")

    mode = state.get("processing_mode")
    if mode:
        print(f"  Band mode:  {mode.value}")

    results = state.get("sonification_results", {})
    if results:
        print("\n  Outputs:")
        for band, info in results.items():
            print(f"    [{band}]")
            if info.get("wav"):
                print(f"      Audio: {info['wav']}")
            if info.get("final_video"):
                print(f"      Video: {info['final_video']}")

    if state.get("overlay_video"):
        print("    [overlay]")
        print(f"      Video: {state['overlay_video']}")

    print(f"\n  {memory.summary()}")
    print(f"{'='*60}\n")


# endregion


# region Execution
# — CLI ———————————————————————————————————————————————————————————


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Agentic Chandra X-ray sonification pipeline",
        epilog=textwrap.dedent("""\
            examples:
              python -m chandrasonify.agentic_code_base
              python -m chandrasonify.agentic_code_base --obs-id 932
              python -m chandrasonify.agentic_code_base --band-mode dual_band
              python -m chandrasonify.agentic_code_base --replay
        """),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-o", "--obs-id", type=str, default=None, help="Observation ID to process"
    )
    parser.add_argument(
        "-e", "--evt2", type=str, default=None, help="Direct path to EVT2 FITS file"
    )
    parser.add_argument(
        "-b",
        "--band-mode",
        type=str,
        default=None,
        choices=["full_band", "dual_band", "triple_band"],
        help="Force a band processing mode",
    )
    parser.add_argument(
        "--no-animation", action="store_true", help="Skip animation rendering"
    )
    parser.add_argument(
        "--replay", action="store_true", help="Replay from saved run_config.json"
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Start fresh run (don't ask to replay existing config)",
    )
    parser.add_argument(
        "--browse", action="store_true", help="Browse sonification catalog"
    )
    parser.add_argument(
        "--workspace",
        type=str,
        default=str(Path(__file__).resolve().parent.parent),
        help="Workspace root directory",
    )
    return parser.parse_args()


# — 5. Main Pipeline ———————————————————————————————————————————————


def run_pipeline(args: argparse.Namespace) -> None:
    """Run the agentic pipeline.

    Args:
        args (argparse.Namespace): Command-line arguments.
    """
    workspace = args.workspace
    print_config()

    # — Catalog ——————————————————————————————————————————————————
    catalog = SonificationCatalog(workspace)
    if args.browse:
        selected = catalog.browse_interactive()
        if selected:
            args.obs_id = selected
        else:
            print("Nothing selected.")
            return

    # — File Selection ———————————————————————————————————————————
    if args.evt2:
        evt2_path = Path(args.evt2)
        if not evt2_path.exists():
            print(f"File not found: {evt2_path}")
            sys.exit(1)
    else:
        selector = FileSelection(workspace)
        evt2_path = selector.select_file(obs_id=args.obs_id)
        if evt2_path is None:
            print("No file selected.")
            return

    obs_id = extract_obs_id(evt2_path)
    out_dir = Path(f"output_{obs_id}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  Processing: {evt2_path.name}")
    print(f"  OBS_ID:     {obs_id}")
    print(f"  Output:     {out_dir}/")

    # — RunConfig for Reproducibility ————————————————————————————
    run_config = RunConfig.load_or_create(obs_id, out_root=".", fresh=args.fresh)
    if args.replay:
        run_config.replay = True
        print("  Mode: REPLAY (using saved decisions)")

    # — Create Agent System ———————————————————————————————————————
    memory = AgentMemory()

    researcher = ObservationResearcher(memory)
    detector = DetectionOptimizer(memory)
    strategist = BandStrategist(memory)
    expert = SonificationExpert(memory)
    overlay_composer = OverlayComposer(memory)
    evaluator = QualityEvaluator(memory)

    director = PipelineDirector(
        memory,
        {
            "observation_researcher": researcher,
            "detection_optimizer": detector,
            "band_strategist": strategist,
            "sonification_expert": expert,
            "quality_evaluator": evaluator,
        },
        run_config=run_config,
    )

    # — Pipeline State ————————————————————————————————————————————
    state: dict[str, Any] = {
        "obs_id": obs_id,
        "evt2_path": str(evt2_path),
        "output_dir": str(out_dir),
        "run_config": run_config,
        "show_animation": not args.no_animation,
        "forced_band_mode": args.band_mode,
        "execution_history": ExecutionHistory(),
        "phase_completions": {},
        # legacy booleans
        "metadata_loaded": False,
        "detection_complete": False,
        "band_mode_selected": False,
        "sonification_complete": False,
        # data
        "metadata": None,
        "src_data_full": None,
        "num_sources": 0,
        "x_bounds": None,
        "detection_params": None,
        "processing_mode": None,
        "validity_report": None,
        "detection_rerun_attempted": False,
        # band tracking (populated after band_strategist runs)
        "bands_to_sonify": [],  # list of (band_name, energy_spec, band_label)
        "bands_pending_detection": [],  # bands that still need detection
        "src_data_per_band": {},  # band_name → (band_label, src_data, x_bounds)
        "bands_sonified": [],  # completed band names
        "sonification_results": {},
    }

    # replay: skip metadata if already saved
    if "metadata" in run_config:
        state["metadata_loaded"] = True
        state["metadata"] = run_config.get("metadata")
        state["phase_completions"]["metadata"] = {
            "step": 0,
            "timestamp": time.time(),
            "source": "replay",
        }
        print("  → Loaded metadata from saved run_config (replay mode)")

    # — Director-Driven Agentic Loop (all phases) ————————————————
    max_steps = 25
    for step in range(1, max_steps + 1):
        decision = director.decide_next_action(state)
        action = decision.get("action", "finish")

        if action == "finish":
            print(
                f"\n  [Step {step}] Director: finish — "
                f"{decision.get('reasoning', '')}"
            )
            break

        agent_name = decision.get("agent", "")
        if agent_name not in director.agents:
            print(f"  [Step {step}] ⚠ Unknown agent '{agent_name}'. Stopping.")
            break

        print(f"\n  [Step {step}] Director → {agent_name}")

        # ── observation_researcher ───────────────────────────────
        if agent_name == "observation_researcher":
            result = researcher.run(
                {
                    "evt2_path": state["evt2_path"],
                    "execution_history": state["execution_history"],
                }
            )
            if result.get("success"):
                director.director_memory.record_outcome(
                    step,
                    {
                        "target": result["metadata"].target_name,
                        "object_type": result["metadata"].object_type,
                        "simbad_ok": result["metadata"].object_type is not None,
                    },
                )
                state["metadata"] = result["metadata"]
                state["metadata_loaded"] = True
                mark_phase_complete(state, "metadata", step)
                state["execution_history"].record(
                    agent_name,
                    decision="Extracted and enriched observation metadata",
                    parameters={
                        "target": result["metadata"].target_name,
                        "object_type": result["metadata"].object_type,
                    },
                    reasoning="Gathered FITS headers and SIMBAD enrichment",
                )
                run_config.log(
                    "metadata", result["metadata"].model_dump(), source="fits_header"
                )

        # ── detection_optimizer ──────────────────────────────────
        elif agent_name == "detection_optimizer":
            # Are we doing per-band detection, or the initial full-band pass?
            pending = state.get("bands_pending_detection", [])
            if pending:
                # Per-band: detect the next pending band
                band_name, energy_spec, band_label = pending[0]
                det_dir = str(out_dir / f"detection_{band_name}")
            else:
                # Initial full-band detection
                band_name, energy_spec, band_label = (
                    "full",
                    ENERGY_BANDS["full"],
                    BAND_LABELS["full"],
                )
                det_dir = str(out_dir / "detection_full")

            result = detector.run(
                {
                    "metadata": state["metadata"],
                    "evt2_path": state["evt2_path"],
                    "out_dir": det_dir,
                    "band_name": band_name,
                    "energy_spec": energy_spec,
                    "band_label": band_label,
                    "run_config": run_config,
                    "execution_history": state["execution_history"],
                }
            )

            if result.get("success"):
                src_data = result["src_data"]
                x_bounds = result.get("x_bounds")
                n_src = result["num_sources"]
                params = result.get("params", {})

                # Run validity checks
                validity = run_validity_checks(
                    src_data=src_data,
                    significance_threshold=getattr(
                        params, "significance_threshold", 1e-6
                    ),
                    exposure=state["metadata"].exposure_time,
                )
                print(
                    f"  Validity: {validity['n_passed']}/{validity['n_total']} "
                    f"checks passed | failed: {validity['checks_failed'] or 'none'}"
                )

                if pending:
                    # Store per-band result
                    state["src_data_per_band"][band_name] = (
                        band_label,
                        src_data,
                        x_bounds,
                    )
                    state["bands_pending_detection"] = pending[1:]
                    director.director_memory.record_outcome(
                        step,
                        {
                            "band": band_name,
                            "num_sources": n_src,
                            "validity": validity["checks_failed"] or "ok",
                        },
                    )
                else:
                    # Full-band result
                    state["src_data_full"] = src_data
                    state["num_sources"] = n_src
                    state["x_bounds"] = x_bounds
                    state["detection_params"] = params
                    state["detection_complete"] = True
                    state["validity_report"] = validity
                    if validity["should_rerun_detection"]:
                        print("  ⚠ Validity suggests rerun may be needed")
                    mark_phase_complete(state, "detection", step)
                    director.director_memory.record_outcome(
                        step,
                        {
                            "num_sources": state["num_sources"],
                            "scales": getattr(params, "wavdetect_scales", "?"),
                            "threshold": getattr(params, "significance_threshold", "?"),
                            "validity_failed": validity["checks_failed"] or "none",
                            "should_rerun": validity["should_rerun_detection"],
                        },
                    )
                    run_config.log(
                        "detection_params",
                        {
                            "wavdetect_scales": getattr(params, "wavdetect_scales", ""),
                            "significance_threshold": getattr(
                                params, "significance_threshold", 0
                            ),
                            "num_sources": state["num_sources"],
                            "reasoning": getattr(params, "reasoning", ""),
                        },
                        source="llm",
                    )

                state["execution_history"].record(
                    agent_name,
                    decision=f"Detected {n_src} sources in {band_label} band",
                    parameters={
                        "band": band_name,
                        "wavdetect_scales": getattr(params, "wavdetect_scales", "?"),
                        "significance_threshold": getattr(
                            params, "significance_threshold", "?"
                        ),
                        "num_sources": state["num_sources"],
                    },
                    reasoning=getattr(params, "reasoning", ""),
                )
            else:
                state["execution_history"].record(
                    agent_name,
                    decision=f"Detection failed for {band_name}",
                    reasoning=result.get("error", "unknown error"),
                )

        # ── band_strategist ──────────────────────────────────────
        elif agent_name == "band_strategist":
            result = strategist.run(
                {
                    "metadata": state["metadata"],
                    "num_sources": state.get("num_sources", 0),
                    "run_config": run_config,
                    "forced_band_mode": state.get("forced_band_mode"),
                    "execution_history": state["execution_history"],
                }
            )
            if result.get("success"):
                mode = result["processing_mode"]
                state["processing_mode"] = mode
                state["band_mode_selected"] = True
                mark_phase_complete(state, "band_strategy", step)

                # Populate band tracking
                bands = get_bands_for_mode(mode)
                state["bands_to_sonify"] = bands

                if mode == ProcessingMode.FULL_BAND:
                    # Reuse full-band detection already done
                    state["src_data_per_band"]["full"] = (
                        BAND_LABELS["full"],
                        state["src_data_full"],
                        state["x_bounds"],
                    )
                    state["bands_pending_detection"] = []
                else:
                    # Each band needs its own detection
                    state["bands_pending_detection"] = list(bands)
                    state["src_data_per_band"] = {}

                director.director_memory.record_outcome(
                    step,
                    {
                        "mode": mode.value,
                        "bands": [b[0] for b in bands],
                        "needs_detection": mode != ProcessingMode.FULL_BAND,
                    },
                )
                state["execution_history"].record(
                    agent_name,
                    decision=f"Selected {mode.value}",
                    parameters={
                        "mode": mode.value,
                        "num_sources": state.get("num_sources", 0),
                    },
                    reasoning="Evaluated spectral complexity and exposure",
                )
                run_config.log("processing_mode", mode.value, source="llm")

        # ── sonification_expert ──────────────────────────────────
        elif agent_name == "sonification_expert":
            # Find the next band that is detected but not yet sonified
            sonified = set(state.get("bands_sonified", []))
            detected = set(state.get("src_data_per_band", {}).keys())
            pending_soni = [
                b
                for b in state.get("bands_to_sonify", [])
                if b[0] in detected and b[0] not in sonified
            ]
            if not pending_soni:
                print("  ⚠ sonification_expert called but no bands ready; skipping")
                continue

            band_name, energy_spec, band_label = pending_soni[0]
            band_label_stored, src_data, x_bounds = state["src_data_per_band"][
                band_name
            ]

            # Compute shared x_bounds across all detected bands for alignment
            all_bounds = [
                v[2] for v in state["src_data_per_band"].values() if v[2] is not None
            ]
            if all_bounds:
                shared_x_bounds = (
                    min(b[0] for b in all_bounds),
                    max(b[1] for b in all_bounds),
                )
            else:
                shared_x_bounds = x_bounds

            soni_dir = str(out_dir / f"sonification_{band_name}")
            soni_result = expert.run(
                {
                    "metadata": state["metadata"],
                    "src_data": src_data,
                    "out_dir": soni_dir,
                    "band_name": band_name,
                    "band_label": band_label_stored,
                    "x_bounds": shared_x_bounds,
                    "show_animation": state.get("show_animation", True),
                    "run_config": run_config,
                    "expected_duration": DURATION,
                    "execution_history": state["execution_history"],
                }
            )

            if soni_result.get("success"):
                state["sonification_results"][band_name] = soni_result
                state["bands_sonified"].append(band_name)
                director.director_memory.record_outcome(
                    step,
                    {
                        "band": band_name,
                        "duration": soni_result.get("duration", "?"),
                        "wav": bool(soni_result.get("wav")),
                    },
                )
                state["execution_history"].record(
                    agent_name,
                    decision=f"Sonified {band_label_stored} band",
                    parameters={
                        "band_name": band_name,
                        "duration": soni_result.get("duration", "?"),
                        "num_sources": len(src_data),
                    },
                    reasoning=soni_result.get(
                        "reasoning", ""
                    ),  # ← was hardcoded string before
                )

                # Per-band mux (silent video + audio → final video)
                band_wav = soni_result.get("wav")
                band_silent_video = soni_result.get("silent_video")
                if band_wav and band_silent_video:
                    band_video_out = str(
                        Path(band_silent_video).parent / "sonification_with_audio.mp4"
                    )
                    mux = tool_mux_audio_video(
                        video_path=band_silent_video,
                        audio_path=band_wav,
                        out_path=band_video_out,
                    )
                    if mux.success:
                        soni_result["final_video"] = band_video_out
                        print(f"  ✓ Muxed: {band_video_out}")
                    else:
                        print(f"  ⚠ Mux failed: {mux.error}")

        else:
            print(f"  [Step {step}] ⚠ Unhandled agent '{agent_name}'; skipping")

    else:
        print(f"  ⚠ Reached max steps ({max_steps}); proceeding with current state")

    state["sonification_complete"] = True

    # Collect wav paths for overlay
    all_wav_paths: dict[str, str] = {
        band: info["wav"]
        for band, info in state["sonification_results"].items()
        if info.get("wav")
    }
    src_data_per_band = {
        band: (lbl, sd)
        for band, (lbl, sd, _) in state.get("src_data_per_band", {}).items()
    }

    # Shared x_bounds for overlay
    all_bounds = [
        v[2] for v in state.get("src_data_per_band", {}).values() if v[2] is not None
    ]
    shared_x_bounds = (
        (min(b[0] for b in all_bounds), max(b[1] for b in all_bounds))
        if all_bounds
        else state.get("x_bounds")
    )

    # — Phase 5: Multi-Band Overlay (when > 1 band) ————————————

    if len(src_data_per_band) > 1:
        print(f"\n{'#'*70}")
        print("  PHASE 5: Multi-band overlay")
        print(f"{'#'*70}")

        # Get the actual sonification duration from results
        first_band = list(src_data_per_band.keys())[0]
        soni_duration = (
            state["sonification_results"].get(first_band, {}).get("duration", 60.0)
        )

        overlay_result = overlay_composer.run(
            {
                "src_data_dict": src_data_per_band,
                "obs_id": obs_id,
                "out_dir": str(out_dir),
                "duration": soni_duration,
                "x_bounds": shared_x_bounds,
                "wav_paths": all_wav_paths,
            }
        )
        if overlay_result.get("success"):
            state["overlay_video"] = overlay_result.get("overlay_video")
            print("  ✓ Overlay video with audio created")
        else:
            print("  ⚠ Overlay composition failed: " f"{overlay_result.get('error')}")

    # — Phase 6: Quality Report and Summary —————————————————————

    # Quality evaluation
    eval_ctx: dict[str, Any] = {}
    detection_params = state.get("detection_params")
    if detection_params is not None and state.get("metadata"):
        eval_ctx["detection"] = {
            "num_sources": state["num_sources"],
            "exposure_time": state["metadata"].exposure_time,
            "threshold": detection_params.significance_threshold,
        }
    soni_eval: dict[str, Any] = {}
    for band, info in state.get("sonification_results", {}).items():
        if info.get("wav"):
            soni_eval[band] = {
                "wav": info["wav"],
                "duration": 60.0,
                "num_sources": state.get("num_sources", 0),
            }
    if soni_eval:
        eval_ctx["sonifications"] = soni_eval

    if eval_ctx:
        report = evaluator.run(eval_ctx)
        evals = report.get("evaluations", {})
        print(evaluator.generate_report(evals))

    # Catalog entry
    meta = state.get("metadata")
    catalog.add_entry(
        obs_id=obs_id,
        target_name=meta.target_name if meta else "",
        evt2_path=str(evt2_path),
        notes=f"mode={mode.value}, " f"sources={state.get('num_sources', 0)}",
    )

    # Save combined source coordinates
    if state.get("src_data_full") is not None:
        tool_print_source_coordinates(
            src_data=state["src_data_full"],
            band_label="Full Band",
            save_path=str(out_dir / "source_coords_full.txt"),
        )

    # RunConfig summary
    if run_config:
        print(f"\n  RunConfig: {run_config.summary()}")

    # Final summary
    print_summary(state, memory)


# — Entrypoint ———————————————————————————————————————————————————

if __name__ == "__main__":
    parsed_args = parse_args()
    run_pipeline(parsed_args)
# endregion
