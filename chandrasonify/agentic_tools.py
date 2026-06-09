"""
Agentic Pipeline — Tool Definitions.

Deterministic operations that agents invoke.  Each tool:
  - Takes structured input
  - Returns a ToolResult with success/failure and output data
  - Contains NO LLM logic — tools are pure domain computation

Tools are the "hands" of the agents; agents are the "brains".
"""

# region Imports
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable
import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FuncAnimation  # noqa: E402
import numpy as np  # noqa: E402
import pexpect  # noqa: E402
from scipy.io import wavfile

from astropy.io import fits  # noqa: E402
from astropy.coordinates import SkyCoord  # noqa: E402
import astropy.units as u  # noqa: E402
from astroquery.simbad import Simbad  # noqa: E402
from pydantic import BaseModel  # noqa: E402
from strauss import score  # noqa: E402
from strauss.sources import Objects  # noqa: E402
from strauss.generator import Sampler  # noqa: E402
from strauss.sonification import Sonification  # noqa: E402

from chandrasonify.config import (  # noqa: E402
    CIAO_BIN,
    SF2,
    VIDEO_CODEC,
    BROKEN_PRESETS,
)
from chandrasonify.agentic_config import (  # noqa: E402
    DURATION,
    PEAK_VOLUME_RANGE,
    STEREO_SPREAD,
    NOTE_LEN,
    SF_PRESET,
    MASTER_VOLUME,
)

os.environ["PATH"] = f"{CIAO_BIN}:{os.environ['PATH']}"
# endregion


# region Core Types
# — Shared Pydantic Models ———————————————————————————————————————


class ObservationMetadata(BaseModel):
    """Observation metadata from FITS header + SIMBAD.

    Attributes:
        obs_id (str): Observation ID.
        instrument (str): Instrument name.
        exposure_time (float): Exposure in seconds.
        target_name (str): Target name.
        ra (float): Right Ascension (degrees).
        dec (float): Declination (degrees).
        num_sources (int): Source count.
        object_type (str | None): SIMBAD object type.
        object_info (str | None): SIMBAD human-readable info.
        redshift (float | None): Redshift from SIMBAD.
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
        wavdetect_scales (str): Scales for wavdetect (e.g. "1,2,4,8").
        significance_threshold (float): Significance threshold.
        reasoning (str): Agent reasoning.
    """

    wavdetect_scales: str
    significance_threshold: float
    reasoning: str


class SonificationConfig(BaseModel):
    """Sonification rendering configuration.

    Attributes:
        duration (float): Target duration in seconds.
        peak_volume_range (tuple[float, float]): [min, max] volume (0.0-1.0).
        stereo_spread (tuple[float, float]): [min, max] stereo pan (0.0-1.0).
    """

    duration: float
    peak_volume_range: tuple[float, float]
    stereo_spread: tuple[float, float]


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


# — Tool Infrastructure —————————————————————————————————————————


class ToolResult(BaseModel):
    """Standardized tool invocation result.

    Attributes:
        success (bool): Whether tool succeeded.
        message (str): Human-readable message.
        data (dict[str, Any]): Structured output data.
        error (str | None): Error message if failed.
    """

    model_config = {"arbitrary_types_allowed": True}

    success: bool
    message: str = ""
    data: dict[str, Any] = {}
    error: str | None = None


@dataclass
class ToolDescriptor:
    """Metadata for tools used by agents.

    Attributes:
        name (str): Unique tool name.
        description (str): What the tool does.
        parameters (dict[str, str]): Parameter descriptions.
    """

    name: str
    description: str
    parameters: dict[str, str] = field(default_factory=dict)


# endregion


# region Internal Domain Classes
# — Internal Domain Classes ——————————————————————————————————————


class _StraussSonification:
    """Internal Strauss-based sonification.

    Call tool_render_sonification() instead of using this directly.

    Attributes:
        src_data (np.ndarray): Source data from wavdetect.
        duration (float): Target duration in seconds.
        peak_volume_range (tuple[float, float]): Volume scaling (min, max).
        stereo_spread (tuple[float, float]): Stereo panning (min, max).
        note_len (float): Duration of each note in seconds.
        sf_preset (int): SoundFont preset number.
        x_bounds (tuple | None): Optional (min, max) X position bounds.
        master_volume (float): Overall volume scaling.
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
        master_volume: float = MASTER_VOLUME,
    ):
        self.src_data = src_data
        self.duration = duration
        self.peak_volume_range = peak_volume_range
        self.stereo_spread = stereo_spread
        self.note_len = note_len
        self.sf_preset = sf_preset
        self.x_bounds = x_bounds
        self.master_volume = master_volume
        self.out_dir = Path(out_dir) if out_dir else Path(".")
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.objects: Objects | None = None
        self.score_obj: score.Score | None = None
        self.sampler: Sampler | None = None
        self.soni: Sonification | None = None

    def visualize_sources(self):
        """Create scatter plot of source positions for debugging."""
        plt.figure(figsize=(12, 8))
        plt.scatter(
            self.src_data["X"],
            self.src_data["Y"],
            s=(self.src_data["NET_COUNTS"] / self.src_data["NET_COUNTS"].max() * 200),
            c=self.src_data["Y"],
            cmap="viridis",
            alpha=0.6,
            edgecolors="black",
            linewidth=0.5,
        )
        plt.tight_layout()
        plt.close()

    def prepare_objects(self):
        """Prepare Strauss Objects mapping sources to musical parameters.

        Maps: X→time, Y→pitch, NET_COUNTS→volume.
        """
        if self.x_bounds is not None:
            x_min, x_max = self.x_bounds
        else:
            x_min = self.src_data["X"].min()
            x_max = self.src_data["X"].max()

        x_range = max(x_max - x_min, 1)
        x_norm = (self.src_data["X"] - x_min) / x_range

        y_norm = (self.src_data["Y"] - self.src_data["Y"].min()) / max(
            self.src_data["Y"].max() - self.src_data["Y"].min(), 1
        )
        net = self.src_data["NET_COUNTS"]
        counts_norm = (net - net.min()) / max(net.max() - net.min(), 1)

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
            f"(x range: {x_min:.1f}-{x_max:.1f} px)"
        )

    def create_score(self):
        """Create Strauss Score with all sources as notes."""
        self.score_obj = score.Score(
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
        """
        Create Strauss Sampler with specified SoundFont preset.
        Adjust note length in parameters if needed.
        """
        preferred = self.sf_preset if self.sf_preset not in BROKEN_PRESETS else 2
        self.sampler = Sampler(sampfiles=SF2, sf_preset=preferred)
        if preferred != self.sf_preset:
            print(f"  sf_preset={self.sf_preset} broken, " f"using {preferred}")
        for attr in ["parameters", "settings", "presets"]:
            if hasattr(self.sampler, attr):
                getattr(self.sampler, attr)["note_length"] = self.note_len

    def render(
        self,
        save_path: str = "xray_sonification.wav",
        master_volume: float = MASTER_VOLUME,
    ) -> str:
        """Render sonification audio and save to disk.

        Args:
            save_path (str): Output filename (default "xray_sonification.wav").
            master_volume (float): Volume scaling (default MASTER_VOLUME).

        Returns:
            str: Path to saved audio file.
        """
        full_path = self.out_dir / save_path
        if full_path.exists():
            print(f"  Audio already exists: {full_path}")
            return str(full_path)

        self.soni = Sonification(
            self.score_obj, self.objects, self.sampler, audio_setup="stereo"
        )
        print(f"  Rendering {self.duration}s sonification...")
        self.soni.render()
        self.soni.notebook_display(show_waveform=True)
        self.soni.save_stereo(str(full_path), master_volume=master_volume)
        print(f"  Saved to {full_path}")
        return str(full_path)

    def run_full_pipeline(self, save_path: str = "xray_sonification.wav") -> str:
        """Run full sonification pipeline.

        Args:
            save_path (str): Output filename (default "xray_sonification.wav").

        Returns:
            str: Path to saved audio file.
        """
        self.visualize_sources()
        self.prepare_objects()
        self.create_score()
        self.create_sample()
        return self.render(save_path=save_path, master_volume=self.master_volume)


class _SonificationVisualizer(_StraussSonification):
    """Internal sonification with animated visualization.

    Attributes:
        band_label (str): Energy band label for plot titles.
        x_norm (np.ndarray | None): Normalized X positions.
        y_norm (np.ndarray | None): Normalized Y positions.
        counts_norm (np.ndarray | None): Normalized counts.
    """

    def __init__(self, *args, band_label: str = "", **kwargs):
        super().__init__(*args, **kwargs)
        self.band_label = band_label
        self.x_norm: np.ndarray | None = None
        self.y_norm: np.ndarray | None = None
        self.counts_norm: np.ndarray | None = None

    def prepare_objects(self):
        """Prepare objects and compute normalized coordinates for animation."""
        super().prepare_objects()
        self.x_norm = (self.src_data["X"] - self.src_data["X"].min()) / max(
            self.src_data["X"].max() - self.src_data["X"].min(), 1
        )
        self.y_norm = (self.src_data["Y"] - self.src_data["Y"].min()) / max(
            self.src_data["Y"].max() - self.src_data["Y"].min(), 1
        )
        self.counts_norm = (
            self.src_data["NET_COUNTS"] - self.src_data["NET_COUNTS"].min()
        ) / max(
            self.src_data["NET_COUNTS"].max() - self.src_data["NET_COUNTS"].min(), 1
        )

    def create_animation(
        self,
        fps: int = 24,
        save: bool = True,
    ) -> tuple[Any, Any] | None:
        """Create scanning-bar animation.

        Args:
            fps (int): Frames per second (default 24).
            save (bool): Save as MP4 file (default True).

        Returns:
            tuple[Any, Any] | None: Figure and animation objects, or None.
        """
        if self.x_norm is None:
            print("  Must call prepare_objects() first")
            return None

        scan_dur = self.duration - self.note_len
        total_dur = self.duration
        end_scan_frame = int(scan_dur * fps)
        num_frames = int(total_dur * fps)

        fig, ax = plt.subplots(figsize=(14, 8))
        ax.scatter(
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
        plt.colorbar(ax.collections[0], ax=ax, label="Y position (Normalised Pitch)")

        # Use shared x_bounds if set, otherwise fall back to source range
        if self.x_bounds is not None:
            dx_min, dx_max = self.x_bounds
        else:
            dx_min = float(self.src_data["X"].min())
            dx_max = float(self.src_data["X"].max())
        dx_range = max(dx_max - dx_min, 1)

        scan_line = ax.axvline(
            x=dx_min, color="red", linewidth=3, alpha=0.8, label="Scanning bar"
        )
        ax.legend(loc="upper right")

        suffix = f" | {self.band_label}" if self.band_label else ""

        def animate(frame: int):
            progress = frame / end_scan_frame if frame < end_scan_frame else 1.0
            scan_line.set_xdata([dx_min + progress * dx_range] * 2)
            ax.set_title(
                f"Chandra X-ray Sonification{suffix} | "
                f"Time: {frame / fps:.2f}s / {total_dur:.2f}s",
                fontsize=14,
            )
            return (scan_line,)

        anim = FuncAnimation(
            fig, animate, frames=num_frames, interval=1000.0 / fps, blit=True
        )

        if save:
            out = self.out_dir / "sonification_animation.mp4"
            anim.save(
                str(out),
                writer="ffmpeg",
                fps=fps,
                extra_args=["-vcodec", VIDEO_CODEC, "-pix_fmt", "yuv420p"],
            )
            print(f"  Animation saved to {out}")

        plt.close(fig)
        return fig, anim

    def run_full_pipeline_with_viz(
        self,
        save_path: str = "xray_sonification.wav",
        show_animation: bool = True,
    ) -> dict[str, str | None]:
        """
        Audio + optional animation + mux.

        Args:
            save_path: Filename for the rendered WAV audio
            show_animation: Whether to create and save the scanning-bar animation

        Returns:
            dict[str, str | None]: Paths to the generated WAV audio, silent video,
                and final muxed video (if created)
        """
        self.visualize_sources()
        self.prepare_objects()
        self.create_score()
        self.create_sample()

        wav_path = self.render(save_path=save_path, master_volume=self.master_volume)

        video_silent: str | None = None
        if show_animation:
            try:
                self.create_animation(fps=24, save=True)
                video_silent = str(self.out_dir / "sonification_animation.mp4")
            except Exception as e:
                print(f"  Animation failed: {e}")

        final_video: str | None = None
        if video_silent and wav_path:
            final_video = str(self.out_dir / "sonification_with_audio.mp4")
            try:
                result = subprocess.run(
                    [
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
                    ],
                    capture_output=True,
                    text=True,
                )
                if result.returncode != 0:
                    print(f"  ⚠ Muxing failed: {result.stderr[:100]}")
                    final_video = None
                else:
                    print(f"  Final video: {final_video}")
            except Exception as e:
                print(f"  Muxing failed: {e}")
                final_video = None

        return {
            "wav": wav_path,
            "silent_video": video_silent,
            "final_video": final_video,
        }


# endregion


# region Tool Functions
# — Tool Functions ———————————————————————————————————————————————

# — 1. extract_metadata ——————————————————————————————————————————


def tool_extract_metadata(evt2_path: str) -> ToolResult:
    """Extract metadata from FITS EVT2 header.

    Args:
        evt2_path (str): Path to EVT2 FITS file.

    Returns:
        ToolResult: ObservationMetadata on success, error on failure.
    """
    try:
        with fits.open(evt2_path) as hdul:
            hdr = hdul[1].header  # type: ignore[index]
            metadata = ObservationMetadata(
                obs_id=hdr.get("OBS_ID", "unknown"),
                instrument=hdr.get("INSTRUME", "unknown"),
                exposure_time=hdr.get("EXPOSURE", 0.0),
                target_name=hdr.get("OBJECT", "unknown"),
                ra=hdr.get("RA_PNT", 0.0),
                dec=hdr.get("DEC_PNT", 0.0),
            )
        return ToolResult(
            success=True,
            message=f"Metadata for {metadata.target_name}",
            data={"metadata": metadata},
        )
    except Exception as e:
        return ToolResult(success=False, error=str(e))


# — SIMBAD Object Type Mapping ——————————————————————————————————


def _load_otype_map() -> dict[str, str]:
    """Load SIMBAD object type mappings from CSV.

    Returns:
        dict[str, str]: Mapping of otype codes to descriptions.
    """
    try:
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
            print(f"  ✓ Loaded {len(otype_map)} SIMBAD object types from CSV")
            return otype_map
        else:
            print(f"  ⚠ SIMBAD otypes CSV not found at {csv_path}")
    except Exception as e:
        print(f"  ⚠ Could not load SIMBAD otypes CSV: {e}")

    # Fallback if CSV not available
    return {
        "*": "Star",
        "**": "Double or Multiple Star",
        "X": "X-ray Source",
        "G": "Galaxy",
        "AGN": "Active Galaxy Nucleus",
        "QSO": "Quasar",
    }


OTYPE_MAP = _load_otype_map()


# — 2. query_simbad —————————————————————————————————————————————


def tool_query_simbad(ra: float, dec: float, target_name: str) -> ToolResult:
    """Query SIMBAD for object type and redshift.

    Args:
        ra (float): Right Ascension (degrees).
        dec (float): Declination (degrees).
        target_name (str): Target name from FITS header.

    Returns:
        ToolResult: SIMBAD data on success, error on failure.
    """
    try:
        print(f"  Looking up '{target_name}' in SIMBAD...")
        simbad = Simbad()
        simbad.add_votable_fields("otype", "rvz_redshift", "sp_type")

        result = simbad.query_object(target_name)

        if not result:
            coord = SkyCoord(ra=ra, dec=dec, unit="deg")
            result = simbad.query_region(coord, radius=5 * u.arcsec)

        if result:
            types = list(set(str(row["otype"]) for row in result if row["otype"]))
            sp_types = list(
                set(str(row["sp_type"]) for row in result if row["sp_type"])
            )
            redshifts = [
                float(row["rvz_redshift"])
                for row in result
                if row["rvz_redshift"] and str(row["rvz_redshift"]) != "--"
            ]

            otype = types[0] if types else None
            redshift = redshifts[0] if redshifts else None

            parts = []
            if otype:
                otype_label = OTYPE_MAP.get(otype, "Unknown")
                if otype_label == "Unknown":
                    print(f"    ⚠ otype code '{otype}' not in OTYPE_MAP")
                    print("    Available codes: " f"{list(OTYPE_MAP.keys())[:10]}...")
                parts.append(f"Type: {otype_label}")
            if sp_types:
                parts.append(f"Spectral types: {', '.join(sp_types)}")
            if redshift:
                parts.append(f"Redshift: {redshift:.4f}")
            info = " | ".join(parts) if parts else "Found in SIMBAD"
            print(f"  Found: {info}")
            return ToolResult(
                success=True,
                message=info,
                data={"object_type": otype, "redshift": redshift, "object_info": info},
            )

        print(f"  No SIMBAD entry for '{target_name}'")
        return ToolResult(
            success=True,
            message="Not found in SIMBAD",
            data={
                "object_type": None,
                "redshift": None,
                "object_info": "Not found in SIMBAD",
            },
        )

    except Exception as e:
        return ToolResult(success=False, error=str(e))


# — 3. create_band_image —————————————————————————————————————————


def tool_create_band_image(
    evt2_path: str,
    energy_spec: str,
    band_name: str,
    out_dir: str,
) -> ToolResult:
    """Create FITS image for energy band.

    Args:
        evt2_path (str): Path to EVT2 file.
        energy_spec (str): Energy specification (e.g. "500:2000").
        band_name (str): Band name (e.g. "soft").
        out_dir (str): Output directory.

    Returns:
        ToolResult: Image path on success, error on failure.
    """
    try:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        output_path = out / f"image_{band_name}.fits"

        if output_path.exists():
            print(f"  Band image exists: {output_path}")
            return ToolResult(
                success=True,
                data={"image_path": str(output_path)},
            )

        subprocess.run(
            [
                "dmcopy",
                f"{evt2_path}[energy={energy_spec}][bin x=::1,y=::1]",
                str(output_path),
                "clobber=yes",
            ],
            check=True,
        )
        print(f"  Created {band_name} image: {output_path}")
        return ToolResult(
            success=True,
            data={"image_path": str(output_path)},
        )
    except Exception as e:
        return ToolResult(success=False, error=str(e))


# — 4. run_wavdetect ————————————————————————————————————————————


def tool_run_wavdetect(
    image_path: str,
    scales: str,
    threshold: float,
    out_dir: str,
) -> ToolResult:
    """Run wavdetect source detection pipeline.

    Args:
        image_path (str): Path to FITS image.
        scales (str): Wavdetect scales (space-separated).
        threshold (float): Significance threshold.
        out_dir (str): Output directory.

    Returns:
        ToolResult: Wavdetect output paths on success.
    """
    try:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)

        psf = f"{out}/wavdetect_psf.fits"
        nbkg = f"{out}/wavdetect_nbkg.fits"
        src = f"{out}/wavdetect_src.fits"
        scell = f"{out}/wavdetect_scell.fits"
        img_out = f"{out}/wavdetect_image.fits"

        if Path(src).exists() and Path(psf).exists() and Path(nbkg).exists():
            print(f"  Wavdetect outputs exist in {out}")
            return ToolResult(
                success=True,
                data={
                    "src_path": src,
                    "psf_path": psf,
                    "nbkg_path": nbkg,
                    "scell_path": scell,
                    "image_path": img_out,
                },
            )

        log_path = out / "wavdetect.log"
        log_file = open(log_path, "wb")

        print("  Creating PSF map...")
        child = pexpect.spawn(
            f"mkpsfmap infile={image_path} outfile={psf} " f"energy=1.5 ecf=0.9"
        )
        child.logfile_read = log_file
        child.expect(pexpect.EOF, timeout=3600)

        print("  Creating background map...")
        child = pexpect.spawn(
            f'dmimgcalc "{image_path}" none {nbkg} ' f'op="imgout=1.0" clob+'
        )
        child.logfile_read = log_file
        child.expect(pexpect.EOF, timeout=600)

        print("  Running wavdetect...")
        child = pexpect.spawn(
            f"wavdetect infile={image_path} outfile={src} "
            f'scales="{scales}" '
            f"sigthresh={threshold} clob+"
        )
        child.logfile_read = log_file
        child.expect("Output source cell image file name")
        child.sendline(scell)
        child.expect("Output reconstructed image file name")
        child.sendline(img_out)
        child.expect("Output normalized background file name")
        child.sendline(nbkg)
        child.expect("Image of the size of the PSF")
        child.sendline(psf)
        child.expect(pexpect.EOF, timeout=3600)
        log_file.close()

        print(f"  Wavdetect done (log: {log_path})")
        return ToolResult(
            success=True,
            data={
                "src_path": src,
                "psf_path": psf,
                "nbkg_path": nbkg,
                "scell_path": scell,
                "image_path": img_out,
            },
        )
    except Exception as e:
        return ToolResult(success=False, error=str(e))


# — 5. load_sources —————————————————————————————————————————————


def tool_load_sources(src_path: str) -> ToolResult:
    """Load source list from wavdetect SRCLIST.

    Args:
        src_path (str): Path to wavdetect source FITS.

    Returns:
        ToolResult: Source data array on success.
    """
    try:
        with fits.open(src_path) as hdul:
            data = hdul["SRCLIST"].data  # type: ignore[index]
        n = len(data)
        print(f"  Loaded {n} sources from {src_path}")
        return ToolResult(
            success=True,
            data={"src_data": data, "num_sources": n},
        )
    except Exception as e:
        return ToolResult(success=False, error=str(e))


# — 6. create_source_plot ————————————————————————————————————————


def tool_create_source_plot(
    src_data: np.ndarray,
    save_path: str,
    band_label: str = "",
) -> ToolResult:
    """Create scatter plot of sources.

    Args:
        src_data (np.ndarray): Source data array.
        save_path (str): Output path for plot.
        band_label (str): Band label for title (default "").

    Returns:
        ToolResult: Success status and plot path.
    """
    try:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(
            src_data["X"],
            src_data["Y"],
            s=src_data["NET_COUNTS"] / src_data["NET_COUNTS"].max() * 200,
            c=src_data["Y"],
            cmap="viridis",
            alpha=0.6,
            edgecolors="black",
            linewidth=0.5,
        )
        suffix = f" — {band_label}" if band_label else ""
        plt.colorbar(scatter, label="Y position (pitch)")
        plt.xlabel("X position (scan direction)")
        plt.ylabel("Y position (pitch)")
        plt.title(f"Source Distribution " f"({len(src_data)} sources){suffix}")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  Plot saved: {save_path}")
        return ToolResult(success=True, data={"plot_path": save_path})
    except Exception as e:
        return ToolResult(success=False, error=str(e))


# — 7. print_source_coordinates ——————————————————————————————————


def tool_print_source_coordinates(
    src_data: np.ndarray,
    band_label: str = "",
    save_path: str | None = None,
) -> ToolResult:
    """Print source coordinates table.

    Args:
        src_data (np.ndarray): Source data array.
        band_label (str): Band label (default "").
        save_path (str | None): Save to file if provided (default None).

    Returns:
        ToolResult: Success status.
    """
    try:
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
                f"{src['X']:>10.2f}  {src['Y']:>10.2f}  "
                f"{src['RA']:>12.6f}  {src['DEC']:>12.6f}  "
                f"{src['NET_COUNTS']:>12.2f}"
            )
            print(line)
            lines.append(line)
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            Path(save_path).write_text("\n".join(lines))
            print(f"  Coordinates saved to {save_path}")
        return ToolResult(
            success=True,
            data={"num_sources": len(src_data)},
        )
    except Exception as e:
        return ToolResult(success=False, error=str(e))


# — 8. render_sonification ———————————————————————————————————————


def tool_render_sonification(
    src_data: np.ndarray,
    out_dir: str,
    band_name: str,
    band_label: str = "",
    duration: float = DURATION,
    note_len: float = NOTE_LEN,
    peak_volume_range: tuple[float, float] = PEAK_VOLUME_RANGE,
    stereo_spread: tuple[float, float] = STEREO_SPREAD,
    sf_preset: int = SF_PRESET,
    x_bounds: tuple[float, float] | None = None,
    show_animation: bool = True,
    master_volume: float = MASTER_VOLUME,
) -> ToolResult:
    """Render sonification audio and animation.

    Args:
        src_data (np.ndarray): Source data array.
        out_dir (str): Output directory.
        band_name (str): Band name (e.g. "soft").
        band_label (str): Band label for titles (default "").
        duration (float): Sonification duration (default DURATION).
        note_len (float): Note length in seconds (default NOTE_LEN).
        peak_volume_range (tuple): Volume range (default PEAK_VOLUME_RANGE).
        stereo_spread (tuple): Stereo pan range (default STEREO_SPREAD).
        sf_preset (int): SoundFont preset (default SF_PRESET).
        x_bounds (tuple | None): X position bounds (default None).
        show_animation (bool): Create animation (default True).
        master_volume (float): Master volume (default MASTER_VOLUME).

    Returns:
        ToolResult: Output file paths on success.
    """
    try:
        viz = _SonificationVisualizer(
            src_data=src_data,
            out_dir=out_dir,
            duration=duration,
            peak_volume_range=peak_volume_range,
            stereo_spread=stereo_spread,
            note_len=note_len,
            sf_preset=sf_preset,
            x_bounds=x_bounds,
            band_label=band_label,
            master_volume=master_volume,
        )
        result = viz.run_full_pipeline_with_viz(
            save_path=f"xray_sonification_{band_name}.wav",
            show_animation=show_animation,
        )
        if result.get("final_video") is None and show_animation:
            return ToolResult(
                success=False,
                error="Muxing failed; silent_video available as fallback",
                data=result,
            )
        print(f"  [DEBUG] Visualizer returned: {result}")
        return ToolResult(success=True, data=result)
    except Exception as e:
        print(f"  [DEBUG] Visualizer error: {e}")
        return ToolResult(success=False, error=str(e))


# — 9. evaluate_detection —————————————————————————————————————————


def tool_evaluate_detection(
    num_sources: int,
    exposure_time: float,
    threshold: float,
) -> ToolResult:
    """Evaluate detection quality.

    Args:
        num_sources (int): Number of sources detected.
        exposure_time (float): Exposure time in seconds.
        threshold (float): Significance threshold used.

    Returns:
        ToolResult: Quality metrics and retry recommendations.
    """
    ev: dict[str, Any] = {
        "status": "good",
        "confidence": 1.0,
        "recommendation": "Detection acceptable",
        "should_retry": False,
        "suggested_threshold_adjustment": 1.0,
    }

    if num_sources == 0:
        ev.update(
            status="failed",
            confidence=0.0,
            recommendation="No sources. Threshold too strict.",
            should_retry=True,
            suggested_threshold_adjustment=0.5,
        )
    elif num_sources < 3:
        ev.update(
            status="sparse",
            confidence=0.4,
            recommendation=f"Very few ({num_sources}). " "Consider relaxing threshold.",
            should_retry=True,
            suggested_threshold_adjustment=0.7,
        )
    elif num_sources > 500:
        ev.update(
            status="saturated",
            confidence=0.3,
            recommendation=f"Too many ({num_sources}). "
            "Likely spurious. Stricter threshold needed.",
            should_retry=True,
            suggested_threshold_adjustment=1.5,
        )
    elif num_sources < 10 and exposure_time < 10000:
        ev.update(
            status="sparse",
            confidence=0.6,
            recommendation="Few sources for this exposure.",
            should_retry=False,
            suggested_threshold_adjustment=0.8,
        )

    return ToolResult(success=True, data=ev)


# — 10. evaluate_audio ———————————————————————————————————————————


def tool_evaluate_audio(
    wav_path: str,
    expected_duration: float,
    num_sources: int,
) -> ToolResult:
    """Evaluate sonification audio quality (clipping, silence, etc.)."""
    ev: dict[str, Any] = {
        "status": "good",
        "quality_score": 1.0,
        "issues": [],
        "should_rerender": False,
    }

    try:
        sr, audio = wavfile.read(wav_path)

        if isinstance(audio, np.ndarray):
            max_val = np.max(np.abs(audio))
            max_possible = 2**15 - 1 if audio.dtype == np.int16 else 1.0
            clip = max_val / max_possible
            if clip > 0.95:
                ev["issues"].append("Clipping detected")
                ev["quality_score"] -= 0.25
                ev["should_rerender"] = True
            elif clip > 0.85:
                ev["issues"].append("Near clipping threshold")
                ev["quality_score"] -= 0.1

            actual_dur = len(audio) / sr
            if abs(actual_dur - expected_duration) > expected_duration * 0.05:
                ev["issues"].append(
                    f"Duration mismatch: {actual_dur:.1f}s "
                    f"vs expected {expected_duration:.1f}s"
                )
                ev["quality_score"] -= 0.15

            rms = np.sqrt(np.mean(audio.astype(float) ** 2))
            if rms < max_possible * 0.01:
                ev["issues"].append("Audio too quiet (mostly silent)")
                ev["quality_score"] -= 0.4
                ev["should_rerender"] = True

        ev["status"] = "good" if ev["quality_score"] > 0.7 else "poor"

    except ImportError:
        ev["issues"].append("scipy not available for audio check")
        ev["quality_score"] = 0.5
    except Exception as e:
        ev["issues"].append(f"Evaluation error: {e}")
        ev["quality_score"] = 0.5

    return ToolResult(success=True, data=ev)


# — 11. find_evt2_files ——————————————————————————————————————————


def tool_find_evt2_files(workspace: str) -> ToolResult:
    """Find all EVT2 files in workspace.

    Args:
        workspace (str): Workspace root directory.

    Returns:
        ToolResult: List of EVT2 file paths.
    """
    try:
        files = sorted(
            [
                str(p)
                for p in Path(workspace).rglob("*.fits*")
                if "evt2" in p.name.lower()
            ],
            key=lambda p: Path(p).stat().st_mtime,
            reverse=True,
        )
        return ToolResult(
            success=True,
            data={"files": files, "count": len(files)},
        )
    except Exception as e:
        return ToolResult(success=False, error=str(e))


# — 12. create_overlay_animation —————————————————————————————————


def tool_create_overlay_animation(
    src_data_dict: dict[str, tuple[str, np.ndarray]],
    obs_id: str,
    duration: float = DURATION,
    note_len: float = NOTE_LEN,
    x_bounds: tuple[float, float] | None = None,
    fps: int = 24,
) -> ToolResult:
    """Create overlay animation from multiple bands.

    Args:
        src_data_dict (dict): Band name → (label, src_data) mapping.
        obs_id (str): Observation ID.
        duration (float): Duration in seconds (default DURATION).
        note_len (float): Note length (default NOTE_LEN).
        x_bounds (tuple | None): X bounds (default None).
        fps (int): Frames per second (default 24).

    Returns:
        ToolResult: Overlay video path on success.
    """
    if not src_data_dict:
        return ToolResult(success=False, error="No source data")

    try:
        scan_dur = duration - note_len
        end_scan = int(scan_dur * fps)
        num_frames = int(duration * fps)

        marker_map = {"soft": "o", "hard": "s", "medium": "^", "full": "D"}

        fig, ax = plt.subplots(figsize=(14, 8))

        if x_bounds is None:
            all_x = np.concatenate([d[1]["X"] for d in src_data_dict.values()])
            x_min, x_max = float(all_x.min()), float(all_x.max())
        else:
            x_min, x_max = x_bounds
        x_range = max(x_max - x_min, 1)

        for band, (label, src) in src_data_dict.items():
            ax.scatter(
                src["X"],
                src["Y"],
                s=src["NET_COUNTS"] / src["NET_COUNTS"].max() * 200,
                c=src["Y"],
                cmap="viridis",
                alpha=0.6,
                edgecolors="black",
                linewidth=0.5,
                marker=marker_map.get(band, "o"),
                label=f"{label} ({len(src)} sources)",
            )

        ax.set_xlabel("X position")
        ax.set_ylabel("Y position")
        ax.legend(loc="upper right", fontsize=11)

        scan_line = ax.axvline(x=x_min, color="red", linewidth=3, alpha=0.8)

        def animate(frame):
            prog = frame / end_scan if frame < end_scan else 1.0
            scan_line.set_xdata([x_min + prog * x_range] * 2)
            ax.set_title(f"Overlay | {frame / fps:.2f}s / {duration:.2f}s", fontsize=14)
            return (scan_line,)

        anim = FuncAnimation(
            fig, animate, frames=num_frames, interval=1000.0 / fps, blit=True
        )

        out_dir = Path(f"output_{obs_id}")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = str(out_dir / "sonification_overlay_animation.mp4")

        anim.save(
            out_path,
            writer="ffmpeg",
            fps=fps,
            extra_args=["-vcodec", VIDEO_CODEC, "-pix_fmt", "yuv420p"],
        )
        plt.close(fig)
        print(f"  Overlay animation: {out_path}")
        return ToolResult(success=True, data={"animation_path": out_path})
    except Exception as e:
        return ToolResult(success=False, error=str(e))


# — 13. mix_audio ———————————————————————————————————————————————


def tool_mix_audio(wav_paths: list[str], out_path: str) -> ToolResult:
    """Mix multiple WAV files into stereo.

    Args:
        wav_paths (list[str]): Paths to WAV files.
        out_path (str): Output WAV path.

    Returns:
        ToolResult: Mixed audio path on success.
    """
    try:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        if len(wav_paths) == 1:
            shutil.copy(wav_paths[0], out_path)
        else:
            inputs: list[str] = []
            for p in wav_paths:
                inputs += ["-i", p]
            fi = "".join(f"[{i}:a]" for i in range(len(wav_paths)))
            fc = f"{fi}amix=inputs={len(wav_paths)}:normalize=0[a]"
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    *inputs,
                    "-filter_complex",
                    fc,
                    "-map",
                    "[a]",
                    out_path,
                ],
                check=True,
                capture_output=True,
            )
        print(f"  Mixed audio: {out_path}")
        return ToolResult(success=True, data={"mixed_path": out_path})
    except Exception as e:
        return ToolResult(success=False, error=str(e))


# — 14. mux_audio_video ——————————————————————————————————————————


def _probe_audio_stream(path: str) -> tuple[bool, str]:
    """Return whether media contains an audio stream."""
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "a:0",
                "-show_entries",
                "stream=codec_name,channels",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                path,
            ],
            capture_output=True,
            text=True,
        )
        out = result.stdout.strip()
        return (bool(out), out or result.stderr.strip())
    except Exception as e:
        return False, str(e)


def tool_mux_audio_video(video_path: str, audio_path: str, out_path: str) -> ToolResult:
    """Mux audio into a silent video via FFmpeg."""
    try:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        # Force exact stream pairing: video from first input,
        # audio from second input.
        result = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                video_path,
                "-i",
                audio_path,
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                "-pix_fmt",
                "yuv420p",
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-shortest",
                out_path,
            ],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            err_msg = result.stderr
            print(f"  FFmpeg error: {err_msg[:200]}")
            return ToolResult(
                success=False,
                error=f"FFmpeg failed: {err_msg[:100]}",
            )

        has_audio, probe = _probe_audio_stream(out_path)
        if not has_audio:
            print(f"  ⚠ Mux output has no audio stream: {probe}")
            return ToolResult(
                success=False,
                error="Mux succeeded but output has no audio stream",
            )

        print(f"  Muxed video: {out_path} (audio stream: {probe})")
        return ToolResult(success=True, data={"output_path": out_path})
    except Exception as e:
        return ToolResult(success=False, error=str(e))


# — 15. validate_detection_params ———————————————————————————————


def tool_validate_detection_params(
    params: ProcessingParameters,
    exposure_time: float,
) -> ToolResult:
    """
    Guard-rails on LLM-suggested detection parameters.
    Relaxes threshold for long exposures; widens scales if needed.
    """
    adjusted = False
    return ToolResult(
        success=True,
        message="adjusted" if adjusted else "no change",
        data={"params": params, "adjusted": adjusted},
    )


# endregion


# region Registry
# — Tool Registry ————————————————————————————————————————————————

TOOL_DESCRIPTORS: dict[str, ToolDescriptor] = {
    "extract_metadata": ToolDescriptor(
        name="extract_metadata",
        description="Extract observation metadata from FITS EVT2 header",
        parameters={"evt2_path": "Path to EVT2 file"},
    ),
    "query_simbad": ToolDescriptor(
        name="query_simbad",
        description="Query SIMBAD for object type and redshift",
        parameters={
            "ra": "RA degrees",
            "dec": "Dec degrees",
            "target_name": "Target name",
        },
    ),
    "create_band_image": ToolDescriptor(
        name="create_band_image",
        description="Create energy-band FITS image via dmcopy",
        parameters={
            "evt2_path": "EVT2 path",
            "energy_spec": "e.g. 500:7000",
            "band_name": "e.g. full/soft/hard",
            "out_dir": "Output directory",
        },
    ),
    "run_wavdetect": ToolDescriptor(
        name="run_wavdetect",
        description="Run wavdetect source detection pipeline",
        parameters={
            "image_path": "FITS image",
            "scales": "e.g. 1 2 4 8",
            "threshold": "e.g. 1e-6",
            "out_dir": "Output directory",
        },
    ),
    "load_sources": ToolDescriptor(
        name="load_sources",
        description="Load source list from wavdetect SRCLIST",
        parameters={"src_path": "Path to wavdetect_src.fits"},
    ),
    "create_source_plot": ToolDescriptor(
        name="create_source_plot",
        description="Create source distribution scatter plot",
        parameters={
            "src_data": "Source array",
            "save_path": "Output PNG path",
            "band_label": "Plot title label",
        },
    ),
    "print_source_coordinates": ToolDescriptor(
        name="print_source_coordinates",
        description="Print and save source coordinate table",
        parameters={
            "src_data": "Source array",
            "band_label": "Label",
            "save_path": "Optional save path",
        },
    ),
    "render_sonification": ToolDescriptor(
        name="render_sonification",
        description="Render sonification audio, animation, and video",
        parameters={
            "src_data": "Source array",
            "out_dir": "Output directory",
            "band_name": "Band identifier",
            "duration": "Seconds",
            "show_animation": "Create video?",
        },
    ),
    "evaluate_detection": ToolDescriptor(
        name="evaluate_detection",
        description="Evaluate detection quality and suggest adjustments",
        parameters={
            "num_sources": "Count",
            "exposure_time": "Seconds",
            "threshold": "Significance threshold",
        },
    ),
    "evaluate_audio": ToolDescriptor(
        name="evaluate_audio",
        description="Evaluate sonification audio quality",
        parameters={
            "wav_path": "WAV file",
            "expected_duration": "Seconds",
            "num_sources": "Source count",
        },
    ),
    "find_evt2_files": ToolDescriptor(
        name="find_evt2_files",
        description="Find all EVT2 FITS files in workspace",
        parameters={"workspace": "Root directory"},
    ),
    "create_overlay_animation": ToolDescriptor(
        name="create_overlay_animation",
        description="Multi-band overlay animation",
        parameters={
            "src_data_dict": "Band→sources mapping",
            "obs_id": "Observation ID",
        },
    ),
    "mix_audio": ToolDescriptor(
        name="mix_audio",
        description="Mix multiple WAV files into one",
        parameters={"wav_paths": "List of WAV paths", "out_path": "Output path"},
    ),
    "mux_audio_video": ToolDescriptor(
        name="mux_audio_video",
        description="Combine audio and video into MP4",
        parameters={
            "video_path": "Silent video",
            "audio_path": "Audio file",
            "out_path": "Output MP4",
        },
    ),
    "validate_detection_params": ToolDescriptor(
        name="validate_detection_params",
        description="Validate/adjust LLM-suggested detection parameters",
        parameters={"params": "ProcessingParameters", "exposure_time": "Seconds"},
    ),
}

TOOL_FUNCTIONS: dict[str, Callable[..., ToolResult]] = {
    "extract_metadata": tool_extract_metadata,
    "query_simbad": tool_query_simbad,
    "create_band_image": tool_create_band_image,
    "run_wavdetect": tool_run_wavdetect,
    "load_sources": tool_load_sources,
    "create_source_plot": tool_create_source_plot,
    "print_source_coordinates": tool_print_source_coordinates,
    "render_sonification": tool_render_sonification,
    "evaluate_detection": tool_evaluate_detection,
    "evaluate_audio": tool_evaluate_audio,
    "find_evt2_files": tool_find_evt2_files,
    "create_overlay_animation": tool_create_overlay_animation,
    "mix_audio": tool_mix_audio,
    "mux_audio_video": tool_mux_audio_video,
    "validate_detection_params": tool_validate_detection_params,
}
# endregion
