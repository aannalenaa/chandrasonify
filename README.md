# chandrasonify: Automated X-Ray Sonification Pipeline

Convert Chandra X-Ray Observatory observations into audio representations through automated sonification.
This project implements three processing pipelines—manual, LLM-assisted, and agentic—to transform X-ray point source detections into synchronised audio-visual output.

A sample sonification output is provided as a download in `example/`.

---

## Overview

**chandrasonify** processes level-2 event files (evt2) from the Chandra X-Ray Observatory (CXO) and transforms point source detections into sound.
Detected sources are mapped to audio parameters: sky position to stereo field and timing, declination to pitch, and net count rate to volume.
The pipeline produces WAV audio files and optionally MP4 video overlays showing source positions synchronised to playback.

Three pipeline variants are provided, sharing the same underlying processing steps but differing in how parameters are selected:

| Pipeline | Parameter selection | LLM server required |
|---|---|---|
| `manual_code.py` | Static config file (`manual_config.py`) | No |
| `llm_code.py` | Single LLM call per decision point | Yes |
| `agentic_code_base.py` | Multi-agent reasoning with reflection loops | Yes |

---

## Requirements

- **CIAO 4.18** — Chandra Interactive Analysis of Observations ([installation guide](https://cxc.cfa.harvard.edu/ciao/download/ciao_dmg.html))
- **Python 3.11+**
- **llama-cpp-python** with a locally hosted LLM (for LLM-assisted and agentic pipelines only)
- **FluidR3\_GM.sf2** soundfont ([download from MuseScore](https://ftp.osuosl.org/pub/musescore/soundfont/MuseScore_General/MuseScore_General.sf2) or equivalent)
- **FFmpeg** (for video overlay output)

---

## Installation

### Step 1 — Install and initialise CIAO

Follow the MacOS [CIAO installation guide](https://cxc.cfa.harvard.edu/ciao/download/ciao_dmg.html).
The usability cannot be guaranteed on other OS.
CIAO must be sourced in your shell before running the pipeline:

```bash
source /path/to/ciao-4.18/bin/ciao.sh
```

### Step 2 — Create a virtual environment inside CIAO

To avoid dependency conflicts, create the project environment inside the CIAO Python installation:

```bash
cd /path/to/ciao-4.18
python -m venv chandrasonify_env
source chandrasonify_env/bin/activate
```

### Step 3 — Install Python dependencies

```bash
pip install -r requirements.txt
```

Key dependencies and their roles in the pipeline:

- `astropy` — FITS file handling and coordinate transformations
- `astroquery` — SIMBAD database queries for object classification
- `strauss` — parameter-driven sonification framework; maps detected source properties to audio parameters
- `pydantic` — runtime validation of LLM-generated parameter responses and observation metadata models
- `langchain-openai` — LLM integration via OpenAI-compatible REST API
- `matplotlib` — source distribution scatter plots and the scanning-bar animation overlays
- `numpy` — numerical array operations on source data (normalisation, coordinate transforms)
- `scipy` — audio quality evaluation (`scipy.io.wavfile` for clipping and silence checks)
- `pexpect` — programmatic control of CIAO's interactive CLI tools; `wavdetect` prompts for output filenames interactively and `pexpect` handles these responses without a human operator
- `llama-cpp-python` — local LLM model server (provides the OpenAI-compatible REST endpoint)

### Step 4 — Configure system paths

Edit `config.py` to match your installation:

```python
CIAO_BIN = "/path/to/ciao-4.18/bin"   # path to CIAO bin directory
SF2      = "FluidR3_GM.sf2"            # path to soundfont file
VIDEO_CODEC = "h264_videotoolbox"      # adjust for non-Apple hardware
```

If using a different soundfont, run the preset detection utility first to identify broken presets and update `BROKEN_PRESETS` in `config.py`:

```bash
python sf2_broken_detection.py
```

### Step 5 (LLM/agentic pipelines only) — Start the local model server

The LLM-assisted and agentic pipelines require a locally hosted language model served via an OpenAI-compatible REST API.
The model used in this work is `Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf`:

```bash
bash start_up_server.sh
```

The script starts the server on `http://localhost:8000/v1`.
Adjust the model path, port, and GPU offload settings in `start_up_server.sh` to match your hardware.
The manual pipeline has no server dependency.

---

## Data

Download Chandra level-2 event files from the [Chandra Data Archive](https://cda.harvard.edu/chaser/) by searching for an Observation ID (ObsID).
Place the `evt2.fits` file (or `evt2.fits.gz`) in a directory named after the ObsID:

```
data/
└── 22304/
    └── primary/
        └── acisf22304N007_evt2.fits.gz
```

The pipeline searches recursively for any `.fits` or `.fits.gz` file containing `evt2` in the filename.

---

## Usage

### Single observation

```bash
# Manual pipeline (no LLM server required)
python -m chandrasonify.manual_code -o 22304

# LLM-assisted pipeline
python -m chandrasonify.llm_code -o 22304

# Agentic pipeline
python -m chandrasonify.agentic_code_base -o 22304
```

If `-o` is omitted, the pipeline enters an interactive file-selection mode.

### Command-line flags

**All pipelines share:**

| Flag | Description |
|---|---|
| `-o / --obs-id ID` | ObsID to process |
| `--fresh` | Delete existing outputs and restart from scratch |
| `--no-animation` | Produce WAV output only; skip video muxing |

**Manual and LLM pipelines additionally support:**

| Flag | Description |
|---|---|
| `-s / --save-dir DIR` | Custom output directory |
| `--resume` | Force resume from a previous interrupted run |
| `-i / --interactive` | Prompt for confirmation at each pipeline step |
| `--band-mode full\|dual\|triple` | Override band mode selection |
| `--check` | Browse previously processed observations |
| `--check-id ID` | Inspect results for a specific previous run |
| `--debug` | Enable verbose debug output |

**Agentic pipeline additionally supports:**

| Flag | Description |
|---|---|
| `-e / --evt2 PATH` | Provide EVT2 file path directly |
| `--replay` | Replay decisions from a saved `run_config.json` |
| `--browse` | Browse previously processed observations |
| `--band-mode full_band\|dual_band\|triple_band` | Override band mode selection |
| `--workspace DIR` | Set workspace root directory |

Note: the agentic pipeline uses `--replay` where the manual and LLM pipelines use `--resume`, and `--browse` where they use `--check`. Band mode values also differ: the agentic pipeline expects underscored forms (`full_band`, `dual_band`, `triple_band`).

### Batch processing

Edit the `obs_ids` array in `run_script.sh` and uncomment the desired pipeline variant:

```bash
obs_ids=(22304 839 3955 9369)

# Uncomment one:
# python -m chandrasonify.agentic_code_base -o "$obs_id"
# python -m chandrasonify.llm_code -o "$obs_id"
# python -m chandrasonify.manual_code -o "$obs_id"
```

Then run:

```bash
bash run_script.sh
```

For the LLM-assisted and agentic pipelines, the model server must be running for the full duration of the batch job.
Start it with `bash start_up_server.sh` before launching the batch script.
The manual pipeline has no server dependency and can be batched without a running model.

---

## Output

Each processed observation produces output under `output_<ID>/`:

```
output_22304/
├── run_config.json                                # parameters used (for reproducibility)
├── source_coords_full.txt                         # full-band source coordinate table
├── preprocessing/
│   └── image_full.fits                            # full-band binned image
├── detection_full/
│   ├── wavdetect_src.fits                         # wavdetect source list
│   ├── wavdetect_psf.fits                         # PSF map
│   ├── wavdetect_nbkg.fits                        # background map
│   ├── source_distribution_full.png               # source scatter plot
│   └── source_coords_full.txt                     # source coordinate table
├── sonification_full/
│   ├── xray_sonification_full.wav                 # audio output
│   ├── sonification_animation.mp4                 # silent video
│   └── sonification_with_audio.mp4                # final muxed video
│
│   (for dual-band runs, additionally:)
├── detection_soft/ and detection_hard/            # per-band detection products
├── sonification_soft/ and sonification_hard/      # per-band audio and video
├── sonification_overlay_animation.mp4             # combined overlay (silent)
└── sonification_overlay_with_audio.mp4            # combined overlay with mixed audio
```

The `run_config.json` records all parameters selected during the run under the `decisions` key, so that results can be inspected and reproduced.
For the agentic pipeline, each agent decision is additionally logged with its source (`llm`, `manual`, or `fits_header`) and a timestamp.

---

## Project structure

```
chandrasonify/
├── chandrasonify/
│   ├── config.py                # system-wide paths and hardware settings
│   ├── run_config.py            # per-observation runtime state and replay logic
│   │
│   ├── manual_code.py           # manual pipeline
│   ├── manual_config.py         # static parameters for manual pipeline
│   │
│   ├── llm_code.py              # LLM-assisted pipeline
│   ├── llm_config.py            # fallback defaults for LLM pipeline
│   │
│   ├── agentic_code_base.py     # agentic pipeline orchestrator
│   ├── agentic_agents.py        # agent implementations
│   ├── agentic_tools.py         # tool definitions (CIAO, SIMBAD, STRAUSS wrappers)
│   ├── agentic_prompts.py       # LLM prompts for agent reasoning
│   └── agentic_config.py        # fallback defaults for agentic pipeline
│
├── sf2_broken_detection.py      # utility to identify incompatible MIDI presets
├── simbad_otypes.csv            # SIMBAD object type abbreviation lookup
├── run_script.sh                # batch processing script
├── start_up_server.sh           # LLM server startup script
├── requirements.txt             # Python dependencies
└── examples/                    # sample sonification output
```

---

## Agentic pipeline: agent overview

The agentic pipeline uses six specialised agents coordinated by a central director:

| Agent | Role |
|---|---|
| `ObservationResearcher` | Extracts FITS header metadata and queries SIMBAD |
| `DetectionOptimizer` | Selects wavdetect scales and significance threshold; retries up to 3 times |
| `BandStrategist` | Decides between full-, dual-, and triple-band processing |
| `SonificationExpert` | Selects sonification parameters; retries up to 2 times |
| `OverlayComposer` | Renders multi-band overlay animation and muxes audio with FFmpeg |
| `QualityEvaluator` | Validates final outputs against internal quality criteria |
| `PipelineDirector` | Orchestrates agent invocation order and manages shared memory |

Each agent follows a think → act → reflect loop: an LLM call determines the action, a tool executes it deterministically, and the result is evaluated before proceeding.

---

## Troubleshooting

**CIAO tools not found**
Ensure CIAO is sourced before running the pipeline: `source /path/to/ciao-4.18/bin/ciao.sh`

**LLM server connection refused**
Start the server with `bash start_up_server.sh` and verify it is reachable: `curl http://localhost:8000/v1/models`

**Soundfont not found**
Place `FluidR3_GM.sf2` in the project root or update the `SF2` path in `config.py`.

**Incompatible MIDI preset**
Some presets are not supported by FluidR3. Run the detection utility and update `BROKEN_PRESETS` in `config.py`:
```bash
python sf2_broken_detection.py
```

**No evt2 file found**
Confirm that the observation directory contains a file matching `*evt2*.fits` or `*evt2*.fits.gz`.

**wavdetect hangs**
`pexpect` drives wavdetect interactively and waits for specific prompt strings. If a run hangs, check `detection_*/wavdetect.log` for the last line emitted and verify the CIAO version matches 4.18.

---

## Citation

```bibtex
@mastersthesis{alber2026chandrasonify,
  author = {Alber, Annalena},
  title  = {Towards a Multi-Agent System for the Automated Sonification
             of {Chandra} {X-Ray} Point Source Detections},
  school = {University of Zurich},
  year   = {2026}
}
```

---

## License

This project was developed as part of a Master's thesis at the University of Zurich.
See `LICENSE` for details.
