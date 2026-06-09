# Chandrasonify: X-Ray Data Sonification Pipeline

Convert Chandra X-ray Observatory observations into audio representations through intelligent sonification. This project implements three processing pipelines—manual, LLM-assisted, and fully agentic—to transform X-ray source data into immersive audio-visual experiences.

## Overview

**Chandrasonify** is a system designed for astronomical sonification. It processes observations from the Chandra X-Ray Observatory (CXO) and transforms source detection data into synchronized audio and video output. The project explores both deterministic and AI-driven approaches to parameter selection, enabling researchers to:

- Sonify X-ray sources with automatic parameter tuning
- Generate audio-visual representations of astronomical data
- Use contextual metadata from SIMBAD to inform sonification strategies

## Key Features

### Processing Pipeline Variants

1. **Manual Pipeline** (`manual_code.py`)
   - Hardcoded sonification and detection parameters
   - Deterministic, reproducible output
   - Best for baseline comparisons and validation

2. **LLM-Assisted Pipeline** (`llm_code.py`)
   - Language model decides parameters based on observation metadata
   - Requires local LLM server (llama-server)
   - Balances automation with interpretability

3. **Agentic Pipeline** (`agentic_code_base.py`)
   - Multiple autonomous agents collaborate via reasoning loops
   - Requires local LLM server (llama-server)
   - Agents handle detection optimization, sonification, and band strategy
   - Think → Act → Reflect architecture for agent decision-making

### Multi-Band Processing

Splits X-ray data into e.g. soft (500–2000 eV) and hard (2000–7000 eV) energy bands when source density warrants it, enabling clearer sonification of complex observations.

### Integrated Astronomy Context

- **SIMBAD Integration**: Fetches object classification and metadata
- **Coordinate Resolution**: Automatically extracts and processes source positions
- **FITS Handling**: Native support for Chandra FITS files

### Comprehensive Audio-Visual Output

- High-quality audio synthesis using FluidR3 soundfont
- Synchronized video overlays showing source positions and energy bands
- Customizable sonification parameters

## Project Structure

```
chandrasonify/
├── chandrasonify/
│   ├── __init__.py                      # Package definition
│   ├── config.py                        # System configuration
│   ├── run_config.py                    # Runtime configuration for observation processing
│   │
│   ├── manual_code.py                   # Deterministic pipeline
│   ├── manual_config.py                 # Manual pipeline parameter defaults
│   │
│   ├── llm_code.py                      # LLM-assisted pipeline
│   ├── llm_config.py                    # LLM pipeline parameter fallbacks
│   │
│   ├── agentic_code_base.py             # Main agentic orchestrator
│   ├── agentic_agents.py                # Agent implementations
│   ├── agentic_config.py                # Agentic pipeline parameter defaults
│   ├── agentic_tools.py                 # Tool definitions for agent invocation
│   └── agentic_prompts.py               # LLM prompts for agent reasoning
│
├── sf2_broken_detection.py              # Utility to identify incompatible MIDI presets
├── FluidR3_GM.sf2                       # Soundfont file for audio synthesis (download separately)
├── simbad_otypes.csv                    # SIMBAD object type classification reference
├── Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf  # Language model (download separately)
│
├── run_script.sh                        # Bash script for batch processing
├── start_up_server.sh                   # Helper to launch llama-server
├── requirements.txt                     # Python package dependencies
└── README.md                            # This file
```

## Installation & Setup

### Prerequisites

- **CIAO 4.18+**: Chandra Interactive Analysis of Observations ([download](https://cxc.cfa.harvard.edu/ciao/))
- **Python 3.11+** with pip
- **Server**: Local LLM server for LLM/agentic pipelines

### Step 1: Install Python Dependencies

```bash
pip install -r requirements.txt
```

Key dependencies:
- `astropy` – FITS and coordinate handling
- `astroquery` – SIMBAD queries
- `strauss` – Sonification framework
- `pydantic` – Configuration validation
- `langchain-openai` – LLM integration
- `matplotlib` – Visualization
- `pexpect` – Process interaction
- `numpy` – Numerical computing

### Step 2: Configure System Paths

Edit `config.py` to match your system:

```python
CIAO_BIN = "/Applications/ciao-4.18/bin"  # CIAO installation path
SF2 = "FluidR3_GM.sf2"                      # Soundfont
VIDEO_CODEC = "h264_videotoolbox"
```

### Step 3 (For LLM/Agentic Pipelines): Start LLM Server

```bash
# Install ollama or use another llama-server provider
llama-server -m Llama-3.1-8B-Instruct-Q8_K_M.gguf --port 8000
```

Adjust the model path and port as needed. The LLM endpoint should be `http://localhost:8000/v1`.

## Usage

### Quick Start

Process a single Chandra observation using the default (agentic) pipeline:

```bash
python -m chandrasonify.agentic_code_base -o 22304
```

### Running the Three Pipelines

#### Manual Pipeline (Fixed Parameters)

```bash
python -m chandrasonify.manual_code -o 22304
```

Parameters are defined in `manual_config.py`. Modify them for different sonification styles.

#### LLM-Assisted Pipeline

```bash
python -m chandrasonify.llm_code -o 22304
```

The LLM analyzes observation metadata and SIMBAD context to select detection and sonification parameters dynamically.

#### Agentic Pipeline (Recommended)

```bash
python -m chandrasonify.agentic_code_base -o 22304
```

Autonomous agents orchestrate the entire workflow:
1. **ObservationResearcher** gathers SIMBAD context and metadata
2. **DetectionOptimizer** refines wavelet detection parameters
3. **BandStrategist** decides between single-band and dual-band processing
4. **SonificationExpert** optimizes audio rendering parameters
5. **QualityEvaluator** assesses output quality and suggests improvements

### Batch Processing

Edit `run_script.sh` and specify multiple observation IDs:

```bash
obs_ids=(22304 23103 19920)
```

Then run:

```bash
bash run_script.sh
```

### Command-Line Options

For the agentic pipeline, use:

```bash
python -m chandrasonify.agentic_code_base --help
```

Common options:
- `-o, --obs-id ID` – Observation ID to process
- `--fresh` – Delete existing output and reprocess
- `--data-dir DIR` – Custom data directory
- `--output-dir DIR` – Custom output directory

## Configuration

### Manual/LLM/Agentic Configs

Each pipeline has its own configuration file:

| File | Purpose |
|------|---------|
| `manual_config.py` | Static parameters for manual pipeline |
| `llm_config.py` | Fallback defaults for LLM pipeline |
| `agentic_config.py` | Fallback defaults for agentic pipeline |
| `run_config.py` | Observation-level settings (paths, band preferences) |
| `config.py` | System-wide settings (CIAO, soundfont, codecs) |

### Key Sonification Parameters

- **DURATION**: Length of audio output in seconds (default: 60.0)
- **PEAK_VOLUME_RANGE**: (min, max) volume scaling (default: 0.3–1.0)
- **STEREO_SPREAD**: (min, max) left-right stereo positioning (default: 0.3–0.7)
- **NOTE_LEN**: Duration of each sonified source note in seconds (default: 1.0)
- **SF_PRESET**: MIDI instrument preset (0–127; default: 1)
- **MASTER_VOLUME**: Global output gain (default: 0.6)

### Detection Parameters

- **WAVDETECT_SCALES**: Wavelet scales for source detection (default: "1 2 4 8")
- **SIGNIFICANCE_THRESHOLD**: Detection significance level (default: 1e-6)

### Band Strategy

- **DUAL_BAND_THRESHOLD**: If source count exceeds this, use dual-band processing (default: 25)
- **ENERGY_BANDS**: Energy ranges for soft/hard bands (default: soft 500–2000 eV, hard 2000–7000 eV)

## Output

For each observation, the pipeline generates:

```
output/obs_<ID>/
├── evt2_clean.fits              # Cleaned event list
├── sources.txt                  # Detected sources and properties
├── sources_<BAND>.txt           # Band-specific source lists
├── detection_image.png          # X-ray source map
├── sonification_params.json     # Selected parameters (for reproducibility)
├── audio.wav                    # Final audio output
├── overlay_<BAND>.mp4           # Video with overlay (if enabled)
├── final_<BAND>.mp4             # Muxed audio + video
└── logs/                        # Processing logs
```

## Troubleshooting

### "CIAO not found"

Ensure `CIAO_BIN` is correctly configured in `config.py` and CIAO is installed:

```bash
source /Applications/ciao-4.18/bin/ciao.sh
which wavdetect
```

### "LLM connection refused"

Start the LLM server:

```bash
llama-server -m <model.gguf> --port 8000
```

Verify connection:

```bash
curl http://localhost:8000/v1/models
```

### "Soundfont file not found"

Ensure `FluidR3_GM.sf2` exists in the current directory or update `SF2` path in `config.py`.

### "Index error with MIDI preset"

Some presets are incompatible with FluidR3. Run the detection utility:

```bash
python -m chandrasonify.sf2_broken_detection
```

This updates `BROKEN_PRESETS` in `config.py`. These presets are automatically skipped during sonification.

### "Observation data not found"

Download Chandra data using:

```bash
# Example with Chandra archive tools
chandra_repro <OBS_ID>
```

Ensure the processed event file (`evt2_clean.fits`) is in the expected directory.

## Architecture Highlights

### Agent Design Pattern

The agentic pipeline implements a **think → act → reflect** loop:

1. **Think**: Agent LLM prompt analyzes the current state and decides next action
2. **Act**: Execute the selected tool deterministically
3. **Reflect**: Evaluate tool output; decide to proceed or retry

Agent types in the pipeline:
- **ObservationResearcher**: Gathers context (SIMBAD, metadata)
- **DetectionOptimizer**: Tunes wavelet detection parameters
- **BandStrategist**: Chooses processing strategy
- **SonificationExpert**: Optimizes audio parameters
- **PipelineDirector**: Orchestrates phase transitions
- **QualityEvaluator**: Assesses and improves output

### Tool Registry

Tools are defined in `agentic_tools.py` with JSON schemas, allowing LLMs to invoke them with proper argument validation. Each tool returns structured `ToolResult` objects.

## Citation

If you use this repo in your research, please cite:

```bibtex
@thesis{chandrasonify2026,
  author={Annalena Alber},
  title={Towards a Multi-Agent System for the Automated Sonification of Chandra X-Ray Point SOurce Detections},
  school={University of Zurich},
  year={2026}
}
```

## License

This project is part of a Master's thesis at the Univerity of Zurich. Check the LICENSE file for details.

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a pull request

## Authors

- **Primary Author**: Annalena Alber
- **Advisors**: Jason Armitage

## Contact

For questions or support:
- **Email**: annalena.alber@uzh.ch
- **GitHub Issues**: [Link to issues]