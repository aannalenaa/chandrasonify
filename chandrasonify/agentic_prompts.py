"""
Agentic Pipeline — LLM Prompts.

Centralized prompt templates for all LLM-powered agents.
Organized by agent for easy discovery and maintenance.
"""

from chandrasonify.agentic_config import (
    DUAL_BAND_THRESHOLD,
    DURATION,
)

# region Detection Prompts
# — Wavdetect Parameter Suggestion —————————————————————————————————


def prompt_detection_params(
    target_name: str,
    instrument: str,
    exposure_time: float,
    object_type: str | None = None,
    object_info: str | None = None,
    redshift: float | None = None,
    history_context: str = "",
) -> str:
    """LLM prompt for detection parameters.

    Args:
        target_name (str): Object name.
        instrument (str): Instrument name.
        exposure_time (float): Exposure duration in seconds.
        object_type (str | None): SIMBAD object type (default None).
        object_info (str | None): SIMBAD info (default None).
        redshift (float | None): Redshift (default None).
        history_context (str): Execution history (default "").

    Returns:
        str: Formatted LLM prompt.
    """
    history_str = f"{history_context}\n\n" if history_context else ""

    return (
        "As an X-ray astronomy expert, select optimal wavdetect "
        "source-detection parameters for this observation.\n\n"
        "Observation:\n"
        f"Target: {target_name}\n"
        f"Instrument: {instrument}\n"
        f"Exposure: {exposure_time:.0f}s\n"
        f"Object type: {object_type or 'unknown'}\n\n"
        f"Object info: {object_info or 'none'}\n"
        f"Redshift: {redshift or 'unknown'}\n"
        f"{history_str}"
        "Choose wavdetect_scales (space-separated ints, e.g. '1 2 4 8') "
        "and significance_threshold (float, e.g. 1e-6). "
        "For exposures > 10000s, consider including scale 16 "
        "(captures faint extended emission)\n"
        "For exposures > 5000s, consider keeping significance_threshold"
        ">= 1e-6 (stricter values produce too many spurious sources)\n"
        "Typical scales: '1 2 4 8', '1 2 4 8 16'\n"
        "Explain your reasoning.\n\n"
        "Respond with your answer structured as valid JSON: "
        '{"wavdetect_scales": "1 2 4 8", "significance_threshold": 1e-6, '
        '"reasoning": "..."}'
    )


# endregion


# region Band Strategy Prompts
# — Band Mode Selection —————————————————————————————————————————————


def prompt_band_strategy(
    target_name: str,
    instrument: str,
    exposure_time: float,
    num_sources: int,
    object_type: str | None = None,
    history_context: str = "",
) -> str:
    """LLM prompt for band strategy selection.

    Args:
        target_name (str): Object name.
        instrument (str): Instrument name.
        exposure_time (float): Exposure duration in seconds.
        num_sources (int): Sources detected in full band.
        object_type (str | None): SIMBAD object type (default None).
        history_context (str): Execution history (default "").

    Returns:
        str: Formatted LLM prompt.
    """
    history_str = f"{history_context}\n\n" if history_context else ""

    return (
        "You are an X-ray astronomy expert "
        "choosing an energy-band strategy.\n\n"
        "Observation:\n"
        f"Target: {target_name}\n"
        f"Instrument: {instrument}\n"
        f"Exposure: {exposure_time:.0f}s\n"
        f"Sources detected: {num_sources}\n"
        f"Object type: {object_type or 'unknown'}\n\n"
        f"{history_str}"
        "Options:\n"
        '"full_band"   — single 0.5-7 keV image (simple, good default)\n'
        '"dual_band"   — soft (0.5-2 keV) + hard (2-7 keV) '
        "(highlights spectral differences)\n"
        '"triple_band" — soft + medium + hard (maximum detail, slower)\n\n'
        "Choose based on source count, scientific interest, "
        "exposure depth.\n"
        f"Use dual_band when ≥ {DUAL_BAND_THRESHOLD} sources.\n"
        "Use triple_band only for very deep, scientifically rich fields.\n\n"
        "Respond with ONLY valid JSON:\n"
        '{{"processing_mode": "full_band", "reasoning": "..."}}'
    )


# endregion


# region Sonification Prompts
# — Sonification Parameter Suggestion —————————————————————————————


def prompt_sonification_params(
    target_name: str,
    num_sources: int,
    exposure_time: float,
    object_type: str | None = None,
    expected_duration: float = DURATION,
    history_context: str = "",
) -> str:
    """LLM prompt for sonification parameters.

    Args:
        target_name (str): Object name.
        num_sources (int): Number of sources to sonify.
        exposure_time (float): Exposure time in seconds.
        object_type (str | None): SIMBAD object type (default None).
        expected_duration (float): Target duration (default DURATION).
        history_context (str): Execution history (default "").

    Returns:
        str: Formatted LLM prompt.
    """
    history_str = f"{history_context}\n\n" if history_context else ""

    return (
        "You are a sonification expert converting X-ray data to sound.\n\n"
        "Observation:\n"
        f"Target: {target_name}\n"
        f"Sources: {num_sources}\n"
        f"Exposure: {exposure_time:.0f}s\n"
        f"Object type: {object_type or 'unknown'}\n"
        f"{history_str}"
        "Choose sonification parameters. Guidelines:\n"
        f"- duration: prefer {expected_duration:.0f}s (range 30-120s)\n"
        "- peak_volume_range: [min, max] in 0.0-1.0\n"
        "- stereo_spread: [min, max] in 0.0-1.0\n"
        "- note_len: 0.5-3.0s (shorter for dense fields)\n"
        "- sf_preset: soundfont preset index (eg. 1-8=piano, 9-16=percussion, "
        "17-24=organ, 25-32=guitar, 33-40=bass, 41-48=strings, 49-56=ensemble,"
        " 57-64=brass, 65-72=reed, 73-80=pipe, 81-99=synth)\n"
        "- master_volume: 0.3-0.9\n\n"
        "Respond with ONLY valid JSON:\n"
        '{{"duration": {expected_duration:.0f}, "peak_volume_range": [0.3, 1.0], '
        '"stereo_spread": [0.3, 0.7], "note_len": 1.0, "sf_preset": 2, '
        '"master_volume": 0.6, "reasoning": "..."}}'
    )


# endregion


# region Reflection Prompts
# — Detection Failure Reflection —————————————————————————————————


def prompt_detection_failure_reflection(
    target_name: str,
    exposure_time: float,
    object_type: str | None,
    quality_info: dict[str, int | float | str],
    failure_history: str,
) -> str:
    """LLM prompt for adjusting detection after failures.

    Args:
        target_name (str): Object name.
        exposure_time (float): Exposure time in seconds.
        object_type (str | None): Object type from SIMBAD.
        quality_info (dict[str, int | float | str]): Detection quality metrics.
        failure_history (str): Summary of failed attempts.

    Returns:
        str: Formatted LLM prompt.
    """

    return (
        "You are an X-ray astronomy expert.\n"
        "Previous attempts:\n"
        f"{failure_history}\n\n"
        f"Observation: {target_name}, "
        f"{exposure_time:.0f}s, "
        f"{object_type or 'unknown'}\n\n"
        f"num_sources: {quality_info.get('num_sources', '?')}\n"
        f"exposure_time: {quality_info.get('exposure_time', '?'):.0f}s\n"
        f"threshold: {quality_info.get('threshold', '?')}\n"
        "Guidelines:\n"
        "0 sources: threshold too strict, halve it\n"
        "<3 sources: likely too strict\n"
        ">500 sources: likely spurious, tighten threshold\n"
        "10-100 sources: usually good\n"
        "Should we retry? Suggest adjusted parameters "
        "or confirm detection is acceptable.\n\n"
        "Suggest ADJUSTED parameters. Respond with ONLY valid JSON:\n"
        '{"wavdetect_scales": "...", "significance_threshold": ..., '
        '"reasoning": "...", "should_retry": true/false}'
    )


# endregion


# region Director Prompts
# — Pipeline Director Decision —————————————————————————————————


def prompt_director_decision(
    state_observation: str,
    execution_history_str: str = "",
    director_log: str = "",
) -> str:
    """LLM prompt for pipeline director decision.

    Args:
        state_observation (str): Current pipeline state.
        execution_history_str (str): Execution history (default "").
        director_log (str): Director log (default "").

    Returns:
        str: Formatted LLM prompt.
    """
    history_section = (
        f"EXECUTION HISTORY:\n{execution_history_str}\n\n"
        if execution_history_str
        else ""
    )

    return (
        "You are the Pipeline Director for an X-ray data sonification pipeline.\n\n"
        "CURRENT STATE:\n"
        f"{state_observation}\n\n"
        f"{history_section}"
        "PRIOR DECISIONS:\n"
        f"{director_log}\n\n"
        "AVAILABLE AGENTS:\n"
        "  observation_researcher — extract FITS metadata + SIMBAD enrichment\n"
        "  detection_optimizer    — run wavdetect source detection\n"
        "  band_strategist        — select full/dual/triple band processing mode\n"
        "  sonification_expert    — render audio/video for one band\n\n"
        "EXECUTION ORDER:\n"
        "  observation_researcher → detection_optimizer → band_strategist\n"
        "  → detection_optimizer can and should be called again "
        "if there are dual or triple bands post band_strategist, even "
        "if the phase has been marked as complete due to prior full band\n"
        "  → sonification_expert (once per band)\n"
        "→ finish\n\n"
        "Do NOT rerun observation_researcher if 'metadata' phase is marked complete.\n"
        "Make sure to call the detection_optimizer if "
        "the sonification_expert reports pending detections; "
        "else you get a loop."
        "YOU MAY deviate or rerun if results justify it:\n"
        "  - Rerun detection_optimizer if validity checks failed or source count\n"
        "    is 0, <3, or >500. Only rerun once.\n"
        "  - Rerun band_strategist if source count changed significantly after\n"
        "    a detection rerun.\n"
        "  - Rerun observation_researcher only if SIMBAD returned nothing and\n"
        "    object type is needed for band strategy.\n\n"
        "SONIFICATION:\n"
        "  - After band_strategy is set, delegate sonification_expert once per band.\n"
        "  - Check processing mode, if more than one band is selected run "
        "detection_optimizer for all bands!\n"
        "  - Check 'Sonified' — delegate sonification_expert for "
        "each band not yet done.\n"
        "  - Only finish when 'Still needed' is empty.\n\n"
        "WHEN TO FINISH:\n"
        "  - All bands are sonified (Still needed: none)\n"
        "  - Do NOT finish if validity failed and you have not yet retried detection\n"
        "  - Do NOT finish if any band is detected but not sonified\n\n"
        "Respond with ONLY valid JSON:\n"
        '{"action": "delegate", "agent": "<agent_name>", "reasoning": "..."}\n'
        "OR\n"
        '{"action": "finish", "reasoning": "..."}'
    )


# endregion
