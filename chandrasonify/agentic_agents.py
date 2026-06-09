"""
Agentic Pipeline — Agent Definitions.

LLM-powered agents that select and invoke tools autonomously.
Each agent implements a think → act → reflect loop:
  1. Think   — LLM decides which tool to call and with what parameters
  2. Act     — Execute the tool (deterministic)
  3. Reflect — Evaluate the result; retry or move on

Agents are the "brains"; tools (agentic_tools.py) are the "hands".
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, SecretStr, TypeAdapter, ValidationError

from chandrasonify.agentic_tools import (
    ObservationMetadata,
    ProcessingMode,
    ProcessingParameters,
    ToolResult,
    TOOL_DESCRIPTORS,
    TOOL_FUNCTIONS,
)
from chandrasonify.agentic_config import (
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
from chandrasonify.agentic_prompts import (
    prompt_detection_params,
    prompt_band_strategy,
    prompt_sonification_params,
    prompt_detection_failure_reflection,
    prompt_director_decision,
)

from chandrasonify.run_config import RunConfig

# region Foundations
# — LLM Setup —————————————————————————————————————————————————————

LLM = ChatOpenAI(
    model="local-model",
    base_url="http://localhost:8000/v1",
    api_key=SecretStr("not-needed"),
    temperature=0.5,
    # max_tokens=1024,  # not supported by all providers
    timeout=60.0,
)


def invoke_llm_with_retry(
    prompt: str,
    max_retries: int = 3,
    backoff: float = 2.0,
) -> str:
    """Invoke LLM with exponential backoff retry.

    Args:
        prompt (str): Prompt to send.
        max_retries (int): Maximum retry attempts.
        backoff (float): Base backoff in seconds.

    Returns:
        str: Response content or empty string on failure.
    """
    for attempt in range(1, max_retries + 1):
        try:
            result = LLM.invoke(prompt)
            return str(result.content)
        except Exception as e:
            if attempt == max_retries:
                print(f"  LLM failed after {max_retries} attempts: {e}")
                return ""
            wait = backoff**attempt
            print(f"  LLM attempt {attempt} failed ({e}); retrying in {wait}s …")
            time.sleep(wait)
    return ""


def _escape_newlines_in_strings(s: str) -> str:
    """Escape newlines in JSON string values.

    Args:
        s (str): JSON string.

    Returns:
        str: JSON with escaped newlines in strings.
    """

    def repl(match):
        """Escape newlines in matched JSON string.

        Args:
            match (re.Match): Regex match object.

        Returns:
            str: Escaped string value.
        """
        inner = match.group(1).replace("\n", "\\n")
        return f'"{inner}"'

    return re.sub(r'"([^"\\]*(?:\\.[^"\\]*)*)"', repl, s, flags=re.DOTALL)


def extract_json(text: str) -> dict:
    """Extract JSON from LLM response text.

    Args:
        text (str): Raw LLM response.

    Returns:
        dict: Extracted JSON as dictionary.
    """
    if not text:
        raise ValueError("Empty LLM response")

    patterns = [
        r"```json\s*(\{.*?\})\s*```",
        r"```\s*(\{.*?\})\s*```",
        r"(\{.*?\})",
    ]

    for pattern in patterns:
        m = re.search(pattern, text, re.DOTALL)
        if m:
            candidate = m.group(1).strip()
            candidate = _escape_newlines_in_strings(candidate)

            try:
                return json.loads(candidate)
            except json.JSONDecodeError as e:
                print("  [DEBUG extract_json] failed candidate:")
                print(candidate)
                print("  error:", e)

    raise ValueError("No valid JSON found in LLM response")


# — Execution History ———————————————————————————————————————————


class ExecutionHistory:
    """Tracks agent executions for LLM context enrichment.

    Attributes:
        entries (list[dict]): Execution records with agent, decision,
            parameters, and reasoning.
    """

    def __init__(self):
        self.entries: list[dict[str, Any]] = []

    def record(
        self,
        agent_name: str,
        decision: str | None = None,
        parameters: dict[str, Any] | None = None,
        reasoning: str | None = None,
    ) -> None:
        """Record an agent execution.

        Args:
            agent_name (str): Agent name.
            decision (str | None): Decision description.
            parameters (dict[str, Any] | None): Decision parameters.
            reasoning (str | None): Agent reasoning.
        """
        self.entries.append(
            {
                "agent": agent_name,
                "decision": decision,
                "parameters": parameters or {},
                "reasoning": reasoning,
            }
        )

    def format_for_llm(self) -> str:
        """Format history for LLM context.

        Returns:
            str: Formatted execution history.
        """
        if not self.entries:
            return "(No previous decisions yet; this is the first agent.)"

        lines = ["\nPrevious Execution History:"]
        for i, entry in enumerate(self.entries, 1):
            agent = entry["agent"]
            decision = entry.get("decision", "(no decision)")
            params = entry.get("parameters", {})
            reasoning = entry.get("reasoning", "")

            lines.append(f"\n  Step {i}: {agent}")
            lines.append(f"    Decision: {decision}")

            if params:
                param_strs = [f"{k}={v}" for k, v in params.items()]
                lines.append(f"    Parameters: {', '.join(param_strs)}")

            if reasoning:
                lines.append(f"    Reasoning: {reasoning}")

        return "\n".join(lines)

    def clear(self) -> None:
        """Clear all history entries."""
        self.entries.clear()


# — Agent Memory —————————————————————————————————————————————————


class AgentMemory:
    """Cross-agent memory tracking tool invocations and outcomes.

    Attributes:
        entries (list[dict]): Tool invocation records.
        detection_attempts (list[dict]): Detection attempt logs.
        sonification_attempts (list[dict]): Sonification attempt logs.
    """

    def __init__(self):
        self.entries: list[dict[str, Any]] = []
        self.detection_attempts: list[dict[str, Any]] = []
        self.sonification_attempts: list[dict[str, Any]] = []

    # -- general -------------------------------------------------
    def log(self, agent: str, tool: str, result: ToolResult, context: str = ""):
        """Log a tool invocation.

        Args:
            agent (str): Agent name.
            tool (str): Tool name.
            result (ToolResult): Tool result.
            context (str): Optional context.
        """
        self.entries.append(
            {
                "agent": agent,
                "tool": tool,
                "success": result.success,
                "message": result.message,
                "error": result.error,
                "context": context,
            }
        )

    # -- detection helpers ---------------------------------------
    def log_detection_attempt(
        self,
        scales: str,
        threshold: float,
        num_sources: int,
        quality_status: str,
    ) -> None:
        """Log a detection attempt.

        Args:
            scales (str): Wavdetect scales.
            threshold (float): Significance threshold.
            num_sources (int): Sources detected.
            quality_status (str): Quality status.
        """
        self.detection_attempts.append(
            {
                "scales": scales,
                "threshold": threshold,
                "num_sources": num_sources,
                "quality": quality_status,
            }
        )

    def get_working_strategy(self) -> dict | None:
        """Get most recent successful detection strategy.

        Returns:
            dict | None: Parameters of last successful attempt, or None.
        """
        for att in reversed(self.detection_attempts):
            if att["quality"] in ("good",):
                return att
        return None

    def get_failure_context(self) -> str:
        """Summarize failed detection attempts.

        Returns:
            str: Summary of failed attempts.
        """
        failed = [a for a in self.detection_attempts if a["quality"] != "good"]
        if not failed:
            return "No previous failures."
        lines = []
        for i, a in enumerate(failed, 1):
            lines.append(
                f"  Attempt {i}: scales={a['scales']}, "
                f"threshold={a['threshold']}, "
                f"sources={a['num_sources']}, "
                f"quality={a['quality']}"
            )
        return "\n".join(lines)

    # -- sonification helpers ------------------------------------
    def log_sonification_attempt(
        self,
        band: str,
        config: dict,
        quality: dict,
    ):
        """Log a sonification attempt.

        Args:
            band (str): Frequency band.
            config (dict): Sonification configuration.
            quality (dict): Quality metrics.
        """
        self.sonification_attempts.append(
            {
                "band": band,
                "config": config,
                "quality": quality,
            }
        )

    # -- summary ------------------------------------------------
    def summary(self) -> str:
        """Summarize memory for debugging or LLM context.

        Returns:
            str: Memory summary.
        """
        ok = sum(1 for e in self.entries if e["success"])
        fail = len(self.entries) - ok
        return (
            f"Memory: {len(self.entries)} actions "
            f"({ok} ok, {fail} failed), "
            f"{len(self.detection_attempts)} detection attempts, "
            f"{len(self.sonification_attempts)} sonification attempts"
        )


# — LLM Response Models —————————————————————————————————————————


class DetectionParamsResponse(BaseModel):
    """LLM response for detection parameters."""
    wavdetect_scales: str
    significance_threshold: float
    reasoning: str


class SonificationParamsResponse(BaseModel):
    """LLM response for sonification parameters."""
    duration: float
    peak_volume_range: list[float]
    stereo_spread: list[float]
    note_len: float
    sf_preset: int
    master_volume: float
    reasoning: str


class ProcessingModeResponse(BaseModel):
    """LLM response for processing mode selection."""
    processing_mode: str
    reasoning: str


class PipelineActionResponse(BaseModel):
    """LLM response for pipeline action decision."""
    action: str  # "delegate" | "finish"
    agent: str  # agent name when delegating
    reasoning: str


# — Base Agent ———————————————————————————————————————————————————


class BaseAgent:
    """
    Base class for all pipeline agents.

    Sub-classes override ``run()`` to implement domain-specific
    think → act → reflect loops.

    Attributes:
        name (str): Unique name of the agent.
        role_prompt (str): Description of the agent's role for LLM context.
        available_tools (list[str]): List of tool names the agent can use.
    """

    name: str = "base"
    role_prompt: str = ""
    available_tools: list[str] = []

    def __init__(self, memory: AgentMemory):
        self.memory = memory

    # -- utilities -----------------------------------------------
    def call_tool(self, tool_name: str, **kwargs) -> ToolResult:
        """
        Execute a tool and log in memory.

        Args:
            tool_name (str): The name of the tool to execute.
            **kwargs: Arguments to pass to the tool function.

        Returns:
            ToolResult: The result of the tool execution,
                including success status, data, and error message if any.
        """
        func = TOOL_FUNCTIONS.get(tool_name)
        if func is None:
            return ToolResult(success=False, error=f"Unknown tool: {tool_name}")
        result = func(**kwargs)
        self.memory.log(self.name, tool_name, result)
        return result

    def _tools_for_prompt(self) -> str:
        """
        Format available tool descriptions for LLM context.

        Returns:
            str: Formatted descriptions of available tools.
        """
        lines = []
        for t_name in self.available_tools:
            desc = TOOL_DESCRIPTORS.get(t_name)
            if desc:
                lines.append(f"  {desc.name}: {desc.description}")
        return "\n".join(lines) or "(none)"

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        """
        Execute agent logic. Override in subclass.

        Args:
            context (dict): Input context for the agent, containing necessary data
                and parameters.

        Returns:
            dict: Output context after agent execution, which may include results and
                decisions for downstream agents.

        Attention, this method must be overridden by subclasses to implement specific
        agent logic. The base implementation raises NotImplementedError to enforce this.
        """
        raise NotImplementedError


# endregion


# region Concrete Agents
# — Concrete Agents —————————————————————————————————————————————

# — ObservationResearcher ————————————————————————————————————


class ObservationResearcher(BaseAgent):
    """
    Loads FITS metadata and enriches it via SIMBAD.
    Provides scientific context for downstream agents.

    Attributes:
        name (str): Unique name of the agent.
        available_tools (list[str]): List of tool names the agent can use.
    """

    name = "observation_researcher"
    available_tools = ["extract_metadata", "query_simbad"]

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        """
        Extract metadata from the FITS header and enrich it with SIMBAD data.

        Args:
            context (dict): Input context containing 'evt2_path' key with the path to
                the evt2 FITS file.

        Returns:
            dict: A dictionary containing 'success' status and either 'metadata' on
                success or 'error' message on failure.
        """
        evt2_path: str = context["evt2_path"]

        print(f"\n{'=' * 60}")
        print(f"[{self.name}] Extracting observation metadata …")
        print(f"{'=' * 60}")

        # Act 1 — extract metadata
        meta_result = self.call_tool("extract_metadata", evt2_path=evt2_path)
        if not meta_result.success:
            return {"success": False, "error": meta_result.error}

        metadata: ObservationMetadata = meta_result.data["metadata"]
        print(f"  Target: {metadata.target_name} (OBS_ID {metadata.obs_id})")
        print(
            f"  Instrument: {metadata.instrument}, "
            f"Exposure: {metadata.exposure_time:.0f}s"
        )
        print(f"  RA={metadata.ra:.4f}, Dec={metadata.dec:.4f}")

        # Act 2 — SIMBAD enrichment
        simbad_result = self.call_tool(
            "query_simbad",
            ra=metadata.ra,
            dec=metadata.dec,
            target_name=metadata.target_name,
        )
        if simbad_result.success:
            metadata.object_type = simbad_result.data.get("object_type")
            metadata.redshift = simbad_result.data.get("redshift")
            metadata.object_info = simbad_result.data.get("object_info")

        # Reflect: "I now have complete metadata."
        print(
            f"  → Metadata complete"
            f"{' (SIMBAD enriched)' if metadata.object_type else ''}"
        )

        return {"success": True, "metadata": metadata}


# — DetectionOptimizer ———————————————————————————————————————


class DetectionOptimizer(BaseAgent):
    """
    Suggests wavdetect parameters via LLM, runs detection, evaluates
    quality, and retries with adjusted parameters when needed.
    Multi-turn think → act → reflect loop (max 3 attempts).

    Attributes:
        name (str): Unique name of the agent.
        available_tools (list[str]): List of tool names the agent can use.
    """

    name = "detection_optimizer"
    available_tools = [
        "create_band_image",
        "run_wavdetect",
        "load_sources",
        "evaluate_detection",
        "validate_detection_params",
        "create_source_plot",
        "print_source_coordinates",
    ]

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        """
        Optimize source detection by iteratively adjusting parameters based on
        quality evaluation.

        Args:
            context (dict): Input context containing necessary data such as
            'metadata', 'evt2_path', 'out_dir', and optional parameters for band
            selection and execution history.

        Returns:
            dict: A dictionary containing 'success' status, detected 'src_data',
            'num_sources', used 'params', and 'x_bounds' of sources on success; or
            'error' message on failure.
        """
        metadata: ObservationMetadata = context["metadata"]
        evt2_path: str = context["evt2_path"]
        out_dir: str = context["out_dir"]
        band_name: str = context.get("band_name", "full")
        energy_spec: str = context.get("energy_spec", ENERGY_BANDS["full"])
        band_label: str = context.get(
            "band_label", BAND_LABELS.get(band_name, band_name)
        )
        run_config = context.get("run_config")
        max_attempts: int = context.get("max_detection_attempts", 3)
        execution_history: ExecutionHistory = context.get(
            "execution_history", ExecutionHistory()
        )

        print(f"\n{'=' * 60}")
        print(f"[{self.name}] Optimising detection for {band_label} band")
        print(f"{'=' * 60}")

        params = self._suggest_params(metadata, run_config, execution_history)

        for attempt in range(1, max_attempts + 1):
            print(f"\n  --- Detection attempt {attempt}/{max_attempts} ---")
            print(
                f"  Scales: {params.wavdetect_scales}, "
                f"Threshold: {params.significance_threshold}"
            )

            img_result = self.call_tool(
                "create_band_image",
                evt2_path=evt2_path,
                energy_spec=energy_spec,
                band_name=band_name,
                out_dir=out_dir,
            )
            if not img_result.success:
                return {"success": False, "error": img_result.error}

            wd_result = self.call_tool(
                "run_wavdetect",
                image_path=img_result.data["image_path"],
                scales=params.wavdetect_scales,
                threshold=params.significance_threshold,
                out_dir=out_dir,
            )
            if not wd_result.success:
                return {"success": False, "error": wd_result.error}

            src_result = self.call_tool(
                "load_sources",
                src_path=wd_result.data["src_path"],
            )
            if not src_result.success:
                return {"success": False, "error": src_result.error}

            n_src = src_result.data["num_sources"]

            # — REFLECT: Evaluate Quality —————————————————————————
            ev = self.call_tool(
                "evaluate_detection",
                num_sources=n_src,
                exposure_time=metadata.exposure_time,
                threshold=params.significance_threshold,
            )
            quality = ev.data
            status = quality.get("status", "unknown")

            self.memory.log_detection_attempt(
                scales=params.wavdetect_scales,
                threshold=params.significance_threshold,
                num_sources=n_src,
                quality_status=status,
            )

            print(f"  Sources found: {n_src}  Quality: {status}")

            if not quality.get("should_retry", False):
                # Success path
                break

            if attempt < max_attempts:
                params = self._reflect_and_adjust(params, quality, metadata, attempt)
                print(f"  LLM reflection reasoning: {params.reasoning}")
                execution_history.record(
                    self.name,
                    decision=f"Adjusted detection params (attempt {attempt + 1})",
                    parameters={
                        "wavdetect_scales": params.wavdetect_scales,
                        "significance_threshold": params.significance_threshold,
                    },
                    reasoning=params.reasoning,
                )
            # else fall through with whatever we have

        # Post-detection artefacts
        src_data = src_result.data["src_data"]

        plot_path = f"{out_dir}/source_distribution_{band_name}.png"
        self.call_tool(
            "create_source_plot",
            src_data=src_data,
            save_path=plot_path,
            band_label=band_label,
        )
        self.call_tool(
            "print_source_coordinates",
            src_data=src_data,
            band_label=band_label,
            save_path=f"{out_dir}/source_coords_{band_name}.txt",
        )

        return {
            "success": True,
            "src_data": src_data,
            "num_sources": n_src,
            "params": params,
            "x_bounds": (
                (
                    float(src_data["X"].min()),
                    float(src_data["X"].max()),
                )
                if n_src > 0
                else None
            ),
        }

    # -- LLM-powered parameter suggestion -----------------------
    def _suggest_params(
        self,
        metadata: ObservationMetadata,
        run_config: RunConfig | None,
        execution_history: ExecutionHistory | None = None,
    ) -> ProcessingParameters:
        """
        Ask LLM for detection parameters (with replay support).

        Args:
            metadata (ObservationMetadata): The metadata of the observation, containing
                details such as target name, instrument, exposure time, and object type.
            run_config (RunConfig | None): Optional configuration for replaying previous
                runs. If provided and in replay mode, it will return saved parameters
                instead of invoking the LLM.
            execution_history (ExecutionHistory | None): Optional execution history to
                provide context to the LLM for better parameter suggestion.

        Returns:
            ProcessingParameters: The suggested detection parameters, including
                wavdetect scales, significance threshold, and reasoning.
        """
        if run_config and run_config.replay and run_config.has("detection_params"):
            saved = run_config.get("detection_params")
            print("  Replaying saved detection params")
            return ProcessingParameters(**saved)

        history_context = ""
        if execution_history:
            history_context = execution_history.format_for_llm() + "\n\n"

        prompt = prompt_detection_params(
            target_name=metadata.target_name,
            instrument=metadata.instrument,
            exposure_time=metadata.exposure_time,
            object_type=metadata.object_type,
            history_context=history_context.rstrip(),
        )

        raw = invoke_llm_with_retry(prompt)
        parsed = extract_json(raw)

        try:
            adapter = TypeAdapter(DetectionParamsResponse)
            resp = adapter.validate_python(parsed)
            params = ProcessingParameters(
                wavdetect_scales=resp.wavdetect_scales,
                significance_threshold=resp.significance_threshold,
                reasoning=resp.reasoning,
            )
            print(
                f"  LLM suggested: scales={params.wavdetect_scales}, "
                f"threshold={params.significance_threshold}"
            )
        except (ValidationError, Exception) as e:
            print(f"  ⚠ LLM parse error: {e}")
            print(f"  [DEBUG] Raw LLM response: {raw}")
            print(f"  [DEBUG] Parsed JSON: {parsed}")
            print("  Using defaults")
            params = ProcessingParameters(
                wavdetect_scales=WAVDETECT_SCALES,
                significance_threshold=SIGNIFICANCE_THRESHOLD,
                reasoning="Fallback defaults",
            )

        val = self.call_tool(
            "validate_detection_params",
            params=params,
            exposure_time=metadata.exposure_time,
        )
        if val.success and val.data.get("adjusted"):
            params = val.data["params"]

        if run_config and not run_config.replay:
            run_config.log(
                "detection_params",
                {
                    "wavdetect_scales": params.wavdetect_scales,
                    "significance_threshold": params.significance_threshold,
                    "reasoning": params.reasoning,
                },
                source="llm",
            )

        return params

    # -- LLM reflection on failure --------------------------------
    def _reflect_and_adjust(
        self,
        params: ProcessingParameters,
        quality: dict,
        metadata: ObservationMetadata,
        attempt: int,
    ) -> ProcessingParameters:
        """
        Ask LLM to adjust params based on failure analysis.

        Args:
            params (ProcessingParameters): The current detection parameters that were
                used in the failed attempt.
            quality (dict): The quality evaluation of the detection attempt, containing
                status and issues.
            metadata (ObservationMetadata): The metadata of the observation, providing
                context for the LLM to understand the failure.
            attempt (int): The current attempt number, which can be used by the LLM to
                understand how many retries have been made.

        Returns:
            ProcessingParameters: The new detection parameters suggested by the LLM for
                the next attempt.
        """
        failure_ctx = self.memory.get_failure_context()
        prompt = prompt_detection_failure_reflection(
            target_name=metadata.target_name,
            exposure_time=metadata.exposure_time,
            object_type=metadata.object_type,
            quality_info=quality,
            failure_history=failure_ctx,
        )

        raw = invoke_llm_with_retry(prompt)
        parsed = extract_json(raw)

        adapter = TypeAdapter(DetectionParamsResponse)
        resp = adapter.validate_python(parsed)
        new_params = ProcessingParameters(
            wavdetect_scales=resp.wavdetect_scales,
            significance_threshold=resp.significance_threshold,
            reasoning=resp.reasoning,
        )
        print(
            f"  LLM adjustment: scales={new_params.wavdetect_scales}, "
            f"threshold={new_params.significance_threshold}"
        )
        return new_params


# — BandStrategist ———————————————————————————————————————————


class BandStrategist(BaseAgent):
    """
    Decides the energy-band processing mode (full / dual / triple)
    using LLM reasoning on metadata + initial detection results.

    Attributes:
        name (str): Unique name of the agent.
        available_tools (list[str]): List of tool names the agent can use.
    """

    name = "band_strategist"
    available_tools = []  # Pure reasoning agent

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        """
        Decide on the energy-band processing strategy based on observation
        metadata, number of sources detected, and execution history.

        Args:
            context (dict): Input context containing 'metadata',
                optional 'num_sources', 'run_config',
                'forced_band_mode', and 'execution_history'.

        Returns:
            dict: A dictionary containing 'success' status,
                selected 'processing_mode', and 'reasoning' for the decision on
                success; or 'error' message on failure.
        """
        metadata: ObservationMetadata = context["metadata"]
        num_sources: int = context.get("num_sources", 0)
        run_config = context.get("run_config")
        forced_mode: str | None = context.get("forced_band_mode")
        execution_history: ExecutionHistory = context.get(
            "execution_history", ExecutionHistory()
        )

        print(f"\n{'=' * 60}")
        print(f"[{self.name}] Selecting energy-band strategy …")
        print(f"{'=' * 60}")

        # Forced via CLI
        if forced_mode:
            mode = ProcessingMode(forced_mode)
            print(f"  Forced mode: {mode.value}")
            return {"success": True, "processing_mode": mode}

        # Replay
        if run_config and run_config.replay and run_config.has("processing_mode"):
            mode = ProcessingMode(run_config.get("processing_mode"))
            print(f"  Replaying: {mode.value}")
            return {"success": True, "processing_mode": mode}

        # Think — LLM reasoning
        mode, reasoning = self._suggest_mode(metadata, num_sources, execution_history)
        print(f"  LLM reasoning: {reasoning}")
        if run_config and not run_config.replay:
            run_config.log(
                "processing_mode",
                {"mode": mode.value, "reasoning": reasoning},
                source="llm",
            )
        return {"success": True, "processing_mode": mode, "reasoning": reasoning}

    def _suggest_mode(
        self,
        metadata: ObservationMetadata,
        num_sources: int,
        execution_history: ExecutionHistory | None = None,
    ) -> tuple[ProcessingMode, str]:
        """
        Ask LLM to select processing mode.

        Args:
            metadata (ObservationMetadata): The metadata of the observation,
                providing context for the LLM to make an informed decision
                about the processing mode.
            num_sources (int): The number of sources detected in the initial
                detection step, which can influence the decision on whether to use
                full-band or dual-band processing.
            execution_history (ExecutionHistory | None): Optional execution
                history to provide context to the LLM about previous decisions
                and outcomes, which can help it make a more informed decision
                on the processing mode

        Returns:
            tuple[ProcessingMode, str]: The selected processing mode and the reasoning
                behind the decision.
        """
        history_context = ""
        if execution_history:
            history_context = execution_history.format_for_llm()

        prompt = prompt_band_strategy(
            target_name=metadata.target_name,
            instrument=metadata.instrument,
            exposure_time=metadata.exposure_time,
            num_sources=num_sources,
            object_type=metadata.object_type,
            history_context=history_context,
        )

        raw = invoke_llm_with_retry(prompt)
        parsed = extract_json(raw)
        mode_str = parsed.get("processing_mode", "")
        reasoning = parsed.get("reasoning", "")

        try:
            return ProcessingMode(mode_str), reasoning
        except ValueError:
            if num_sources >= DUAL_BAND_THRESHOLD:
                return ProcessingMode.DUAL_BAND, "Fallback: source count threshold"
            return ProcessingMode.FULL_BAND, "Fallback: default"


# — SonificationExpert ———————————————————————————————————————


class SonificationExpert(BaseAgent):
    """
    Suggests sonification parameters via LLM, renders audio/video,
    evaluates quality, and retries with adjusted parameters.
    Multi-turn think → act → reflect loop (max 2 attempts).

    Attributes:
        name (str): Unique name of the agent.
        available_tools (list[str]): List of tool names the agent can use.
    """

    name = "sonification_expert"
    available_tools = [
        "render_sonification",
        "evaluate_audio",
    ]

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        """
        Optimise sonification parameters by iteratively adjusting based on quality
            evaluation.

        Args:
            context (dict): Input context containing necessary data such as
                'metadata', 'src_data', 'out_dir', and optional parameters for band
                selection, execution history, and expected duration.

        Returns:
            dict: A dictionary containing 'success' status, sonification results such as
                'wav'and 'final_video' paths, and reasoning for the decisions made
                on success; or 'error' message on failure.
        """
        metadata: ObservationMetadata = context["metadata"]
        src_data = context["src_data"]
        out_dir: str = context["out_dir"]
        band_name: str = context.get("band_name", "full")
        band_label: str = context.get(
            "band_label", BAND_LABELS.get(band_name, band_name)
        )
        x_bounds = context.get("x_bounds")
        show_animation: bool = context.get("show_animation", True)
        run_config = context.get("run_config")
        expected_duration: float = context.get("expected_duration", DURATION)
        max_attempts: int = context.get("max_sonification_attempts", 2)

        print(f"\n{'=' * 60}")
        print(f"[{self.name}] Sonifying {band_label} band …")
        print(f"{'=' * 60}")

        # Think — suggest parameters
        execution_history: ExecutionHistory = context.get(
            "execution_history", ExecutionHistory()
        )
        config = self._suggest_params(
            metadata, src_data, run_config, expected_duration, execution_history
        )

        for attempt in range(1, max_attempts + 1):
            print(f"\n  --- Sonification attempt {attempt}/{max_attempts} ---")
            print(
                f"  Duration: {config['duration']}s, "
                f"Preset: {config['sf_preset']}, "
                f"Volume: {config['master_volume']}"
            )

            # Act — render
            result = self.call_tool(
                "render_sonification",
                src_data=src_data,
                out_dir=out_dir,
                band_name=band_name,
                band_label=band_label,
                duration=config["duration"],
                note_len=config["note_len"],
                peak_volume_range=tuple(config["peak_volume_range"]),
                stereo_spread=tuple(config["stereo_spread"]),
                sf_preset=config["sf_preset"],
                x_bounds=x_bounds,
                show_animation=show_animation,
                master_volume=config["master_volume"],
            )
            print(
                f"  [DEBUG] Tool result: success={result.success}, "
                f"data keys={list(result.data.keys()) if result.data else 'None'}"
            )
            if not result.success:
                return {"success": False, "error": result.error}

            wav_path = result.data.get("wav")
            print(f"  [DEBUG] wav_path from result: {wav_path}")

            # Reflect — evaluate audio quality
            if wav_path:
                ev = self.call_tool(
                    "evaluate_audio",
                    wav_path=wav_path,
                    expected_duration=config["duration"],
                    num_sources=len(src_data),
                )
                quality = ev.data
                self.memory.log_sonification_attempt(
                    band=band_name, config=config, quality=quality
                )
                print(
                    f"  Audio quality: {quality.get('status', '?')} "
                    f"(score={quality.get('quality_score', '?')})"
                )

                if quality.get("issues"):
                    for issue in quality["issues"]:
                        print(f"    ⚠ {issue}")

                if not quality.get("should_rerender", False):
                    break

                if attempt < max_attempts:
                    config = self._adjust_params(config, quality)
            else:
                break

        return {
            "success": True,
            "band_name": band_name,
            "band_label": band_label,
            "duration": config["duration"],
            "reasoning": config.get("reasoning", ""),  # ← add
            "wav": result.data.get("wav"),
            "silent_video": result.data.get("silent_video"),
            "final_video": result.data.get("final_video"),
        }

    # -- LLM-powered parameter suggestion -----------------------
    def _suggest_params(
        self,
        metadata: ObservationMetadata,
        src_data,
        run_config=None,
        expected_duration: float = DURATION,
        execution_history: ExecutionHistory | None = None,
    ) -> dict[str, Any]:
        """
        Ask LLM for sonification parameters.

        Args:
            metadata (ObservationMetadata): The metadata of the observation, providing
                context for the LLM to suggest appropriate sonification parameters based
                on the characteristics of the target and observation.
            src_data: The source data detected in the previous step, which can influence
                the sonification parameters such as duration and volume.
            run_config (RunConfig | None): Optional configuration for replaying previous
                runs. If provided and in replay mode, it will return saved parameters
                    instead of invoking the LLM.
            expected_duration (float): The expected duration of the sonification, which
                can be used by the LLM to suggest parameters that fit within
                this duration.
            execution_history (ExecutionHistory | None): Optional execution history to
                provide context to the LLM about previous decisions and outcomes, which
                can help it make more informed suggestions for the sonification
                parameters.

        Returns:
            dict[str, Any]: A dictionary containing the suggested sonification
                parameters, including duration, peak volume range, stereo spread,
                note length, soundfont preset, master volume, and reasoning for the
                suggestions.
        """
        key = "sonification_params"
        if run_config and run_config.replay and run_config.has(key):
            saved = run_config.get(key)
            print("  Replaying saved sonification params")
            return saved

        n_src = len(src_data) if src_data is not None else 0
        history_context = ""
        if execution_history:
            history_context = execution_history.format_for_llm()

        prompt = prompt_sonification_params(
            target_name=metadata.target_name,
            num_sources=n_src,
            exposure_time=metadata.exposure_time,
            object_type=metadata.object_type,
            expected_duration=expected_duration,
            history_context=history_context,
        )

        raw = invoke_llm_with_retry(prompt)
        parsed = extract_json(raw)

        try:
            adapter = TypeAdapter(SonificationParamsResponse)
            resp = adapter.validate_python(parsed)
            config = {
                "duration": resp.duration,
                "peak_volume_range": resp.peak_volume_range,
                "stereo_spread": resp.stereo_spread,
                "note_len": resp.note_len,
                "sf_preset": resp.sf_preset,
                "master_volume": resp.master_volume,
                "reasoning": resp.reasoning,
            }
            print(f"  LLM reasoning: {resp.reasoning}preset={config['sf_preset']}")
        except (ValidationError, Exception) as e:
            print(f"  ⚠ LLM parse error: {e}")
            print(f"  [DEBUG] Raw LLM response: {raw[:500]}")
            print(f"  [DEBUG] Parsed JSON: {parsed}")
            print("  Using defaults")
            config = {
                "duration": expected_duration,
                "peak_volume_range": list(PEAK_VOLUME_RANGE),
                "stereo_spread": list(STEREO_SPREAD),
                "note_len": NOTE_LEN,
                "sf_preset": SF_PRESET,
                "master_volume": MASTER_VOLUME,
            }

        if run_config and not run_config.replay:
            run_config.log(key, config, source="llm")

        return config

    # -- Adjust on quality issues --------------------------------
    def _adjust_params(
        self,
        config: dict[str, Any],
        quality: dict,
    ) -> dict[str, Any]:
        """
        Adjust parameters based on quality evaluation.

        Args:
            config (dict[str, Any]): The current sonification parameters that were used
                in the failed attempt.
            quality (dict): The quality evaluation of the sonification attempt,
                containing status and issues that can guide the adjustments needed for
                the next attempt.

        Returns:
            dict[str, Any]: The new sonification parameters adjusted based on the
                quality issues identified.
        """
        issues = quality.get("issues", [])
        new = dict(config)

        for issue in issues:
            issue_lower = issue.lower()
            if "clipping" in issue_lower:
                new["master_volume"] = config["master_volume"] * 0.7
                new["peak_volume_range"] = [
                    config["peak_volume_range"][0] * 0.8,
                    config["peak_volume_range"][1] * 0.8,
                ]
                print(f"  Reduced volume: {new['master_volume']:.2f}")
            elif "silent" in issue_lower or "quiet" in issue_lower:
                new["master_volume"] = min(config["master_volume"] * 1.4, 0.95)
                print(f"  Increased volume: {new['master_volume']:.2f}")

        return new


# — OverlayComposer —————————————————————————————————————————


class OverlayComposer(BaseAgent):
    """
    Ensures overlay artefacts exist and composes final overlay video.
    If overlay animation is missing, creates it, then mixes WAV tracks
    and muxes audio into the overlay video.

    Attributes:
        name (str): Unique name of the agent.
        available_tools (list[str]): List of tool names the agent can use.
    """

    name = "overlay_composer"
    available_tools = [
        "create_overlay_animation",
        "mix_audio",
        "mux_audio_video",
    ]

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        """
        Ensure overlay artefacts exist and compose final overlay video.

        Args:
            context (dict): Input context containing necessary data such as
                'src_data_dict', 'obs_id', 'out_dir', 'duration', 'x_bounds'
                and 'wav_paths'.

        Returns:
            dict: A dictionary containing 'success' status, paths to 'animation',
                'mixed_wav', and 'overlay_video' on success; or 'error' message
                on failure.
        """
        src_data_dict = context["src_data_dict"]
        obs_id: str = context["obs_id"]
        out_dir = Path(context.get("out_dir", f"output_{obs_id}"))
        duration: float = float(context.get("duration", DURATION))
        x_bounds = context.get("x_bounds")
        wav_paths: dict[str, str] = context.get("wav_paths", {})

        print(f"\n{'=' * 60}")
        print(f"[{self.name}] Building overlay video …")
        print(f"{'=' * 60}")

        overlay_anim = out_dir / "sonification_overlay_animation.mp4"
        if overlay_anim.exists():
            anim_path = str(overlay_anim)
            print(f"  Overlay animation exists: {anim_path}")
        else:
            print("  Overlay animation missing; creating...")
            anim_result = self.call_tool(
                "create_overlay_animation",
                src_data_dict=src_data_dict,
                obs_id=obs_id,
                duration=duration,
                x_bounds=x_bounds,
            )
            if not anim_result.success:
                return {"success": False, "error": anim_result.error}
            anim_path = anim_result.data.get("animation_path")
            if not anim_path:
                return {
                    "success": False,
                    "error": "Overlay animation path missing",
                }

        if not wav_paths:
            return {
                "success": False,
                "error": "No WAV paths available for overlay mix",
            }

        mixed_wav = str(out_dir / "mixed_sonification.wav")
        mix_result = self.call_tool(
            "mix_audio",
            wav_paths=list(wav_paths.values()),
            out_path=mixed_wav,
        )
        if not mix_result.success:
            return {"success": False, "error": mix_result.error}

        overlay_video = str(out_dir / "overlay_with_audio.mp4")
        mux_result = self.call_tool(
            "mux_audio_video",
            video_path=anim_path,
            audio_path=mixed_wav,
            out_path=overlay_video,
        )
        if not mux_result.success:
            return {"success": False, "error": mux_result.error}

        return {
            "success": True,
            "animation_path": anim_path,
            "mixed_wav": mixed_wav,
            "overlay_video": overlay_video,
        }


# — QualityEvaluator —————————————————————————————————————————


class QualityEvaluator(BaseAgent):
    """
    Validates pipeline outputs and provides structured feedback
    that other agents can consume for improvement.

    Attributes:
        name (str): Unique name of the agent.
        available_tools (list[str]): List of tool names the agent can use.
    """

    name = "quality_evaluator"
    available_tools = ["evaluate_detection", "evaluate_audio"]

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        """
        Evaluate all completed outputs.

        Args:
            context (dict): Input context containing results to evaluate, such
                as 'detection' results and 'sonifications' details.

        Returns:
            dict: A dictionary containing 'success' status and a nested
                'evaluations' dictionary with the results of each evaluation,
                or an 'error' message on failure.
        """
        results: dict[str, Any] = {"success": True, "evaluations": {}}

        if "detection" in context:
            det = context["detection"]
            ev = self.call_tool(
                "evaluate_detection",
                num_sources=det["num_sources"],
                exposure_time=det["exposure_time"],
                threshold=det["threshold"],
            )
            results["evaluations"]["detection"] = ev.data

        if "sonifications" in context:
            for band, info in context["sonifications"].items():
                wav = info.get("wav")
                if wav:
                    ev = self.call_tool(
                        "evaluate_audio",
                        wav_path=wav,
                        expected_duration=info.get("duration", DURATION),
                        num_sources=info.get("num_sources", 0),
                    )
                    results["evaluations"][f"audio_{band}"] = ev.data

        return results

    def generate_report(self, evaluations: dict) -> str:
        """
        Generate a human-readable quality report.
        Args:
            evaluations (dict): A dictionary containing evaluation results for different
                components, such as detection and audio quality.

        Returns:
            str: A formatted string report summarizing the quality evaluations,
                including status, confidence, quality scores, identified issues,
                and recommendations for each evaluated component.
        """
        lines = ["\n" + "=" * 60, "Quality Evaluation Report", "=" * 60]
        for name, ev in evaluations.items():
            lines.append(f"\n  {name}:")
            lines.append(f"    Status:  {ev.get('status', '?')}")
            if "confidence" in ev:
                lines.append(f"    Confidence: {ev['confidence']:.0%}")
            if "quality_score" in ev:
                lines.append(f"    Quality: {ev['quality_score']:.0%}")
            if "issues" in ev:
                for issue in ev["issues"]:
                    lines.append(f"    ⚠ {issue}")
            if "recommendation" in ev:
                lines.append(f"    Recommendation: {ev['recommendation']}")
        return "\n".join(lines)


class DirectorMemory:
    """
    Records the director's own decisions and their outcomes.
    Separate from AgentMemory (tool calls) and ExecutionHistory (agent actions).

    Attributes:
        decisions (list[dict]): A list of dictionaries, each representing a
            decision made by the director, containing keys such as 'step',
            'agent', 'reasoning', and 'outcome'.
    """

    def __init__(self):
        """
        Initialize the DirectorMemory with an empty list of decisions.
        """
        self.decisions: list[dict[str, Any]] = []

    def record(
        self,
        step: int,
        agent: str,
        reasoning: str,
        outcome: dict[str, Any] | None = None,
    ) -> None:
        """
        Record a decision made by the director, including the step number, the
        agent delegated to, the reasoning behind the decision, and an optional outcome.

        Args:
            step (int): The step number of the decision, used to track the sequence of
                decisions.
            agent (str): The agent delegated to.
            reasoning (str): The reasoning behind the decision.
            outcome (dict[str, Any] | None): An optional outcome of the decision.

        Returns:
            None: This method does not return anything; it updates the internal state of
                the DirectorMemory by appending a new decision record to the decisions
                list.
        """
        self.decisions.append(
            {
                "step": step,
                "agent": agent,
                "reasoning": reasoning,
                "outcome": outcome or {},
            }
        )

    def record_outcome(self, step: int, outcome: dict[str, Any]) -> None:
        """
        Attach outcome to an existing decision after the agent runs.

        Args:
            step (int): The step number of the decision to which the outcome should be
                attached. This is used to identify the specific decision in the
                decisions list that corresponds to this step.
            outcome (dict[str, Any]): The outcome of the decision, typically containing
            results or feedback from the agent that was delegated to.

        Returns:
            None: This method does not return anything; it updates the internal state of
            the DirectorMemory by finding the decision record that matches the given
            step number and updating its 'outcome' field with the provided outcome data.
        """
        for d in reversed(self.decisions):
            if d["step"] == step:
                d["outcome"] = outcome
                return

    def format_for_prompt(self) -> str:
        """
        Format the decision log for LLM context.

        Returns:
            str: A formatted string representing the director's decision log,
                which can be included in the prompt for the LLM to provide
                context on past decisions, reasoning, and outcomes. If there
                are no recorded decisions, it returns a message indicating that
                there are no previous director decisions.
        """
        if not self.decisions:
            return "(No previous director decisions.)"
        lines = ["Director decision log:"]
        for d in self.decisions:
            lines.append(
                f"  Step {d['step']}: delegated to {d['agent']}\n"
                f"    Reason:  {d['reasoning']}\n"
                f"    Outcome: {d['outcome'] or 'pending'}"
            )
        return "\n".join(lines)


# — PipelineDirector —————————————————————————————————————————


class PipelineDirector(BaseAgent):
    """
    Orchestrates the entire pipeline by reasoning about current state
    and delegating work to specialised agents.

    Uses LLM to decide the next action with a deterministic
    state-machine fallback when the LLM is unavailable or unreliable.

    Attributes:
        name (str): Unique name of the agent.
        available_tools (list[str]): List of tool names the agent can use.
            In this case, it delegates via agents rather than using tools directly.
    """

    name = "pipeline_director"
    available_tools = []  # Delegates via agents, not tools directly

    def __init__(
        self,
        memory: AgentMemory,
        agents: dict[str, BaseAgent],
        run_config=None,
    ):
        """
        Initialize the PipelineDirector with its own memory, a dictionary of available
        agents to delegate to, and an optional run configuration for logging and replay.

        Args:

        """
        super().__init__(memory)
        self.agents = agents
        self.run_config = run_config
        self.director_memory = DirectorMemory()
        self._step_counter = 0

    # -- Decision making ----------------------------------------
    def decide_next_action(self, state: dict[str, Any]) -> dict[str, Any]:
        """ """
        obs = self._observe(state)
        history_context = state.get(
            "execution_history", ExecutionHistory()
        ).format_for_llm()
        director_log = self.director_memory.format_for_prompt()

        prompt = prompt_director_decision(
            state_observation=obs,
            execution_history_str=history_context,
            director_log=director_log,
        )

        raw = invoke_llm_with_retry(prompt)
        try:
            parsed = extract_json(raw)
        except ValueError:
            print("  [director] LLM parse failed; using fallback")
            return self._fallback_decision(state)

        action = parsed.get("action", "")
        agent = parsed.get("agent", "")

        if action == "finish":
            return {"action": "finish", "reasoning": parsed.get("reasoning", "Done")}

        if action == "delegate" and agent in self.agents:
            print(f"\n[director] → delegate to {agent}")
            print(f"  Reasoning: {parsed.get('reasoning', '')}")
            if action == "delegate":
                self._step_counter += 1
                self.director_memory.record(
                    step=self._step_counter,
                    agent=agent,
                    reasoning=parsed.get("reasoning", ""),
                )
                if self.run_config:
                    self.run_config.log(
                        f"director_step_{self._step_counter}",
                        {"agent": agent, "reasoning": parsed.get("reasoning", "")},
                        source="director",
                    )
            return parsed

        # Unknown action or agent name — fallback
        print(
            f"  [director] Unrecognised action/agent ('{action}'/'{agent}'); fallback"
        )
        return self._fallback_decision(state)

    # -- Deterministic fallback ----------------------------------
    def _fallback_decision(self, state: dict[str, Any]) -> dict[str, Any]:
        """ """
        completions = state.get("phase_completions", {})
        validity = state.get("validity_report", {})

        if "metadata" not in completions:
            return {
                "action": "delegate",
                "agent": "observation_researcher",
                "reasoning": "Fallback: metadata needed first",
            }

        if "detection" not in completions:
            return {
                "action": "delegate",
                "agent": "detection_optimizer",
                "reasoning": "Fallback: initial detection needed",
            }

        # Validity-driven rerun (once only)
        if validity.get("should_rerun_detection") and not state.get(
            "detection_rerun_attempted"
        ):
            state["detection_rerun_attempted"] = True
            return {
                "action": "delegate",
                "agent": "detection_optimizer",
                "reasoning": "Fallback: validity checks suggest rerun",
            }

        if "band_strategy" not in completions:
            return {
                "action": "delegate",
                "agent": "band_strategist",
                "reasoning": "Fallback: band strategy needed",
            }

        # Per-band detection still pending
        if state.get("bands_pending_detection"):
            return {
                "action": "delegate",
                "agent": "detection_optimizer",
                "reasoning": "Fallback: per-band detection pending",
            }

        # Sonification still pending
        sonified = set(state.get("bands_sonified", []))
        detected = set(state.get("src_data_per_band", {}).keys())
        if detected - sonified:
            return {
                "action": "delegate",
                "agent": "sonification_expert",
                "reasoning": "Fallback: bands detected but not yet sonified",
            }

        return {"action": "finish", "reasoning": "Fallback: all phases complete"}

    # -- Main orchestration loop --------------------------------
    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        """
        Main agentic loop.

        The director observes state, thinks about what to do next,
        delegates to the appropriate agent, and updates state.
        Repeats until all steps are complete or max iterations hit.


        Args:
            context (dict): Initial context for the pipeline, which may include
                observation metadata, execution history, and any pre-filled state
                that the director can use to make informed decisions about which agents
                to delegate to and in what order.

        Returns:
            dict: The final state after the director has completed its orchestration,
                which may include results from the various agents it delegated to,
                as well as any final decisions or outcomes that were recorded in
                the director's memory throughout the process.
        """
        state = dict(context)  # mutable copy
        max_iterations = 20

        for iteration in range(1, max_iterations + 1):
            print(f"\n{'#' * 60}")
            print(f"  Pipeline iteration {iteration}/{max_iterations}")
            print(f"{'#' * 60}")

            # 1. Decide
            decision = self.decide_next_action(state)
            action = decision.get("action", "finish")

            if action == "finish":
                print(
                    f"\n[director] Pipeline complete: {decision.get('reasoning', '')}"
                )
                break

            if action != "delegate":
                print(f"  Unknown action '{action}'; finishing.")
                break

            agent_name = decision["agent"]
            agent = self.agents.get(agent_name)
            if agent is None:
                print(f"  Unknown agent '{agent_name}'; falling back to state machine")
                decision = self._fallback_decision(state)
                agent_name = decision["agent"]
                agent = self.agents.get(agent_name)
                if agent is None:
                    break

            # 2. Delegate
            result = agent.run(state)

            # 3. Update state based on which agent ran
            state = self._update_state(state, agent_name, result)

        return state

    def _update_state(
        self,
        state: dict[str, Any],
        agent_name: str,
        result: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Update the pipeline state based on the results returned by the agent that was
        just run. This method takes the current state, the name of the agent that was
        executed, and the result returned by that agent, and updates the state
        dictionary with new information or flags that indicate which phases have been
        completed, what data has been produced, and any other relevant information that
        will inform future decisions made by the director in subsequent iterations of
        the pipeline orchestration loop.

        Args:
            state (dict[str, Any]): The current state of the pipeline before the agent's
                results are incorporated. This state may include information about
                completed phases, detected sources, sonification results, and any other
                relevant data that the director uses to make decisions.
            result (dict[str, Any]): The output returned by the agent that was just
                executed, which may contain new data, flags indicating completion of
                certain phases, or any other relevant information that should
            agent_name (str): The name of the agent that was just executed, which can be
                used to determine how to interpret the results and what parts of the
                state to update.

        Returns:
            dict[str, Any]: The updated state of the pipeline after incorporating the
                results from the executed agent, which will be used in the next
                iteration of the director's decision-making process.
        """
        if not result.get("success", False):
            print(f"  ⚠ {agent_name} failed: {result.get('error')}")
            return state

        if agent_name == "observation_researcher":
            state["metadata"] = result["metadata"]
            state["metadata_loaded"] = True
            state["metadata_phase_complete"] = True

        elif agent_name == "detection_optimizer":
            state["src_data"] = result["src_data"]
            state["num_sources"] = result["num_sources"]
            state["detection_params"] = result.get("params")
            state["x_bounds"] = result.get("x_bounds")
            state["detection_complete"] = True

        elif agent_name == "band_strategist":
            state["processing_mode"] = result["processing_mode"]
            state["band_mode_selected"] = True

        elif agent_name == "sonification_expert":
            band = result.get("band_name", "full")
            if "sonification_results" not in state:
                state["sonification_results"] = {}
            state["sonification_results"][band] = result
            # Check if sonification is complete for all needed bands
            state["sonification_complete"] = self._all_bands_sonified(state)

        elif agent_name == "quality_evaluator":
            state["quality_report"] = result.get("evaluations", {})

        return state

    def _all_bands_sonified(self, state: dict[str, Any]) -> bool:
        """
        Check if sonification is complete for all needed bands based on the processing
        mode.

        Args:
            state (dict[str, Any]): The current state of the pipeline, which should
                include information about the processing mode and which bands have
                been sonified.

        Returns:
            bool: True if sonification is complete for all needed bands based on the
                processing mode, False otherwise. The method checks the processing mode
                (full, dual, or triple band) and verifies that the corresponding
                sonification results are present in the state to determine if the
                sonification phase can be considered complete.
        """
        mode = state.get("processing_mode", ProcessingMode.FULL_BAND)
        done = set(state.get("sonification_results", {}).keys())

        if mode == ProcessingMode.FULL_BAND:
            return "full" in done
        elif mode == ProcessingMode.DUAL_BAND:
            return {"soft", "hard"}.issubset(done)
        elif mode == ProcessingMode.TRIPLE_BAND:
            return {"soft", "medium", "hard"}.issubset(done)
        return bool(done)

    def _observe(self, state: dict[str, Any]) -> str:
        """
        Observe the current state and return a formatted string summarising key
        information for the director's decision-making process. This method extracts
        relevant details from the state, such as observation metadata,
        phase completions, detection results, band processing status, and recent
        decisions recorded in the director's memory, and compiles them into a
        human-readable format that can be included in the prompt for the LLM to provide
        context on the current situation of the pipeline.

        Args:
            state (dict[str, Any]): The current state of the pipeline, which may include
                information about the observation, completed phases, detection results,
                band processing status, and any other relevant data that the director
                needs to make informed decisions about which agents to delegate to next.

        Returns:
            str: A formatted string summarizing the current state of the pipeline,
                including key details that are relevant for the director's
                decision-making process. This string is intended to provide a clear and
                concise overview of the current situation, which can be used as context
                in the LLM prompt when deciding the next action.
        """
        lines = [f"  OBS_ID: {state.get('obs_id', '?')}"]

        # Phase completions
        completions = state.get("phase_completions", {})
        lines.append(f"\n  Completed phases: {', '.join(completions.keys()) or 'none'}")
        for phase, info in completions.items():
            lines.append(
                f"    {phase}: step {info.get('step', '?')} "
                f"(can be rerun if results were poor)"
            )

        # Metadata
        if state.get("metadata_loaded"):
            meta = state.get("metadata")
            if meta:
                lines.append(f"\n  METADATA:")
                lines.append(f"    Target:      {meta.target_name}")
                lines.append(f"    Instrument:  {meta.instrument}")
                lines.append(f"    Exposure:    {meta.exposure_time:.0f}s")
                lines.append(f"    Object type: {meta.object_type or 'unknown'}")
                lines.append(
                    f"    Redshift:    {meta.redshift if meta.redshift else 'unknown'}"
                )

        # Detection + validity
        if state.get("num_sources") is not None and state.get("detection_complete"):
            lines.append(f"\n  DETECTION (full band):")
            lines.append(f"    Sources found: {state['num_sources']}")
            params = state.get("detection_params")
            if params:
                lines.append(f"    Scales:    {params.wavdetect_scales}")
                lines.append(f"    Threshold: {params.significance_threshold}")
            validity = state.get("validity_report")
            if validity:
                lines.append(
                    f"    Validity:  "
                    f"{validity['n_passed']}/{validity['n_total']} checks passed"
                )
                if validity["checks_failed"]:
                    lines.append(
                        f"    Failed:    {', '.join(validity['checks_failed'])}"
                    )
                if validity["should_rerun_detection"]:
                    lines.append("    ⚠ Validity recommends rerunning detection")
                lines.append(
                    f"    Edge frac: {validity['edge_frac']:.2f}  "
                    f"Expected FP: {validity['expected_fp']:.1f}"
                )

        # Band strategy
        if state.get("processing_mode"):
            lines.append(f"\n  BAND STRATEGY: {state['processing_mode'].value}")

        # Band tracking
        bands_to_sonify = state.get("bands_to_sonify", [])
        if bands_to_sonify:
            band_names = [b[0] for b in bands_to_sonify]
            pending_det = [b[0] for b in state.get("bands_pending_detection", [])]
            detected = list(state.get("src_data_per_band", {}).keys())
            sonified = state.get("bands_sonified", [])
            lines.append(f"\n  BAND PROCESSING:")
            lines.append(f"    All bands:       {', '.join(band_names)}")
            lines.append(f"    Pending detect:  {', '.join(pending_det) or 'none'}")
            lines.append(f"    Detected:        {', '.join(detected) or 'none'}")
            lines.append(f"    Sonified:        {', '.join(sonified) or 'none'}")
            remaining = set(band_names) - set(sonified)
            if remaining:
                lines.append(f"    Still needed:    {', '.join(remaining)}")

        lines.append(f"\n  {self.memory.summary()}")
        lines.append(f"\n  DIRECTOR LOG:\n{self.director_memory.format_for_prompt()}")

        return "\n".join(lines)


# endregion
