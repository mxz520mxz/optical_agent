"""
Optical Lens Design Agent
─────────────────────────
An AI agent that designs optical lenses from natural language descriptions.

Workflow:
  1. User describes desired lens in natural language
  2. Agent interprets requirements and designs initial structure
  3. Agent runs DeepLens gradient-based optimization
  4. Agent evaluates optical performance
  5. Agent decides whether to modify structure (add/remove elements)
  6. Agent repeats until performance goals are met or max iterations reached
  7. Agent generates final design report

Usage:
  python agent.py
  python agent.py --description "设计一个50mm f/1.8全画幅相机镜头"
  python agent.py --description "Design a 24mm wide angle lens for APS-C sensor" --max-iter 3
  python agent.py --provider openai --model gpt-4o -d "50mm f/2.8 lens"
  python agent.py --provider local --base-url http://localhost:11434/v1 --model qwen2.5:72b -d "50mm lens"
"""

import argparse
import json
import os
import sys
from typing import Any

import lens_tools
import llm_provider

# ─── Tool Definitions for Claude ──────────────────────────────────────────────

TOOLS: list[dict] = [
    {
        "name": "design_initial_structure",
        "description": (
            "Generate an initial lens structure from optical specifications. "
            "Call this first to create a starting design."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "focal_length": {
                    "type": "number",
                    "description": "Target focal length in mm (e.g., 50.0)",
                },
                "fnum": {
                    "type": "number",
                    "description": "Target f-number, e.g., 2.8 for f/2.8",
                },
                "fov_deg": {
                    "type": "number",
                    "description": "Full diagonal field of view in degrees (e.g., 46 for 50mm full-frame)",
                },
                "sensor_size": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Sensor [width, height] in mm, e.g., [36.0, 24.0] for full-frame",
                },
                "num_elements": {
                    "type": "integer",
                    "description": "Number of lens elements to use (optional, auto-selected if not given)",
                },
                "lens_type": {
                    "type": "string",
                    "description": (
                        "Preferred lens type: 'singlet', 'doublet', 'triplet', "
                        "'double_gauss', 'telephoto', 'wide_angle'"
                    ),
                },
                "wavelengths": {
                    "type": "string",
                    "description": "Wavelength range: 'visible' (default) or 'nir'",
                },
            },
            "required": ["focal_length", "fnum"],
        },
    },
    {
        "name": "run_optimization",
        "description": (
            "Run DeepLens gradient-based optimization to improve lens performance. "
            "Call this after design_initial_structure or after modifying structure."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "lens_id": {
                    "type": "string",
                    "description": "Lens ID returned by design_initial_structure or a previous tool call (e.g. 'lens_a1b2c3d4')",
                },
                "iterations": {
                    "type": "integer",
                    "description": "Number of optimization iterations (default: 2000)",
                },
                "learning_rates": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Learning rates for [d, c, k, a] parameters. Default: [1e-3, 1e-4, 1e-1, 1e-4]",
                },
                "decay": {
                    "type": "number",
                    "description": "Learning rate decay factor (default: 0.02)",
                },
            },
            "required": ["lens_id"],
        },
    },
    {
        "name": "evaluate_lens",
        "description": (
            "Evaluate the optical performance of a lens. "
            "Returns RMS spot size, distortion, chromatic aberration, and MTF estimates."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "lens_id": {
                    "type": "string",
                    "description": "Lens ID to evaluate (e.g. 'lens_a1b2c3d4')",
                },
            },
            "required": ["lens_id"],
        },
    },
    {
        "name": "add_lens_element",
        "description": (
            "Add a new lens element to improve performance. "
            "Use when current design cannot meet requirements after optimization."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "lens_id": {
                    "type": "string",
                    "description": "Lens ID of the current design (e.g. 'lens_a1b2c3d4')",
                },
                "position": {
                    "type": "string",
                    "enum": ["front", "rear", "middle"],
                    "description": (
                        "'front' to add before first element, "
                        "'rear' to add after last element, "
                        "'middle' to add near aperture stop"
                    ),
                },
                "material": {
                    "type": "string",
                    "description": (
                        "Glass material as 'n/V' (refractive index / Abbe number). "
                        "Crown examples: '1.5168/64.2' (N-BK7), '1.6180/63.4' (N-LAK9). "
                        "Flint examples: '1.6200/36.4' (N-F2), '1.7847/25.7' (N-SF11)"
                    ),
                },
                "element_type": {
                    "type": "string",
                    "enum": ["biconvex", "biconcave", "meniscus", "plano_convex"],
                    "description": "Shape of the new element",
                },
            },
            "required": ["lens_id"],
        },
    },
    {
        "name": "remove_lens_element",
        "description": (
            "Remove a lens element when the design is over-engineered "
            "or to simplify while maintaining performance."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "lens_id": {
                    "type": "string",
                    "description": "Lens ID of the current design (e.g. 'lens_a1b2c3d4')",
                },
                "element_index": {
                    "type": "integer",
                    "description": "1-based index of the element to remove (front-to-back)",
                },
            },
            "required": ["lens_id", "element_index"],
        },
    },
    {
        "name": "save_lens",
        "description": "Save the final lens design to a JSON or ZMX file.",
        "input_schema": {
            "type": "object",
            "properties": {
                "lens_id": {
                    "type": "string",
                    "description": "Lens ID to save (e.g. 'lens_a1b2c3d4')",
                },
                "filename": {
                    "type": "string",
                    "description": "Output filename without extension (default: 'final_lens')",
                },
                "fmt": {
                    "type": "string",
                    "enum": ["json", "zmx"],
                    "description": "Output format: 'json' (default) or 'zmx' (Zemax)",
                },
            },
            "required": ["lens_id"],
        },
    },
    {
        "name": "generate_report",
        "description": (
            "Generate a formatted design report summarizing the final lens. "
            "Call this as the last step."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "lens_id": {
                    "type": "string",
                    "description": "Lens ID of the final design (e.g. 'lens_a1b2c3d4')",
                },
                "metrics": {
                    "type": "object",
                    "description": "Evaluation metrics dict from evaluate_lens",
                },
            },
            "required": ["lens_id", "metrics"],
        },
    },
]

# ─── Tool Dispatcher ───────────────────────────────────────────────────────────

TOOL_FUNCTIONS: dict[str, Any] = {
    "design_initial_structure": lens_tools.design_initial_structure,
    "run_optimization": lens_tools.run_optimization,
    "evaluate_lens": lens_tools.evaluate_lens,
    "add_lens_element": lens_tools.add_lens_element,
    "remove_lens_element": lens_tools.remove_lens_element,
    "save_lens": lens_tools.save_lens,
    "generate_report": lens_tools.generate_report,
}


def execute_tool(name: str, inputs: dict) -> str:
    """Execute a tool and return its result as a JSON string."""
    fn = TOOL_FUNCTIONS.get(name)
    if fn is None:
        return json.dumps({"error": f"Unknown tool: {name}"})
    try:
        result = fn(**inputs)
        return json.dumps(result, indent=2, default=str)
    except Exception as e:
        return json.dumps({"error": str(e), "tool": name})


# ─── System Prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert optical lens designer with deep knowledge of geometric optics, aberration theory, and computational lens design. Your task is to design camera lenses based on user requirements using a systematic workflow.

## Design Workflow

Follow these steps in order:

1. **Interpret** the user's description and extract:
   - Focal length (mm)
   - F-number (e.g., f/2.8)
   - Field of view (degrees) or sensor format
   - Special requirements (low distortion, low chromatic aberration, compact size, etc.)

2. **Design initial structure** using `design_initial_structure` with appropriate parameters.

3. **Optimize** the lens using `run_optimization`. Use 1000-3000 iterations for a good result.

4. **Evaluate** performance using `evaluate_lens`. Check:
   - RMS spot size (< 10 µm is excellent, < 20 µm is good, > 30 µm needs improvement)
   - Distortion (< 1% is excellent, < 3% is acceptable)
   - Chromatic aberration (< 5 µm is good)

5. **Decide** based on evaluation:
   - If metrics meet requirements → proceed to step 6
   - If RMS spot is too large at edge fields → `add_lens_element` at 'rear' position
   - If chromatic aberration is high → `add_lens_element` with a flint glass material
   - If center AND edge RMS are both poor → try a completely different structure
   - Maximum 3 structural modifications per design session

6. **Iterate**: After any structural change, re-run optimization (step 3) and re-evaluate (step 4).

7. **Save** the final design using `save_lens`.

8. **Report** using `generate_report` to summarize results.

## Glass Material Guide
- Crown glasses (low dispersion, positive elements):
  - N-BK7: '1.5168/64.2' (standard crown, affordable)
  - N-LAK9: '1.6910/54.7' (dense crown, compact designs)
  - N-LAK21: '1.6400/60.1' (lanthanum crown)
- Flint glasses (high dispersion, negative elements for chromatic correction):
  - N-F2: '1.6200/36.4' (standard flint)
  - N-SF11: '1.7847/25.7' (dense flint)
  - N-SF5: '1.6727/32.1' (medium flint)

## Lens ID Memory System

All tools use **lens IDs** (short strings like `"lens_a1b2c3d4"`) instead of full JSON configurations.
- `design_initial_structure` returns a `lens_id` — use this in all subsequent calls
- Each tool that modifies a lens returns a **new** `lens_id` — always use the latest ID
- Never pass raw JSON configs to tools; always pass the `lens_id` string

## Important Notes
- Always reason step by step about which modification will help most
- If DeepLens is not installed, optimization returns mock results — still follow the workflow
- Explain your reasoning at each decision point in plain language
- Be concise but informative in your progress updates
"""

# ─── Main Agent Loop ──────────────────────────────────────────────────────────

def run_agent(
    user_description: str,
    max_struct_iter: int = 3,
    verbose: bool = True,
    provider: str = "anthropic",
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
) -> str:
    """
    Run the lens design agent for a given user description.

    Args:
        user_description: Natural language description of the desired lens
        max_struct_iter: Maximum structural modification iterations
        verbose: Whether to print progress to stdout
        provider: LLM provider — "anthropic" (default), "openai", or "local"
        model: Model name (defaults to provider's default if not given)
        base_url: API base URL (required for "local"; e.g. http://localhost:11434/v1)
        api_key: Override API key (otherwise read from environment variable)

    Returns:
        Final report text
    """
    resolved_model = model or llm_provider.DEFAULT_MODELS.get(provider, "")
    client = llm_provider.make_client(provider, api_key=api_key, base_url=base_url)

    messages = [
        {
            "role": "user",
            "content": (
                f"{user_description}\n\n"
                f"(Maximum structural modifications allowed: {max_struct_iter}. "
                "Please follow the complete design workflow and generate a final report.)"
            ),
        }
    ]

    if verbose:
        print("\n" + "=" * 60)
        print("OPTICAL LENS DESIGN AGENT")
        print(f"Provider: {provider}  Model: {resolved_model}")
        print("=" * 60)
        print(f"Requirement: {user_description}")
        print("=" * 60 + "\n")

    final_report = ""
    iteration = 0
    max_iterations = 20  # safety limit on total API calls

    while iteration < max_iterations:
        iteration += 1

        if verbose:
            print(f"[Agent call {iteration}] Thinking...", end=" ", flush=True)

        resp = llm_provider.call_llm(
            client=client,
            provider=provider,
            model=resolved_model,
            system=SYSTEM_PROMPT,
            messages=messages,
            tools=TOOLS,
            max_tokens=8192,
        )

        if verbose:
            print(f"stop_reason={resp['stop_reason']}")

        # Append assistant turn to history
        llm_provider.append_assistant_message(messages, provider, resp["raw"])

        # Print text output
        if resp["text"] and verbose:
            print(f"\n[Agent]: {resp['text']}\n")
            if "report" in resp["text"].lower() and "╔" in resp["text"]:
                final_report = resp["text"]

        # Check stop condition
        if resp["stop_reason"] == "end_turn":
            break

        if resp["stop_reason"] != "tool_use":
            if verbose:
                print(f"Unexpected stop reason: {resp['stop_reason']}")
            break

        # Execute all tool calls
        tool_results = []
        for tc in resp["tool_calls"]:
            tool_name = tc["name"]
            tool_input = tc["input"]

            if verbose:
                print(f"  → Tool: {tool_name}({_fmt_tool_args(tool_input)})")

            result_str = execute_tool(tool_name, tool_input)

            if verbose:
                try:
                    result_dict = json.loads(result_str)
                    summary = result_dict.get("summary", result_dict.get("error", ""))
                    if summary:
                        print(f"    ✓ {summary[:120]}")
                    if "report" in result_dict:
                        print(result_dict["report"])
                        final_report = result_dict["report"]
                except Exception:
                    pass

            tool_results.append({"id": tc["id"], "name": tool_name, "content": result_str})

        llm_provider.append_tool_results(messages, provider, tool_results)

    if verbose:
        print("\n" + "=" * 60)
        print("DESIGN SESSION COMPLETE")
        print("=" * 60)

    return final_report


def _fmt_tool_args(inputs: dict) -> str:
    """Format tool arguments for display."""
    parts = []
    for k, v in inputs.items():
        if k == "metrics" and isinstance(v, dict):
            parts.append("metrics=<...>")
        else:
            parts.append(f"{k}={repr(v)}")
    return ", ".join(parts)


# ─── CLI Entry Point ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Optical Lens Design Agent — design lenses from natural language",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python agent.py
  python agent.py --description "设计一个50mm f/1.8全画幅相机镜头"
  python agent.py --description "Design a compact 24mm f/2.8 wide-angle for APS-C"
  python agent.py --description "telephoto 200mm f/4 for wildlife photography" --max-iter 3
  python agent.py --provider openai --model gpt-4o -d "50mm f/2.8 standard lens"
  python agent.py --provider local --base-url http://localhost:11434/v1 --model qwen2.5:72b -d "50mm lens"
        """,
    )
    parser.add_argument(
        "--description", "-d",
        type=str,
        default=None,
        help="Natural language description of the desired lens",
    )
    parser.add_argument(
        "--max-iter", "-m",
        type=int,
        default=3,
        help="Maximum number of structural modification iterations (default: 3)",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress verbose output",
    )
    parser.add_argument(
        "--provider", "-p",
        type=str,
        default="anthropic",
        choices=["anthropic", "openai", "local"],
        help="LLM provider: anthropic (default), openai, or local",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=(
            "Model name to use (default per provider: "
            "claude-opus-4-6 / gpt-4o / qwen2.5:72b)"
        ),
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="API base URL for local models (e.g. http://localhost:11434/v1)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Override API key (otherwise reads from environment variable)",
    )
    args = parser.parse_args()

    # Interactive mode if no description given
    if args.description is None:
        print("Optical Lens Design Agent")
        print("─" * 40)
        print("Describe the lens you want to design (in Chinese or English):")
        print("Examples:")
        print("  设计一个50mm f/1.8全画幅标准镜头")
        print("  Design a wide-angle 24mm f/2.8 lens for APS-C sensor")
        print("  telephoto 200mm f/4 with minimal chromatic aberration")
        print()
        description = input("> ").strip()
        if not description:
            print("No description provided. Exiting.")
            sys.exit(1)
    else:
        description = args.description

    report = run_agent(
        user_description=description,
        max_struct_iter=args.max_iter,
        verbose=not args.quiet,
        provider=args.provider,
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
    )

    if args.quiet and report:
        print(report)


if __name__ == "__main__":
    main()
