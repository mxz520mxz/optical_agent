# Optical Lens Design Agent — Developer Guide

## Project Overview

An AI agent that designs camera lenses from natural language descriptions, using DeepLens for differentiable optical simulation.

## File Map

| File | Role |
|------|------|
| `agent.py` | Main agentic loop, tool schemas, CLI entry point |
| `llm_provider.py` | Multi-provider LLM abstraction (Anthropic / OpenAI / local) |
| `lens_tools.py` | Tool implementations called by the agent |
| `lens_memory.py` | Lens config storage — maps short IDs to on-disk JSON |
| `lens_templates.py` | Template selection & scaling logic |
| `templates/*.json` | Starting optical structures (singlet → double_gauss) |
| `DeepLens/` | Git-cloned DeepLens repo (not committed, see setup.sh) |
| `results/` | Output directory (not committed) |
| `results/memory/` | Lens config store (auto-created by lens_memory) |

## Lens ID Memory System

**Problem solved**: Full `lens_config` dicts (1700–2500 chars each) would bloat the LLM message history across multiple design iterations.

**Solution**: All tools store configs on disk and pass short IDs instead.

```
design_initial_structure(...)  →  {"lens_id": "lens_a1b2c3d4", ...}
run_optimization(lens_id=...)  →  {"lens_id": "lens_b5e6f7g8", ...}  # new ID
evaluate_lens(lens_id=...)     →  {metrics...}  # no lens_id in output
```

**Key rules:**
- Every tool that **creates or modifies** a lens returns a **new** `lens_id`
- The agent always passes the **latest** `lens_id` to the next tool
- `results/memory/<lens_id>.json` holds the full config on disk
- `lens_memory.py` is the only place that reads/writes these files

## Adding a New Tool

1. Implement the function in `lens_tools.py`:
   - Accept `lens_id: str` as first param (not `lens_config: dict`)
   - Call `lens_memory.load(lens_id)` to get the config
   - Call `lens_memory.store(new_config)` and return the new `lens_id`

2. Add the tool schema to `TOOLS` in `agent.py`:
   - Use `"type": "string"` for the `lens_id` parameter
   - Add to `TOOL_FUNCTIONS` dispatcher

## LLM Provider Configuration

```bash
# Anthropic (default)
export ANTHROPIC_API_KEY="..."
python agent.py -d "50mm f/1.8 lens"

# OpenAI
export OPENAI_API_KEY="..."
python agent.py --provider openai --model gpt-4o -d "50mm lens"

# Local (Ollama)
python agent.py --provider local --base-url http://localhost:11434/v1 --model qwen2.5:72b -d "50mm lens"
```

Supported providers are defined in `llm_provider.DEFAULT_MODELS`.

## DeepLens Setup

```bash
bash setup.sh        # clone DeepLens + pip install -r requirements.txt
```

Or manually:
```bash
git clone https://github.com/vccimaging/DeepLens.git
pip install -r requirements.txt
```

`lens_tools.py` auto-detects `./DeepLens/` and adds it to `sys.path`. If DeepLens is absent, all tools run in mock mode.

## Quick Test

```bash
# Verify memory module
python -c "
import lens_memory
lid = lens_memory.store({'surfaces': [], 'foclen': 50.0}, 'test')
assert lens_memory.load(lid)['foclen'] == 50.0
print('OK:', lid)
"

# Verify tool pipeline (mock mode, no API key needed for imports)
python -c "
import lens_tools
r = lens_tools.design_initial_structure(50.0, 2.8, fov_deg=46)
print('lens_id:', r['lens_id'])
r2 = lens_tools.run_optimization(r['lens_id'], iterations=10)
print('optimized:', r2['lens_id'])
r3 = lens_tools.evaluate_lens(r2['lens_id'])
print('rms:', r3['rms_spot_um'])
"
```
