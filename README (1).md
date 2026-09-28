# ARC-AGI-3 submission — ProgramSynthesisKitAgent

Self-contained agent file: `program_synthesis_agent.py`
(no dependencies beyond numpy; arcengine used when available).

## Install into the official kit

```bash
git clone https://github.com/arcprize/ARC-AGI-3-Agents
cd ARC-AGI-3-Agents
pip install -e .
cp /path/to/program_synthesis_agent.py .
```

Register the agent (e.g. in `agents/__init__.py`):

```python
from program_synthesis_agent import ProgramSynthesisKitAgent
AVAILABLE_AGENTS["program_synthesis"] = ProgramSynthesisKitAgent
```

Then follow the kit README to run it (API key via ARC_API_KEY).

## Contest submission
Fill in the Google Form: https://forms.gle/wMLZrEFGDh33DhzV9
