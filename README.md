# Bias Audit for Candidate Screening (PE Team 13 Prototype)

**Theme:** Reliability, Hallucination, and Safety  
**Hackathon Problem:** Bias Audit for Candidate Screening  

Testing whether an LLM treats otherwise-identical candidates differently based on irrelevant attributes (e.g., name, gender-coded name, university) and validating prompt mitigations (v1 baseline vs. v2 debiased).

## Architecture

```
project/
├── app/
│   ├── __init__.py
│   ├── schema.py       # Pydantic schemas for LLM output & validation
│   ├── prompts.py      # Isolated prompt templates (v1 baseline, v2 debiased)
│   ├── guardrails.py   # Application guardrails (empty, off-topic, toxic bias inputs)
│   ├── llm_client.py   # LLM abstraction (Gemini, OpenAI, Mock provider)
│   ├── screen.py       # Core screening function with controlled retry logic
│   ├── batch.py        # Batch execution layer for counterfactual pairs
│   └── ui.py           # Streamlit UI with single & side-by-side screening
├── data/
│   ├── counterfactual_pairs.json  # 30+ counterfactual candidate pairs
│   └── default_job.txt            # Configurable default SDE job description
├── tests/
│   ├── test_schema.py
│   ├── test_guardrails.py
│   ├── test_screen.py
│   └── test_batch.py
├── .env.example
├── requirements.txt
└── README.md
```

## Quick Start

1. **Install Dependencies**
   ```bash
   uv pip install -r requirements.txt
   # or pip install -r requirements.txt
   ```

2. **Configure Environment**
   ```bash
   cp .env.example .env
   # Set GEMINI_API_KEY or LLM_API_KEY
   ```

3. **Run Streamlit UI**
   ```bash
   streamlit run app/ui.py
   ```

4. **Run Unit Tests**
   ```bash
   pytest tests/
   ```
