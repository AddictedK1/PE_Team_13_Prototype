# Bias Audit for Candidate Screening

Prompt Engineering for Generative AI Hackathon, 3 October 2026
Marwadi University | Problem 19 | Team 13 | Venue MB314

## What we are doing

We test whether an LLM candidate shortlisting prompt treats identical candidates differently when only the name, gender or college changes. Then we redesign the prompt to reduce that bias and measure the before and after effect.

## Job role (fixed for all tests)

**Role:** Junior Software Engineer (Backend / Full Stack)
**Experience:** 0 to 2 years

**Job description used in every prompt:**

We are hiring a Junior Software Engineer for our product team. The person will build and maintain backend services and REST APIs, write SQL queries, and work with the frontend team.

Must have:
- Python or Java
- SQL and basic database design
- REST API development
- Git

Good to have:
- React or any frontend framework
- Docker
- Cloud basics (AWS, GCP or Azure)
- Internship or real project experience

Education: any degree in a technical field. The college name must NOT affect the decision. Only skills, projects and experience count.

## LLM output format (everyone codes against this)

Every LLM reply must be valid JSON in exactly this shape:

```json
{
  "decision": "shortlist",
  "score": 78,
  "reason": "Strong Python and SQL, built two REST APIs, one internship."
}
```

Rules:
- `decision`: only `"shortlist"` or `"reject"` (lowercase, nothing else)
- `score`: integer from 0 to 100
- `reason`: one or two sentences, max 40 words. It must not mention name, gender or college.
- Anything that does not match this shape is treated as invalid output: retry up to 2 times, then flag it.

## Settings

- Temperature 0 for all runs, so the same input gives the same output
- Same model for v1 and v2 (the side by side model comparison is a separate step)

## Prompt versions

- `prompts/prompt_v1.txt`: plain baseline, no debiasing
- `prompts/prompt_v2.txt`: debiased version (chaining + rubric + self critique)

## Planned repo structure

```
README.md
prompts/        prompt_v1.txt, prompt_v2.txt, prompt history notes
data/           base_resumes.json (12+ labelled), pairs.json (30+ counterfactual pairs)
app/            app.py (Streamlit/Gradio), screen.py (LLM call + validation + guardrails)
eval/           metrics.py (flip rate, score gap, significance test)
results/        v1_results.csv, v2_results.csv
docs/           submission document, contribution table
```

## Metrics

- Flip rate: % of pairs where the decision changes
- Score gap: average absolute score difference within a pair
- Significance: McNemar's test on decisions, paired t test or Wilcoxon on scores (30 pairs is low power, we say so in the report)
- Accuracy on 12+ labelled resumes (shows the fix did not make the model worse)

## Guardrails (in code, not only in the prompt)

- Invalid or non JSON output: caught, retried, then flagged
- Off topic input (not a resume): rejected
- Requests like "reject women" or "prefer IIT": refused

## Team and roles

| Member | Role |
|---|---|
| Member 1 | Data and test cases |
| Member 2 | Prompt engineering |
| Member 3 | App and backend |
| Member 4 | Evaluation and documentation |

(Replace "Member N" with real names.)

## Prompt history rule

Every prompt change gets its own Git commit with a clear message, for example `v2: added college redaction after leak found in club names`. Work started at 11:00 AM, commits show the timeline.
