"""Streamlit User Interface for Bias Audit Candidate Screening.

Supports:
1. Candidate resume upload as PDF with stable text extraction and session state
2. Single candidate screening (v1 baseline or v2 debiased)
3. Side-by-side v1 vs v2 comparison on the same resume
4. Configurable job description and guardrail violation reporting
5. Batch counterfactual evaluation across 30+ pairs
"""

from __future__ import annotations

import io
import json
import logging
import os
import sys
from pathlib import Path

# Ensure project root is in sys.path for robust imports across all working directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
from app.llm_client import get_llm_client
from app.pdf_parser import (
    CorruptedPDFError,
    EmptyPDFError,
    PDFExtractionError,
    PDFSizeLimitError,
    extract_resume_text,
)
from app.prompts import DEFAULT_JOB_DESCRIPTION, DEFAULT_PROMPT_TEMPLATE
from app.schema import ScreeningResult
from app.screen import screen, screen_resume_pdf

logger = logging.getLogger(__name__)


def display_result_card(res: ScreeningResult, title: str):
    """Render a structured screening result card without emojis."""
    st.markdown(f"### {title}")

    if not res.valid:
        st.error("**Validation / Guardrail Status: BLOCKED / INVALID**")
        st.markdown(f"**Error Category:** `{res.error or 'unknown_failure'}`")
        st.markdown(f"**Reason:** {res.reason}")
        if res.retried:
            st.info("System attempted 1 controlled retry before returning invalid.")
        st.caption("Note: System failures are safely recorded as invalid and NOT as a candidate rejection.")
        return

    # Valid candidate evaluation
    if res.decision == "shortlist":
        st.success("**Decision: SHORTLIST**")
    else:
        st.error("**Decision: REJECT**")

    metric_col1, metric_col2 = st.columns(2)
    with metric_col1:
        st.metric(label="Suitability Score", value=f"{res.score}/100")
    with metric_col2:
        st.metric(label="Validation Status", value="PASSED", delta="Retried" if res.retried else "First Attempt")

    st.markdown("#### Evaluation Reason:")
    st.info(res.reason)

    if res.raw_response:
        with st.expander("View Raw LLM Output"):
            st.code(res.raw_response, language="json")


def main():
    """Main Streamlit execution entrypoint."""
    try:
        st.set_page_config(
            page_title="Bias Audit - Candidate Screening",
            layout="wide",
        )
    except Exception:
        pass

    st.title("Bias Audit for Candidate Screening")
    st.caption("Reliability, Hallucination and Safety Hackathon | PE Team 13 Prototype")

    # Session State Initialization for robust PDF persistence across reruns
    if "uploaded_resume_name" not in st.session_state:
        st.session_state.uploaded_resume_name = None
    if "extracted_resume_text" not in st.session_state:
        st.session_state.extracted_resume_text = ""
    if "resume_ready" not in st.session_state:
        st.session_state.resume_ready = False
    if "pdf_error_message" not in st.session_state:
        st.session_state.pdf_error_message = None

    # Sidebar Configuration
    st.sidebar.header("Configuration")
    mode = st.sidebar.radio(
        "Evaluation Mode",
        [
            "Side-by-Side Audit (v1 vs v2)",
            "Single Version Screening",
            "Batch Counterfactual Audit (30+ Pairs)",
        ],
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("LLM Provider")
    provider_choice = st.sidebar.selectbox(
        "Provider",
        ["Auto-detect", "gemini", "openai", "mock"],
        index=0,
    )
    provider_arg = None if provider_choice == "Auto-detect" else provider_choice

    # Main Layout
    col_jd, col_resume = st.columns(2)

    with col_jd:
        st.subheader("1. Job Description")
        job_desc = st.text_area(
            "Enter or edit the target job requirements:",
            value=DEFAULT_JOB_DESCRIPTION,
            height=280,
        )

    with col_resume:
        st.subheader("2. Candidate Resume")
        uploaded_file = st.file_uploader(
            "Upload candidate resume as PDF",
            type=["pdf"],
            help="Accepted format: PDF",
        )

        st.text_area(
            "Gemini prompt template",
            value=DEFAULT_PROMPT_TEMPLATE,
            height=260,
            key="gemini_prompt_template",
        )

        if uploaded_file is not None:
            # Process new file or reprocess if filename changed
            if (
                st.session_state.uploaded_resume_name != uploaded_file.name
                or not st.session_state.resume_ready
            ):
                try:
                    if hasattr(uploaded_file, "seek"):
                        uploaded_file.seek(0)
                    text = extract_resume_text(uploaded_file)
                    st.session_state.uploaded_resume_name = uploaded_file.name
                    st.session_state.extracted_resume_text = text
                    st.session_state.resume_ready = True
                    st.session_state.pdf_error_message = None
                except EmptyPDFError:
                    st.session_state.uploaded_resume_name = uploaded_file.name
                    st.session_state.extracted_resume_text = ""
                    st.session_state.resume_ready = False
                    st.session_state.pdf_error_message = (
                        "Could not extract text from this PDF. Please upload a text-based resume PDF."
                    )
                except CorruptedPDFError:
                    st.session_state.uploaded_resume_name = uploaded_file.name
                    st.session_state.extracted_resume_text = ""
                    st.session_state.resume_ready = False
                    st.session_state.pdf_error_message = (
                        "Could not read this PDF. Please upload a valid text-based resume."
                    )
                except PDFSizeLimitError as e:
                    st.session_state.uploaded_resume_name = uploaded_file.name
                    st.session_state.extracted_resume_text = ""
                    st.session_state.resume_ready = False
                    st.session_state.pdf_error_message = str(e)
                except Exception as e:
                    logger.exception("Unexpected error extracting PDF")
                    st.session_state.uploaded_resume_name = uploaded_file.name
                    st.session_state.extracted_resume_text = ""
                    st.session_state.resume_ready = False
                    st.session_state.pdf_error_message = (
                        f"Could not read this PDF. Please upload a valid text-based resume."
                    )
        else:
            # File uploader cleared by user
            if st.session_state.uploaded_resume_name is not None:
                st.session_state.uploaded_resume_name = None
                st.session_state.extracted_resume_text = ""
                st.session_state.resume_ready = False
                st.session_state.pdf_error_message = None

        # Display upload feedback and collapsible text preview
        if st.session_state.uploaded_resume_name:
            st.markdown(f"Uploaded file:\n`{st.session_state.uploaded_resume_name}`")

        if st.session_state.pdf_error_message:
            st.error(st.session_state.pdf_error_message)
        elif st.session_state.resume_ready and st.session_state.extracted_resume_text:
            st.success("Status: Resume loaded successfully.")
            with st.expander("Extracted Resume Text", expanded=False):
                st.text(st.session_state.extracted_resume_text)

    # Execution flows
    if mode == "Single Version Screening":
        st.markdown("---")
        st.subheader("3. Select Prompt and Screen")
        version = st.selectbox("Prompt Version", ["v1 (Baseline)", "v2 (Debiased)"])
        version_key = "v1" if "v1" in version else "v2"

        if st.button("Screen Candidate", type="primary", use_container_width=True):
            if not st.session_state.resume_ready or not st.session_state.extracted_resume_text.strip():
                if st.session_state.pdf_error_message:
                    st.error(st.session_state.pdf_error_message)
                else:
                    st.warning("Please upload a candidate resume PDF.")
            else:
                with st.spinner(f"Running screening with {version_key.upper()}..."):
                    try:
                        client = get_llm_client(provider=provider_arg)
                        result = screen(
                            resume=st.session_state.extracted_resume_text,
                            prompt_version=version_key,
                            job_description=job_desc,
                            client=client,
                            custom_prompt=st.session_state.gemini_prompt_template,
                        )
                        st.markdown("---")
                        display_result_card(result, f"Screening Result ({version_key.upper()})")
                    except Exception as e:
                        st.error(f"Application error: {e}")

        if st.button("Evaluate Uploaded PDF With Gemini", type="primary", use_container_width=True):
            if uploaded_file is None:
                st.warning("Please upload a PDF resume before evaluating.")
            else:
                with st.spinner("Sending the uploaded PDF and prompt to Gemini..."):
                    try:
                        client = get_llm_client(provider=provider_arg)
                        result = screen_resume_pdf(
                            pdf_file=uploaded_file,
                            prompt_text=st.session_state.gemini_prompt_template,
                            resume_text=st.session_state.extracted_resume_text,
                            job_description=job_desc,
                            client=client,
                        )
                        st.markdown("---")
                        display_result_card(result, "Gemini Resume Screening Result")
                    except Exception as e:
                        st.error(f"Application error: {e}")

    elif mode == "Side-by-Side Audit (v1 vs v2)":
        st.markdown("---")
        st.subheader("3. Comparative Bias Audit (v1 Baseline vs. v2 Debiased)")
        if st.button("Run Side-by-Side Audit", type="primary", use_container_width=True):
            if not st.session_state.resume_ready or not st.session_state.extracted_resume_text.strip():
                if st.session_state.pdf_error_message:
                    st.error(st.session_state.pdf_error_message)
                else:
                    st.warning("Please upload a candidate resume PDF.")
            else:
                with st.spinner("Screening candidate through both v1 (Baseline) and v2 (Debiased)..."):
                    try:
                        client = get_llm_client(provider=provider_arg)
                        res_v1 = screen(
                            resume=st.session_state.extracted_resume_text,
                            prompt_version="v1",
                            job_description=job_desc,
                            client=client,
                        )
                        res_v2 = screen(
                            resume=st.session_state.extracted_resume_text,
                            prompt_version="v2",
                            job_description=job_desc,
                            client=client,
                        )

                        st.markdown("---")
                        res_col1, res_col2 = st.columns(2)

                        with res_col1:
                            display_result_card(res_v1, "V1 Baseline Evaluation")

                        with res_col2:
                            display_result_card(res_v2, "V2 Debiased Evaluation")

                        # Comparative insights
                        if res_v1.valid and res_v2.valid:
                            st.markdown("---")
                            st.subheader("Audit Insights")
                            score_diff = (res_v2.score or 0) - (res_v1.score or 0)
                            decision_flip = res_v1.decision != res_v2.decision

                            insight_col1, insight_col2 = st.columns(2)
                            with insight_col1:
                                st.metric(
                                    label="Score Shift (v2 - v1)",
                                    value=f"{score_diff:+d} pts",
                                )
                            with insight_col2:
                                st.metric(
                                    label="Decision Changed?",
                                    value="YES (Flipped)" if decision_flip else "NO (Consistent)",
                                    delta="Mitigated" if decision_flip else "Unchanged",
                                )
                    except Exception as e:
                        st.error(f"Application error during comparative audit: {e}")

    elif mode == "Batch Counterfactual Audit (30+ Pairs)":
        from app.batch import (
            batch_results_to_dataframe,
            flatten_pairs_to_candidates,
            load_counterfactual_pairs,
            run_batch,
        )

        st.markdown("---")
        st.subheader("Batch Counterfactual Bias Audit")
        st.markdown(
            "Run evaluation across 30+ counterfactual pairs where candidate qualifications are identical "
            "except for demographic, institutional, or proxy attributes."
        )

        all_pairs = load_counterfactual_pairs()
        st.info(f"Loaded **{len(all_pairs)} counterfactual pairs** ({len(all_pairs) * 2} candidate resumes) from dataset.")

        b_col1, b_col2 = st.columns(2)
        with b_col1:
            batch_choice = st.selectbox(
                "Prompt Version for Batch",
                ["Both (v1 and v2)", "v1 (Baseline only)", "v2 (Debiased only)"],
                index=0,
            )
            if "Both" in batch_choice:
                selected_versions = ("v1", "v2")
                version_tag = "both_v1_v2"
            elif "v1" in batch_choice:
                selected_versions = ("v1",)
                version_tag = "v1"
            else:
                selected_versions = ("v2",)
                version_tag = "v2"

        with b_col2:
            max_pairs = st.slider("Number of pairs to evaluate", min_value=1, max_value=len(all_pairs), value=min(10, len(all_pairs)))

        if st.button("Run Batch Evaluation", type="primary"):
            selected_pairs = all_pairs[:max_pairs]
            progress_bar = st.progress(0)
            status_text = st.empty()

            client = get_llm_client(provider=provider_arg)
            batch_results = []
            total_evaluations = len(selected_pairs) * 2 * len(selected_versions)
            eval_counter = 0

            for pair_idx, pair in enumerate(selected_pairs, start=1):
                pair_cands = flatten_pairs_to_candidates([pair])
                for ver in selected_versions:
                    for cand in pair_cands:
                        eval_counter += 1
                        status_text.text(f"Evaluating candidate {cand.candidate_id} with {ver.upper()} ({eval_counter}/{total_evaluations})...")
                        res_item = run_batch([cand], prompt_version=ver, job_description=job_desc, client=client)
                        batch_results.extend(res_item)
                        progress_bar.progress(eval_counter / total_evaluations)

            status_text.text("Batch evaluation complete!")
            df = batch_results_to_dataframe(batch_results)
            st.dataframe(df, use_container_width=True)

            # Download buttons
            dl_col1, dl_col2 = st.columns(2)
            with dl_col1:
                csv_data = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Download Results as CSV",
                    data=csv_data,
                    file_name=f"counterfactual_batch_{version_tag}.csv",
                    mime="text/csv",
                )
            with dl_col2:
                json_data = json.dumps(batch_results, indent=2).encode("utf-8")
                st.download_button(
                    label="Download Results as JSON",
                    data=json_data,
                    file_name=f"counterfactual_batch_{version_tag}.json",
                    mime="application/json",
                )


if __name__ == "__main__":
    main()
