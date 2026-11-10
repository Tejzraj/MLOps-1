"""Run from the repository root with python -m streamlit run frontend/app.py."""

import logging

import streamlit as st

from frontend.api_client import PredictionError, mock_prediction, predict
from frontend.config import Settings, load_settings
from frontend.question_bank import Question, load_questions


def render_home(settings: Settings) -> None:
    """Introduce the practice workflow and the MVP's evaluation boundary."""
    st.title(settings.title)
    st.caption("INTERVIEW PREPARATION · DATA STRUCTURES")
    st.header("Build confidence, one answer at a time.")
    st.write(
        "Practice explaining core concepts clearly. Choose a question, write your answer, "
        "then compare your explanation with a curated reference."
    )
    for column, title, description in zip(
        st.columns(3),
        ("01 · Choose", "02 · Explain", "03 · Reflect"),
        (
            "Explore essential data structures.",
            "Answer in your own words.",
            "Review the category and reference.",
        ),
        strict=True,
    ):
        with column, st.container(border=True):
            st.subheader(title)
            st.write(description)
    st.info("Open Practice in the sidebar to begin. No sign-in required.")
    st.write(
        "Practice questions are curated educational content, separate from the Mohler dataset. "
        "Mock mode demonstrates the interface only. API mode displays the category returned by FastAPI."
    )


def render_practice(settings: Settings, questions: list[Question], mode: str) -> None:
    """Collect an answer and display a result only for its matching input snapshot."""
    st.title("Practice room")
    st.write("Focus on the concept, explain the tradeoffs, and include an example when useful.")
    by_id = {question.question_id: question for question in questions}
    question_id = st.selectbox(
        "Choose a question",
        options=list(by_id),
        format_func=lambda key: by_id[key].topic,
        key="question_id",
    )
    question = by_id[question_id]
    with st.container(border=True):
        st.caption(question.topic.upper())
        st.subheader(question.question)
    candidate_answer = st.text_area(
        "Your answer",
        height=220,
        max_chars=settings.max_answer_chars,
        placeholder="Explain the idea in your own words…",
        key=f"answer_{question_id}",
    )
    demo_label = settings.mock_label
    if mode == "mock":
        demo_label = st.selectbox(
            "Simulated result category",
            settings.quality_labels,
            index=settings.quality_labels.index(settings.mock_label),
            key="demo_label",
            help="Chosen by you for the demo. Your answer is not scored.",
        )
    snapshot = (
        question_id,
        candidate_answer,
        mode,
        demo_label,
        settings.backend_url,
        settings.predict_path,
        settings.timeout_seconds,
    )
    if st.session_state.get("result_snapshot") != snapshot:
        st.session_state.pop("result", None)
    if st.button("Evaluate Answer", type="primary", key="evaluate"):
        st.session_state.pop("result", None)
        if not candidate_answer.strip():
            st.warning("Please write an answer before evaluating.")
        else:
            try:
                with st.spinner(
                    "Preparing simulated result…" if mode == "mock" else "Waiting for FastAPI…"
                ):
                    result = (
                        mock_prediction(demo_label, settings)
                        if mode == "mock"
                        else predict(question, candidate_answer, settings)
                    )
                st.session_state["result"] = result
                st.session_state["result_snapshot"] = snapshot
            except PredictionError as exc:
                st.error(str(exc))
                st.info(
                    "No result was generated. You can retry, or explicitly select Mock mode in the sidebar."
                )
    result = st.session_state.get("result")
    if result is not None:
        with st.container(border=True):
            st.subheader("Evaluation result")
            if result.simulated:
                st.warning(
                    "SIMULATED RESPONSE · Demo only. This category was selected by you; your answer was not scored."
                )
            else:
                st.caption(f"Source: FastAPI backend · Model version: {result.model_version}")
                st.warning(
                    "Experimental quality prediction, not a factual-correctness check. "
                    "Results may be incorrect and require human review."
                )
            st.subheader(result.quality_label)
        st.subheader("Compare your explanation")
        candidate_column, reference_column = st.columns(2)
        with candidate_column:
            st.markdown("**Your answer**")
            st.text(candidate_answer)
        with reference_column:
            st.markdown("**Reference answer**")
            st.write(question.reference_answer)
        st.caption(
            "Review the definition, operations, complexity and tradeoffs. No similarity score is computed here."
        )
    with st.expander("Study the reference answer"):
        st.write(question.reference_answer)


def main() -> None:
    """Initialize configuration, navigation and explicit backend mode controls."""
    st.set_page_config(page_title="Interview Studio", page_icon="💬", layout="wide")
    try:
        settings = load_settings()
        questions = load_questions(settings.question_bank_path)
    except ValueError as exc:
        logging.getLogger(__name__).error("Frontend initialization failed: %s", exc)
        st.error(str(exc))
        st.info("Check frontend/params.yaml or FRONTEND_PARAMS, then restart the app.")
        st.stop()
    logging.basicConfig(level=settings.log_level)
    with st.sidebar:
        st.title(settings.title)
        st.caption("Practice. Explain. Improve.")
        page = st.radio("Navigation", ["Home", "Practice", "How it works"], key="navigation")
        st.divider()
        mode = st.radio(
            "Evaluation mode",
            ["mock", "api"],
            index=["mock", "api"].index(settings.mode),
            format_func=lambda value: "Mock · simulated" if value == "mock" else "API · FastAPI",
            key="mode",
        )
        if mode == "api":
            st.caption(f"Backend: {settings.backend_url}")
            st.caption(f"Timeout: {settings.timeout_seconds:g} seconds")
        st.caption("Phase 2 · Interview preparation MVP")
    if mode == "mock":
        st.warning(
            "MOCK MODE — All evaluation responses are simulated. No model is connected and no answer is scored."
        )
    else:
        st.info("API MODE — Answers are sent to FastAPI. Connection failures are shown as errors.")
    if page == "Home":
        render_home(settings)
    elif page == "Practice":
        render_practice(settings, questions, mode)
    else:
        st.title("How it works")
        st.write("Choose Practice, select a topic, write an answer and press Evaluate Answer.")
        st.write("Results use four categories: " + ", ".join(settings.quality_labels) + ".")
        st.write(
            "In Mock mode, you choose a simulated category to preview the result display. No grading takes place."
        )
        st.write(
            "In API mode, only the backend evaluates answers. The frontend performs no ML inference or scoring."
        )
        st.write(
            "Answers stay in the current Streamlit session and are sent to the backend only when you evaluate in API mode."
        )
        st.write(
            "Set BACKEND_URL, BACKEND_TIMEOUT_SECONDS and FRONTEND_MODE before starting Streamlit. See frontend/README.md for commands and the verified API contract."
        )


if __name__ == "__main__":
    main()
