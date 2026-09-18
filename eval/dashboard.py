"""
Evaluation dashboard -- visualizes results produced by eval/evaluate.py.

Run:
    streamlit run eval/dashboard.py
"""
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config

st.set_page_config(page_title="RAG Evaluation Dashboard", page_icon="📊", layout="wide")
st.title("📊 RAG Evaluation Dashboard")
st.caption("Retrieval relevance, answer quality, confidence calibration, and latency for the Enterprise Knowledge Assistant.")

results_path = Path(config.EVAL_RESULTS_PATH)
if not results_path.exists():
    st.warning(
        "No evaluation results found yet. Run the evaluation harness first:\n\n"
        "`python -m eval.evaluate`"
    )
    st.stop()

with open(results_path) as f:
    data = json.load(f)

summary = data["summary"]
results = data["results"]
df = pd.DataFrame(results)
# Flatten nested score dicts into top-level columns for easy charting/display.
if "rouge" in df.columns:
    df["rouge_l"] = df["rouge"].apply(lambda v: v["rougeL"] if isinstance(v, dict) else None)
if "bertscore" in df.columns:
    df["bertscore_f1"] = df["bertscore"].apply(lambda v: v["f1"] if isinstance(v, dict) else None)

# ---------------- Top-line metrics ----------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Questions evaluated", summary["num_questions"])
c2.metric("Retrieval hit rate", f"{summary['retrieval_hit_rate']*100:.0f}%")
c3.metric("Mean Reciprocal Rank", summary["mean_reciprocal_rank"])
coverage = summary.get("avg_answer_keyword_coverage")
c4.metric("Answer keyword coverage", f"{coverage*100:.0f}%" if coverage is not None else "N/A")

c5, c6, c7, c8 = st.columns(4)
rouge_l = summary.get("avg_rouge_l")
c5.metric("Avg ROUGE-L", f"{rouge_l:.2f}" if rouge_l is not None else "N/A")
bertscore_f1 = summary.get("avg_bertscore_f1")
c6.metric("Avg BERTScore F1", f"{bertscore_f1:.2f}" if bertscore_f1 is not None else "N/A")
judge_score = summary.get("avg_llm_judge_score")
c7.metric("Avg LLM Judge (1-5)", f"{judge_score:.2f}" if judge_score is not None else "N/A")
c8.metric("Avg latency", f"{summary['avg_latency_sec']}s")

st.divider()

col1, col2 = st.columns(2)
with col1:
    st.subheader("Confidence Distribution")
    conf_df = pd.DataFrame(
        list(summary["confidence_distribution"].items()), columns=["Confidence", "Count"]
    )
    st.bar_chart(conf_df.set_index("Confidence"))

with col2:
    st.subheader("Latency per Question")
    st.bar_chart(df.set_index("id")["latency_sec"])

st.divider()
st.subheader("Answer-Quality Metrics by Question")
st.caption(
    "ROUGE-L and BERTScore F1 are 0-1 lexical/semantic overlap with the reference answer. "
    "LLM Judge is a 1-5 correctness rating, shown here divided by 5 so all three sit on the same axis."
)
score_cols = [c for c in ["rouge_l", "bertscore_f1", "llm_judge_score"] if c in df.columns]
if score_cols:
    chart_df = df.set_index("id")[score_cols].copy()
    if "llm_judge_score" in chart_df.columns:
        chart_df["llm_judge_score"] = chart_df["llm_judge_score"] / 5
    chart_df = chart_df.rename(columns={
        "rouge_l": "ROUGE-L",
        "bertscore_f1": "BERTScore F1",
        "llm_judge_score": "LLM Judge (/5)",
    })
    st.bar_chart(chart_df)
else:
    st.info("No reference answers (`expected_answer`) found in the testset, so ROUGE/BERTScore/LLM-judge weren't computed.")

st.divider()
st.subheader("Per-Question Results")

def hit_icon(v):
    return "✅" if v else "❌"

display_df = df.copy()
display_df["retrieval_hit"] = display_df["retrieval_hit"].apply(hit_icon)
display_df["keyword_coverage"] = display_df["keyword_coverage"].apply(
    lambda v: f"{v*100:.0f}%" if v is not None else "N/A"
)
for col in ["rouge_l", "bertscore_f1"]:
    if col in display_df.columns:
        display_df[col] = display_df[col].apply(lambda v: f"{v:.2f}" if v is not None else "N/A")
if "llm_judge_score" in display_df.columns:
    display_df["llm_judge_score"] = display_df["llm_judge_score"].apply(
        lambda v: f"{v}/5" if v is not None else "N/A"
    )

table_cols = ["id", "question", "retrieval_hit", "keyword_coverage", "confidence", "latency_sec"]
table_cols += [c for c in ["rouge_l", "bertscore_f1", "llm_judge_score"] if c in display_df.columns]
st.dataframe(
    display_df[table_cols],
    use_container_width=True,
    hide_index=True,
)

st.divider()
st.subheader("Inspect a Question")
selected_id = st.selectbox("Select question", df["id"].tolist())
row = df[df["id"] == selected_id].iloc[0]

st.markdown(f"**Question:** {row['question']}")
st.markdown(f"**Answer:**\n\n{row['answer']}")
if row.get("reference_answer"):
    st.markdown(f"**Reference (gold) answer:**\n\n{row['reference_answer']}")
st.markdown(f"**Expected source:** `{row['expected_source']}`  |  **Retrieved sources:** `{row['retrieved_sources']}`")
st.markdown(f"**Confidence:** {row['confidence']} ({row['confidence_score']})")
st.markdown(f"**Matched keywords:** {row['matched_keywords']} / expected {row['expected_keywords']}")

if row.get("rouge") or row.get("bertscore") or row.get("llm_judge_score") is not None:
    st.markdown("**Answer-quality scores:**")
    sc1, sc2, sc3 = st.columns(3)
    rouge = row.get("rouge")
    sc1.metric("ROUGE-L", f"{rouge['rougeL']:.2f}" if rouge else "N/A")
    bertscore = row.get("bertscore")
    sc2.metric("BERTScore F1", f"{bertscore['f1']:.2f}" if bertscore else "N/A")
    sc3.metric("LLM Judge", f"{row['llm_judge_score']}/5" if row.get("llm_judge_score") is not None else "N/A")
    if row.get("llm_judge_reasoning"):
        st.markdown(f"*Judge reasoning: {row['llm_judge_reasoning']}*")
