"""Streamlit inbox view for the code-mixed support triage assistant.

Same model + logic as app.py (Gradio); this file exists so the project can be
deployed for free on Streamlit Community Cloud.
"""
import streamlit as st

from triage.predict import triage, DEFAULT_THRESHOLD, _get_model
from eval.evaluate import run_eval

st.set_page_config(page_title="Code-Mixed Support Triage", page_icon="📥", layout="wide")

EXAMPLES = [
    "Enna order innum varala, refund venum",
    "Mera refund abhi tak nahi aaya, kab milega?",
    "Payment cut aagiduchu but order confirm aagala udane sollunga",
    "Ye fraud hai, paise wapas do warna police mein jaunga",
    "Mujhe agent se baat karni hai",
    "ok",
]


@st.cache_resource(show_spinner="Loading model (the first start can take a minute)...")
def load_model():
    _get_model()
    return True


@st.cache_data(show_spinner=False)
def cached_eval(threshold):
    return run_eval(threshold)


def set_example(text):
    st.session_state["msg"] = text
    st.session_state["run"] = True


def start_run():
    st.session_state["run"] = True


load_model()

st.title("📥 Code-Mixed Support Triage")
st.write("Tanglish / Hinglish inbox: intent, urgency, draft reply, and human "
         "escalation when the model is unsure.")

threshold = st.slider("Escalation threshold (escalate when intent confidence is below this)",
                      0.10, 0.90, float(DEFAULT_THRESHOLD), 0.05)

tab_inbox, tab_eval = st.tabs(["Inbox", "Test set results"])

with tab_inbox:
    st.text_area("Customer message", key="msg", height=100,
                 placeholder="e.g. Enna order innum varala, refund venum")
    st.button("Triage", type="primary", on_click=start_run)

    st.caption("Examples")
    cols = st.columns(3)
    for i, ex in enumerate(EXAMPLES):
        cols[i % 3].button(ex, key=f"ex{i}", on_click=set_example, args=(ex,))

    message = st.session_state.get("msg", "")
    if st.session_state.get("run") and message.strip():
        r = triage(message, threshold)
        c1, c2, c3 = st.columns(3)
        c1.metric("Intent", r["intent"] or "UNCLEAR", f"{r['intent_confidence']:.0%} confidence",
                  delta_color="off")
        c2.metric("Urgency", r["urgency"], f"{r['urgency_confidence']:.0%} confidence",
                  delta_color="off")
        c3.metric("Language", r["language"])
        if r["escalate"]:
            st.error(f"🚨 ESCALATE TO HUMAN: {r['reason']}")
        else:
            st.success(f"✅ Auto-handled: {r['reason']}")
        st.subheader("Draft reply")
        st.info(r["draft_reply"])

with tab_eval:
    res, m = cached_eval(threshold)
    st.write(f"**{m['n']} team-written Tanglish/Hinglish messages**")
    a, b, c, d = st.columns(4)
    a.metric("Intent accuracy", f"{m['intent_accuracy']:.1%}")
    b.metric("Urgency accuracy", f"{m['urgency_accuracy']:.1%}")
    c.metric("Escalation rate", f"{m['escalation_rate']:.1%}")
    d.metric("Accuracy when auto-handled", f"{m['intent_accuracy_auto_handled']:.1%}")
    st.caption(f"Wrong intents caught by escalation: "
               f"{m['wrong_intents_caught_by_escalation']}/{m['wrong_intents_total']}")
    table = res.copy()
    table["intent_ok"] = table["intent_ok"].map({True: "✅", False: "❌"})
    table["urgency_ok"] = table["urgency_ok"].map({True: "✅", False: "❌"})
    table["escalated"] = table["escalated"].map({True: "🚨", False: ""})
    st.dataframe(table)
