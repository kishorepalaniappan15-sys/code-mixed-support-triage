"""Gradio inbox view for the code-mixed support triage assistant."""
import gradio as gr
import pandas as pd

from triage.predict import triage, DEFAULT_THRESHOLD
from eval.evaluate import run_eval

EXAMPLES = [
    "Enna order innum varala, refund venum",
    "Mera refund abhi tak nahi aaya, kab milega?",
    "Payment cut aagiduchu but order confirm aagala udane sollunga",
    "Ye fraud hai, paise wapas do warna police mein jaunga",
    "Mujhe agent se baat karni hai",
    "ok",
]


def classify(message, threshold):
    if not message or not message.strip():
        return "Type a customer message and press **Triage**.", "", ""
    r = triage(message, threshold)
    intent = r["intent"] or "UNCLEAR"
    summary = (
        f"### Intent: `{intent}`  (confidence {r['intent_confidence']:.0%})\n"
        f"### Urgency: **{r['urgency']}**  (confidence {r['urgency_confidence']:.0%})\n"
        f"Detected language: {r['language']}"
    )
    if r["escalate"]:
        flag = f"## 🚨 ESCALATE TO HUMAN\n{r['reason']}"
    else:
        flag = f"## ✅ Auto-handled\n{r['reason']}"
    return summary, r["draft_reply"], flag


def show_eval(threshold):
    res, m = run_eval(threshold)
    md = (
        f"**{m['n']} team-written Tanglish/Hinglish messages**  \n"
        f"Intent accuracy: **{m['intent_accuracy']:.1%}** · "
        f"Urgency accuracy: **{m['urgency_accuracy']:.1%}** · "
        f"Escalation rate: **{m['escalation_rate']:.1%}**  \n"
        f"Accuracy on auto-handled messages: **{m['intent_accuracy_auto_handled']:.1%}** · "
        f"Wrong intents caught by escalation: **{m['wrong_intents_caught_by_escalation']}/{m['wrong_intents_total']}**"
    )
    table = res.copy()
    table["intent_ok"] = table["intent_ok"].map({True: "✅", False: "❌"})
    table["urgency_ok"] = table["urgency_ok"].map({True: "✅", False: "❌"})
    table["escalated"] = table["escalated"].map({True: "🚨", False: ""})
    return md, table


with gr.Blocks(title="Code-Mixed Support Triage") as demo:
    gr.Markdown("# 📥 Code-Mixed Support Triage\n"
                "Tanglish / Hinglish inbox: intent, urgency, draft reply, and "
                "human escalation when the model is unsure.")
    threshold = gr.Slider(0.1, 0.9, value=DEFAULT_THRESHOLD, step=0.05,
                          label="Escalation threshold (escalate when intent confidence is below this)")

    with gr.Tab("Inbox"):
        msg = gr.Textbox(lines=3, label="Customer message",
                         placeholder="e.g. Enna order innum varala, refund venum")
        btn = gr.Button("Triage", variant="primary")
        summary = gr.Markdown()
        reply = gr.Textbox(label="Draft reply", lines=4)
        flag = gr.Markdown()
        gr.Examples(EXAMPLES, inputs=msg)
        btn.click(classify, [msg, threshold], [summary, reply, flag])
        msg.submit(classify, [msg, threshold], [summary, reply, flag])

    with gr.Tab("Test set results"):
        ev_btn = gr.Button("Re-run evaluation")
        ev_md = gr.Markdown()
        ev_table = gr.Dataframe(interactive=False, wrap=True)
        ev_btn.click(show_eval, [threshold], [ev_md, ev_table])
        demo.load(show_eval, [threshold], [ev_md, ev_table])

if __name__ == "__main__":
    demo.launch()
