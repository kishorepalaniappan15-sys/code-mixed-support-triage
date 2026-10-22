"""Run the classifier on the team-written test set and report metrics."""
import pandas as pd
from triage.predict import triage, DEFAULT_THRESHOLD

TEST_PATH = "data/test_set.csv"


def run_eval(threshold: float = DEFAULT_THRESHOLD):
    df = pd.read_csv(TEST_PATH)
    rows = []
    for _, r in df.iterrows():
        out = triage(r["text"], threshold)
        rows.append({
            "message": r["text"], "language": r["language"],
            "expected_intent": r["expected_intent"], "predicted_intent": out["intent"],
            "intent_conf": out["intent_confidence"],
            "expected_urgency": r["expected_urgency"], "predicted_urgency": out["urgency"],
            "escalated": out["escalate"],
            "intent_ok": out["intent"] == r["expected_intent"],
            "urgency_ok": out["urgency"] == r["expected_urgency"],
        })
    res = pd.DataFrame(rows)
    auto = res[~res["escalated"]]
    metrics = {
        "n": len(res),
        "intent_accuracy": res["intent_ok"].mean(),
        "urgency_accuracy": res["urgency_ok"].mean(),
        "escalation_rate": res["escalated"].mean(),
        "intent_accuracy_auto_handled": auto["intent_ok"].mean() if len(auto) else float("nan"),
        "wrong_intents_caught_by_escalation": int(((~res["intent_ok"]) & res["escalated"]).sum()),
        "wrong_intents_total": int((~res["intent_ok"]).sum()),
    }
    return res, metrics


if __name__ == "__main__":
    res, m = run_eval()
    res.to_csv("eval/results.csv", index=False)
    print(f"Test messages:                {m['n']}")
    print(f"Intent accuracy:              {m['intent_accuracy']:.1%}")
    print(f"Urgency accuracy:             {m['urgency_accuracy']:.1%}")
    print(f"Escalation rate:              {m['escalation_rate']:.1%}")
    print(f"Accuracy on auto-handled:     {m['intent_accuracy_auto_handled']:.1%}")
    print(f"Wrong intents escalated:      {m['wrong_intents_caught_by_escalation']}/{m['wrong_intents_total']}")
    print("Saved eval/results.csv")
