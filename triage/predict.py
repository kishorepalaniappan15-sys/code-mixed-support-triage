"""Intent + urgency classification, draft reply, and human-escalation logic."""
import os
import joblib
import numpy as np

from triage.normalize import normalize, detect_language

MODEL_PATH = "models/triage.joblib"
DEFAULT_THRESHOLD = 0.45  # intent confidence below this -> escalate
MARGIN_THRESHOLD = 0.08   # top1 - top2 below this -> ambiguous -> escalate

# ---- urgency (rule-based scorer over normalised text) -----------------------
HIGH_WORDS = {"fraud", "scam", "legal", "lawyer", "court", "police", "worst",
              "cheat", "cheated", "harassment", "urgent", "immediately",
              "chargeback", "sue", "emergency", "hack", "hacked"}
MEDIUM_WORDS = {"quickly", "still", "again", "late", "delayed", "damaged",
                "wrong", "stuck", "deducted", "failed", "days", "weeks",
                "times", "complaint"}
MEDIUM_PHRASES = ["not arrived", "not received", "not picking up",
                  "not possible", "not working", "not done"]

REPLIES = {
    "DELIVERY": "I'm sorry your order is delayed. I'm checking the live tracking for your parcel and will share the exact status and a delivery date shortly.",
    "REFUND": "I understand you're waiting on a refund. I've noted your request and will confirm the refund status and the expected credit date.",
    "CANCEL": "I can help cancel your order. Please confirm the order number and I'll process the cancellation if it hasn't shipped yet.",
    "ACCOUNT": "I can help with your account. For security, please verify your registered email or phone number and I'll guide you through the fix.",
    "PAYMENT": "Sorry about the payment trouble. Please share the transaction ID and I'll check whether the amount was captured and reverse any duplicate charge.",
    "INVOICE": "I'll send your invoice to your registered email. Please share the order number and any GST details you need on it.",
    "SHIPPING": "Happy to help with shipping. Please share your order number and pincode and I'll confirm the charges or update the address if it hasn't shipped.",
    "ORDER": "Thanks for letting us know about your order. Please share the order number and I'll check it and sort out the right fix (status, replacement or return).",
    "CONTACT": "I'm sorry you couldn't reach us. I've noted your request and a support agent will contact you on your registered number.",
    "SUBSCRIPTION": "I can help with your subscription. Please confirm the registered email and I'll make the change before the next renewal.",
    "FEEDBACK": "Thank you for your feedback. I've shared it with our team and we appreciate you taking the time.",
}
GREETING = {"Tanglish": "Vanakkam!", "Hinglish": "Namaste!", "English": "Hello!"}

_model = None


def _get_model():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            from triage.train import train_and_save
            try:
                train_and_save(offline=False)
            except Exception as exc:  # no internet -> seed data only
                print("Bitext unavailable, training on seed data only:", exc)
                train_and_save(offline=True)
        _model = joblib.load(MODEL_PATH)
    return _model


def score_urgency(clean: str):
    tokens = set(clean.split())
    high = sorted(tokens & HIGH_WORDS)
    medium = sorted(tokens & MEDIUM_WORDS) + [p for p in MEDIUM_PHRASES if p in clean]
    if high:
        return "High", min(0.95, 0.80 + 0.05 * len(high)), high
    if medium:
        return "Medium", min(0.90, 0.65 + 0.07 * len(medium)), medium
    return "Low", 0.70, []


def triage(message: str, threshold: float = DEFAULT_THRESHOLD) -> dict:
    clean = normalize(message)
    lang = detect_language(message)
    urgency, urg_conf, urg_hits = score_urgency(clean)

    if len(clean.split()) < 2:
        return {"intent": None, "intent_confidence": 0.0, "urgency": urgency,
                "urgency_confidence": round(urg_conf, 2), "language": lang,
                "escalate": True,
                "reason": "Message too short or unclear to classify.",
                "draft_reply": f"{GREETING[lang]} Could you share a few more details so we can help you?"}

    model = _get_model()
    probs = model.predict_proba([clean])[0]
    classes = model.classes_
    order = np.argsort(probs)[::-1]
    top, second = order[0], order[1]
    conf = float(probs[top])
    margin = float(probs[top] - probs[second])

    reasons = []
    if conf < threshold:
        reasons.append(f"Intent confidence {conf:.2f} is below the {threshold:.2f} threshold")
    elif margin < MARGIN_THRESHOLD:
        reasons.append(f"Ambiguous between {classes[top]} and {classes[second]}")
    if urgency == "High":
        reasons.append(f"high-urgency wording ({', '.join(urg_hits)})")

    intent = str(classes[top])
    escalate = bool(reasons)
    reply = f"{GREETING[lang]} {REPLIES[intent]}"
    if escalate:
        reply += " I'm also passing this to a human agent who will follow up personally."

    return {"intent": intent, "intent_confidence": round(conf, 3),
            "urgency": urgency, "urgency_confidence": round(urg_conf, 2),
            "language": lang, "escalate": escalate,
            "reason": (("; ".join(reasons))[0].upper() + ("; ".join(reasons))[1:] + ".") if reasons else "Confident, handled automatically.",
            "draft_reply": reply}


if __name__ == "__main__":
    import sys
    print(triage(" ".join(sys.argv[1:]) or "Enna order innum varala"))
