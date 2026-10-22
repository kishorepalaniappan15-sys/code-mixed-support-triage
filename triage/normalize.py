"""Light cleanup of code-mixed (Tanglish / Hinglish) text.

Romanised Tamil/Hindi words are mapped to English equivalents so the model,
which is trained mostly on English Bitext data, can still pick up the intent.
"""
import re

HINGLISH = {
    "mera": "my", "meri": "my", "mere": "my", "mujhe": "me", "nahi": "not",
    "nahin": "not", "nai": "not", "abhi": "now", "tak": "until", "kab": "when",
    "milega": "get", "mila": "received", "aaya": "arrived", "aayi": "arrived",
    "aayega": "arrive", "paisa": "money", "paise": "money", "wapas": "back",
    "chahiye": "want", "bhej": "send", "bhejo": "send", "galat": "wrong",
    "kharab": "damaged", "toota": "damaged", "kyun": "why", "kyu": "why",
    "kya": "what", "kaise": "how", "band": "stop", "badalna": "change",
    "badalni": "change", "dikha": "show", "bhugtan": "payment",
    "shikayat": "complaint", "jaldi": "quickly", "turant": "immediately",
    "dhokha": "fraud", "din": "days", "hafte": "weeks", "kat": "deducted",
    "baar": "times", "dobara": "again", "batao": "tell", "karna": "do",
    "karni": "do", "kar": "do", "baat": "talk", "nikla": "turned out",
    "ship": "ship", "mein": "in", "ke": "", "liye": "", "wala": "", "hai": "",
    "hain": "", "ka": "", "ki": "", "ko": "", "se": "", "ho": "", "hua": "",
    "raha": "", "rahi": "", "rahe": "", "gaya": "", "gayi": "", "gaye": "",
    "warna": "otherwise", "bahut": "very", "achhi": "good", "accha": "good",
}

TANGLISH = {
    "enna": "what", "ennoda": "my", "enoda": "my", "en": "my", "innum": "still",
    "varala": "not arrived", "vara": "arrive", "varum": "arrive",
    "varudhu": "coming", "venum": "want", "venam": "do not want",
    "panna": "do", "pannanum": "do", "pannunga": "please do", "panni": "do",
    "pannidunga": "please do", "epo": "when", "eppo": "when",
    "evlo": "how much", "kidaikala": "not received", "kidaikum": "will get",
    "kidaikkum": "will get", "kaasu": "money", "panam": "money",
    "thirumba": "back", "kudunga": "give", "anuppunga": "send",
    "thappa": "wrong", "illa": "not", "ila": "not", "mosadi": "fraud",
    "mosam": "fraud", "maathunga": "change", "maathanum": "change",
    "nikkuthu": "stuck", "aagala": "not done", "sollunga": "tell",
    "enakku": "me", "irukku": "is", "iruku": "is", "pochu": "gone",
    "vandhuchu": "arrived", "vanthuchu": "arrived", "udane": "immediately",
    "seekiram": "quickly", "naal": "days", "vaaram": "weeks",
    "cut": "deducted", "romba": "very", "niruthunga": "stop",
    "mudiyala": "not possible", "edukala": "not picking up",
    "utha": "pick", "pesanum": "talk", "yaarum": "nobody",
    "edukuven": "will take", "eduthutinga": "took",
}

LEXICON = {**HINGLISH, **TANGLISH}

_PUNCT = re.compile(r"[^\w\s']", re.UNICODE)
_REPEAT = re.compile(r"(.)\1{2,}")


def _tokens(text: str):
    text = str(text).lower().strip()
    text = _REPEAT.sub(r"\1\1", text)
    return _PUNCT.sub(" ", text).split()


def normalize(text: str) -> str:
    tokens = [LEXICON.get(tok, tok) for tok in _tokens(text)]
    return re.sub(r"\s+", " ", " ".join(tokens)).strip()


def detect_language(text: str) -> str:
    toks = _tokens(text)
    ta = sum(t in TANGLISH for t in toks)
    hi = sum(t in HINGLISH for t in toks)
    if ta == 0 and hi == 0:
        return "English"
    return "Tanglish" if ta >= hi else "Hinglish"


if __name__ == "__main__":
    for s in ["Enna order innum varala, refund venum!!",
              "Mera refund abhi tak nahi aaya, kab milega?"]:
        print(s, "->", normalize(s), "|", detect_language(s))
