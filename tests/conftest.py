"""Shared test helpers. All data here is SYNTHETIC and used only to test code paths;
it is never used for reported evaluation results."""
import random

import pandas as pd
import pytest

_VOCAB = {
    "sports": ["खेल", "गोल", "टोली", "खेलाडी", "जित", "प्रतियोगिता", "मैदान"],
    "economy": ["बैंक", "ब्याजदर", "बजार", "शेयर", "मूल्य", "अर्थ", "लगानी"],
    "health": ["अस्पताल", "रोग", "उपचार", "स्वास्थ्य", "औषधि", "चिकित्सक", "खोप"],
}
_FILLER = ["आज", "भने", "गरेको", "छ", "को", "मा", "सम्बन्धी"]


def synthetic_frame(per_class: int = 40, seed: int = 0) -> pd.DataFrame:
    """Small labelled dataset where each class has its own vocabulary (synthetic, for tests only)."""
    rng = random.Random(seed)
    rows = []
    for label, words in _VOCAB.items():
        for i in range(per_class):
            body = rng.sample(words, 4) + rng.sample(_FILLER, 3)
            rng.shuffle(body)
            rows.append({"text": " ".join(body) + f" समाचार नम्बर {label}{i}", "label": label})
    return pd.DataFrame(rows)


@pytest.fixture
def frame() -> pd.DataFrame:
    return synthetic_frame()
