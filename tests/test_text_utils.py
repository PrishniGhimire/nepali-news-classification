import re

from src.text_utils import clean_text, tokenize


def test_tokenize_keeps_devanagari_words_intact():
    assert tokenize("नेपाली क्रिकेट टोलीले जित्यो।") == ["नेपाली", "क्रिकेट", "टोलीले", "जित्यो"]


def test_default_regex_word_pattern_would_split_words():
    # Why we use a custom tokenizer: \w skips vowel signs such as "ा" and "ी".
    assert len(re.findall(r"\w+", "नेपाली")) > 1


def test_clean_text_strips_urls_html_zero_width_and_digits():
    raw = "देखें https://example.com <b>खबर</b>​   १२३"
    assert clean_text(raw) == "देखें खबर 123"


def test_clean_text_merges_zwj_spelling_variants():
    assert clean_text("गर्‍यो") == clean_text("गर्यो")
