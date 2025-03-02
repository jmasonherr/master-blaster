import string
from modifications import mangle_sentence  # Replace your_module


def test_basic_mangling():
    sentence = "Grocery shopping at the local market"
    mangled = mangle_sentence(sentence)
    assert sentence != mangled


def test_mangling_level_increase():
    sentence = "Simple test sentence"
    mangled_level1 = mangle_sentence(sentence, mangling_level=1)
    mangled_level3 = mangle_sentence(sentence, mangling_level=3)
    assert mangled_level1 != mangled_level3
    assert abs(len(mangled_level1) - len(mangled_level3)) <= len(sentence) * 2


def test_empty_sentence():
    assert mangle_sentence("") == ""


def test_single_word_sentence():
    sentence = "Shopping"
    mangled = mangle_sentence(sentence)
    assert sentence != mangled


def test_mangling_consistency():
    sentence = "Another test phrase"
    for _ in range(10):
        mangle_sentence(sentence)


def test_randomness_within_bounds():
    sentence = "A very long test sentence to see if strange characters appear."
    mangled = mangle_sentence(sentence, mangling_level=3)
    for char in mangled:
        assert (
            char.lower() in string.ascii_lowercase
            or char == " "
            or char.upper() in string.ascii_uppercase
        )


def test_mangling_length():
    sentence = "This is a sentence."
    mangled = mangle_sentence(sentence)
    assert abs(len(sentence) - len(mangled)) < len(sentence)
