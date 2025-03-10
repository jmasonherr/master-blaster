import random
import string


def drop_word_text_augmentation(text: str) -> str:
    """
    Simple text augmentation for consistency regularization.
    This randomly drops words to create a slightly modified version of the input text.

    Args:
        text (str): Original text to augment

    Returns:
        str: Augmented text with some words randomly dropped
    """
    # Split text into words
    words = text.split()

    # For very short texts, don't perform augmentation
    if len(words) <= 3:
        return text

    # Randomly drop words with 10% probability
    drop_prob = 0.1
    augmented_words = [w for w in words if random.random() > drop_prob]

    # Safety check: Make sure we don't drop too many words
    # If we've dropped more than 20% of words, randomly select 80% to keep
    if len(augmented_words) < len(words) * 0.8:
        num_to_keep = int(len(words) * 0.8)
        indices_to_keep = random.sample(range(len(words)), num_to_keep)
        augmented_words = [words[i] for i in sorted(indices_to_keep)]

    return " ".join(augmented_words)


def mangle_sentence(sentence: str, mangling_level=1):
    """
    Mangles a sentence to resemble bank transaction descriptions.

    Args:
        sentence: The input sentence (string).
        mangling_level: An integer controlling the degree of mangling. Higher values result in more aggressive mangling.

    Returns:
        A mangled string.
    """

    words = sentence.split()
    mangled_words = []

    for word in words:
        mangled_word = word.lower()

        # Random abbreviation or removal of letters
        if random.random() < 0.3 * mangling_level:
            if len(mangled_word) > 4:
                start = random.randint(0, len(mangled_word) - 3)
                end = random.randint(start + 1, len(mangled_word))
                mangled_word = mangled_word[start:end]

        # Random letter removal
        if random.random() < 0.2 * mangling_level and len(mangled_word) > 2:
            remove_index = random.randint(0, len(mangled_word) - 1)
            mangled_word = (
                mangled_word[:remove_index] + mangled_word[remove_index + 1 :]
            )

        # Random letter replacement
        if random.random() < 0.15 * mangling_level and len(mangled_word) > 1:
            replace_index = random.randint(0, len(mangled_word) - 1)
            new_char = random.choice(string.ascii_lowercase)
            mangled_word = (
                mangled_word[:replace_index]
                + new_char
                + mangled_word[replace_index + 1 :]
            )

        # Random word concatenation
        if random.random() < 0.1 * mangling_level and len(mangled_words) > 0:
            previous_word = mangled_words.pop()
            mangled_word = previous_word + mangled_word

        mangled_words.append(mangled_word)

    # Random capitalization (like acronyms)
    if random.random() < 0.2 * mangling_level:
        random_word_index = random.randint(0, len(mangled_words) - 1)
        mangled_words[random_word_index] = mangled_words[random_word_index].upper()

    return " ".join(mangled_words)


if __name__ == "__main__":
    # Example usage
    sentences = [
        "Grocery shopping at the local market.",
        "Online payment for a subscription service.",
        "Transfer to savings account.",
        "Purchase of new electronic device.",
        "Restaurant dinner with friends.",
    ]

    for s in sentences:
        print(f"Original: {s}")
        print(f"Mangled (level 1): {mangle_sentence(s)}")
        print(f"Mangled (level 2): {mangle_sentence(s, mangling_level=2)}")
        print(f"Mangled (level 3): {mangle_sentence(s, mangling_level=3)}")
        print("-" * 20)
