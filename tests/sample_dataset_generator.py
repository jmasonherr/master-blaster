import pandas as pd
import numpy as np
from pathlib import Path


def generate_sample_sentiment_data():
    """
    Generate a sample sentiment dataset for testing purposes.
    This creates a synthetic dataset with positive and negative examples.
    """
    # Create directory if it doesn't exist
    data_dir = Path("./data")
    data_dir.mkdir(exist_ok=True)

    # Define sample texts for positive and negative sentiment
    positive_texts = [
        "This movie was absolutely amazing, I loved every minute of it.",
        "One of the best films I've seen in years, highly recommended.",
        "Brilliant performances by the entire cast, a true masterpiece.",
        "The story was captivating from start to finish.",
        "I couldn't stop smiling throughout the entire film.",
        "A perfect blend of comedy and drama, with excellent direction.",
        "The cinematography was stunning and the script was well-written.",
        "This film exceeded all my expectations, a must-see.",
        "The characters were well-developed and the plot was engaging.",
        "I was thoroughly entertained and would watch it again.",
        "A beautiful story that touched my heart.",
        "The director did an outstanding job with this film.",
        "Fantastic movie that kept me engaged the whole time.",
        "This deserves all the awards it has received.",
        "One of my favorite movies of all time now.",
        "The acting was superb and the story was compelling.",
        "I laughed and cried - a truly emotional experience.",
        "A cinematic masterpiece that will be remembered for years.",
        "Incredible soundtrack that perfectly complemented the story.",
        "I can't recommend this movie enough.",
        "A brilliant film that tackles important issues.",
        "The chemistry between the actors was amazing.",
        "The special effects were breathtaking.",
        "A powerful story told in a beautiful way.",
        "This film really moved me emotionally.",
    ]

    negative_texts = [
        "This movie was a complete waste of time.",
        "I couldn't wait for it to end, truly disappointing.",
        "The plot made no sense and the acting was terrible.",
        "One of the worst films I've seen in a long time.",
        "I regret spending money on this movie.",
        "The characters were flat and uninteresting.",
        "A boring story with predictable outcomes.",
        "The dialogue was cringe-worthy throughout.",
        "I nearly fell asleep multiple times watching this.",
        "The pacing was off and the story dragged on forever.",
        "This film was a major disappointment.",
        "Poor direction and sloppy editing ruined this movie.",
        "I wouldn't recommend this to anyone.",
        "The script was lazy and full of plot holes.",
        "I've seen better acting in student films.",
        "This movie was painful to sit through.",
        "Nothing about this film worked for me.",
        "A pretentious mess with no redeeming qualities.",
        "The special effects looked cheap and unconvincing.",
        "I want those two hours of my life back.",
        "This is the kind of movie that gives cinema a bad name.",
        "Confusing plot and annoying characters throughout.",
        "The ending was particularly disappointing.",
        "It tried too hard and failed miserably.",
        "Avoid this movie at all costs.",
    ]

    # Create a larger dataset by repeating with minor modifications
    expanded_positive = []
    expanded_negative = []

    for text in positive_texts:
        expanded_positive.append(text)
        expanded_positive.append(
            text.replace("movie", "film").replace("amazing", "fantastic")
        )
        expanded_positive.append(
            text.replace("loved", "enjoyed").replace("brilliant", "excellent")
        )

    for text in negative_texts:
        expanded_negative.append(text)
        expanded_negative.append(
            text.replace("movie", "film").replace("terrible", "awful")
        )
        expanded_negative.append(
            text.replace("disappointing", "frustrating").replace(
                "worst", "most terrible"
            )
        )

    # Create combined lists
    all_texts = expanded_positive + expanded_negative
    all_labels = ["positive"] * len(expanded_positive) + ["negative"] * len(
        expanded_negative
    )

    # Create synthetic loss values
    # Some clean examples (low loss) and some noisy examples (high loss)
    all_losses = []
    for i in range(len(all_texts)):
        # 80% clean, 20% noisy
        if np.random.random() < 0.8:
            all_losses.append(np.random.uniform(0.1, 0.3))  # Clean
        else:
            all_losses.append(np.random.uniform(0.7, 0.9))  # Noisy

    # Create DataFrame
    df = pd.DataFrame({"text": all_texts, "label": all_labels, "loss": all_losses})

    # Shuffle the data
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Save to CSV
    output_path = data_dir / "sample_sentiment.csv"
    df.to_csv(output_path, index=False)

    print(f"Generated sample sentiment dataset with {len(df)} examples")
    print(f"Saved to {output_path}")

    return output_path


if __name__ == "__main__":
    generate_sample_sentiment_data()
