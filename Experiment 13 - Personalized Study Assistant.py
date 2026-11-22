# Experiment 13: Personalized Study Assistant with CSV Retrieval

import re

import pandas as pd

# Upload the CSV in Google Colab. For local execution, place the CSV next to this script.
try:
    from google.colab import files
except ImportError:
    files = None

CSV_FILENAME = "experiment_13_study_materials.csv"

if files is not None:
    uploaded = files.upload()
    if uploaded:
        CSV_FILENAME = next(iter(uploaded))

df = pd.read_csv(CSV_FILENAME)


def tokenize(text):
    """Convert text into a set of lowercase alphabetic tokens."""
    return set(re.findall(r"[a-zA-Z]+", str(text).lower()))


def retrieve_material(query):
    """Find the study material with the greatest query-token overlap."""
    query_words = tokenize(query)
    if not query_words:
        return None, 0

    scores = []
    searchable_columns = ["Topic", "Concept", "Explanation", "Keywords"]
    for _, row in df.iterrows():
        searchable_text = " ".join(str(row.get(column, "")) for column in searchable_columns)
        material_words = tokenize(searchable_text)
        scores.append(len(query_words & material_words) / len(query_words))

    if not scores or max(scores) == 0:
        return None, 0

    best_index = scores.index(max(scores))
    return df.iloc[best_index], scores[best_index]


def study_assistant(query):
    """Print study notes and quiz details for the closest matching CSV record."""
    material, score = retrieve_material(query)
    if material is None:
        print("No matching material found. Try another concept.")
        return

    print("\n----- PERSONALIZED STUDY NOTES -----")
    print("Topic       :", material.get("Topic", ""))
    print("Concept     :", material.get("Concept", ""))
    print("Explanation :", material.get("Explanation", ""))
    print("Match score :", round(score * 100, 2), "%")

    print("\n----- QUICK QUIZ -----")
    print("Question   :", material.get("Quiz_Question", ""))
    print("Difficulty :", material.get("Difficulty", ""))
    print("Answer     :", material.get("Answer", ""))


available_concepts = df["Concept"].dropna().astype(str).tolist() if "Concept" in df else []
print("Available concepts:", ", ".join(available_concepts))
query = input("\nEnter a topic or concept to study: ")
study_assistant(query)

print("\nStudy assistant session completed.")
print("Note: Matching is based on CSV keyword overlap, not a live Generative AI model.")
