# Experiment 10: Prompt Design and Evaluation

import pandas as pd

# Upload the input CSV in Google Colab. For local execution, place the CSV next to this script.
try:
    from google.colab import files
except ImportError:
    files = None

CSV_FILENAME = "experiment_10_prompt_evaluation.csv"

if files is not None:
    uploaded = files.upload()
    if uploaded:
        CSV_FILENAME = next(iter(uploaded))

df = pd.read_csv(CSV_FILENAME)


def evaluate_prompt(prompt):
    """Score a prompt against a simple five-feature checklist."""
    text = str(prompt).lower()
    features = {
        "Role": any(word in text for word in ["act as", "you are"]),
        "Context": any(
            word in text
            for word in ["student", "manager", "college", "project", "report"]
        ),
        "Clear task": any(
            word in text
            for word in ["explain", "draft", "review", "summarize", "compare"]
        ),
        "Constraints": any(
            word in text for word in ["words", "bullet", "under", "three", "five"]
        ),
        "Output format": any(
            word in text for word in ["table", "subject", "bullet points", "example"]
        ),
    }
    return sum(features.values()), features


print("----- PROMPT DESIGN AND EVALUATION -----")
for _, row in df.iterrows():
    weak_score, _ = evaluate_prompt(row["Weak_Prompt"])
    improved_score, features = evaluate_prompt(row["Improved_Prompt"])

    print(f"\nPrompt ID: {row['Prompt_ID']} | Task: {row['Task']}")
    print(f"Weak prompt score : {weak_score}/5")
    print(f"Improved prompt score : {improved_score}/5")
    satisfied = ", ".join(key for key, value in features.items() if value)
    print("Satisfied features :", satisfied or "None")
    print("Evaluation :", "Improved" if improved_score > weak_score else "Needs review")

# Calculate checklist scores for the overall summary.
df["Weak_Score"] = df["Weak_Prompt"].apply(lambda prompt: evaluate_prompt(prompt)[0])
df["Improved_Score"] = df["Improved_Prompt"].apply(
    lambda prompt: evaluate_prompt(prompt)[0]
)

print("\n----- OVERALL SUMMARY -----")
print("Average weak prompt score :", round(df["Weak_Score"].mean(), 2))
print("Average improved prompt score:", round(df["Improved_Score"].mean(), 2))
print(
    "\nEvaluation completed. This is a rule-based prompt-quality checklist, "
    "not an LLM evaluation."
)
