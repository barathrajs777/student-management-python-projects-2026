# Experiment 11: Role-Based and Example-Based Prompting

import pandas as pd

# Upload the CSV in Google Colab. For local execution, place the CSV next to this script.
try:
    from google.colab import files
except ImportError:
    files = None

CSV_FILENAME = "experiment_11_prompting_techniques.csv"

if files is not None:
    uploaded = files.upload()
    if uploaded:
        CSV_FILENAME = next(iter(uploaded))

df = pd.read_csv(CSV_FILENAME)

positive_words = {"quick", "perfectly", "excellent", "helpful", "love", "early", "great"}
negative_words = {"poor", "dull", "stopped", "broken", "slow", "unavailable"}


def sentiment_demo(sentence):
    """Return a simple rule-based sentiment label for a sentence."""
    words = set(str(sentence).lower().replace(".", "").replace(",", "").split())
    score = len(words & positive_words) - len(words & negative_words)
    return "Positive" if score > 0 else "Negative" if score < 0 else "Neutral"


def construct_prompt(row):
    """Build a prompt using the row's role, task, input, and technique."""
    role = row["Role"]
    task = row["Task"]
    text = row["Input_Text"]
    technique = row["Technique"]

    if technique == "Zero-Shot":
        return f"You are a {role}. {task}: {text}"
    if technique == "One-Shot":
        return (
            f"You are a {role}. {task}. Example: "
            f"{row['Example_Input']} -> {row['Example_Output']}. New input: {text}"
        )
    if technique == "Few-Shot":
        return (
            f"You are a {role}. {task}. Examples: "
            "Excellent service -> Positive; Poor quality -> Negative. "
            f"New input: {text}"
        )
    return f"Act as a {role}. {task}: {text}"


print("----- ROLE-BASED AND EXAMPLE-BASED PROMPTING -----")
for _, row in df.iterrows():
    prompt = construct_prompt(row)
    task = str(row["Task"]).lower()
    input_text = str(row["Input_Text"])

    if "sentiment" in task:
        response = sentiment_demo(input_text)
    elif "priority" in task:
        response = "High" if "unavailable" in input_text.lower() else "Low"
    else:
        response = "Suggested rewrite: Please share the project file at your earliest convenience."

    print(f"\nID: {row['Example_ID']} | Technique: {row['Technique']}")
    print("Constructed prompt:", prompt)
    print("Python demonstration response:", response)

print(
    "\nDemonstration completed. Responses use predefined Python rules, "
    "not a live LLM."
)
