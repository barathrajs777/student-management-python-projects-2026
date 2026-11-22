# Experiment 12: Technical Content and Python Program Generation

from pathlib import Path
import re
import textwrap

import pandas as pd

# Upload the CSV in Google Colab. For local execution, place the CSV next to this script.
try:
    from google.colab import files
except ImportError:
    files = None

CSV_FILENAME = "experiment_12_technical_generation.csv"

if files is not None:
    uploaded = files.upload()
    if uploaded:
        CSV_FILENAME = next(iter(uploaded))

df = pd.read_csv(CSV_FILENAME)

templates = {
    "CSV Data Analysis": '''
        import pandas as pd

        def column_average(filename, column):
            data = pd.read_csv(filename)
            if column not in data.columns:
                raise ValueError("Column not found")
            values = pd.to_numeric(data[column], errors="coerce").dropna()
            if values.empty:
                raise ValueError("No numeric values available")
            return round(values.mean(), 2)
    ''',
    "Temperature Conversion": '''
        def celsius_to_fahrenheit(celsius):
            if not isinstance(celsius, (int, float)):
                raise TypeError("Temperature must be numeric")
            return round((celsius * 9 / 5) + 32, 2)
    ''',
    "Password Validation": '''
        def check_password(password):
            checks = {
                "Minimum 8 characters": len(password) >= 8,
                "Uppercase": any(char.isupper() for char in password),
                "Lowercase": any(char.islower() for char in password),
                "Digit": any(char.isdigit() for char in password),
                "Special character": any(not char.isalnum() for char in password),
            }
            for rule, passed in checks.items():
                print(rule, ":", "Pass" if passed else "Fail")
            return all(checks.values())
    ''',
    "List Statistics": '''
        def list_statistics(values):
            if not values:
                raise ValueError("List cannot be empty")
            return {
                "minimum": min(values),
                "maximum": max(values),
                "range": max(values) - min(values),
            }
    ''',
    "Text Analysis": '''
        import string

        def analyze_text(text):
            cleaned = text.translate(str.maketrans("", "", string.punctuation)).lower()
            words = cleaned.split()
            return {"total_words": len(words), "unique_words": len(set(words))}
    ''',
}

output_dir = Path("generated_technical_content")
output_dir.mkdir(exist_ok=True)

print("----- TECHNICAL CONTENT AND PYTHON PROGRAM GENERATION -----")
for _, row in df.iterrows():
    topic = row["Topic"]
    if topic not in templates:
        raise ValueError(f"No code template available for topic: {topic}")

    code = textwrap.dedent(templates[topic]).lstrip()
    explanation = (
        f"Topic: {topic}\nAudience: {row['Audience']}\n"
        f"Task: {row['Task_Description']}\n"
        f"Requirements: {row['Requirements']}\n"
        "Approach: validate inputs, process data and present a clear result."
    )

    # Keep generated filenames safe and portable.
    file_stem = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(row["Task_ID"]).lower()).strip("_")
    if not file_stem:
        raise ValueError("Task_ID must contain at least one letter or number")

    (output_dir / f"{file_stem}.py").write_text(code, encoding="utf-8")
    (output_dir / f"{file_stem}_notes.txt").write_text(explanation, encoding="utf-8")

    print(f"\nTask ID: {row['Task_ID']} | Topic: {topic}")
    print(explanation)
    print("Saved:", f"{file_stem}.py", "and", f"{file_stem}_notes.txt")

print("\nTotal generated files:", len(list(output_dir.iterdir())))
print("Template-based generation completed. No external LLM is called.")
