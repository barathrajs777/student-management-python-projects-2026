# Experiment 8: Data Cleaning and EDA
# Using Pandas and Matplotlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Upload the CSV in Google Colab. For local execution, place the CSV next to this script.
try:
    from google.colab import files
except ImportError:
    files = None

CSV_FILENAME = "experiment_8_student_performance.csv"

if files is not None:
    uploaded = files.upload()
    if uploaded:
        CSV_FILENAME = next(iter(uploaded))

df = pd.read_csv(CSV_FILENAME)

print("----- ORIGINAL DATASET -----")
print(df.head())
print("\nDataset Shape:", df.shape)

print("\n----- MISSING VALUES -----")
print(df.isnull().sum())
print("\nDuplicate Records:", df.duplicated().sum())

# Handle missing numerical values with each column's median.
num_cols = df.select_dtypes(include=np.number).columns
for col in num_cols:
    median = df[col].median()
    if pd.notna(median):
        df[col] = df[col].fillna(median)

# Handle missing categorical values with each column's most frequent value.
cat_cols = df.select_dtypes(include="object").columns
for col in cat_cols:
    mode = df[col].mode()
    if not mode.empty:
        df[col] = df[col].fillna(mode[0])

# Remove duplicate records.
df.drop_duplicates(inplace=True)

# Calculate average score and classify student performance.
score_cols = ["Assignment_Score", "Exam_Score"]
df["Average_Score"] = df[score_cols].mean(axis=1)
df["Performance"] = pd.cut(
    df["Average_Score"],
    bins=[-1, 50, 70, 85, 100],
    labels=["Needs Improvement", "Average", "Good", "Excellent"],
)

print("\n----- CLEANED DATASET -----")
print(df.head(10))
print("\nCleaned Dataset Shape:", df.shape)

# Statistical analysis.
print("\n----- STATISTICAL SUMMARY -----")
print(df.describe().round(2))

# Department-wise analysis.
dept_analysis = df.groupby("Department")["Exam_Score"].mean()
print("\n----- DEPARTMENT-WISE AVERAGE -----")
print(dept_analysis.round(2))

# Performance distribution.
print("\n----- PERFORMANCE DISTRIBUTION -----")
print(df["Performance"].value_counts())

# Visualization 1: Department-wise average exam score.
plt.figure(figsize=(8, 5))
dept_analysis.plot(kind="bar", color="skyblue", edgecolor="black")
plt.title("Department-wise Average Exam Score")
plt.xlabel("Department")
plt.ylabel("Average Exam Score")
plt.xticks(rotation=0)
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()

# Visualization 2: Student performance distribution.
plt.figure(figsize=(8, 5))
df["Performance"].value_counts().plot(kind="bar", color="orange", edgecolor="black")
plt.title("Student Performance Distribution")
plt.xlabel("Performance Category")
plt.ylabel("Number of Students")
plt.xticks(rotation=15)
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()

print("\nData Cleaning and EDA Completed Successfully!")
