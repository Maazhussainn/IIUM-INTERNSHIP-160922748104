import os
import random
import csv
from pathlib import Path

# ----------------------------
# Q5 Functions
# ----------------------------
def compare_strings(str1: str, str2: str):
    """
    Compare two strings character by character.
    Returns:
        similarity %,
        total matches,
        total mismatches,
        list of detailed comparison lines.
    """
    matches = 0
    comparison_lines = []
    max_len = max(len(str1), len(str2))  # Ensure we cover both strings fully

    for i in range(max_len):
        c1 = str1[i] if i < len(str1) else "-"
        c2 = str2[i] if i < len(str2) else "-"
        status = "✓" if c1 == c2 else "✗"
        if status == "✓":
            matches += 1
        comparison_lines.append(f"Pos {i+1}: {c1} vs {c2} → {status}")

    similarity = (matches / max_len) * 100
    return similarity, matches, max_len - matches, comparison_lines


def get_unique_filename(folder: str, base_name: str = "Q5_report", ext: str = ".txt") -> str:
    """Generate a unique filename inside folder (avoids overwriting)."""
    filename = os.path.join(folder, base_name + ext)
    counter = 1
    while os.path.exists(filename):
        filename = os.path.join(folder, f"{base_name}_{counter}{ext}")
        counter += 1
    return filename

# ----------------------------
# Q6 Functions
# ----------------------------

NUM_PLATES = 1000
PLATE_LENGTH = 9
SIMILARITY_THRESHOLD = 70  # percentage threshold for match

# Save results in Google Drive path
RESULT_FOLDER = "/content/drive/MyDrive/Assignment internship/result/Q6"
os.makedirs(RESULT_FOLDER, exist_ok=True)


def generate_plate():
    """Generate a synthetic Indian license plate-like string."""
    state_code = random.choice("MHDLRJGUPKNCH") + random.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    digits = "".join(random.choices("0123456789", k=2))
    letters = "".join(random.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZ", k=2))
    number = "".join(random.choices("0123456789", k=3))
    return f"{state_code}{digits}{letters}{number}"[:PLATE_LENGTH]


# --- Generate plates ---
plates = [generate_plate() for _ in range(NUM_PLATES)]

# --- Compare plates pairwise ---
output_file = get_unique_filename(RESULT_FOLDER, base_name="license_plate_similarity", ext=".csv")

with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Plate1", "Plate2", "Similarity(%)", "Matches", "Mismatches", "MatchAboveThreshold"])

    for i in range(len(plates)):
        for j in range(i + 1, len(plates)):
            plate1 = plates[i]
            plate2 = plates[j]
            similarity, matches, mismatches, _ = compare_strings(plate1, plate2)
            match_flag = "YES" if similarity >= SIMILARITY_THRESHOLD else "NO"
            writer.writerow([plate1, plate2, f"{similarity:.2f}", matches, mismatches, match_flag])

print(f"✅ Completed! Report saved at: {output_file}")
