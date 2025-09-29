import os

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
    max_len = max(len(str1), len(str2))  # Ensure both strings compared fully

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
    """
    Generate a unique filename in the given folder.
    Example: Q5_report.txt → Q5_report_1.txt → Q5_report_2.txt
    """
    filename = os.path.join(folder, base_name + ext)
    counter = 1
    while os.path.exists(filename):
        filename = os.path.join(folder, f"{base_name}_{counter}{ext}")
        counter += 1
    return filename


def main():
    # ✅ Google Drive path for saving results
    result_path = "/content/drive/MyDrive/Assignment internship/result/Q5"

    # Create result folder if not exists
    os.makedirs(result_path, exist_ok=True)

    # --- Input handling ---
    while True:
        str1 = input("Enter first string (6–10 chars): ").strip()
        if 6 <= len(str1) <= 10:
            break
        else:
            print("❌ Error: First string must be between 6–10 characters. Try again.\n")

    while True:
        str2 = input("Enter second string (6–10 chars): ").strip()
        if 6 <= len(str2) <= 10:
            break
        else:
            print("❌ Error: Second string must be between 6–10 characters. Try again.\n")

    # --- Compare the strings ---
    similarity, matches, mismatches, details = compare_strings(str1, str2)

    # --- Get unique file name ---
    output_file = get_unique_filename(result_path)

    # --- Write report ---
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("String Similarity Report\n")
        f.write("========================\n\n")
        f.write(f"String 1: {str1} (Length: {len(str1)})\n")
        f.write(f"String 2: {str2} (Length: {len(str2)})\n\n")

        f.write(f"Total Matches: {matches}\n")
        f.write(f"Total Mismatches: {mismatches}\n")
        f.write(f"Similarity: {similarity:.2f}%\n\n")

        f.write("Character-by-Character Comparison:\n")
        f.write("----------------------------------\n")
        for line in details:
            f.write(line + "\n")

    # --- Print final result ---
    print(f"\n✅ Similarity: {similarity:.2f}%")
    print(f"✅ Report saved to: {output_file}")


if __name__ == "__main__":
    main()
