from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
file_path = BASE_DIR / "data" / "raw" / "opensooq_mobile.csv"
line_to_inspect = 645

with open(file_path, 'rb') as f:
    for i, line in enumerate(f):
        if i >= line_to_inspect - 5 and i <= line_to_inspect + 5:
            print(f"Line {i+1}:")
            print(line)
            print("-" * 20)
