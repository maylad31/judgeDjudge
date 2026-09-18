from datasets import load_dataset
import csv

# Load only 10 rows from RubricBench
dataset = load_dataset("donjoey/rubricbench", split="train[:10]")
rows = [dict(item) for item in dataset]

# Convert to required CSV format
with open("input.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f, fieldnames=["task", "rubrics", "ideal_response", "negative_response"]
    )
    writer.writeheader()

    for item in rows:
        if item["label"] == 0:
            ideal = item["response_a"]
            negative = item["response_b"]
        else:
            ideal = item["response_b"]
            negative = item["response_a"]

        writer.writerow(
            {
                "task": item["instruction"],
                "rubrics": item["rubrics"],
                "ideal_response": ideal,
                "negative_response": negative,
            }
        )

print(f"Converted {len(rows)} rows to input.csv")
