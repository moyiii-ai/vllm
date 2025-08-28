import math
import random

INPUT_FILE = "narrativeqa.jsonl"
OUTPUT_FILE_1 = "narrativeqa_part1.jsonl"
OUTPUT_FILE_2 = "narrativeqa_part2.jsonl"

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    lines = f.readlines()

random.shuffle(lines)

total_requests = len(lines)
half_requests = math.ceil(total_requests / 2)

part1_lines = lines[:half_requests]
part2_lines = lines[half_requests:]

with open(OUTPUT_FILE_1, "w", encoding="utf-8") as f:
    f.writelines(part1_lines)

with open(OUTPUT_FILE_2, "w", encoding="utf-8") as f:
    f.writelines(part2_lines)

print(f"Total prompts: {total_requests}")
print(f"Part1: {len(part1_lines)} prompts -> {OUTPUT_FILE_1}")
print(f"Part2: {len(part2_lines)} prompts -> {OUTPUT_FILE_2}")
