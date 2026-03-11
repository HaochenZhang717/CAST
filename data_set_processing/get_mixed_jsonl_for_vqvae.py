import os
from pathlib import Path


# def concat_jsonl(file1, file2, output_file):
#     with open(output_file, "w", encoding="utf-8") as outfile:
#         for path in [file1, file2]:
#             with open(path, "r", encoding="utf-8") as infile:
#                 for line in infile:
#                     line = line.strip()
#                     if line:   # avoid empty lines
#                         outfile.write(line + "\n")

# Example
# concat_jsonl("file1.jsonl", "file2.jsonl", "merged.jsonl")


from pathlib import Path

root = Path("./anomaly_indices")

for subfolder in root.iterdir():
    if subfolder.is_dir():
        subfolder = str(subfolder).split("/")[-1]
        file1 = f"./anomaly_indices/{subfolder}/V_segments.jsonl"
        file2 = f"./normal_indices_144/{subfolder}/normal_144.jsonl"
        output = f"./mixed_indices/{subfolder}/mixed.jsonl"
        os.makedirs(f"./mixed_indices/{subfolder}", exist_ok=True)
        # if file1.exists() and file2.exists():
        #     print(f"Merging in {subfolder.name}")

        with open(output, "w", encoding="utf-8") as outfile:
            for file in [file1, file2]:
                with open(file, "r", encoding="utf-8") as infile:
                    for line in infile:
                        line = line.strip()
                        if line:
                            outfile.write(line + "\n")

print("All done.")

