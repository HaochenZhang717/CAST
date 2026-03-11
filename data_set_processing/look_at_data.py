from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

npz_folder = Path("./npz_files")

all_lengths = []

def compute_segment_lengths(labels):
    lengths = []
    current = 0
    for v in labels:
        if v == 1:
            current += 1
        else:
            if current > 0:
                lengths.append(current)
                current = 0
    if current > 0:
        lengths.append(current)
    return lengths

for file in npz_folder.glob("*.npz"):
    data = np.load(file)
    labels = data["anomaly_label"]
    all_lengths.extend(compute_segment_lengths(labels))

all_lengths = np.array(all_lengths)

print("Total segments:", len(all_lengths))
print("Mean length:", all_lengths.mean())
print("Max length:", all_lengths.max())

plt.figure()
plt.hist(all_lengths, bins=30)
plt.xlabel("Anomaly Segment Length")
plt.ylabel("Frequency")
plt.title("Histogram of Anomaly Segment Lengths")
plt.show()