from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

npz_folder = Path("./npz_files")

all_lengths = []

anomaly_list = []
for file in npz_folder.glob("*.npz"):
    data = np.load(file)
    labels = data["anomaly_label"]
    print(file)
    print(data['signal'].shape)
    print(data['anomaly_label'].shape)
    anomaly_list.append(data['signal'][labels==1])
    # all_lengths.extend(compute_segment_lengths(labels))


np.save("anomaly_segments.npy", anomaly_list)
# all_lengths = np.array(all_lengths)
#
# print("Total segments:", len(all_lengths))
# print("Mean length:", all_lengths.mean())
# print("Max length:", all_lengths.max())
#
# plt.figure()
# plt.hist(all_lengths, bins=30)
# plt.xlabel("Anomaly Segment Length")
# plt.ylabel("Frequency")
# plt.title("Histogram of Anomaly Segment Lengths")
# plt.show()