import os
import numpy as np
import pandas as pd

# Folder containing CSV files
# csv_folder = "/Users/zhc/Downloads/DataFromWenjia/NYISO_load_anomaly_2024_interpolated_files/30min"
csv_folder = "/Users/zhc/Downloads/DataFromWenjia/PJM_load_anomaly_2024_interpolated_files/30min"
# csv_folder = "/Users/zhc/Downloads/DataFromWenjia/SPP_2021_csv_Feb_interpolated_files/30min"
# csv_folder = "/Users/zhc/Downloads/DataFromWenjia/interpolated_outputs/30min"

out_path = "./"
# 输出目录（可选）
output_folder = os.path.join(out_path, "npz_files")
os.makedirs(output_folder, exist_ok=True)

# 遍历所有 CSV 文件
for filename in os.listdir(csv_folder):
    if filename.endswith(".csv"):
        csv_path = os.path.join(csv_folder, filename)

        print(f"Processing: {filename}")

        # 读取 CSV
        df = pd.read_csv(csv_path)

        # 提取列
        for key in df.keys():
            if 'load' in key:
                signal = df[key].values.astype(np.float32)
            if 'demand' in key:
                signal = df[key].values.astype(np.float32)
            if 'anomaly' in key:
                anomaly_label = df[key].values.astype(np.int64)
        # signal = df["load"].values.astype(np.float32)
        # signal = df["demand"].values.astype(np.float32)



        # 构造输出文件名
        base_name = os.path.splitext(filename)[0]
        npz_path = os.path.join(output_folder, base_name + ".npz")

        # 保存
        np.savez(npz_path,
                 signal=signal.reshape(-1, 1),
                 anomaly_label=anomaly_label)

print("All files processed.")