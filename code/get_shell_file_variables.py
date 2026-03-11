# import os
# import json
# from pathlib import Path
#
# folder = Path("../data_set_processing/npz_files")
# files = sorted(folder.glob("*.npz"))
# paths = [str(f) for f in files]
# data_paths_string = "DATA_PATHS='" + json.dumps(paths) + "'"
# print(data_paths_string)
#
#
#
#
#
# folder = Path("../data_set_processing/mixed_indices")
# files = sorted(folder.rglob("*.jsonl"))
# paths = [str(f) for f in files]
# data_paths_string = "PRETRAIN_INDICES_PATHS='" + json.dumps(paths) + "'"
# print(data_paths_string)
#
#
#
# folder = Path("../data_set_processing/ts_with_anomaly_indices")
# files = sorted(folder.rglob("*.jsonl"))
# paths = [str(f) for f in files]
# data_paths_string = "FINETUNE_TRAIN_INDICES_PATHS='" + json.dumps(paths) + "'"
# print(data_paths_string)
#
# folder = Path("../data_set_processing/anomaly_indices")
# files = sorted(folder.rglob("*.jsonl"))
# paths = [str(f) for f in files]
# data_paths_string = "ANOMALY_INDICES_FOR_SAMPLE='" + json.dumps(paths) + "'"
# print(data_paths_string)
#
# folder = Path("../data_set_processing/normal_indices_300")
# files = sorted(folder.rglob("*.jsonl"))
# paths = [str(f) for f in files]
# data_paths_string = "NORMAL_INDICES_FOR_SAMPLE='" + json.dumps(paths) + "'"
# print(data_paths_string)
#


import os
import json
from pathlib import Path

def print_var(folder_path, var_name, pattern):
    folder = Path(folder_path)
    files = sorted(folder.rglob(pattern))
    paths = [str(f) for f in files]
    data_paths_string = f"{var_name}='" + json.dumps(paths, separators=(",", ":")) + "'"
    print(data_paths_string)


print_var("../data_set_processing/npz_files", "DATA_PATHS", "*.npz")
print_var("../data_set_processing/mixed_indices", "PRETRAIN_INDICES_PATHS", "*.jsonl")
print_var("../data_set_processing/ts_with_anomaly_indices", "FINETUNE_TRAIN_INDICES_PATHS", "*.jsonl")
print_var("../data_set_processing/anomaly_indices", "ANOMALY_INDICES_FOR_SAMPLE", "*.jsonl")
print_var("../data_set_processing/normal_indices_300", "NORMAL_INDICES_FOR_SAMPLE", "*.jsonl")