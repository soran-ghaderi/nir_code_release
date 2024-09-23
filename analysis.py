import pandas as pd
from datasets import load_from_disk
import json
import os
import re

from collections import defaultdict

layer_indices = [1, 10, 23, "orig"]


subset_name = "processed_Meta-Llama-3.1-8B-Instruct-evals__mbpp__details_results"

dataset = load_from_disk(f"data/{subset_name}")
dfo = dataset.to_pandas()
print(dfo.columns)

from typing import Any, Union


def sort_filenames(filenames):
    def extract_number(filename):
        # Extract the last number from the filename
        match = re.search(r"(\d+)\.json$", filename)
        return int(match.group(1)) if match else 0

    # Sort the filenames based on the extracted number
    return sorted(filenames, key=extract_number)


def find_last_checkpoints(directory, layer):
    files = []
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            if "_" + str(layer) + "_" in filename:
                files.append(filename)
    return sort_filenames(files)[-1]


# def extract_json_data(filename):
#     all_keys = {}
#     stats = defaultdict(int)
#     extracted_data = []
#
#     with open(filename, "r") as file:
#         try:
#             data = json.load(file)
#             print("data.keys(): ", data.keys())
#             k1 = dict(data.items())
#             print('k1["results"].keys(): ', k1["results"].keys())
#             results = dict(k1["results"])
#             print("loop over results.keys(): ")
#             # for i, k in enumerate(results.keys()):
#             #     print(i, results[k][0], "\n", "--" * 20)
#
#             for i, (k, v) in enumerate(results.items()):
#                 # if i > 0:
#                 #     break
#                 print("index: ", i)
#                 print("k: ", k)
#                 print("v: ", v[0] if type(v[0]) is int else v[0][:15])
#                 print("len v: ", len(v), v.count(None))
#                 print("--" * 40)
#                 extracted_data.append(
#                     {
#                         "columns": k,
#                         "value": v if v else None,
#                     }
#                 )
#
#         except json.JSONDecodeError:
#             stats["invalid_json_files"] += 1
#
#     return stats
#
#
# checkpoint_dir = "checkpoints-old"
# last_file = find_last_checkpoints(checkpoint_dir, "orig")
# path = f"checkpoints-old/{last_file}"
# extract_json_data(path)


def extract_json_data(filename):
    with open(filename, "r") as file:
        data = json.load(file)

    results = data["results"]

    columns = ["instance_id", "query", "context", "test_cases"]
    for suffix in ["1", "10", "23", "orig"]:
        columns.extend(
            [
                f"final_output_{suffix}",
                f"extracted_functions_{suffix}",
                f"trajectories_and_context_{suffix}",
                f"context_expansion_{suffix}",
            ]
        )

    df_data = defaultdict(list)

    for i in range(max(len(v) for v in results.values())):
        for col in columns:
            value = results.get(col, [])
            df_data[col].append(value[i] if i < len(value) else None)

    return pd.DataFrame(df_data)


checkpoint_dir = "checkpoints-old"
last_file = find_last_checkpoints(checkpoint_dir, "orig")
print("last file: ", last_file)
path = f"{checkpoint_dir}/{last_file}"
df = extract_json_data(path)

print(df.head())
print(df.info())

df.to_csv("structured_checkpoint_data.csv", index=False)

#  ========================================================================


def extract_python_function(text):
    if pd.isna(text):
        return None
    pattern = r"((?:def|class)\s+[\w\d_]+\s*\([^)]*\):(?:\n(?:    .*(?:\n|$))*)*)"
    matches = re.findall(pattern, text, re.MULTILINE)
    return "\n\n".join(matches) if matches else None


def extract_imports(text):
    if pd.isna(text):
        return None
    pattern = r"^(import\s+.*|from\s+.*\s+import\s+.*)$"
    matches = re.findall(pattern, text, re.MULTILINE)
    return "\n".join(matches) if matches else None


def process_dataframe(df):
    new_df = df.copy()
    layer_indices = ["1", "10", "23", "orig"]

    for idx in layer_indices:
        final_output_col = f"final_output_{idx}"
        extracted_functions_col = f"extracted_functions_{idx}"

        if (
            final_output_col in new_df.columns
            and extracted_functions_col in new_df.columns
        ):
            imports = new_df[final_output_col].apply(extract_imports)
            functions = new_df[final_output_col].apply(extract_python_function)

            new_df[extracted_functions_col] = imports.combine_first(functions)

    return new_df


new_df = process_dataframe(df)

print(new_df[["final_output_1", "extracted_functions_1"]].head())
print(new_df[["final_output_10", "extracted_functions_10"]].head())
print(new_df[["final_output_23", "extracted_functions_23"]].head())
print(new_df[["final_output_orig", "extracted_functions_orig"]].head())

new_df.to_csv("processed_checkpoint_data.csv", index=False)
