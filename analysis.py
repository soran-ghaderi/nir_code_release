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


def extract_function_name(text):
    if pd.isna(text):
        return None
    pattern = r"(?:def|class|assert)\s+([\w\d_]+)\s*\("
    match = re.search(pattern, text)
    return match.group(1) if match else None


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
import pandas as pd
import numpy as np
from typing import List, Callable
import ast
import re


# def code_correctness(func: str, test_cases: str) -> float:
#     # Placeholder for code execution and test case validation
#     # This would need to be implemented with proper code execution safety measures
#     pass
#
#
# def syntactic_correctness(func: str) -> bool:
#     try:
#         ast.parse(func)
#         return True
#     except SyntaxError:
#         return False
#
#
# def cyclomatic_complexity(func: str) -> int:
#     # Placeholder for cyclomatic complexity calculation
#     # This would need to be implemented using an appropriate library or custom logic
#     pass
#
#
# def halstead_metrics(func: str) -> dict:
#     # Placeholder for Halstead complexity measures
#     # This would calculate program vocabulary, program length, volume, difficulty, and effort
#     pass
#
#
# def function_name_consistency(query: str, func: str) -> float:
#     # Extract function name from the code
#     func_name_match = re.search(r"def\s+(\w+)", func)
#     if not func_name_match:
#         return 0.0
#     func_name = func_name_match.group(1)
#
#     # Calculate similarity between query and function name
#     # This is a simplified version and could be improved with more sophisticated NLP techniques
#     query_words = set(query.lower().split())
#     func_name_words = set(re.findall(r"[a-z]+", func_name.lower()))
#     return len(query_words.intersection(func_name_words)) / len(
#         query_words.union(func_name_words)
#     )
#
#
# def calculate_metrics(df: pd.DataFrame) -> pd.DataFrame:
#     metrics = []
#
#     for layer_idx in ["1", "10", "23", "orig"]:
#         extracted_funcs_col = f"extracted_functions_{layer_idx}"
#
#         if extracted_funcs_col not in df.columns:
#             continue
#
#         layer_metrics = {
#             f"response_rate_{layer_idx}": df[extracted_funcs_col].notna().mean(),
#             f"syntactic_correctness_{layer_idx}": df[extracted_funcs_col]
#             .apply(syntactic_correctness)
#             .mean(),
#             f"avg_cyclomatic_complexity_{layer_idx}": df[extracted_funcs_col]
#             .apply(cyclomatic_complexity)
#             .mean(),
#             f"avg_function_name_consistency_{layer_idx}": df.apply(
#                 lambda row: function_name_consistency(
#                     row["query"], row[extracted_funcs_col]
#                 ),
#                 axis=1,
#             ).mean(),
#             # Add more metrics here
#         }
#
#         # Halstead metrics
#         halstead = (
#             df[extracted_funcs_col].apply(halstead_metrics).agg(["mean", "median"])
#         )
#         for metric, values in halstead.items():
#             layer_metrics[f"avg_{metric}_{layer_idx}"] = values["mean"]
#             layer_metrics[f"median_{metric}_{layer_idx}"] = values["median"]
#
#         metrics.append(layer_metrics)
#
#     return pd.DataFrame(metrics)
#
#
# # Usage
# # df = pd.read_csv("your_data.csv")
# metrics_df = calculate_metrics(df)
# print(metrics_df)

import pandas as pd
import numpy as np
import ast
import re
from typing import List, Callable
from radon.metrics import h_visit
from radon.complexity import cc_visit


def safe_execute(func: str, test_cases: str) -> bool:
    # This is a placeholder. In a real scenario, you'd need a sandboxed environment
    # to safely execute code. For now, we'll just check if the function compiles.
    try:
        compile(func, "<string>", "exec")
        return True
    except:
        return False


def syntactic_correctness(func: str) -> bool:
    try:
        ast.parse(func)
        return True
    except SyntaxError:
        return False


def cyclomatic_complexity(func: str) -> int:
    try:
        return max((cc.complexity for cc in cc_visit(func)), default=0)
    except:
        return 0


def calculate_name_similarity(name1: str, name2: str) -> float:
    if not name1 or not name2:
        return 0.0
    name1_words = set(re.findall(r"[a-z]+", name1.lower()))
    name2_words = set(re.findall(r"[a-z]+", name2.lower()))
    return len(name1_words.intersection(name2_words)) / len(
        name1_words.union(name2_words)
    )


# def check_function_name_consistency(
#     test_cases: str, generated_function: str, use_similarity: bool = False
# ) -> Tuple[float, str, str]:
#     test_func_name = extract_function_name(test_cases)
#     generated_func_name = extract_function_name(generated_function)
#
#     if use_similarity:
#         consistency = calculate_name_similarity(test_func_name, generated_func_name)
#     else:
#         consistency = (
#             float(test_func_name == generated_func_name)
#             if test_func_name and generated_func_name
#             else 0.0
#         )
#
#     return consistency, test_func_name, generated_func_name
#
#
# # Function to apply to DataFrame
# def apply_function_name_consistency(
#     df: pd.DataFrame, layer_idx: str, use_similarity: bool = False
# ) -> pd.DataFrame:
#     results = df.apply(
#         lambda row: check_function_name_consistency(
#             row["test_cases"], row[f"extracted_functions_{layer_idx}"], use_similarity
#         ),
#         axis=1,
#     )
#
#     df[f"function_name_consistency_{layer_idx}"] = [r[0] for r in results]
#     df[f"test_function_name_{layer_idx}"] = [r[1] for r in results]
#     df[f"generated_function_name_{layer_idx}"] = [r[2] for r in results]
#
#     return df


def check_function_name_consistency(test_cases: str, generated_function: str) -> float:
    test_func_name = extract_function_name(test_cases)
    generated_func_name = extract_function_name(generated_function)

    if test_func_name and generated_func_name:
        print("yes equal: ", test_func_name, generated_func_name)
        return float(test_func_name == generated_func_name)

    print("were not equal: ", test_func_name, generated_func_name)
    return 0.0


# def function_name_consistency(query: str, test_cases: str) -> float:
#     func_name = extract_function_name(test_cases)
#     func_name_match = func_name in query
#     print(func_name_match)
#     if not func_name_match:
#         return 0.0
#     func_name = func_name_match.group(1)
#     query_words = set(query.lower().split())
#     func_name_words = set(re.findall(r"[a-z]+", func_name.lower()))
#     return len(query_words.intersection(func_name_words)) / len(
#         query_words.union(func_name_words)
#     )


def code_length(func: str) -> dict:
    lines = func.split("\n")
    return {"lines": len(lines), "characters": len(func)}


def comment_analysis(func: str) -> dict:
    lines = func.split("\n")
    comment_lines = sum(1 for line in lines if line.strip().startswith("#"))
    return {
        "comment_lines": comment_lines,
        "comment_ratio": comment_lines / len(lines) if lines else 0,
    }


def calculate_metrics(row: pd.Series, layer_idx: str) -> dict:
    func = row[f"extracted_functions_{layer_idx}"]
    if pd.isna(func):
        return {
            f"{metric}_{layer_idx}": np.nan
            for metric in [
                "syntactic_correctness",
                "cyclomatic_complexity",
                "function_name_consistency",
                "lines",
                "characters",
                "comment_lines",
                "comment_ratio",
                "h1",
                "h2",
                "N1",
                "N2",
                "vocabulary",
                "length",
                "volume",
                "difficulty",
                "effort",
            ]
        }

    metrics = {
        f"syntactic_correctness_{layer_idx}": int(syntactic_correctness(func)),
        f"cyclomatic_complexity_{layer_idx}": cyclomatic_complexity(func),
        f"function_name_consistency_{layer_idx}": check_function_name_consistency(
            row["test_cases"], func
        ),
    }

    length_metrics = code_length(func)
    metrics.update({f"{k}_{layer_idx}": v for k, v in length_metrics.items()})

    comment_metrics = comment_analysis(func)
    metrics.update({f"{k}_{layer_idx}": v for k, v in comment_metrics.items()})

    try:
        h_metrics = h_visit(func)
        metrics.update({f"{k}_{layer_idx}": v for k, v in h_metrics.items()})
    except:
        metrics.update(
            {
                f"{k}_{layer_idx}": np.nan
                for k in [
                    "h1",
                    "h2",
                    "N1",
                    "N2",
                    "vocabulary",
                    "length",
                    "volume",
                    "difficulty",
                    "effort",
                ]
            }
        )

    return metrics


def process_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    new_df = df.copy()

    for layer_idx in ["1", "10", "23", "orig"]:
        new_df[f"response_rate_{layer_idx}"] = (
            new_df[f"extracted_functions_{layer_idx}"].notna().astype(int)
        )

        metrics = new_df.apply(lambda row: calculate_metrics(row, layer_idx), axis=1)
        for metric in metrics.iloc[0].keys():
            new_df[metric] = metrics.apply(lambda x: x[metric])

    return new_df


# Load the original DataFrame
df = pd.read_csv("structured_checkpoint_data.csv")

# Process the DataFrame and add new metrics
new_df = process_dataframe(df)

# Save the new DataFrame
new_df.to_csv("codellm_metrics.csv", index=False)

# Print summary of new metrics
print(new_df.info())

# test_out = """\ndef find_ways(n):\n    values = [4, 2, 1]"""
# test_case = """\nassert find_ways(n) == [4, 2]\n"""
# print(extract_function_name(test_case))
# out = check_function_name_consistency(test_out, test_case)
# print("out: ", out)
