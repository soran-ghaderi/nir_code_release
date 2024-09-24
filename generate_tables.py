import pandas as pd
import numpy as np

# Load the DataFrame
df = pd.read_csv("codellm_metrics.csv")


def generate_latex_table(data, caption, label):
    latex = f"\\begin{{table}}[h]\n\\centering\n\\caption{{{caption}}}\n\\label{{{label}}}\n"
    latex += "\\begin{tabular}{l" + "c" * len(data.columns) + "}\n\\hline\n"
    latex += "Metric & " + " & ".join(data.columns) + " \\\\\n\\hline\n"
    for index, row in data.iterrows():
        latex += (
            f"{index} & "
            + " & ".join([f"{x:.4f}" if isinstance(x, float) else str(x) for x in row])
            + " \\\\\n"
        )
    latex += "\\hline\n\\end{tabular}\n\\end{table}\n"
    return latex


# Response Rates
response_rates = df[[f"response_rate_{i}" for i in ["1", "10", "23", "orig"]]].mean()
response_rates_data = pd.DataFrame(
    {
        "Layer 1": [response_rates["response_rate_1"], 249],
        "Layer 10": [response_rates["response_rate_10"], 249],
        "Layer 23": [response_rates["response_rate_23"], 249],
        "Original": [response_rates["response_rate_orig"], 249],
    },
    index=["Response Rate", "Sample Size"],
)
response_rates_latex = generate_latex_table(
    response_rates_data, "Response Rates Across Different Layers", "tab:response_rates"
)

# Code Quality Metrics
code_quality_data = pd.DataFrame(
    {
        "Layer 1": [
            df["syntactic_correctness_1"].mean(),
            df["function_name_consistency_1"].mean(),
        ],
        "Layer 10": [
            df["syntactic_correctness_10"].mean(),
            df["function_name_consistency_10"].mean(),
        ],
        "Layer 23": [
            df["syntactic_correctness_23"].mean(),
            df["function_name_consistency_23"].mean(),
        ],
        "Original": [
            df["syntactic_correctness_orig"].mean(),
            df["function_name_consistency_orig"].mean(),
        ],
    },
    index=["Syntactic Correctness", "Function Name Consistency"],
)
code_quality_latex = generate_latex_table(
    code_quality_data,
    "Code Quality Metrics Across Different Layers",
    "tab:code_quality",
)

# Code Complexity Metrics
code_complexity_data = pd.DataFrame(
    {
        "Layer 1": [df["cyclomatic_complexity_1"].mean()],
        "Layer 10": [df["cyclomatic_complexity_10"].mean()],
        "Layer 23": [df["cyclomatic_complexity_23"].mean()],
        "Original": [df["cyclomatic_complexity_orig"].mean()],
    },
    index=["Cyclomatic Complexity"],
)
code_complexity_latex = generate_latex_table(
    code_complexity_data,
    "Code Complexity Metrics Across Different Layers",
    "tab:code_complexity",
)

# Code Structure Metrics
code_structure_data = pd.DataFrame(
    {
        "Layer 1": [
            df["lines_1"].mean(),
            df["characters_1"].mean(),
            df["comment_lines_1"].mean(),
            df["comment_ratio_1"].mean(),
        ],
        "Layer 10": [
            df["lines_10"].mean(),
            df["characters_10"].mean(),
            df["comment_lines_10"].mean(),
            df["comment_ratio_10"].mean(),
        ],
        "Layer 23": [
            df["lines_23"].mean(),
            df["characters_23"].mean(),
            df["comment_lines_23"].mean(),
            df["comment_ratio_23"].mean(),
        ],
        "Original": [
            df["lines_orig"].mean(),
            df["characters_orig"].mean(),
            df["comment_lines_orig"].mean(),
            df["comment_ratio_orig"].mean(),
        ],
    },
    index=["Lines of Code", "Characters", "Comment Lines", "Comment Ratio"],
)
code_structure_latex = generate_latex_table(
    code_structure_data,
    "Code Structure Metrics Across Different Layers",
    "tab:code_structure",
)

# Print LaTeX code for all tables
print(response_rates_latex)
print(code_quality_latex)
print(code_complexity_latex)
print(code_structure_latex)
