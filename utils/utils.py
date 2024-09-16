import logging
import random
import re
from typing import List

import numpy as np
import pandas as pd
import torch
from rich.logging import RichHandler
from datasets import Dataset

import configs


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def logger():
    if configs.USE_RICH:
        logging.basicConfig(
            level=configs.logging_level,
            format="%(message)s",
            datefmt="[%X]",
            handlers=[RichHandler(rich_tracebacks=True)],
        )
        logger = logging.getLogger("rich")
    else:

        logger = logging.getLogger(__name__)
        logging.basicConfig(level=configs.logging_level)
    return logger


def extract_test_cases(text):
    # Pattern to match assert statements
    pattern = r"assert\s+[\w_]+\(.*?\).*?(?=[\n<]|$)"

    # Find all matches
    test_cases = re.findall(pattern, text)

    # Group test cases by task
    grouped_tests = []
    current_group = []

    for test in test_cases:
        if current_group and not test.startswith(current_group[-1].split("(")[0]):
            grouped_tests.append(current_group)
            current_group = []
        current_group.append(test)

    if current_group:
        grouped_tests.append(current_group)
        # print("test cases len: ", len(grouped_tests))
        logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.ERROR)
        logger.info(f"test cases len: {len(grouped_tests)}")

    return grouped_tests


# text = '''<|start_header_id|>user<|end_header_id|>\n\nYou are an expert Python programmer, and here is your task:\nWrite a function to find the similar elements from the given two tuple lists.\nYour code should pass the following tests:\nassert similar_elements((3, 4, 5, 6),(5, 7, 4, 10)) == (4, 5)\nassert similar_elements((1, 2, 3, 4),(5, 4, 3, 7)) == (3, 4)\nassert similar_elements((11, 12, 14, 13),(17, 15, 14, 13)) == (13, 14)<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n```python\ndef similar_elements(test_tup1, test_tup2):\n res = tuple(set(test_tup1) & set(test_tup2))\n return (res) \n```<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nYou are an expert Python programmer, and here is your task:\nWrite a python function to identify non-prime numbers.\nYour code should pass the following tests:\nassert is_not_prime(2) == False\nassert is_not_prime(10) == True\nassert is_not_prime(35) == True<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n```python\nimport math\ndef is_not_prime(n):\n result = False\n for i in range(2,int(math.sqrt(n)) + 1):\n if n % i == 0:\n result = True\n return result\n```<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nYou are an expert Python programmer, and here is your task:\nWrite a function to find the largest integers from a given list of numbers using heap queue algorithm.\nYour code should pass the following tests:\nassert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],3)==[85, 75, 65] \nassert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],2)==[85, 75] \nassert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],5)==[85, 75, 65, 58, 35]<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n```python\nimport heapq as hq\ndef heap_queue_largest(nums,n):\n largest_nums = hq.nlargest(n, nums)\n return largest_nums\n```<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nYou are an expert Python programmer, and here is your task:\nWrite a function to create the next bigger number by rearranging the digits of a given number.\nYour code should pass the following tests:\nassert rearrange_bigger(12)==21\nassert rearrange_bigger(10)==False\nassert rearrange_bigger(102)==120<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n```python"'''
# out = extract_test_cases(text)
# print(out[-1])


def extract_functions(text):
    # Extract imports
    import_pattern = r"^(?:from\s+[\w.]+\s+import\s+(?:[\w.]+(?:\s*,\s*[\w.]+)*|\*)|import\s+(?:[\w.]+(?:\s*,\s*[\w.]+)*))(?:\s+as\s+[\w.]+)?"
    imports = re.findall(import_pattern, text, re.MULTILINE)

    # Extract functions
    function_pattern = r"(def\s+\w+\s*\(.*?\):(?:\s*['\"][\s\S]*?['\"])?\s*(?:(?!def\s)[\s\S])*?(?=\ndef|\Z))"
    functions = re.findall(function_pattern, text, re.MULTILINE | re.DOTALL)

    def clean_code(code):
        # Remove docstrings
        code = re.sub(r'"""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\'', "", code)
        # Remove comments
        code = re.sub(r"#.*", "", code)
        # Remove empty lines and trailing whitespace
        code = "\n".join(line for line in code.splitlines() if line.strip())
        return code

    cleaned_imports = [clean_code(imp) for imp in imports]
    cleaned_functions = [clean_code(func) for func in functions]

    # Combine imports and functions
    cleaned_code = "\n".join(cleaned_imports)
    if cleaned_imports and cleaned_functions:
        cleaned_code += "\n\n"
    cleaned_code += "\n\n".join(cleaned_functions)

    return cleaned_code


def extract_sections(text, tag):
    pattern = f"<{tag}>(.*?)</{tag}>"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        if text:
            return extract_functions(text)
        return f"Context expansion section not found. The original text: {text}"


def add_parsed_functions_to_dataset(
    dataset: Dataset, results_df: pd.DataFrame, layer_indices: List[int]
) -> Dataset:
    # Convert results DataFrame to a dictionary
    results_dict = results_df.to_dict("records")

    # Create a dictionary to store new columns
    new_columns = {
        f"final_output_layer_{layer}": [None] * len(dataset) for layer in layer_indices
    }
    new_columns.update(
        {
            f"extracted_functions_layer_{layer}": [None] * len(dataset)
            for layer in layer_indices
        }
    )

    # Populate new columns
    for result in results_dict:
        instance_id = result["instance_id"]
        layer = result["layer_idx"]
        if 0 <= instance_id < len(dataset):
            new_columns[f"final_output_layer_{layer}"][instance_id] = result[
                "final_output"
            ]
            new_columns[f"extracted_functions_layer_{layer}"][instance_id] = result[
                "extracted_functions"
            ]

    # Create a new dataset with only the new columns
    new_dataset = Dataset.from_dict(new_columns)

    # Combine the original dataset with the new dataset
    updated_dataset = concatenate_datasets([dataset, new_dataset], axis=1)

    return updated_dataset


from datasets import Dataset, concatenate_datasets


def add_parsed_functions_to_dataset2(
    dataset: Dataset, results_df: pd.DataFrame, layer_indices: List[int]
) -> Dataset:
    # Convert results DataFrame to a dictionary
    results_dict = results_df.to_dict("records")

    # Create a dictionary to store new columns
    new_columns = {
        f"final_output_layer_{layer}": [None] * len(dataset) for layer in layer_indices
    }
    new_columns.update(
        {
            f"extracted_functions_layer_{layer}": [None] * len(dataset)
            for layer in layer_indices
        }
    )

    # Populate new columns
    for result in results_dict:
        instance_id = result["instance_id"]
        layer = result["layer_idx"]
        if 0 <= instance_id < len(dataset):
            new_columns[f"final_output_layer_{layer}"][instance_id] = result[
                "final_output"
            ]
            new_columns[f"extracted_functions_layer_{layer}"][instance_id] = result[
                "extracted_functions"
            ]

    # Create a new dataset with only the new columns
    new_dataset = Dataset.from_dict(new_columns)

    # Combine the original dataset with the new dataset
    updated_dataset = concatenate_datasets([dataset, new_dataset], axis=1)

    return updated_dataset
