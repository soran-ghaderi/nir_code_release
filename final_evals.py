import re

from transformers import AutoConfig

from datasets import Dataset

from tqdm import tqdm

from typing import List, Dict
import pandas as pd
from collections import defaultdict
import traceback
import configs
from framework_with_template import AdvancedLLaMACRVFramework

from utils import logger
from utils import (
    extract_functions,
    extract_sections,
    add_parsed_functions_to_dataset,
)

from utils.loading_model import CustomTransformerLoader
import torch

# from rich import print
# from rich.console import Console

# Set up logging and console
# console = Console()
logger = logger()

# seed = 42
# set_seed(seed)

model_urls = {
    "llama31": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "llama3": "meta-llama/Meta-Llama-3-8B-Instruct",
}
model_path = model_urls["llama31"]
tokenizer_path = model_path
hf_token = "your token"

config = AutoConfig.from_pretrained(model_path, use_auth_token=hf_token)

# console.rule("[bold red]Loading the Model")

loader = CustomTransformerLoader()
# mp.set_start_method('spawn')
model, tokenizer = loader.load_model(
    model_path=model_path, tokenizer_path=tokenizer_path, hf_token=hf_token
)

crv_layers = configs.CRV_LAYERS

print(":warning: model type: ", type(model))
print("config.hidden_size: ", config.num_hidden_layers)
print("config._attn_implementation: ", config._attn_implementation)

import gc


def clear_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.synchronize()


def move_to_cpu(model):
    model = model.cpu()
    clear_gpu_memory()
    return model


def move_to_gpu(model):
    if torch.cuda.is_available():
        return model.cuda()
    return model


def check_memory(threshold=0.8):
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated()
        memory_reserved = torch.cuda.memory_reserved()
        memory_total = torch.cuda.get_device_properties(0).total_memory
        # print(memory_allocated,memory_reserved,memory_total)
        memory_usage = (memory_allocated + memory_reserved) / memory_total

        if memory_usage > threshold:
            print(memory_allocated, memory_reserved, memory_total, memory_usage)
        return True
    return False


check_memory()
clear_gpu_memory()

import os
import json


def save_checkpoint(
    layer_idx: int,
    instance_index: int,
    results: List[Dict],
    checkpoint_dir: str = "checkpoints",
):
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(
        checkpoint_dir, f"checkpoint_layer_{layer_idx}_instance_{instance_index}.json"
    )

    checkpoint_data = {
        "layer_idx": layer_idx,
        "instance_index": instance_index,
        "results": results,
    }

    with open(checkpoint_path, "w") as f:
        json.dump(checkpoint_data, f)

    print(f"Checkpoint saved at {checkpoint_path}")


def load_checkpoint(checkpoint_dir: str = "checkpoints") -> Dict:
    if os.path.exists(checkpoint_dir):

        checkpoint_files = sorted(
            [f for f in os.listdir(checkpoint_dir) if f.startswith("checkpoint_")]
        )

        if not checkpoint_files:
            return None

        latest_checkpoint = checkpoint_files[-1]
        checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)

        with open(checkpoint_path, "r") as f:
            checkpoint_data = json.load(f)

        checkpoint_data["results"] = defaultdict(list, checkpoint_data["results"])
        print(f"Loaded checkpoint from {checkpoint_path}")
        return checkpoint_data
    else:
        return None


def extract_function_name(text):
    if pd.isna(text):
        return None
    pattern = r"(?:def|class|assert)\s+([\w\d_]+)\s*\("
    match = re.search(pattern, text)
    return match.group(1) if match else None


def check_function_name_consistency(test_cases: str, generated_function: str) -> float:
    test_func_name = extract_function_name(str(test_cases).lower())
    generated_func_name = extract_function_name(str(generated_function).lower())

    if test_func_name in generated_function.lower():
        return (
            float(generated_function.lower().__contains__(test_func_name)),
            test_func_name,
        )
    if test_func_name and generated_func_name:
        # if test_func_name == generated_func_name:
        #     print("yes equal: ", test_func_name, generated_func_name)
        # else:
        # print(
        #     f"not equal: tests; {test_cases}, extracted: {test_func_name}\ngens: {generated_function} extracted: {generated_func_name}"
        # )
        return (
            float(generated_function.lower().__contains__(test_func_name)),
            test_func_name,
        )

    print("were not equal: ", test_func_name, generated_func_name)
    return 0.0, test_func_name


def evaluate_model(
    model,
    tokenizer,
    dataset,
    layer_indices: List[int],
    checkpoint_dir: str = "checkpoints",
    max_retries: int = 2,
) -> pd.DataFrame:
    pr_flag = False
    checkpoint = load_checkpoint(checkpoint_dir)
    max_instances = 0
    dataset_size = len(dataset)

    results = defaultdict(list)

    if checkpoint:
        start_layer_idx = checkpoint["layer_idx"]
        start_instance_index = checkpoint["instance_index"] + 1
        results = defaultdict(list, checkpoint["results"])
        max_instances = len(results["instance_id"])
        # df = pd.read_csv(f"{checkpoint_dir}/latest_results.csv")
    else:
        start_layer_idx = layer_indices[0]
        start_instance_index = 0

    df = dataset
    skipped_dataset = dataset.iloc[start_instance_index:]

    print("layer_indices: ", layer_indices)

    for i, instance in enumerate(
        tqdm(
            skipped_dataset.iterrows(),
            desc=f"Processing instances",
            total=dataset_size - start_instance_index,
        )
    ):
        instance = instance[1]
        for layer_idx in tqdm(
            layer_indices[layer_indices.index(start_layer_idx) :],
            desc=f"Processing layer_idx {i}",
        ):
            framework = AdvancedLLaMACRVFramework(model, tokenizer, layer_idx=layer_idx)

            if start_instance_index >= dataset_size:
                print(
                    f"All instances processed for layer {layer_idx}. Moving to next layer."
                )
                start_instance_index = 0
                continue

            if not pr_flag:
                print(
                    f"start_layer_idx: {start_layer_idx}\nstart_instance_index: {start_instance_index}\ni: {i}\nlayer_idx: {layer_idx}"
                )
            try:
                if (start_instance_index + i + 1) % 5 == 0:
                    # save_checkpoint(
                    #     layer_idx, start_instance_index + i, dict(results), checkpoint_dir
                    # )
                    # Save the dataframe after each row is completed
                    df.to_csv(f"{checkpoint_dir}/latest_results.csv", index=False)
                    save_checkpoint(
                        layer_indices[-1], start_instance_index + i, checkpoint_dir
                    )

                if i % 20 == 0:
                    clear_gpu_memory()

                if check_memory():
                    clear_gpu_memory()
                query = instance["query"][0]
                context = instance["context"][0]
                test_cases = str(instance["extracted_test_cases"])
                if context == "" or query == "":
                    print(
                        f"empty context or query at instance {i}, context: {context}, query: {query}"
                    )
                tmp = "Proposed solution context: "
                extracted_test_func_name = None

                rethink = 0
                for attempt in range(max_retries):
                    try:
                        if rethink == attempt:
                            trajectories_and_context = (
                                framework.generate_thought_trajectories(
                                    query,
                                    context,
                                    test_cases,
                                    max_new_tokens=1000,
                                )
                            )
                        context_expansion = extract_sections(
                            trajectories_and_context, "context_generation"
                        )
                        context_expansion = f"Proposed solution context: {context_expansion}{extract_sections(trajectories_and_context, 'solution')}"

                        hidden_states, seq_len = framework.extract_hidden_states(
                            context_expansion
                        )
                        crv, seq_len = framework.generate_crv(hidden_states, seq_len)

                        final_output = framework.final_generation(
                            query,
                            test_cases,
                            crv,
                            seq_len,
                            max_new_tokens=250,
                            new_feedback=extracted_test_func_name,
                        )
                        print(
                            f"instance {i}, layer {layer_idx}, attempt {attempt+1}/{max_retries}"
                        )
                        print(
                            f"lenghts: trajectories_and_context, {len(trajectories_and_context)}, context_expansion: {len(context_expansion)}, final_output: {len(final_output)}"
                        )
                        is_equal, extracted_test_func_name = (
                            check_function_name_consistency(test_cases, final_output)
                        )
                        if not is_equal:
                            print("did not exist in test sets")
                            print(f"{final_output}, {test_cases}")

                            trajectory_is_equal, extracted_test_func_name = (
                                check_function_name_consistency(
                                    test_cases, trajectories_and_context
                                )
                            )
                            if not trajectory_is_equal:
                                rethink = attempt

                            if attempt < max_retries - 1:
                                continue

                        if len(final_output) < 10:
                            print("output doesn't seem to be correct")
                            if attempt < max_retries - 1:
                                continue

                        extracted_functions = extract_functions(final_output)
                        df.at[i, f"final_output_{layer_idx}"] = final_output
                        df.at[i, f"extracted_functions_{layer_idx}"] = (
                            extracted_functions
                        )
                        df.at[i, f"trajectories_and_context_{layer_idx}"] = (
                            trajectories_and_context
                        )
                        df.at[i, f"context_expansion_{layer_idx}"] = context_expansion
                        df.to_csv(f"{checkpoint_dir}/latest_results.csv", index=False)
                        print(f"saved instance {i}, layer {layer_idx}")
                        # generation_successful = True
                        break
                    except Exception as e:
                        logger.error(
                            f"Error processing example {i} for layer {layer_idx}: {str(e)}"
                        )
                        df.at[i, f"final_output_{layer_idx}"] = str(e)
                        df.at[i, f"extracted_functions_{layer_idx}"] = str(e)
                        df.at[i, f"trajectories_and_context_{layer_idx}"] = str(e)
                        df.at[i, f"context_expansion_{layer_idx}"] = str(e)
                        logger.warning(
                            f"Generation failed for layer {layer_idx}, instance {i}, attempt {attempt + 1}: {str(e)}\nquery: {query}"
                        )
                        if attempt == max_retries - 1:
                            logger.error(
                                f"All retries failed for layer {layer_idx}, instance {i}. Error: {traceback.format_exc()}"
                            )

                        df.to_csv(f"{checkpoint_dir}/latest_results.csv", index=False)

                # trajectories_and_context = framework.generate_thought_trajectories(query, context, test_cases, max_new_tokens=1000, alt_text=alt_text)
                # context_expansion = extract_sections(trajectories_and_context, "context_generation")
                # context_expansion = tmp + context_expansion + extract_sections(trajectories_and_context, "solution")

                # hidden_states, seq_len = framework.extract_hidden_states(context_expansion)
                # crv, seq_len = framework.generate_crv(hidden_states, seq_len)

                # final_output = framework.final_generation(query, test_cases, crv, seq_len, max_new_tokens=250)
                # extracted_functions = extract_functions(final_output)

                # df.at[i, f"final_output_{layer_idx}"] = final_output
                # # df.at[i, f"extracted_functions_{layer_idx}"] = extracted_functions
                # df.at[i, f"trajectories_and_context_{layer_idx}"] = trajectories_and_context
                # df.at[i, f"context_expansion_{layer_idx}"] = context_expansion

                # current_instance = i + start_instance_index
                # if current_instance >= max_instances:
                #     results["instance_id"].append(current_instance)
                #     results["query"].append(query)
                #     results["context"].append(context)
                #     results["test_cases"].append(test_cases)
                #     max_instances = current_instance + 1

                # results[f"final_output_{layer_idx}"].append(final_output)
                # # results[f"extracted_functions_{layer_idx}"].append(extracted_functions)
                # results[f"trajectories_and_context_{layer_idx}"].append(
                #     trajectories_and_context
                # )
                # results[f"context_expansion_{layer_idx}"].append(context_expansion)

                if not pr_flag:
                    print(f"results dict for instance i: {i}")
                    pr_flag = True

            except Exception as e:
                logger.error(
                    f"Error processing example {i} for layer {layer_idx}: {str(e)}"
                )
                df.at[i, f"final_output_{layer_idx}"] = str(e)
                df.at[i, f"extracted_functions_{layer_idx}"] = str(e)
                df.at[i, f"trajectories_and_context_{layer_idx}"] = str(e)
                df.at[i, f"context_expansion_{layer_idx}"] = str(e)
                print(f"Error processing example {start_instance_index + i}: {str(e)}")
                save_checkpoint(
                    layer_idx, start_instance_index + i, None, checkpoint_dir
                )
                raise

        # start_instance_index = 0

    # Ensure all arrays have the same length
    # max_length = max(len(v) for v in results.values())
    # for key in results:
    #     results[key] = results[key] + [None] * (max_length - len(results[key]))

    df = pd.DataFrame(dict(results))  # Convert defaultdict to regular dict
    if "instance_id" in df.columns:
        df.set_index("instance_id", inplace=True)

    # Convert results to DataFrame
    # df = pd.DataFrame(results)
    # df.set_index('instance_id', inplace=True)
    return df


def evaluate_model2(
    model,
    tokenizer,
    dataset: Dataset,
    layer_indices: List[int],
    checkpoint_dir: str = "checkpoints",
) -> pd.DataFrame:
    df = pd.DataFrame()
    dataset_size = len(dataset)
    checkpoint = load_checkpoint(checkpoint_dir)

    if checkpoint:
        start_instance_index = checkpoint["instance_index"] + 1
        df = pd.read_csv(f"{checkpoint_dir}/latest_results.csv")
    else:
        start_instance_index = 0

    for i, instance in enumerate(
        tqdm(
            dataset.skip(start_instance_index),
            desc="Processing instances",
            total=dataset_size - start_instance_index,
        )
    ):
        row = {
            "instance_id": start_instance_index + i,
            "query": instance["query"][0] if instance["query"] else "",
            "context": instance["context"][0] if instance["context"] else "",
            "test_cases": instance["context"],
        }

        for layer_idx in layer_indices:
            framework = AdvancedLLaMACRVFramework(model, tokenizer, layer_idx=layer_idx)
            print(f"instace {i}, layer_idx: {layer_idx}")
            try:
                trajectories_and_context = framework.generate_thought_trajectories(
                    row["query"],
                    row["context"],
                    row["test_cases"],
                    max_new_tokens=1000,
                    alt_text=None,
                )
                context_expansion = extract_sections(
                    trajectories_and_context, "context_generation"
                )
                context_expansion = f"Proposed solution context: {context_expansion}{extract_sections(trajectories_and_context, 'solution')}"

                if context_expansion is None:
                    print(
                        f"instance {i}, layer idx {layer_idx} - no context extracted. the trajectories_and_context: {trajectories_and_context}"
                    )
                hidden_states, seq_len = framework.extract_hidden_states(
                    context_expansion
                )
                crv, seq_len = framework.generate_crv(hidden_states, seq_len)

                final_output = framework.final_generation(
                    row["query"], row["test_cases"], crv, seq_len, max_new_tokens=250
                )
                # extracted_functions = extract_functions(final_output)

                df.at[i, f"final_output_{layer_idx}"] = final_output
                # df.at[i, f"extracted_functions_{layer_idx}"] = (
                #     extracted_functions
                # )
                df.at[i, f"trajectories_and_context_{layer_idx}"] = (
                    trajectories_and_context
                )
                df.at[i, f"context_expansion_{layer_idx}"] = context_expansion

            except Exception as e:
                logger.error(
                    f"Error processing example {i} for layer {layer_idx}: {str(e)}"
                )
                df.at[i, f"final_output_{layer_idx}"] = str(e)
                # df.at[i, f"extracted_functions_{layer_idx}"] = str(e)
                df.at[i, f"trajectories_and_context_{layer_idx}"] = str(e)
                df.at[i, f"context_expansion_{layer_idx}"] = str(e)

        # df = df.append(row, ignore_index=True)

        # Save the dataframe after each row is completed
        df.to_csv(f"{checkpoint_dir}/latest_results.csv", index=False)
        save_checkpoint(layer_indices[-1], start_instance_index + i, checkpoint_dir)

    return df


subset_name = "processed_Meta-Llama-3.1-8B-Instruct-evals__mbpp__details"
# loaded_dataset = load_from_disk(f"data/{subset_name}")
loaded_dataset = pd.read_csv("verified_df_with_test_cases.csv")

# num_examples = 250
# if num_examples is not None:
#     loaded_dataset = loaded_dataset.select(range(num_examples))

# Define layer indices to evaluate
layer_indices = [1, 10, 23, "orig"]
# layer_indices = [32, 'orig']

checkpoint = load_checkpoint()
start_index = 0
results_df = pd.DataFrame()

if checkpoint:
    start_index = checkpoint["instance_index"]
    results_df = checkpoint["results"]
    print(f"Resuming from example {start_index}")

# Evaluate model
# new_results_df = evaluate_model(model, tokenizer, loaded_dataset, layer_indices)
new_results_df = evaluate_model(model, tokenizer, loaded_dataset, layer_indices)
results_df.to_csv("final_results.csv", index=False)
print("Evaluation completed and final results saved.")

print("new_results_df info:")
print(new_results_df.info())
print("\nnew_results_df shape:", new_results_df.shape)
print("\nnew_results_df columns:", new_results_df.columns)
print("\nnew_results_df head:")
print(new_results_df.head())

# Combine previous results with new results
results_df = pd.concat([results_df, new_results_df], ignore_index=True)

# Add parsed functions to the dataset
updated_dataset = add_parsed_functions_to_dataset(
    loaded_dataset, results_df, layer_indices
)

print(f"Type of updated_dataset: {type(updated_dataset)}")
print(f"Number of rows in updated_dataset: {len(updated_dataset)}")

# Save the updated dataset
updated_dataset.save_to_disk(f"data/{subset_name}_results")

# os.remove(os.path.join("checkpoints", "checkpoint.pkl"))
