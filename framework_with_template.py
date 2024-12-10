import configs
from experimental.controller.memory_manager import MemoryManager
from generator.crv_generator import CRVGenerator
from generator.text_generator import TextGenerator

crv_layers = configs.CRV_LAYERS


class AdvancedLLaMACRVFramework:
    def __init__(self, model, tokenizer, layer_idx=10):
        self.model = model
        self.tokenizer = tokenizer
        self.text_generator = TextGenerator(model, tokenizer)
        self.crv_generator = CRVGenerator(
            model, tokenizer, max_length=configs.MAX_LENGTH
        )
        self.memory_manager = MemoryManager(model, max_memories=5)
        self.layer_idx = layer_idx
        self.device = next(model.parameters()).device
        self.text_generator = TextGenerator(model, tokenizer, device=self.device)

    def generate_thought_trajectories(
        self,
        input_query,
        context=None,
        test_cases=None,
        max_new_tokens=1000,
        alt_text=None,
    ):

        if context is None:
            print(
                f"from the generate_thought_trajectories func: \n"
                f"the context is {context} \nfor the input query: {input_query}"
            )
        prompt_template = f"""
        <|begin_of_text|><|start_header_id|>system<|end_header_id|>
        Enable code_interpreter tool.<|eot_id|>
        \n\n
        Your proposed code and names must be consistent with the test cases and pass the following test cases: {test_cases}.

        \n\nYour outputs must follow this structure and make sure you open and close the tags accurately:

        Identify the core components of this problem.
        1. Identify potential edge cases and tricky parts.
        2. Write 2 short test cases for the edge cases and tricky parts.

        <chain_of_thoughts>
        1. you must consider the edge cases according to the problem statement.
        2. Begin with a <thinking> section.
        3. Inside the thinking section:
           a. Write the topic name of the query, the name of the algorithm if necessary.
           b. Draft an answer as an expert.
           b. Briefly analyze the question and outline your approach.
           c. Present a clear plan of steps to solve the problem.
           d. Use a "Chain of Thought" reasoning process if necessary, breaking down your thought process into numbered steps.
        4. Include a <reflection> section for each idea where you:
           a. Review your reasoning.
           b. Check for potential errors or oversights.
           c. Confirm or adjust your conclusion if necessary.
        5. Be sure to close all reflection sections.
        6. Close the thinking section with </thinking>.
        7. Provide your final answer in an <output> section.        
        </chain_of_thoughts>

        <chain_of_thought_selection>
        you must consider the edge cases according to the problem statement and select the most promising chain of thought that solves the edge cases (not necessarily the simplest nor the standard approach).
        </chain_of_thought_selection>

        <solution>
        1. As a Python expert, generate the Python code and make sure it solves the edge cases while keeping it efficient.
        2. the internal steps must produce the required output.
        </solution>

        Include a <reflection> section for the selected solution where if it is not correct, modify or if necessary, rewrite the solution and pay attention to the input problem.
           a. Review your reasoning.
           b. Check for potential errors or oversights according to the problem. you must consider the edge cases according to the problem. Make sure it is not overcomplicated.
           c. Confirm or adjust your conclusion if necessary.
        4. Be sure to close all reflection sections.

        <context_generation>
        1. Rewrite the problem.
        2. Rewrite the edge cases and tricky parts in one short sentence
        2. Generate a very accurate and minimal Python code/pseudocode for the final solution. Ensure that the final solution is minimal and accurate.
        </context_generation>
        
        <|eot_id|>
        <|start_header_id|>assistant<|end_header_id|>\n\n"
        """

        generated_text = self.text_generator.generate_text(
            prompt_template,
            max_new_tokens=max_new_tokens,
            num_return_sequences=1,
            output_file="data/results.csv",
            # stop_sequences=["The end", ".\n\n"],
        )
        # print("generated_thought trajectory: ", generated_text)

        return generated_text

    def extract_hidden_states(self, context):
        best_crv, seq_length = self.crv_generator.generate_crvs(
            context, crv_layers=crv_layers, max_length=configs.MAX_LENGTH
        )
        return best_crv, seq_length  # Return the hidden state and its len

    def generate_crv(self, hidden_states, seq_length):
        # return torch.mean(hidden_states, dim=1)
        return hidden_states, seq_length

    def final_generation(
        self,
        original_query,
        test_cases,
        crv,
        seq_length,
        max_new_tokens=250,
        new_feedback=None,
    ):
        function_name = ""
        if not (new_feedback is None) and not (self.layer_idx == "orig"):
            function_name = "def" + new_feedback + "("

        query = f"""<|start_header_id|>user<|end_header_id|>\n\nYou are an expert Python programmer who writes accurate code; here is your task:\n{original_query}.\nYour code passes the following tests:{test_cases}"<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n```python"""
        # Combine original query and CRV
        self.memory_manager.add_memory(
            crv, seq_length, layer_idx=self.layer_idx, crv_layers=crv_layers
        )

        # model.model.set_post_concat_crv(True)
        self.memory_manager.set_concat_positions(0, start_pos=0, end_pos=seq_length)
        if isinstance(self.layer_idx, int):
            self.memory_manager.apply_memory_to_model(0)
        generated_text = self.text_generator.generate_text(
            query,
            max_new_tokens=max_new_tokens,
            num_return_sequences=1,
            output_file="data/results.csv",
            # stop_sequences=["The end", ".\n\n"],
        )
        return generated_text
