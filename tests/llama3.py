import torch
from transformers.models.llama.modeling_llama import LlamaRMSNorm

from main import RMSNorm


def test_rmsnorm():
    # Create dummy input data
    input_data = torch.randn(4, 8)  # Batch size of 4, input dimension of 8

    # Instantiate the RMSNorm class
    rmsnorm = RMSNorm(dim=8, eps=1e-6)

    # Pass the input data through the RMSNorm layer
    output_data = rmsnorm(input_data)

    # Print input and output data
    print("Input Data:")
    print(input_data)
    print("\nOutput Data:")
    print(output_data)

def test_inheritance():
    rmsnorm = RMSNorm(dim=8, eps=1e-6)

    # Check if RMSNorm is a subclass of LlamaRMSNorm
    is_subclass = issubclass(RMSNorm, LlamaRMSNorm)
    # Check if the instance is an instance of LlamaRMSNorm
    is_instance = isinstance(rmsnorm, LlamaRMSNorm)

    # Print results
    print(f"RMSNorm is a subclass of LlamaRMSNorm: {is_subclass}")
    print(f"rmsnorm is an instance of LlamaRMSNorm: {is_instance}")

    # Run a forward pass to ensure functionality
    input_data = torch.randn(4, 8)  # Batch size of 4, input dimension of 8
    output_data = rmsnorm(input_data)
    print("Input Data:")
    print(input_data)
    print("\nOutput Data:")
    print(output_data)




# Run the test function
if __name__ == "__main__":
    test_rmsnorm()
    test_inheritance()