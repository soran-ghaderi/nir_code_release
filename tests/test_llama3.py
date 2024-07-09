import torch
from transformers.models.llama.modeling_llama import LlamaRMSNorm

from main import RMSNorm, precompute_freqs_cis


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


def test_precompute_freqs_cis():
    # Test parameters
    dim = 16
    end = 10
    theta = 10000.0

    # Run the function
    result = precompute_freqs_cis(dim, end, theta)

    # Check the shape of the output
    expected_shape = (end, dim // 2)
    assert result.shape == expected_shape, f"Expected shape {expected_shape}, but got {result.shape}"

    # Check the dtype of the output
    assert result.dtype == torch.complex64, f"Expected dtype torch.complex64, but got {result.dtype}"

    # Check some basic value properties
    print("Are all magnitudes close to 1?", torch.allclose(result.abs(), torch.ones_like(result.abs()), atol=1e-5))

    assert torch.allclose(result.abs(), torch.ones_like(result.abs()), atol=1e-5), "All magnitudes should be 1"

    # Additional checks can be added here
    print("All tests passed!")
    print(result)
    return result




# Run the test function
if __name__ == "__main__":
    # test_rmsnorm()
    # test_inheritance()
    test_precompute_freqs_cis()