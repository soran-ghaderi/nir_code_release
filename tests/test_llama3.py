import pytest
import torch
from transformers.models.llama.modeling_llama import LlamaRMSNorm

from main import RMSNorm, precompute_freqs_cis, reshape_for_broadcast, apply_rotary_emb


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

def test_reshape_for_broadcast():
    # Test case 1: 3D tensor
    x1 = torch.randn(2, 4, 6)  # Shape: [batch_size, sequence_length, dim]
    freqs_cis1 = torch.randn(4, 6)  # Shape: [sequence_length, dim]
    reshaped1 = reshape_for_broadcast(freqs_cis1, x1)
    assert reshaped1.shape == (1, 4, 6), f"Expected shape (1, 4, 1, 6), but got {reshaped1.shape}"

    # Test case 2: 4D tensor
    x2 = torch.randn(2, 4, 5, 6)  # Shape: [batch_size, sequence_length, channels, dim]
    freqs_cis2 = torch.randn(4, 6)  # Shape: [sequence_length, dim]
    reshaped2 = reshape_for_broadcast(freqs_cis2, x2)
    assert reshaped2.shape == (1, 4, 1, 6), f"Expected shape (1, 4, 1, 1, 6), but got {reshaped2.shape}"

    print("All tests passed.")


# def test_apply_rotary_emb():
#     # Test case: 3D tensors
#     batch_size = 2
#     seq_len = 4
#     dim = 6
#     xq = torch.randn(batch_size, seq_len, dim)
#     xk = torch.randn(batch_size, seq_len, dim)
#     freqs_cis = torch.polar(torch.ones(seq_len, dim // 2),
#                             torch.arange(seq_len * (dim // 2)).reshape(seq_len, dim // 2).float())
#
#     xq_out, xk_out = apply_rotary_emb(xq, xk, freqs_cis)
#
#     assert xq_out.shape == xq.shape, f"Expected xq_out shape {xq.shape}, but got {xq_out.shape}"
#     assert xk_out.shape == xk.shape, f"Expected xk_out shape {xk.shape}, but got {xk_out.shape}"
#
#     # Verify that the rotary embeddings are applied correctly
#     # Check if the output tensors have similar properties to the input
#     assert torch.allclose(xq_out.mean(), xq.mean(), atol=1e-5), "Mean of xq_out differs from xq"
#     assert torch.allclose(xk_out.mean(), xk.mean(), atol=1e-5), "Mean of xk_out differs from xk"
#     assert torch.allclose(xq_out.std(), xq.std(), atol=1e-5), "Std of xq_out differs from xq"
#     assert torch.allclose(xk_out.std(), xk.std(), atol=1e-5), "Std of xk_out differs from xk"
#
#     print("All tests passed.")

@pytest.mark.parametrize("batch_size, seq_len, n_heads, d_head", [
    (1, 10, 4, 64),
    (2, 20, 8, 32),
    (4, 15, 6, 48),
])
def test_apply_rotary_emb(batch_size, seq_len, n_heads, d_head):
    # Arrange
    xq = torch.randn(batch_size, seq_len, n_heads, d_head)
    xk = torch.randn(batch_size, seq_len, n_heads, d_head)
    freqs_cis = torch.randn(seq_len, d_head // 2, dtype=torch.cfloat)

    # Act
    xq_out, xk_out = apply_rotary_emb(xq, xk, freqs_cis)

    # Assert
    assert xq_out.shape == xq.shape
    assert xk_out.shape == xk.shape
    assert xq_out.dtype == xq.dtype
    assert xk_out.dtype == xk.dtype

    # Check if the output is different from the input
    assert not torch.allclose(xq_out, xq)
    assert not torch.allclose(xk_out, xk)

    # Check if the output is consistent
    xq_out2, xk_out2 = apply_rotary_emb(xq, xk, freqs_cis)
    assert torch.allclose(xq_out, xq_out2)
    assert torch.allclose(xk_out, xk_out2)

@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_apply_rotary_emb_dtype(dtype):
    # Arrange
    batch_size, seq_len, n_heads, d_head = 2, 10, 4, 64
    xq = torch.randn(batch_size, seq_len, n_heads, d_head, dtype=dtype)
    xk = torch.randn(batch_size, seq_len, n_heads, d_head, dtype=dtype)
    freqs_cis = torch.randn(seq_len, d_head // 2, dtype=torch.cfloat)

    # Act
    xq_out, xk_out = apply_rotary_emb(xq, xk, freqs_cis)

    # Assert
    assert xq_out.dtype == dtype
    assert xk_out.dtype == dtype

def test_apply_rotary_emb_edge_cases():
    # Test with minimal dimensions
    xq = torch.randn(1, 1, 1, 2)
    xk = torch.randn(1, 1, 1, 2)
    freqs_cis = torch.randn(1, 1, dtype=torch.cfloat)
    xq_out, xk_out = apply_rotary_emb(xq, xk, freqs_cis)
    assert xq_out.shape == xq.shape
    assert xk_out.shape == xk.shape

    # Test with odd d_head (should raise an error)
    xq = torch.randn(1, 10, 4, 63)
    xk = torch.randn(1, 10, 4, 63)
    freqs_cis = torch.randn(10, 31, dtype=torch.cfloat)
    with pytest.raises(RuntimeError):
        apply_rotary_emb(xq, xk, freqs_cis)

def test_apply_rotary_emb_numerical():
    # Arrange
    xq = torch.tensor([[[[1.0, 2.0, 3.0, 4.0]]]])
    xk = torch.tensor([[[[5.0, 6.0, 7.0, 8.0]]]])
    freqs_cis = torch.tensor([[1.0 + 1j, 1.0 + 1j]], dtype=torch.cfloat)

    # Act
    xq_out, xk_out = apply_rotary_emb(xq, xk, freqs_cis)

    # Assert
    expected_xq = torch.tensor([[[[-1.0, 3.0, -1.0, 7.0]]]])
    expected_xk = torch.tensor([[[[-1.0, 11.0, -1.0, 15.0]]]])
    assert torch.allclose(xq_out, expected_xq, atol=1e-6)
    assert torch.allclose(xk_out, expected_xk, atol=1e-6)
# Run the test function
# if __name__ == "__main__":
    # test_rmsnorm()
    # test_inheritance()
    # test_precompute_freqs_cis()
    # test_reshape_for_broadcast()
    # test_apply_rotary_emb()

