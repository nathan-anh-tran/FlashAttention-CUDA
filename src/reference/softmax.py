import torch

# Mix of normal-range values plus two large, close-together values (1000.0, 1000.5)
# to prove the subtract-the-max trick matters: exp(1000) alone overflows float32,
# so a naive (non-stable) softmax would produce NaN on this input.
x = torch.tensor([1.0, 2.0, 3.0, 1000.0, 5.0, -2.0, 1000.5, 0.5])

result = torch.softmax(x, dim=0)

print("Input:", x.tolist())
print("Softmax output:")
for i, val in enumerate(result.tolist()):
    print(f"  [{i}] = {val:.8f}f")
print("Sum (should be 1.0):", result.sum().item())
