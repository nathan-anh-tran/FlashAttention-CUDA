import torch

# Row-wise softmax over a small matrix (3 rows x 8 cols), one row per test case:
#   row 0: mix of normal values + two large, close-together values (1000.0, 1000.5)
#          to prove the subtract-the-max trick matters: exp(1000) alone overflows
#          float32, so a naive (non-stable) softmax would produce NaN here.
#   row 1: plain increasing values — normal-range sanity check.
#   row 2: all zeros — every output should be exactly 1/8 = 0.125, an easy
#          hand-checkable case independent of the overflow story.
x = torch.tensor([
    [1.0, 2.0, 3.0, 1000.0, 5.0, -2.0, 1000.5, 0.5],
    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
])

result = torch.softmax(x, dim=1)  # softmax across columns, independently per row

print("Input:")
print(x.tolist())
print("Softmax output:")
for r in range(result.shape[0]):
    print(f"row {r}:")
    for c, val in enumerate(result[r].tolist()):
        print(f"  [{c}] = {val:.8f}f")
    print(f"  row sum (should be 1.0): {result[r].sum().item()}")
