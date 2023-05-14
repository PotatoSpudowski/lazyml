from typing import List, Optional
from lazyml import LazyBuffer, Tensor

a = LazyBuffer.fromCPU(Tensor([[1, 2, 3, 4, 5], [11, 12, 13, 14, 15], [21, 22, 23, 24, 25], [31, 32, 33, 34, 35], [41, 42, 43, 44, 45]], (5, 5)))
b = LazyBuffer.fromCPU(Tensor([[6, 7, 8, 9, 10], [16, 17, 18, 19, 20], [26, 27, 28, 29, 30], [36, 37, 38, 39, 40], [46, 47, 48, 49, 50]], (5, 5)))

# Add the two LazyBuffer instances
add_result = a.binary_op(b, Tensor.__add__)
print("Addition result:", add_result.toCPU().data)

# Multiply the two LazyBuffer instances
mul_result = a.binary_op(b, Tensor.__mul__)
print("Multiplication result:", mul_result.toCPU().data)

# Apply a unary operation (e.g., negation) on a LazyBuffer instance
neg_result = a.unary_op(Tensor.__neg__)
print("Negation result:", neg_result.toCPU().data)

# Apply a reduce operation (e.g., sum) on a LazyBuffer instance
sum_result = a.reduce_op(Tensor.sum)
print("Sum result:", sum_result.toCPU().data)

# Slice a LazyBuffer instance
sliced_result = a.slice(1, 4)
print("Sliced result:", sliced_result.toCPU().data)