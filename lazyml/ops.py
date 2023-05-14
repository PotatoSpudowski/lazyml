from typing import Callable, Optional, Tuple, Union, Any, List

class ShapeTracker:
    def __init__(self, shape: Tuple[int, ...]):
        self.shape = shape

class Tensor:
    def __init__(self, data: List[List[float]], shape: Tuple[int, int]):
        self.data = data
        self.shape = shape

    def __add__(self, other: 'Tensor') -> 'Tensor':
        result = Tensor([], (0, 0))
        result.shape = self.shape
        result.data = [[self.data[i][j] + other.data[i][j]
                        for j in range(self.shape[1])]
                       for i in range(self.shape[0])]
        return result

    def __mul__(self, other: 'Tensor') -> 'Tensor':
        result = Tensor([], (0, 0))
        result.shape = self.shape
        result.data = [[self.data[i][j] * other.data[i][j]
                        for j in range(self.shape[1])]
                       for i in range(self.shape[0])]
        return result

    def __neg__(self) -> 'Tensor':
        result = Tensor([], (0, 0))
        result.shape = self.shape
        result.data = [[-self.data[i][j] for j in range(self.shape[1])]
                       for i in range(self.shape[0])]
        return result

    def sum(self, axis: Optional[int] = None) -> 'Tensor':
        if axis is None:
            total = sum([sum(row) for row in self.data])
            return Tensor([[total]], (1, 1))
        elif axis == 0:
            result_data = [[sum(self.data[i][j] for i in range(self.shape[0]))]
                           for j in range(self.shape[1])]
            return Tensor(result_data, (1, self.shape[1]))
        else:  # axis == 1
            result_data = [[sum(self.data[i][j] for j in range(self.shape[1]))]
                           for i in range(self.shape[0])]
            return Tensor(result_data, (self.shape[0], 1))

class LazyBuffer:
    def __init__(self, data: Tensor):
        self.data = data
        self.tracker = ShapeTracker(data.shape)

    def realize(self) -> Tensor:
        return self.data

    @classmethod
    def fromCPU(cls, cpu_tensor: Tensor) -> 'LazyBuffer':
        return cls(cpu_tensor)

    def toCPU(self) -> Tensor:
        return self.data

    def unary_op(self, op: Callable[[Tensor], Tensor]) -> 'LazyBuffer':
        return LazyBuffer(op(self.data))

    def binary_op(self, other: 'LazyBuffer', op: Callable[[Tensor, Tensor], Tensor]) -> 'LazyBuffer':
        return LazyBuffer(op(self.data, other.data))

    def reduce_op(self, op: Callable[..., Tensor], axis: Optional[int] = None) -> 'LazyBuffer':
        return LazyBuffer(op(self.data, axis=axis))

    def slice(self, start: int, end: int) -> 'LazyBuffer':
        return LazyBuffer(Tensor([row[start:end] for row in self.data.data], (self.data.shape[0], end - start)))