def divup(a, b):
    return (a + b - 1) // b


def roundup(a, b):
    return divup(a, b) * b


def x(s):
    return "x".join(map(str, s))


class Tensor:
    def __init__(self, shape, grid=[1, 1], strides=None, tilized=False):
        min_noc_granularity = 16
        # Shape is always the logical shape, no constraints / any rank
        self.shape = shape
        # Grid is the logical grid shape that shards this tensor
        self.grid = grid
        # Minimum row major granularity for innermost dimension
        self.row_stride = min_noc_granularity
        # Whether the tensor is tiled or row major
        self.tilized = tilized
        # strides expressed on the canonical shape of this tensor, as though it were flat laid out in host dram
        self.strides = strides if strides is not None else self.calculate_strides()
        assert self.strides[-1] == 1
        # Logical shard shape, this is the physical tensor shape trivially divided by the grid shape
        self.lshard = self.calculate_lshard_shape()
        # Physical shard shape, this is the footprint of the allocation in memory, potentially padded to a tile for ops that need to tilize
        self.pshard = self.calculate_pshard_shape()

    def calculate_strides(self):
        strides = [1]
        for i in range(len(self.shape) - 1):
            dim = self.shape[-i - 1]
            if i == 0:
                dim = roundup(dim, self.row_stride)
            strides.insert(0, dim * strides[0])
        return strides

    def physical_shape(self):
        collapsed = self.shape[0] * self.strides[0] // self.strides[-2]
        return [collapsed, self.strides[-2]]

    def calculate_lshard_shape(self):
        pshape = self.physical_shape()
        return [
            pshape[0] // self.grid[0],
            pshape[1] // self.grid[1],
        ]

    def calculate_pshard_shape(self):
        return [
            roundup(self.lshard[0], 32 if self.tilized else 1),
            roundup(self.lshard[1], 32 if self.tilized else 1),
        ]

    def __str__(self):
        return f"Tensor(shape={x(self.shape)}, strides={x(self.strides)}, pshape={x(self.physical_shape())}, grid={x(self.grid)}, lshard={x(self.lshard)}, pshard={x(self.pshard)})"


print("=== Tightly packed NHW ===")
row_maj = Tensor([2, 6, 6, 3], grid=[2, 1])
print("row_maj:", row_maj)

tilized = Tensor([2, 6, 6, 3], grid=[2, 1], tilized=True)
print("tilized:", tilized)

print()
print("=== N[HW] padded to tile (manual strides) ===")
row_maj = Tensor([2, 6, 6, 3], grid=[2, 1], strides=[1024, 96, 16, 1])
print("row_maj:", row_maj)

tilized = Tensor([2, 6, 6, 3], grid=[2, 1], strides=[1024, 96, 16, 1], tilized=True)
print("tilized:", tilized)

print()
print("=== NH[W] padded to tile (manual strides) ===")
row_maj = Tensor([2, 6, 6, 3], grid=[2, 1], strides=[3072, 512, 16, 1])
print("row_maj:", row_maj)

tilized = Tensor([2, 6, 6, 3], grid=[2, 1], strides=[3072, 512, 16, 1], tilized=True)
print("tilized:", tilized)
