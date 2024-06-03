def divup(a, b):
    return (a + b - 1) // b


def roundup(a, b):
    return divup(a, b) * b


def x(s):
    return "x".join(map(str, s))


def xor(a, b):
    return bool(a) != bool(b)


def dim_map_neg_indexed(dim_map, rank):
    dim_map_neg = {}
    for dim in dim_map.keys():
        if dim < 0:
            dim_map_neg[dim] = dim_map[dim]
        else:
            dim_map_neg[dim - rank] = dim_map[dim]
    return dim_map_neg


class Layout:
    @staticmethod
    def calculate_strides(shape, align_dims):
        align_dims = dim_map_neg_indexed(align_dims, len(shape))
        strides = [1]
        for dim in range(-1, -len(shape) - 1, -1):
            stride = shape[dim] * strides[0]
            align = align_dims.get(dim, 1)
            if dim < -1:
                align *= strides[-2]
            stride = roundup(stride, align)
            strides.insert(0, stride)
        return strides

    @staticmethod
    def replace_strides(old_strides, replacements):
        replacements = dim_map_neg_indexed(replacements, len(old_strides))
        new_strides = []
        factor_out = 1
        factor_in = 1
        for dim in range(-1, -len(old_strides) - 1, -1):
            old_stride = old_strides[dim]
            assert (
                old_stride % factor_out == 0
            ), f"{old_stride} % {factor} != 0, {dim} {old_strides} {new_strides}"
            stride = (old_stride // factor_out) * factor_in
            if dim in replacements:
                stride = replacements[dim]
                factor_out *= old_stride
                factor_in *= stride
            new_strides.insert(0, stride)
        return new_strides

    def __init__(
        self,
        shape,
        strides=None,
        align_dims=None,
        shard_shape=None,
        grid_shape=None,
        tile_shape=[1, 1],
    ):
        assert xor(grid_shape, shard_shape)
        assert xor(align_dims, strides)

        # The logical shape of the tensor, always
        self.shape = shape

        # The strides of the tensor, if not provided, calculate it
        # The strides enable arbitrary dimensions to be padded out to enable dims from [0, N-1]
        # to always be tightly packed
        self.strides = (
            strides
            if strides is not None
            else Layout.calculate_strides(shape, align_dims)
        )
        assert (len(self.shape) + 1) == len(self.strides)

        # The shape of the tile, constrained by Noc gran / FPU
        # [1, N] means row major, [N, N] means tilized
        self.tile_shape = tile_shape

        if shard_shape is not None:
            self.shard_shape = shard_shape
        else:
            inner = self.strides[-2]
            assert self.strides[0] % inner == 0
            outer = self.strides[0] // inner
            self.shard_shape = [
                outer // grid_shape[0],
                inner // grid_shape[1],
            ]
            inferred_grid_shape = self.get_grid_shape()
            assert (
                inferred_grid_shape == grid_shape
            ), f"{inferred_grid_shape} != {grid_shape}"

    def get_shape(self):
        return self.shape

    def get_shard_shape(self):
        return self.shard_shape

    def get_physical_shard_shape(self):
        return [
            roundup(self.shard_shape[0], self.tile_shape[0]),
            roundup(self.shard_shape[1], self.tile_shape[1]),
        ]

    def get_grid_shape(self):
        col = self.strides[-2]
        row = self.strides[0] // col
        return [
            divup(row, self.shard_shape[0]),
            divup(col, self.shard_shape[1]),
        ]

    def __str__(self):
        return f"""Layout(shape={x(self.shape)},
       strides={x(self.strides)},
       shard_shape={x(self.shard_shape)},
       tile_shape={x(self.tile_shape)},
       grid={x(self.get_grid_shape())})"""


class Tensor:
    def __init__(self, layout):
        self.layout = layout

    @classmethod
    def create_from_shard_shape(cls, shape, shard_shape, tile_shape=[1, 1]):
        return Tensor(
            Layout(
                shape,
                align_dims={0: shard_shape[0]},
                shard_shape=shard_shape,
                tile_shape=tile_shape,
            )
        )

    @classmethod
    def create_from_grid_shape(cls, shape, grid_shape, tile_shape=[1, 1]):
        return Tensor(
            Layout(
                shape,
                align_dims={0: grid_shape[0]},
                grid_shape=grid_shape,
                tile_shape=tile_shape,
            )
        )

    @classmethod
    def create_from_manual_align(
        cls, shape, shard_shape, align_dims, tile_shape=[1, 1]
    ):
        return Tensor(
            Layout(
                shape,
                align_dims=align_dims,
                shard_shape=shard_shape,
                tile_shape=tile_shape,
            )
        )

    @classmethod
    def create_from_manual_strides(cls, shape, shard_shape, strides, tile_shape=[1, 1]):
        return Tensor(
            Layout(
                shape, strides=strides, shard_shape=shard_shape, tile_shape=tile_shape
            )
        )

    def __str__(self):
        return f"Tensor({self.layout})"


def eltwise(inputs):
    assert len(inputs) > 0
    for i in range(1, len(inputs)):
        assert inputs[i].layout == inputs[0].layout
    return Tensor(inputs[0].layout)


def matmul(lhs, rhs, log=False):
    lhs_physical_shard_shape = lhs.layout.get_physical_shard_shape()
    rhs_physical_shard_shape = rhs.layout.get_physical_shard_shape()
    lhs_grid_shape = lhs.layout.get_grid_shape()
    rhs_grid_shape = rhs.layout.get_grid_shape()

    assert lhs.layout.tile_shape == rhs.layout.tile_shape
    assert lhs.layout.shard_shape[1] == rhs.layout.shard_shape[0]
    assert lhs_physical_shard_shape[1] == rhs_physical_shard_shape[0]
    assert lhs_grid_shape[1] == rhs_grid_shape[0]

    shape = lhs.layout.shape
    shape[-1] = rhs.layout.shape[-1]
    shard_shape = lhs.layout.shard_shape
    shard_shape[-1] = rhs.layout.shard_shape[-1]
    strides = Layout.replace_strides(lhs.layout.strides, {-2: rhs.layout.strides[-2]})
    out = Tensor.create_from_manual_strides(
        shape, shard_shape, strides, tile_shape=lhs.layout.tile_shape
    )
    if log:
        print(f"matmul({lhs}, {rhs}) -> {out}")
    return out


def conv2d(act, weights, stride=1, padding=0, log=False):
    # (in_channels,kH,kW,out_channels)
    assert len(act.layout.shape) == 4, "Only 4D tensors supported"
    assert len(weights.layout.shape) == 4, "Only 4D tensors supported"
    assert act.layout.tile_shape[0] == 1, "Only row major supported"

    # (N, H, W, C)
    def calc_out_shape(in_shape, kernel_shape, stride, padding):
        return [
            in_shape[0],
            (in_shape[1] + 2 * padding - (kernel_shape[0] - 1) - 1) // stride + 1,
            (in_shape[2] + 2 * padding - (kernel_shape[1] - 1) - 1) // stride + 1,
            kernel_shape[3],
        ]

    out_shape = calc_out_shape(act.layout.shape, weights.layout.shape, stride, padding)
    out = Tensor.create_from_grid_shape(
        out_shape, act.layout.get_grid_shape(), tile_shape=weights.layout.tile_shape
    )
    if log:
        print(f"conv2d({act}, {weights}) -> {out}")
    return out


def group_norm(a, num_groups, num_channels, log=False):
    out = Tensor.create_from_shard_shape(
        a.layout.shape,
        a.layout.shard_shape,
        a.layout.tile_shape,
    )
    if log:
        print(f"group_norm({a}) -> {out}")
    return out


def to_row_major(a, log=False):
    out = Tensor.create_from_shard_shape(
        a.layout.shape,
        a.layout.shard_shape,
        [1, a.layout.shard_shape[1]],
    )
    if log:
        print(f"group_norm({a}) -> {out}")
    return out


print("=== Snippet of SD ===")

a = Tensor.create_from_shard_shape(
    [2, 1, 320], shard_shape=[1, 40], tile_shape=[32, 32]
)
l0 = Tensor.create_from_shard_shape(
    [320, 1280], shard_shape=[40, 160], tile_shape=[32, 32]
)
l1 = Tensor.create_from_shard_shape(
    [1280, 1280], shard_shape=[160, 160], tile_shape=[32, 32]
)

# linear
a = matmul(a, l0, log=True)
a = matmul(a, l1, log=True)

# conv
# (kH, kW, in_channels, out_channels)
w0 = Tensor.create_from_shard_shape(
    [3, 3, 32, 320], shard_shape=[288, 320], tile_shape=[32, 32]
)

w1 = Tensor.create_from_shard_shape(
    [3, 3, 320, 320], shard_shape=[2880, 320], tile_shape=[32, 32]
)

w2 = Tensor.create_from_shard_shape(
    [1, 1, 320, 1536], shard_shape=[320, 192], tile_shape=[32, 32]
)

# (N, H, W, C)
c = Tensor.create_from_grid_shape(
    [2, 64, 64, 32], grid_shape=[64, 1], tile_shape=[1, 32]
)

c = conv2d(c, w0, padding=1, log=True)

c = group_norm(c, num_groups=32, num_channels=32, log=True)

c = to_row_major(c, log=True)
c = conv2d(c, w1, padding=1, log=True)

c = to_row_major(c, log=True)
c = conv2d(c, w2, padding=0, log=True)
