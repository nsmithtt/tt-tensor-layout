# Tensor Layout Spec

Terms:
- `shape`: Always logical shape, n-dimensional
- `strides`: Same as pytorch strides, but this is crucial for describing how n-dimensional data gets packed into a 2D physical layout. This 2D physical layout is always the inner dim (-1) wide and dims [0, N-1] are collapsed into rows derived from strides
- `shard_shape`: Also a logical shape, describes a 2d region that chunks physical_shape . Note this does not need to be a tile multiple
- `physical_shard_shape`: The `shard_shape` padded out to `tile_shape`
- `tile_shape`: A programmable tile shape, though constraints must check that it's compatible with an op's usage, i.e. FPU/Noc compatible
- `grid_shape`: `[divup(strides[0] // strides[-2], shard_shape[0]), divup(strides[-2], shard_shape[0])]`
