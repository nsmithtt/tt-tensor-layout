# Tensor Layout Spec

This proposal outlines an effort to support more flexible tensor layouts in TTNN.  The initial impulse to refactor tensor layout spawned from convolutions which need additional shape tracking, but this is cumbersome and error prone to manually propagate through a model.  The existing TTNN layout is also somewhat ambiguous in some situations, different ops use concepts like sharding and padding in different ways that aren't always compatible or consistent with each other.

Some high level goals:
- Logical shapes: Keep the original tensor shape intact and agnostic to underlying storage layout
- Flexible sharding: Enable flexibility in choosing grid shape, to get better parallelization and avoid resharding
- Logical-Physical Isomorphism: Encode this information with just a few attributes to enable derived conversions from logical to physical layout (and back)
- Explicit: A single source of truth
- Decouple itself from padding

Terms:
- `shape`: Always logical shape, n-dimensional
- `strides`: Same as pytorch strides, but this is crucial for describing how n-dimensional data gets packed into a 2D physical layout. This 2D physical layout is always the inner dim (-1) wide and dims [0, N-1] are collapsed into rows derived from strides
- `shard_shape`: Also a logical shape, describes a 2d region that chunks physical_shape . Note this does not need to be a tile multiple
- `physical_shard_shape`: The `shard_shape` padded out to `tile_shape`
- `tile_shape`: A programmable tile shape, though constraints must check that it's compatible with an op's usage, i.e. FPU/Noc compatible
- `grid_shape`: `[divup(strides[0] // strides[-2], shard_shape[0]), divup(strides[-2], shard_shape[0])]`
