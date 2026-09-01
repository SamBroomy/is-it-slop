# is-it-slop-model

The hierarchical slop classifier network, shared by the [`is-it-slop`](https://crates.io/crates/is-it-slop) inference crate and the training pipeline.

> **Note:** This is an internal crate. For AI text detection, use the main [`is-it-slop`](https://crates.io/crates/is-it-slop) crate instead.

The crate contains only the network definition (a Burn `Module`): a shared per-chunk trunk, a chunk head, and an attention-pooled document head. Keeping it separate lets inference depend on the forward pass alone, with none of the training machinery.

## License

MIT
