"""Fixed incremental coverage; full suites remain the stage acceptance sets.

Each row gives existing pytest nodes and the boundary they retain. No random
sampling, shape rewriting, tolerance change or performance filtering occurs.
"""

from tests.mla.paths import relocated


def row(module, function, parameters, reason):
    prefix = relocated(f"tests/test_flash_attn_npu_v3_mla_{module}.py::{function}")
    return dict(
        reason=reason,
        nodes=[prefix]
        if parameters is None
        else [
            f"{prefix}[{dtype}{'-' + parameter if parameter else ''}]"
            for dtype in ("fp16", "bf16")
            for parameter in parameters
        ],
    )


COVERAGE = {
    "regression-dense": [
        row(
            "dense",
            "test_dense_boundaries",
            (
                "1-1",
                "15-15",
                "16-16",
                "17-17",
                "127-127",
                "128-128",
                "129-129",
                "255-255",
                "256-256",
                "257-17",
                "511-129",
                "512-128",
            ),
            "Vector M/N alignment and one-element tails, both dtypes",
        ),
        row(
            "dense",
            "test_scheduler_and_pipeline",
            ("2", "21", "41"),
            "batches below/above core count and task-ring rotation",
        ),
        row("dense", "test_content_rope_halves", None, "all four content/RoPE halves"),
        row(
            "dense",
            "test_schedule_boundaries",
            None,
            "global empty capacities and rejected overflow",
        ),
        row(
            "multiblock",
            "test_multiblock_accuracy",
            ("shape2", "shape4", "shape10", "shape14"),
            "cross-head M partitions, 511/513 rows, adjusted odd M and more tasks than cores",
        ),
        row(
            "kvblocks",
            "test_two_kv_block_accuracy",
            ("shape0", "shape6", "shape9", "shape10"),
            "first second-KV block, head tail and 1023/1024 boundary with public metadata reuse",
        ),
        row(
            "kvblocks",
            "test_running_max_changes_and_nupdate_clamp",
            ("-8.0", "64.0"),
            "decreasing max and clamp-crossing max update",
        ),
        row(
            "longkv",
            "test_long_kv_accuracy",
            ("shape0", "shape10", "shape14", "shape15"),
            "1025/8193/16383/16384 KV and public func/cache calls",
        ),
        row(
            "longkv",
            "test_long_kv_running_max",
            ("alternating", "tail"),
            "task-ring max updates and final one-element block",
        ),
        row(
            "dense",
            "test_interleaved_shapes_and_runtime_scale",
            None,
            "queued streams, runtime scale and reuse",
        ),
    ],
    "regression-nonpaged": [
        row(
            "bsnd_attention",
            "test_accuracy",
            ("spec0", "spec3", "spec6", "spec7", "spec9"),
            "effective lengths, 511/513 KV, omitted zero suffix, cross-sample ring and all-zero work",
        ),
        row(
            "tnd_attention",
            "test_accuracy",
            ("spec1", "spec6", "spec7", "spec13", "spec15"),
            "nonuniform packed lengths, zero query, ring, empty batch and empty KV",
        ),
        row(
            "tnd_bsnd_attention",
            "test_accuracy",
            ("spec1", "spec6", "spec13", "spec16", "spec18"),
            "physical cache offsets separate from effective lengths, ring and empty capacities",
        ),
        row(
            "used_attention",
            "test_accuracy",
            ("spec3", "spec4", "spec5"),
            "all three used layouts with allocation gaps, poison, zero samples and ring",
        ),
        row(
            "used_attention",
            "test_global_empty_fills_allocation",
            None,
            "all three layouts with zero used Q or KV",
        ),
        row(
            "used_attention",
            "test_each_length_update",
            None,
            "allocation and used descriptor updates",
        ),
        row(
            "bsnd_stride",
            "test_stride_attention",
            ("config2", "config5", "config8", "config9", "config10", "config12", "config18"),
            "v1/v2/v5 independent views, broadcast/overlap and 65536 address boundary",
        ),
        row(
            "tnd_stride",
            "test_stride_attention",
            (
                "config2",
                "config7",
                "config12",
                "config17",
                "config3",
                "config9",
                "config21",
                "config67",
            ),
            "v3/v4/v5 both layouts, independent/broadcast/overlap views and 65536 address boundary",
        ),
    ],
    "regression-paged": [
        *[
            entry
            for module, layout in (("paged_attention", "bnsd"), ("tnd_paged_attention", "ntd"))
            for entry in (
                row(
                    module,
                    "test_task_ring",
                    ("spec0", "spec4", "spec10", "spec12", "spec13", "spec14", "spec15", "spec22"),
                    "P16/128/1024, unaligned P112/528, repeated pages, used, 16K KV, tail and source-adjusted core count",
                ),
                row(
                    module,
                    "test_strides",
                    (layout, "independent", "broadcast", "table_overlap", "combined_65536"),
                    "query/cache/table strides including broadcast and 65536",
                ),
                row(
                    module,
                    "test_empty_cache",
                    None,
                    "empty table/pool/capacity and zero used combinations",
                ),
                row(
                    module,
                    "test_scale_halves_and_max_change",
                    ("0-0.0", "1--0.125", "2-0.25", "3-None"),
                    "zero/negative/default scale, all content/RoPE halves and maximum change",
                ),
                row(
                    module,
                    "test_stream_and_allocator_lifetime",
                    None,
                    "queued streams and allocator lifetime",
                ),
            )
        ],
    ],
    "causal-check": [
        row(
            "causal_attention",
            "test_head_and_empty_boundaries",
            ("7-3-1", "7-7-17", "7-33-129", "129-65-171"),
            "Sq greater/equal/less than Sk, masked rows and head tails",
        ),
        row(
            "causal_attention",
            "test_s2_and_m_tails",
            ("128-511", "129-512", "255-513", "256-1025"),
            "mask crosses KV blocks and M tails",
        ),
        row(
            "causal_attention",
            "test_empty_and_pipeline",
            None,
            "global/partial empty work and pipeline synchronization",
        ),
        row(
            "causal_attention",
            "test_amla_and_masked_max_changes",
            None,
            "small/large amplitude, source precision limits and masked max updates",
        ),
        row(
            "causal_attention",
            "test_interleaved_streams",
            None,
            "stream isolation and mask lifetime",
        ),
    ],
}

INCREMENTAL_SUITES = {
    name: [node for entry in entries for node in entry["nodes"]]
    for name, entries in COVERAGE.items()
}
PARENTS = {
    "regression-dense": "shared-dense",
    "regression-nonpaged": "shared-nonpaged",
    "regression-paged": "shared-paged",
    "causal-check": "causal-attention",
}
