# MLA 当前状态

最近闭环为 **M06c3b：官方 metadata 入口按 TND query 生成混合 v4**。分支为 `feat/mla-cann-migration`，验收日期 **2026-10-10**。唯一下一步为 **[M06c3c：公开只读混合布局 kvcache](plan.md#下一步m06c3c)**，尚未开始。范围与契约见 [design](design.md)；当前闭环提交用 `git log -1 -- docs/mla/status.md` 定位。

## 当前可调用能力

共同边界为 **910B C1V2、FP16/BF16、512+64 共享 KV、Hkv=1、连续布局**，允许非零 storage offset。前向拒绝梯度，Q 使用完整分配段。BSND 每样本 KV 容量及紧凑 TND 的 `max_seqlen_k` 为 0..16384；紧凑 TND 总 Q/KV 各不超过 `INT32_MAX`，总 KV 可以超过 16384。混合记录的 BSND 总容量为 64 位 `B*Sk`。其他字段、地址及资源界限见 [design 6.2](design.md#62-调度字段与边界)。

| 入口或条件 | 当前行为 |
| --- | --- |
| `flash_attn_func` | BSND 整容量前向，内部生成 v1 `int32[401]` |
| `flash_attn_varlen_func` | 紧凑 TND Q/KV，连续同设备 `int32[B+1]` 前缀；每次内部生成并消费 v3 `int32[405+4B]`，max 参数为上界，前缀值在设备校验 |
| `flash_attn_with_kvcache` | 只读 BSND；`cache_seqlens` 接受 None、`0..Sk` 整数或同设备连续 `int32[B]`。无长度且无记录时走 v1，否则走 v2；显式 v2 逐项核对当次长度 |
| `get_scheduler_metadata`，无 Q 前缀 | 64/512 BSND 分支生成 v2 `int32[401+B]`；要求设备连续 `int32[B]` KV 长度 |
| `get_scheduler_metadata`，有 `cu_seqlens_q` | **已接通** TND Q + BSND KV 的 v4 `int32[406+5B]`。Q 前缀须为同设备连续 `int32[B+1]`，KV 长度仍须为 `int32[B]`；不接受 None/整数长度。公开混合前向及 v4 复用仍拒绝 |
| split 与选项 | 仅 `num_splits=1`；func/varlen 默认 1，kvcache/metadata 默认 0，须显式传 1。无 mask、`softcap=0`、`pack_gqa=None`、`sm_margin=0`、`attention_chunk=0` |
| scale 与返回 | scale 缺省 `1/sqrt(576)`，显式值须在 FP32 有限范围；记录不固化 scale。O 与输入同 dtype，LSE 为 FP32；BSND O/LSE 为 `[B,Sq,Hq,512]` / `[B,Hq,Sq]`，TND 为 `[Tq,Hq,512]` / `[Hq,Tq]`。func/varlen 请求 LSE 返回二项，kvcache 返回四项，未 split 的 accumulator 为同设备空 tensor |
| 空容量与空行 | 空 batch/query 保留空 shape，仍校验记录和前缀。非零容量中的零 KV 样本 `O=0、LSE=-inf`，全局空 KV 为 `O=0、LSE=+inf` |

**仍拒绝**公开混合布局前向、used、非连续外层 stride、PA、causal/window、自动或多 split、192/128 prefill，以及追加 KV、内置 rotary 等范围外组合。varlen 不扩展官方签名来提供显式 v3 复用。已有 v2 可克隆或使用连续偏移视图；损坏/过期记录由前向拒绝。fake/compile 已覆盖，完整 graph replay 生命周期留在 M16。目标包含 910C，生产平台仍仅开放 910B。

## 独立组件边界

官方 metadata 入口没有总 Q 参数。生产 [TndBsndMetadataOp](../../csrc/ascend910/flash_attn_npu_3/mla/mla_forward.cpp) 按静态 batch 分配精确字数，复用双事件与 allocator stream 登记；[AICPU 入口](../../csrc/ascend910/flash_attn_npu_3/mla/mla_tnd_bsnd_metadata.aicpu) `ComputeMlaTndBsndMetadataFromPrefix` 在输出/输入描述符检查后读取 Q 尾值，再运行完整解析与调度，无前缀/长度 D2H、`.item()` 或稳态主机同步。

`max_seqlen_q` 为段长上界，`max_seqlen_k` 为物理 Sk；不使用矩形 `B*maxQ*Hq` 限制紧凑 Q。非空总容量绑定全核 blockDim，空 batch/query/KV 容量保留单校验核与零 workspace。B=0 仍要求 Q `[0]` 和空 KV tensor。数值不合法时生成失败记录并清零完整载荷；生成器本身不触发消费者 Trap。旧显式总 Q 生成器的语义及 v4 字段布局不变，详见 [design 5.5](design.md#55-v4tnd-q-与-bsnd-kv-混合记录)。

| 布局 / 协议 | CPU 调度、AICPU、设备校验、独立 attention | 公开入口 |
| --- | --- | --- |
| BSND Q/KV，v1/v2 | 已验收 | 上述 BSND 子集 |
| 紧凑 TND Q/KV，v3 | 已验收 | 上述 varlen 子集 |
| TND Q + BSND KV，v4 | 已验收 | 仅 metadata 生成；前向待 M06c3c |

本步没有修改 attention 的 Kernel / Block / Tile 数值路径。混合布局仍保留 64 位 `b*Sk` 物理 KV 起点，短 KV 不重排后续 cache；其独立 attention 证据保留在 `build/mla-m06c2b2d-mixed-attention/`。

## 最近闭环证据

归档为 **`build/mla-m06c3b-metadata/`**，总索引 `acceptance.json`；`*-command.json` / `*-exit.json` 保存完整命令、环境及退出码。接手核验上一步 4,354 个归档文件、229 个生产源码依赖和 22 个对象，源码与实际加载扩展一致，无未解决的前步失败。

| 验收面 | 结果 | 相对证据路径 |
| --- | --- | --- |
| v4 CPU 生成与消费者 | 183 项通过，ASan/UBSan/LeakSanitizer；包括新推导总量模式、固定源差分、旧协议隔离、缓冲及 64 位边界 | `cpu-metadata.xml`、`cpu-artifacts/` |
| 公开 v4 | 580 项去重通过；双 dtype 真实/fake/C++、精确字数、每 dtype 102 组 CPU/固定源逐字对照、非法值失败记录、空容量、offset、64 位物理容量、别名、更新/跨流/allocator/compile | `public-metadata.xml`、`extra-boundaries-recheck.xml`、`producer-final/` |
| 普通 FA3 与已有 MLA | 1,031 项通过；普通 metadata/compile、官方签名、BSND 精度与生命周期、dense metadata dispatcher | `regressions.xml`、`regression-nodes.json` |
| 最终构建复查 | 896 项通过，含上述 580 项、280 项 BSND 拒绝边界、36 项上一步紧凑 TND 双 dtype 源精度及生命周期最小检查；另 1 项最终 sanitizer 缓冲检查通过 | `final-recheck.xml`、`final-nodes.json`、`cpu-final.xml` |
| 构建 | 13 项通过；实例、依赖、AICPU 隔离及 sanitizer。最终加载路径、对象/源码审计通过 | `build-tests.xml`、`production-build-manifest.json` |

合计 **2,123 项去重 pytest 通过**，重复复查不重复计数。原有 TND producer 的源码/库哈希仍匹配，最小复查复用该库；TND attention probe 已按当前头文件重建。生成链路未转调原 CANN attention。

补充测试首轮因错误消息正则只接受大写 `Tensor` 失败，None 实际已被拒绝；修正后 36 项及最终全组复查通过。首次产物审计因临时 `profile.py` 遮蔽标准库失败，重命名后审计通过。格式化改动已复核，C++ 仅空白变化，随后完成最终重建和复查；所有原始日志保留。

## 生成耗时诊断

性能驱动为归档 `generation_profile.py`，命令见 `performance-command.json`。设备/构建/测试结束后，在健康空闲 910B 卡 2、CPU 168..175 采样；六组固定几何覆盖非均匀、21 样本、长 KV、全零实际 KV、空 Q 与空 KV，双 dtype。预热 5 次，5 轮交替，每轮 20 个 wall 样本和 10 个 profiler 样本。公开生成与独立显式总 Q 生成器均逐字核对 CPU 记录。

下表为逐例中位数范围；Kernel 比值为公开 / 独立显式总 Q 生成器。wall 包含公开分配、dispatcher、事件与同步；独立驱动使用 ctypes 和预分配记录，不能据此比较混合 attention E2E。

| dtype | 公开生成 wall | 公开 AICPU Kernel | Kernel 比值 |
| --- | --- | --- | --- |
| FP16 | 208.625..222.570 µs | 16.460..38.121 µs | 0.972..1.023 |
| BF16 | 219.710..230.285 µs | 16.750..37.921 µs | 0.969..1.025 |

证据在 `performance/`，共 2,400 个 wall 和 1,200 个 AICPU 样本；原始 CANN `op_summary` 与 `kernel_details` 的数量、名称、设备、起始时间和耗时逐行一致。profiler 的可选 `fwk_api_db` 解析报错，**数据库导出未通过**；本步只使用已核对的 Text CSV，原始日志保留。公开混合前向、显式复用及包含生成的 R=1/4 E2E 留在 M06c3c，本步不宣称其性能达标。既有紧凑 TND 公开 E2E 证据仍在 `build/mla-m06c3a-varlen/`。

## 环境与构建

| 项目 | 本步核验 |
| --- | --- |
| Python / PyTorch | `/usr/local/python3.11.15/bin/python3` 3.11.15；torch `2.7.1+cpu`、torch_npu `2.7.1` |
| CANN / 编译器 | `/usr/local/Ascend/cann-9.0.0-beta.2`；bisheng clang 15.0.5 |
| 芯片 | 八张 Ascend910B3；卡 7 公开入口，卡 4 普通/BSND 回归，卡 2 性能。沙箱内 NPU 初始化不可用，设备/构建及完整 sanitizer 在沙箱外执行 |
| CATLASS | `769cd40a8716b28650b6bebb08db4834eea4462f`，工作区干净 |
| 只读 CANN 源仓 | `ca18aee610729ca5270a44e2d84c751bc687fbf0`，原有 67 条删除/未跟踪记录未改变 |
| 最终生产重建 | 22 个对象；仅 `flash_api.o`、`mla_forward.o`、`mla_tnd_bsnd_metadata.o` 改变，其余 19 个哈希不变；229 个生产源码依赖核验通过 |
| 实际加载扩展 | `flash_attn_npu_3/flash_attn_npu_3.cpython-311-aarch64-linux-gnu.so`；SHA256 `cd8bad0e60bfefc4a0fb9d4e73481be8fc6d596696d626df012e16436ab6a52f` |

## 最小复现与交接

完整命令和选择见归档 `*-command.json` / `final-nodes.json`。下列仅作最小复查；先检查健康空闲设备，设置 `MLA_DEVICE` 并使用新目录，不计作完整 2,123 项验收。

```bash
MLA_SCHEDULE_SANITIZE=1 ASAN_OPTIONS=detect_leaks=1 UBSAN_OPTIONS=halt_on_error=1 python3 -m pytest -x -q tests/test_flash_attn_npu_v3_mla_tnd_bsnd_metadata.py -k buffer_guards --basetemp=build/mla-mixed-generator-cpu-recheck
FLASH_ATTN_BUILD_VERSION=v3 FLASH_ATTN_BUILD_NPU=910 python3 setup.py build_ext --inplace
ASCEND_RT_VISIBLE_DEVICES=${MLA_DEVICE:?set MLA_DEVICE} OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 MLA_TND_BSND_METADATA_DEVICE_DIR=build/mla-mixed-generator-device-recheck python3 -m pytest -x -q tests/test_flash_attn_npu_v3_mla_mixed_metadata_public.py -k 'entries_shape_empty_and_offsets or prefix_updates_streams_allocator_and_aliases or public_records_match_cpu_and_pinned_source'
```

质量命令为 `PYTHONPATH=/tmp/mla-m01-precommit:$PYTHONPATH PRE_COMMIT_HOME=/tmp/mla-m01-precommit-cache ci/code_quality.sh --staged`，保留 CANN PYTHONPATH；提交前检查 `git diff --check`、文档链接/锚点及质量钩子。下一次接手先核验当前源码、扩展和证据，再推进唯一下一步 M06c3c。
