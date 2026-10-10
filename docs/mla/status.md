# MLA 当前状态

最近闭环为 **M06d1：实际 Q/KV 使用长度 CPU 归一与调度**。分支为 `feat/mla-cann-migration`，验收日期 **2026-10-10**。唯一下一步为 **[M06d2a：used 记录协议与独立 AICPU 生成](plan.md#下一步m06d2a)**，尚未开始。范围与契约见 [design](design.md)，当前闭环提交用 `git log -1 -- docs/mla/status.md` 定位。

## 当前可调用能力

共同边界为 **910B C1V2、FP16/BF16、512+64 共享 KV、Hkv=1、连续布局**，允许非零 storage offset。前向拒绝梯度，Q 使用完整分配段。BSND 每样本 KV 容量及紧凑 TND 的 `max_seqlen_k` 为 0..16384；紧凑 TND 总 Q/KV 各不超过 `INT32_MAX`，总 KV 可以超过 16384。混合记录的 BSND 总容量为 64 位 `B*Sk`。其他字段、地址及资源界限见 [design 6.2](design.md#62-调度字段与边界)。

| 入口或条件 | 当前实现 |
| --- | --- |
| `flash_attn_func` | BSND 整容量前向，内部生成 v1 `int32[401]` |
| `flash_attn_varlen_func` | 紧凑 TND Q/KV，连续同设备 `int32[B+1]` 前缀；每次内部生成并消费 v3 `int32[405+4B]`，max 参数为上界，前缀值在设备校验 |
| `flash_attn_with_kvcache`，BSND Q | 只读 BSND；`cache_seqlens` 接受 None、`0..Sk` 整数或同设备连续 `int32[B]`。无长度且无记录时走 v1，否则走 v2；显式 v2 逐项核对当次长度 |
| `flash_attn_with_kvcache`，TND Q | 已接通 TND Q/qv + BSND K/V，必须给出同设备连续 `int32[B+1]` Q 前缀及 `max_seqlen_q` 上界。KV 长度接受 None、整数或 tensor；内部生成 v4，或消费显式 `int32[406+5B]`，逐项核对当次 Q 前缀和 KV 长度 |
| `get_scheduler_metadata` | 无 Q 前缀时生成 v2；有 `cu_seqlens_q` 时生成混合 v4。两者都要求设备连续 `int32[B]` KV 长度，不接受 None/整数便利输入 |
| split 与选项 | 仅 `num_splits=1`；func/varlen 默认 1，kvcache/metadata 默认 0，须显式传 1。无 mask、`softcap=0`、`pack_gqa=None`、`sm_margin=0`、`attention_chunk=0` |
| scale 与返回 | scale 缺省 `1/sqrt(576)`，显式值须在 FP32 有限范围；记录不固化 scale。O 与输入同 dtype，LSE 为 FP32；BSND O/LSE 为 `[B,Sq,Hq,512]` / `[B,Hq,Sq]`，TND 为 `[Tq,Hq,512]` / `[Hq,Tq]`。func/varlen 请求 LSE 返回二项，kvcache 返回四项，未 split 的 accumulator 为同设备空 tensor |
| 空容量与空行 | 空 batch/query 保留空 shape，仍校验记录和前缀。非零容量中的零 KV 样本 `O=0、LSE=-inf`，全局空 KV 为 `O=0、LSE=+inf` |

**仍拒绝** used、非连续外层 stride、PA、causal/window、自动或多 split、192/128 prefill，以及追加 KV、内置 rotary 等范围外组合。varlen 不增加显式 v3 复用参数；内部通用混合入口不接受 `max_seqlen_k`。v2/v4 可克隆或使用连续偏移视图；损坏/过期记录由设备拒绝。fake/compile 已覆盖，完整 graph replay 生命周期留在 M16。目标包含 910C，生产平台仍仅开放 910B。

## 本步实现边界

新增 [`ParseUsedLengths` / `BuildUsedSchedule`](../../csrc/ascend910/flash_attn_npu_3/mla/mla_used_schedule.h)，实现位于 [`mla_tnd_schedule.cpp`](../../csrc/ascend910/flash_attn_npu_3/mla/mla_tnd_schedule.cpp)，仅为独立 CPU 组件，覆盖 BSND Q/KV、TND Q/KV、TND Q + BSND KV。

每个样本分别保存 64 位物理 Q/KV 起点、分配长度和实际 used 长度。Q/KV 分别完整校验；used 缺省使用整个分配段，缩短 used 不移动后续样本。BSND 总量须等于物理容量乘 batch，TND 分配前缀须从 0 单调到总容量，max 参数仍为上界；全零 used 不能绕过分配、计数、字段和地址校验。失败不改写输出，也不保留输入指针。

复用既有变长分核与无 split 端点映射。基本块/workspace 使用分配几何，任务/成本/执行块上界使用实际长度；显式 used 数值转换成源非累计 int64，缺省 TND used 保留累计尾值。源 KV 数组等长判定保留，因此 TND 显式整段 used 与缺省 used 可有不同 M 基块，详见 [design 6.2.7](design.md#627-分配长度与-used-的-cpu-归一)。空分配沿用单校验核、零 workspace；非空分配中的全零 used 保留零成本调度与 workspace。

**本步没有 used metadata 协议、AICPU 发射、设备消费者或 attention 输出。** 公开与内部生产入口继续拒绝尚未接通的 used 组合；v1/v2/v3/v4 语义保持不变。共享调度源码由既有 v3/v4 AICPU 生成器编译，因此重建并回归这两条已有链路。attention Kernel / Block / Tile 未修改，本步未重跑性能；最近公开混合 E2E 证据仍在 `build/mla-m06c3c-kvcache/performance-acceptance.json`，不作为 used 能力证据。

## 最近闭环证据

归档为 **`build/mla-m06d1-used/`**，总索引 `acceptance.json`；`*-command.json` / `*-exit.json` 保存命令、环境与退出码。接手核验上一步 **5,183** 个归档文件、**237** 个生产源码依赖和 **24** 个对象；最小复查 **170 项公开混合 kvcache + 1 项 CPU sanitizer** 均通过。中断恢复后再次核对 HEAD 仍为本轮起点、无本轮遗留测试进程，环境、当前 **238** 个依赖、**24** 个对象及实际加载扩展哈希一致，见 `resume-check.json`。

| 验收面 | 当前结果 | 相对证据路径 |
| --- | --- | --- |
| used CPU 归一与调度 | **306 项通过**；U01/T01、可选 Q/KV used、零容量/零 used、更新复用、越界/计数/前缀/字段/地址/workspace；Q/KV 物理起点超过 uint32 | `used-verified.xml`、`used-verified/` |
| 独立固定源对照 | **6,187 组**三方比较及任务/成本枚举一致；原生非 FD **3,579**，FA3 禁止 split 映射 **2,608**；另 **288 组**与旧布局调度逐字段一致 | `used-differential-summary.json`、`used-verified/*-inputs.json` / `*-outputs.json` |
| 既有调度与 metadata 回归 | **861 项通过**；dense/BSND/TND/混合调度、v1..v4 生成以及 v3/v4 消费者；另两项重复边界用例启用独立消费者 sanitizer | `cpu-regression-acceptance.json`、`cpu-verified.xml`、`cpu-verified-*/`、`consumer-sanitized.xml` |
| 普通 FA3 与边界 | **839 项通过**；普通 metadata、compile、官方签名、MLA 契约与拒绝边界 | `ordinary-regressions.xml`、`ordinary-nodes.json` |
| 910B 已有 TND/混合链路 | **792 项通过**；公开混合 metadata、双 dtype 源/数学精度、前缀更新、跨 stream/allocator/别名/compile；used 仍拒绝 | `device-regressions.xml`、`device-nodes.json`、`mixed-attention/`、`tnd-attention/` |
| 构建与产物 | **15 项通过**；新增头的传递依赖失效、已有实例/隔离/缓冲检查；24 个链接对象及加载扩展核验通过 | `build-tests.xml`、`production-build-manifest.json` |

本步最终验收合计 **2,813 项去重 pytest 通过**；接手复查、首轮最小检查和中断/失败轮次不重复计数。调度与 TND/混合生产器可执行文件及独立消费者缓冲检查以 `-fsanitize=address,undefined -fno-sanitize-recover=all`、`ASAN_OPTIONS=detect_leaks=1`、`UBSAN_OPTIONS=halt_on_error=1` 运行，源 oracle 不包含或链接迁移解析器。

归档 `pytest_persistent.py` 保留逐请求 JSON 对应关系，在 EOF 后核验驱动退出码、stderr 与多余输出。used 组 **7 个驱动**全部正常退出；旧路径 861 项断言通过，**34 个驱动**正常退出，但两个 dense 驱动等待 EOF 超时，该轮总退出码保留为 1。修复后完整重跑受影响的 dense 模块及环境初始化检查 **94 项**，并重跑共用源 oracle 的 v1 metadata **22 项**；四个补跑驱动均正常退出，无 sanitizer 报告。补跑覆盖两个失败驱动的全部请求数/输入行数；节点、固定源文件哈希及 metadata 输入/输出摘要均核对一致。这些重复用例不增加总计，汇总判定见 `cpu-regression-acceptance.json`，原始结果保留在各轮 `*-driver-exits.json`。

初轮逐次启动的泄漏扫描即使空输入也约需 4.5 秒，两轮在 27/40 项通过后主动中断，退出码 2。持续进程工具先后修复 `select` 描述符上限、退出汇总变量错误及按无关环境变量重复启动问题；最后定位到测试环境 fork 的 Python worker 继承了驱动 stdin 写管道，导致父进程关闭输入后仍无法收到 EOF。清理本轮闲置 worker 后其余驱动正常退出，工具用 `os.register_at_fork` 在子进程关闭继承管道，再完成上述定向重跑。全过程保留泄漏扫描，未放宽 sanitizer；失败脚本保留为 `pytest_persistent_v{1,2,3}.py`，诊断见 `cpu-exit-diagnosis.json`，中断/失败轮次不单独计作通过。最后一组复查曾因自动审批超时未能启动，恢复后在健康空闲卡 7 完成，见 `approval-block.json` / `metadata-exit-recheck-exit.json`。质量检查发现的 Python 变量命名及格式问题已修正，最终仓库钩子通过。

## 环境与构建

| 项目 | 本步核验 |
| --- | --- |
| Python / PyTorch | `/usr/local/python3.11.15/bin/python3` 3.11.15；torch `2.7.1+cpu`、torch_npu `2.7.1` |
| CANN / 编译器 | `/usr/local/Ascend/cann-9.0.0-beta.2`；bisheng clang 15.0.5 |
| 芯片 | 八张 Ascend910B3；卡 0 用于既有 metadata，健康空闲卡 7 用于已有 MLA 回归及 metadata 补查，卡 4 用于普通 FA3。沙箱内驱动初始化失败，设备/构建及完整 sanitizer 在沙箱外执行 |
| CATLASS | `769cd40a8716b28650b6bebb08db4834eea4462f`，工作区干净 |
| 只读 CANN 源仓 | `ca18aee610729ca5270a44e2d84c751bc687fbf0`，原有 67 条删除/未跟踪记录未改变 |
| 生产重建 | 24 个对象中仅 `mla_tnd_metadata.o`、`mla_tnd_bsnd_metadata.o` 改变，其余 22 个哈希不变；238 个生产源码依赖核验通过 |
| 实际加载扩展 | `flash_attn_npu_3/flash_attn_npu_3.cpython-311-aarch64-linux-gnu.so`；SHA256 `3fca88dd5acbda6317fe019721a9f2fc3a5bf1f7733360e6ca5581a80bce2f43` |

## 最小复现与交接

完整命令、选取节点与证据见归档。下列为最小 CPU 复查，使用新目录；完整 sanitizer 矩阵可按 `used-verified-command.json` 和 `cpu-verified-command.json` 使用修正后的归档持续进程驱动及新目录重跑；退出问题的定向复查命令为 `cpu-exit-recheck-command.json` / `metadata-exit-recheck-command.json`。

```bash
MLA_SCHEDULE_SANITIZE=1 ASAN_OPTIONS=detect_leaks=1 UBSAN_OPTIONS=halt_on_error=1 python3 -m pytest -x -q tests/test_flash_attn_npu_v3_mla_used_schedule.py -k 'u01_t01_physical_spans or source_array_semantics_and_capacity_geometry or per_sample_bounds_and_required_prefixes or wide_addresses_and_active_tile_bound' --basetemp=build/mla-used-cpu-recheck
```

源码/构建变化时重建扩展；已有设备链路的最小复查如下，先检查健康空闲卡并设置 `MLA_DEVICE`：

```bash
FLASH_ATTN_BUILD_VERSION=v3 FLASH_ATTN_BUILD_NPU=910 python3 setup.py build_ext --inplace
ASCEND_RT_VISIBLE_DEVICES=${MLA_DEVICE:?set MLA_DEVICE} OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 MLA_TND_BSND_METADATA_DEVICE_DIR=build/mla-used-legacy-producer-recheck MLA_TND_BSND_ATTENTION_DIR=build/mla-used-legacy-attention-recheck python3 -m pytest -x -q tests/test_flash_attn_npu_v3_mla_mixed_public.py -k 'entries_empty_and_returns or none_integer_lengths or updates_streams_allocator_and_aliases or (source_accuracy and None and spec0)' --basetemp=build/mla-used-legacy-public-recheck
```

质量命令为 `PYTHONPATH=/tmp/mla-m01-precommit:$PYTHONPATH PRE_COMMIT_HOME=/tmp/mla-m01-precommit-cache ci/code_quality.sh --staged`，保留 CANN PYTHONPATH。只暂存本步文件，保留 `kernel_meta/` 未跟踪产物。下次先核验本步证据与最小命令，再推进 M06d2a；M06d2 已在 plan 拆为生成、消费和独立 attention 三步。
