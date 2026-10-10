# MLA 当前状态

当前闭环为 **M06d2a：used 记录协议与独立 AICPU 生成**，2026-10-10 基于 `feat/mla-cann-migration` 的 `78d6284` 完成。唯一下一步为 **[M06d2b：used 记录设备校验消费者](plan.md#下一步m06d2b)**，尚未开始。本步提交用 `git log -1 -- docs/mla/status.md` 定位。

目标契约与实现结构见 [design](design.md)，步骤及完成条件见 [plan](plan.md)。下表只描述当前可调用能力，不把独立 CPU 组件或未来目标计作公开支持。

## 当前可调用能力

共同边界为 **910B C1V2、FP16/BF16、512+64 共享 KV、Hkv=1、连续布局、无梯度**；允许非零 storage offset。Q 使用完整分配段。BSND 每样本 KV 容量与紧凑 TND 的 `max_seqlen_k` 均为 0..16384；紧凑 TND 总 Q/KV 各不超过 `INT32_MAX`，总 KV 可以超过 16384。混合 BSND cache 总容量为 64 位 `B*Sk`。完整字段、地址和资源界限见 [design 6.2](design.md#62-调度字段与边界)。

| 入口 | 输入与长度 | 调度记录 |
| --- | --- | --- |
| `flash_attn_func` | BSND 整容量 | 内部 v1：`int32[401]` |
| `flash_attn_varlen_func` | 紧凑 TND Q/KV；两组同设备连续 `int32[B+1]` 前缀，max 参数为每段上界 | 每次内部 v3：`int32[405+4B]`；无公开显式复用参数 |
| `flash_attn_with_kvcache`，BSND Q | 只读 BSND cache；KV 长度为 None、`0..Sk` 整数或同设备连续 `int32[B]` | 无长度且无记录走 v1；否则内部 v2 或显式 v2：`int32[401+B]` |
| `flash_attn_with_kvcache`，TND Q | TND Q/qv + BSND K/V；必须给出 Q 前缀和 `max_seqlen_q` 上界；KV 长度同上 | 内部 v4 或显式 v4：`int32[406+5B]` |
| `get_scheduler_metadata` | 必须给设备连续 `int32[B]` KV 长度；不接受 None/整数 | 无 Q 前缀生成 v2，有 Q 前缀生成混合 v4 |

- **调用选项：** 仅 `num_splits=1`；func/varlen 默认 1，kvcache/metadata 默认 0，须显式传 1。无 mask，`softcap=0`、`pack_gqa=None`、`sm_margin=0`、`attention_chunk=0`；metadata 的 `has_softcap=False`。scale 默认 `1/sqrt(576)`，显式值须在 FP32 有限范围，记录不固化 scale。
- **输出：** O 与输入同 dtype，LSE 为 FP32。BSND O/LSE 为 `[B,Sq,Hq,512]` / `[B,Hq,Sq]`，TND 为 `[Tq,Hq,512]` / `[Hq,Tq]`。请求 LSE 时 func/varlen 返回二项，kvcache 返回四项；未 split 的两个 accumulator 为同设备空 tensor。
- **记录校验：** 前缀/长度值留在设备检查；显式 v2/v4 逐项匹配当次长度，v4 还匹配 Q 前缀。记录可克隆或使用连续偏移视图，损坏/过期记录由消费者 Trap 拒绝。空输出仍校验记录和长度。
- **空值：** 空 batch/query 保留空 shape；非零容量中的零 KV 样本 `O=0、LSE=-inf`，全局空 KV 为 `O=0、LSE=+inf`。

**仍拒绝：** varlen `seqused_q/k`、Q-used、非连续外层 stride、PA、causal/window、自动或多 split、192/128 prefill，以及 KV 追加、内置 rotary 等范围外组合。已有 BSND cache 的实际 KV 长度通过 `cache_seqlens` 支持，内部通用前向用 `seqused_k` 传递。内部通用混合入口仍拒绝 `max_seqlen_k`。fake/compile 已覆盖，完整 graph replay 留在 M16；目标包含 910C，生产平台检查仍只开放 910B。

以上入口与拒绝边界对应 [Python 包装/fake](../../flash_attn_npu_3/flash_attn_npu_interface.py)、[C++ 通用入口](../../csrc/ascend910/flash_attn_npu_3/flash_api.cpp)、[专用前向](../../csrc/ascend910/flash_attn_npu_3/mla/mla_forward.cpp) 及 [静态校验](../../csrc/ascend910/flash_attn_npu_3/mla/mla_validation.h)；完整生产分派表见 [design 6.5](design.md#65-生产接入与构建)。

## 独立 used 生成

M06d1 的 [`ParseUsedLengths` / `BuildUsedSchedule`](../../csrc/ascend910/flash_attn_npu_3/mla/mla_used_schedule.h) 现由 [v5 生成器](../../csrc/ascend910/flash_attn_npu_3/mla/mla_used_metadata.cpp) 在 CPU/AICPU 共用。覆盖 BSND Q/KV、紧凑 TND Q/KV、TND Q + BSND KV；缩短 used 不移动后续样本，缺省使用完整分配段。

新记录为 **`int32[410+8B]`、version/kind=5/5**，保存完整调度、两轴布局/总容量/max 上界、显式 used 标记，以及每样本的 64 位物理起点、分配长度和实际长度。分配与 used 的源数组语义分别保留，详见 [design 5.7](design.md#57-v5分配与-used-记录) 和 [6.2.7](design.md#627-分配长度与-used-的-cpu-归一)。

[`ComputeMlaUsedMetadata`](../../csrc/ascend910/flash_attn_npu_3/mla/mla_used_metadata.aicpu) 直接读取设备前缀/used，无长度 D2H。安全输出上的生成失败清零八字头后的全部载荷；输出尺寸/地址不安全或输入与输出重叠则直接拒绝写入。四组只读输入可相互别名，空 batch 的显式零长度数组仍保留 used 标记。旧 v1–v4 消费者拒绝 v5，且不改写解码输出。

**目前仅独立生成 probe 发射 v5。尚无 used 设备消费者、attention 输出或公开接入**；扩展链接新对象不代表公开能力已开放。现有公共入口和拒绝边界保持上表所述。

## 最近闭环证据

归档目录为 **`build/mla-m06d2a-used-metadata/`**，总索引 `acceptance.json`；`*-command.json` / `*-exit.json` 保存命令、环境和退出码，`source-snapshots/` 与 `production-final/` 保存本步源码和构建产物。验收共 **1,378 项去重 pytest 通过**，没有跳过项：

| 验收面 | 本次结果 | 证据（相对归档目录） |
| --- | --- | --- |
| 上一步最小复查 | 4 项 used CPU 检查，ASan/UBSan；上一文档静态命令通过，原结果哈希未变 | `baseline-used.xml`、`precheck.json` |
| v5 CPU 与 sanitizer | 182 项；U01/T01、三布局、缺省/显式 used、零容量/零 used、错误/溢出、缓冲重叠、奇数 offset、旧协议隔离、64 位跨度 | `cpu-verified.xml`、`cpu-verified/`、`cpu-verified-driver-exits.json` |
| v5 910B 生成 | 35 项；四数组更新、失败清零与重用、空 batch、别名、跨 stream/allocator、64 位容量 | `device-smoke.xml`（7 项）、`device-verified.xml`（28 项）及同名目录 |
| 固定源差分 | CPU 6,474 条记录：原生非 FD 5,327，FA3 禁止 split 映射 1,147；AICPU 3,168 条：2,622 / 546；完整字段/跨度/任务一致 | `differential-summary.json`、`cpu-verified/`、`device-verified/`、`source-verified/`、`device-source/` |
| v1–v4 AICPU 回归 | 129 项；字段/源对照、错误状态、空值、更新、生命周期 | `legacy-metadata.xml`、`legacy-metadata-tmp/` |
| 普通 FA3 / 接口边界 | 839 项；普通 metadata、compile、固定官方签名、MLA 契约与未开放 used 拒绝 | `ordinary-regressions.xml`、`ordinary-nodes.json` |
| 910B 混合公开链路 | 170 项；双 dtype 源精度、None/整数长度、空值/返回、stream/allocator/别名 | `legacy-public.xml`、`legacy-public-attention/` |
| 构建 / 质量 | 19 项；含 v5 header/args/schedule.cpp 的增量失效；只新增一个链接对象；质量钩子与 diff 检查通过 | `build-tests.xml`、`production-build-manifest.json`、`quality-staged.log` |

CPU 验收使用归档 `pytest_persistent.py`，七个持续驱动正常退出，ASan/UBSan/LSan 无报告。首轮 CPU smoke 中，测试把“非空地址加巨大 count 导致输出区间重叠”误期望为可写失败记录；已按协议拆开重叠拒写与空指针错误计数，并完整补跑。该失败轮次保留在 `cpu-smoke.*`，不计作通过；本次没有遗留失败。

最终哈希核验发现 CPU 测试文件在验收后被 formatter 添加了一行空行；保留了与原 manifest 哈希完全匹配的测试源码，确认前后 AST 一致，见 `test-format-audit.json` / `tested-sources/`。生产源码、对象和扩展哈希均匹配；该差异未被记作功能重跑。

独立生成耗时为双 dtype 各七例，5 次预热、20 次测量：FP16 各例中位数 **261.70–282.98 μs**，BF16 **261.92–269.62 μs**。包含 ctypes、双事件和逐次同步，输入/输出 buffer 预先分配；原始数据见 `device-verified/timing-1.json` / `timing-2.json`。这不是 attention 或公开 E2E。未修改 attention Kernel / Block / Tile，未重跑其性能；最近公开混合 E2E 仍为 `build/mla-m06c3c-kvcache/performance-acceptance.json`，不作为 used 输出能力证据。

## 环境与构建

开发前核对上一功能归档的 **6,074 个文件、237 个未改源码依赖、24 个对象和扩展哈希**；其余一个 Python 依赖仅上一文档提交的 docstring 变化，已由该步静态检查复核。CATLASS 与只读 CANN 源工作区在本次前后保持一致。

| 项目 | 本次实测记录 |
| --- | --- |
| Python / PyTorch | `/usr/local/python3.11.15/bin/python3` 3.11.15；torch `2.7.1+cpu`、torch_npu `2.7.1` |
| CANN / 编译器 | `/usr/local/Ascend/cann-9.0.0-beta.2`；bisheng clang 15.0.5 |
| 芯片与执行环境 | 八张 Ascend910B3；选择健康空闲卡：v5 用 7，普通 FA3 用 4，旧 metadata 用 5，混合前向用 3；构建/设备/完整 sanitizer 在沙箱外运行 |
| CATLASS / 只读 CANN 源 | `769cd40a8716b28650b6bebb08db4834eea4462f` / `ca18aee610729ca5270a44e2d84c751bc687fbf0`；CATLASS 干净，CANN 原有 67 条删除/未跟踪记录保留 |
| 生产重建 | 新增 `mla_used_metadata.o`；原有 24 个对象哈希不变；25 个对象、242 个源码依赖归档 |
| 实际加载扩展 | `flash_attn_npu_3/flash_attn_npu_3.cpython-311-aarch64-linux-gnu.so`；SHA256 `380d5f9e7188ddeb792ca0a13c159a0765a1ae5c56baa2316821ec12eeba18fb` |

## 最小复现与交接

完整命令见归档的 `cpu-verified-command.json`、`device-smoke-command.json`、`device-verified-command.json`、`legacy-metadata-command.json`、`ordinary-regressions-command.json` 和 `legacy-public-command.json`。下列最小 CPU/设备复查使用新的输出目录；实机前先检查健康空闲卡并设置 `MLA_DEVICE`：

```bash
MLA_SCHEDULE_SANITIZE=1 ASAN_OPTIONS=detect_leaks=1 UBSAN_OPTIONS=halt_on_error=1 MLA_USED_METADATA_CPU_DIR=build/mla-m06d2a-used-metadata/recheck-cpu MLA_USED_SCHEDULE_ARTIFACT_DIR=build/mla-m06d2a-used-metadata/recheck-source MLA_PERSISTENT_LABEL=recheck-cpu python3 build/mla-m06d2a-used-metadata/pytest_persistent.py -x -q tests/test_flash_attn_npu_v3_mla_used_metadata.py -k 'buffers_guards or u01_t01 or empty or dtype_static' --basetemp=build/mla-m06d2a-used-metadata/recheck-cpu-tmp

ASCEND_RT_VISIBLE_DEVICES=${MLA_DEVICE:?set MLA_DEVICE} OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 MLA_USED_METADATA_DEVICE_DIR=build/mla-m06d2a-used-metadata/recheck-device python3 -m pytest -x -q tests/test_flash_attn_npu_v3_mla_used_metadata_aicpu.py -k 'failures_empty or four_input or each_array or alias_cross' --basetemp=build/mla-m06d2a-used-metadata/recheck-device-tmp
```

源码/构建变化时先重建：`FLASH_ATTN_BUILD_VERSION=v3 FLASH_ATTN_BUILD_NPU=910 python3 setup.py build_ext --inplace`。质量命令为 `PYTHONPATH=/tmp/mla-m01-precommit:$PYTHONPATH PRE_COMMIT_HOME=/tmp/mla-m01-precommit-cache ci/code_quality.sh --staged`，保留 CANN PYTHONPATH。只暂存本步文件，保留 `kernel_meta/` 未跟踪产物。

下次先核验源码、构建和运行环境，按受影响范围复查后推进 M06d2b；不要把 v5 独立生成视为 used 已可执行 attention。910C 和完整 graph replay 仍未实机验收。
