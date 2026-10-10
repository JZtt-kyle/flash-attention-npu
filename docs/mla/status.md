# MLA 当前状态

最近闭环为 **M06c3a：公开紧凑 TND varlen 与内部 v3 生成**。分支为 `feat/mla-cann-migration`，验收日期 **2026-10-10**。唯一下一步为 **[M06c3b：官方 metadata 入口按 TND query 生成混合 v4](plan.md#下一步m06c3b)**，尚未开始。目标契约与范围见 [design](design.md)；当前闭环提交用 `git log -1 -- docs/mla/status.md` 定位。

## 当前可调用能力

以下均指公开 MLA 入口。共同边界为 **910B C1V2、FP16/BF16、512+64 共享 KV、Hkv=1、连续 BSND 或紧凑 TND**，允许非零 storage offset。梯度拒绝，Q 使用完整分配段。BSND 每样本 KV 容量 `Sk` 和 TND 的 `max_seqlen_k` 均为 0..16384；TND 总 Q/KV 容量各不超过 `INT32_MAX`，总 KV 可以超过 16384。其余字段、地址及资源界限见 [design 6.2](design.md#62-调度字段与边界)。

| 入口或条件 | 当前行为 |
| --- | --- |
| `flash_attn_func` | BSND 整容量前向，内部生成 v1 `int32[401]` |
| `flash_attn_varlen_func` | 紧凑 TND Q/KV，连续同设备 `int32[B+1]` 前缀（B=0 仍为 `[0]`），每次内部生成并消费 v3 `int32[405+4B]`；max 参数为上界，前缀值在设备校验 |
| `flash_attn_with_kvcache` | 只读 BSND；`cache_seqlens` 接受 None（整容量）、`0..Sk` 整数或同设备连续 `int32[B]`，含非均匀及全零长度。无长度且无显式记录时走 v1，否则走 v2 |
| `get_scheduler_metadata` | 64/512 BSND 分支生成 v2 `int32[401+B]`；要求同设备连续 `int32[B]` 长度，不接受 None/整数或 TND query |
| 显式记录复用 | 只读 BSND kvcache 接受 v2，逐项复核当次长度；旧 v1、损坏/过期记录拒绝。允许克隆/连续偏移视图；varlen 无公开显式记录参数，不扩展官方签名 |
| split 与选项 | 仅 `num_splits=1`；func/varlen 默认即为 1，kvcache/metadata 默认 0，须显式传 1。无 mask、`softcap=0`、`pack_gqa=None`、`sm_margin=0`、`attention_chunk=0` |
| scale | 缺省 `1/sqrt(576)`，显式值须在 FP32 有限范围内；记录不固化 scale |
| 输出与返回 | O 与输入同 dtype；BSND LSE 为 FP32 `[B,Hq,Sq]`，TND O/LSE 为 `[Tq,Hq,512]` / `[Hq,Tq]`。func/varlen 请求 LSE 返回二项；kvcache 返回四项，未 split 的两个 accumulator 为同设备空 tensor |
| 空行与空容量 | 非零总容量中的零 KV 样本 `O=0、LSE=-inf`，包括端点外零尾部；全局空 KV `O=0、LSE=+inf`。空 batch/query 保留空 shape，仍校验记录与前缀 |

**公开入口仍拒绝** TND Q + BSND KV 混合布局、used、非连续外层 stride、PA、causal/window、自动或多 split、192/128 prefill 和其他未接通选项。梯度、追加 KV、内置 rotary 等范围外组合也明确拒绝。fake/compile 已覆盖，完整 graph replay 生命周期留在 M16。目标包含 910C，生产平台当前仅开放 910B；合成 C1V1 调度不代表 910C 实机验收。

核验入口：[Python 包装](../../flash_attn_npu_3/flash_attn_npu_interface.py)、[C++ 静态边界](../../csrc/ascend910/flash_attn_npu_3/mla/mla_validation.h)、[生产生成与发射](../../csrc/ascend910/flash_attn_npu_3/mla/mla_forward.cpp)、[TND launcher](../../csrc/ascend910/flash_attn_npu_3/mla/mla_tnd_launch_impl.hpp)。v3 在每个 core 的 TPipe、attention 事件和写入前校验；host 只依据静态容量/max 上界预算 workspace，全核发射后工作核数来自设备记录。沿用双事件和 allocator stream 生命周期，无前缀 D2H 或稳态主机等待。

## 独立组件边界

| 布局 / 协议 | CPU 调度 | AICPU 生成 | 设备校验 | 独立 attention | 公开入口 |
| --- | --- | --- | --- | --- | --- |
| BSND Q/KV，v1/v2 | 已验收 | 已验收 | 已验收 | 已验收 | 上述 BSND 子集 |
| 紧凑 TND Q/KV，v3 | 已验收 | 已验收 | 已验收 | 已验收 | 已开放上述 varlen 子集 |
| TND Q + BSND KV，v4 | 已验收 | 已验收 | 已验收 | 已验收 | 拒绝，待 M06c3b/c |

TND 与混合布局复用 `MlaNonPaged` 的 Kernel / Block / Tile 数值路径。混合布局的五字 span 保持 64 位 `b*Sk` 物理 KV 起点，短 KV 不重排后续 cache；本步没有扩展其公开能力。协议及失败行为见 [v3](design.md#54-v3紧凑-tnd-记录与设备消费)、[v4](design.md#55-v4tnd-q-与-bsnd-kv-混合记录)。

## 最近闭环证据

归档为 **`build/mla-m06c3a-varlen/`**，总索引 `acceptance.json`；`*-command.json` / `*-exit.json` 保存完整命令、环境与退出码。接手时核验上一步 4,098 个归档文件、223 个生产源码依赖及 20 个对象，固定官方 FA3 快照另核验 8 个文件，未发现未解决的前步失败。以下为本步新运行结果。

| 验收面 | 结果 | 相对证据路径 |
| --- | --- | --- |
| TND 消费者与调度边界 | 100 项通过，ASan/UBSan/LeakSanitizer | `cpu-consumer.xml`、`cpu-artifacts/` |
| 公开紧凑 TND | 301 项去重通过；FP16/BF16、固定源/双 reference、空容量/非均匀/多块/offset、真实/fake/C++、非法前缀、跨流/allocator/compile | `public-varlen.xml`、`final-recheck.xml`、`tnd-attention/`、`tnd-producer/` |
| 普通 FA3 与公开 BSND | 923 项通过；包含契约/签名、普通 metadata/compile 与已有 BSND 入口 | `contracts.xml`、`regressions.xml`、`regression-nodes.json` |
| 构建 | 13 项通过；生成实例、AICPU 隔离、依赖与 sanitizer；最终重建和对象/加载路径审计通过 | `build-tests.xml`、`production-build-manifest.json` |

合计 **1,337 项去重 pytest 通过**。格式化后的最终构建另复查 343 项，包含新增的 14 项别名前缀/静态上界测试，其余不重复计数。首轮 **22 个独立进程真实 Trap** 覆盖双 dtype 的起始值、递减、尾值、max 上界和空总容量非法前缀；最终构建追加 4 个复查进程，全部通过，最长报错 **0.134 秒以内**。证据见 `public/`、`final-traps/` 和 `final-traps.xml`。

固定源同调度 O 与非空行 LSE 逐元素一致；独立数学 reference 按逻辑段长切片。非零容量的零 KV 行另验源 `+inf` / FA3 `-inf`，全局空容量按官方契约验收。生产每次调用都生成 v3，不转调原 CANN attention。

## 公开性能与边界

固定矩阵为 [公开 varlen 性能驱动](../../tests/mla_varlen_profile.py) 的十例，双 dtype 相同，包含非均匀/跨块/长 KV、零成本尾部和总 KV 超过 16384。公开前向每次内部生成 v3，R=1/4 均计入完整调用段；当前没有公开 v3 预生成/显式复用入口，因此不报告不存在的复用模式。

同一健康空闲 910B 卡 2、CPU 168..175，预热 5 次，5 轮交替采样、每轮各 20 个 wall 样本及 10 个 profiler 样本。所有本次构建和设备回归结束后重新采集完整矩阵，原始首轮数据保留在 `performance-fp16-packaged/`、`performance-bf16/`。首轮 BF16 后半段出现延迟升高，两个同路径用例未达标；波动原因未单独定位，不将该轮记为通过。

固定源原生分派的五个非 FD 用例（索引 0/4/6/8/9）承担 E2E 门槛，其余五例原生启用尚未迁移的 FD，全部保留并单列能力差距。分类由独立源调度及实际 runtime tiling key 共同核验，见 `native-dispatch-proof.json`。TND 基线沿用固定源码包的 296 个文件和三份 host 库，本步补建原生 FP16/BF16 TND binary；代码与实际加载/分派哈希见 `native-package.json`、`native-device/` 和 `vendor/`，只读 CANN 源仓不变。

最终构建的完整复测结果如下，比值均为 FA3 / 固定源原生 CANN；同路径逐例 R=1/4 均不劣化。

| dtype | 同路径 R=1 E2E 比值 | 同路径 R=4 E2E 比值 | 同路径 attention Kernel 比值 | AICPU 生成中位数 |
| --- | --- | --- | --- | --- |
| FP16 | 0.837..0.965 | 0.769..0.834 | 1.247..4.052 | 24.161 µs |
| BF16 | 0.866..0.932 | 0.788..0.814 | 1.208..4.169 | 24.380 µs |

本步 attention Kernel 含生产全核发射与 v3 校验，Kernel 时间并未优于原生；小型零计算例的比值最高。表中 E2E 包含生成、分配、发射及同步，不能由 Kernel 比值替代。原生 FD 的长 KV 例（索引 3）仍有差距：FP16 R=1/4 为 1.590/1.416，BF16 为 1.626/1.501，本步不宣称 FD 性能达标。

最终证据在 `performance-fp16-quiet/`、`performance-bf16-quiet/`，门槛汇总为 `performance-acceptance.json`。原始 `op_summary` / `kernel_details` 的条数、名称、设备、起始时间和耗时逐行一致，共 2,000 个 attention Kernel、1,000 个 AICPU 生成样本及 8,000 个 wall 样本。既有同调度 TND 组件性能见已核验的 `build/mla-m06c2b2d-mixed-attention/tnd-performance-*/`；本步没有改其数值 Kernel。

## 环境与构建

| 项目 | 本步核验 |
| --- | --- |
| Python / PyTorch | `/usr/local/python3.11.15/bin/python3` 3.11.15；torch `2.7.1+cpu`、torch_npu `2.7.1` |
| CANN / 编译器 | `/usr/local/Ascend/cann-9.0.0-beta.2`；bisheng clang 15.0.5 |
| 芯片 | 八张 Ascend910B3；卡 7 公开 TND，卡 4 普通/BSND 回归，卡 6 最终 Trap，卡 2 性能。沙箱内 NPU 初始化不可用，设备/构建及完整 sanitizer 在沙箱外执行 |
| CATLASS | `769cd40a8716b28650b6bebb08db4834eea4462f`，工作区干净 |
| 只读 CANN 源仓 | `ca18aee610729ca5270a44e2d84c751bc687fbf0`，原有 67 条删除/未跟踪记录未改变 |
| 最终生产重建 | 22 个对象，新增两个 TND dispatcher；旧对象中仅 `flash_api.o`、`mla_forward.o`、`mla_tnd_metadata.o` 改变，其余 17 个哈希不变；229 个生产源码依赖核验通过 |
| 实际加载扩展 | `flash_attn_npu_3/flash_attn_npu_3.cpython-311-aarch64-linux-gnu.so`；SHA256 `58e5b63a540470e8aab9c6fedeb7bfa82a8fe9b1c587bede5a2dc6bdb5864730` |

首次 host 构建因包含设备注解失败，已将纯 TND context 定义移至协议头并重建通过。一次格式化重建在链接前主动中断，避免替换运行中测试所加载的扩展；最终重建后已复查。原生 TND binary 首次未正确装入隔离 vendor，修复目录权限/打包后核验实际分派通过。各次原始失败日志保留，不能仅依据旧 `.so` 认定源码已验证。

## 最小复现与交接

完整命令及测试选择以归档 `*-command.json` 为准。以下只作最小复查，使用新目录并先确认健康空闲设备，设置 `MLA_DEVICE`；不将其计作完整 1,337 项验收。

```bash
MLA_CONSUMER_SANITIZE=1 MLA_SCHEDULE_SANITIZE=1 ASAN_OPTIONS=detect_leaks=1 UBSAN_OPTIONS=halt_on_error=1 python3 -m pytest -x -q tests/test_flash_attn_npu_v3_mla_tnd_metadata_consumer.py --basetemp=build/mla-varlen-cpu-recheck
FLASH_ATTN_BUILD_VERSION=v3 FLASH_ATTN_BUILD_NPU=910 python3 setup.py build_ext --inplace
ASCEND_RT_VISIBLE_DEVICES=${MLA_DEVICE:?set MLA_DEVICE} OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 MLA_TND_ATTENTION_DIR=build/mla-varlen-attention-recheck MLA_TND_METADATA_DEVICE_DIR=build/mla-varlen-producer-recheck python3 -m pytest -x -q tests/test_flash_attn_npu_v3_mla_varlen_public.py -k 'entries_and_return or prefix_updates_cross_stream_and_allocator or alias_prefixes_with_offsets or fake_allocation_bounds or (source_accuracy and None and spec0)'
```

性能命令见归档 `performance-*-quiet-command.json`，须使用核验过的 TND vendor 并设置 `ASCEND_CUSTOM_OPP_PATH`、对应 `LD_LIBRARY_PATH`。源 probe 会核验源码/库哈希，不用旧产物冒充当前构建。

质量命令为 `PYTHONPATH=/tmp/mla-m01-precommit:$PYTHONPATH PRE_COMMIT_HOME=/tmp/mla-m01-precommit-cache ci/code_quality.sh --staged`，保留 CANN PYTHONPATH；`git diff --check`、文档链接/锚点与暂存钩子通过。下一次接手先核对环境、源码与产物，再推进唯一下一步。
