# MLA 当前状态

当前闭环为 **M06d3a：内部 BSND Q-used 接入**，2026-10-10 基于 `feat/mla-cann-migration` 的 `000ce1b` 完成。M06d3 已按入口验收边界拆分，唯一下一步为 **[M06d3b：TND used 入口接入](plan.md#下一步m06d3b)**，尚未开始。本步提交用 `git log -1 -- docs/mla/status.md` 定位。

目标契约与实现结构见 [design](design.md)，步骤及完成条件见 [plan](plan.md)。本次只开放内部 BSND Q-used，公开 varlen used 仍未开放。

## 当前可调用能力

共同边界为 **910B C1V2、FP16/BF16、512+64 共享 KV、Hkv=1、连续布局、无梯度**；允许非零 storage offset。BSND 每样本 KV 容量与紧凑 TND 的 `max_seqlen_k` 均为 0..16384；紧凑 TND 总 Q/KV 各不超过 `INT32_MAX`，总 KV 可以超过 16384。BSND 总容量与物理起点使用 64 位字段。完整界限见 [design 6.2](design.md#62-调度字段与边界)。

| 入口 | 输入与长度 | 调度记录 |
| --- | --- | --- |
| `flash_attn_func` | BSND 整容量 | 内部 v1：`int32[401]` |
| `flash_attn_varlen_func` | 紧凑 TND Q/KV；两组同设备连续 `int32[B+1]` 前缀，max 参数为每段上界 | 每次内部 v3：`int32[405+4B]`；无公开显式复用参数 |
| `flash_attn_with_kvcache`，BSND Q | Q 整容量，只读 BSND cache；KV 长度为 None、`0..Sk` 整数或同设备连续 `int32[B]` | 无长度且无记录走 v1；否则内部 v2 或显式 v2：`int32[401+B]` |
| `flash_attn_with_kvcache`，TND Q | Q 使用完整分配段，KV 为 BSND；必须给 Q 前缀和 `max_seqlen_q` 上界；KV 长度同上 | 内部 v4 或显式 v4：`int32[406+5B]` |
| `get_scheduler_metadata` | 必须给设备连续 `int32[B]` KV 长度，不接受 None/整数 | 无 Q 前缀生成 v2，有 Q 前缀生成混合 v4 |
| **内部 `_flash_attn_forward` / C++ `fwd`，BSND Q/KV** | 给出 `seqused_q:int32[B]`，可选 `seqused_k:int32[B]`；同设备连续，各值为 `0..本轴容量` | 每次内部 v5：`int32[410+8B]`；拒绝显式 metadata；无 Q-used 沿用 v1/v2 |

- **调用选项：** 仅 `num_splits=1`；func/varlen/内部前向默认 1，kvcache/metadata 默认 0，须显式传 1。无 mask，`softcap=0`、`pack_gqa=None`、`sm_margin=0`、`attention_chunk=0`；metadata 的 `has_softcap=False`。scale 默认 `1/sqrt(576)`，显式值须在 FP32 有限范围，记录不固化 scale。
- **输出：** O 与输入同 dtype，LSE 为 FP32。BSND O/LSE 为 `[B,Sq,Hq,512]` / `[B,Hq,Sq]`，TND 为 `[Tq,Hq,512]` / `[Hq,Tq]`。请求 LSE 时 func/varlen 返回二项，kvcache 返回四项；未 split 的两个 accumulator 为同设备空 tensor。
- **记录与长度：** 值留在设备检查，不做 D2H。显式 v2/v4 匹配当次长度，v4 还匹配 Q 前缀；损坏/过期记录由消费者 Trap 拒绝，空输出仍检查记录与长度。v5 逐次生成并完整检查两轴 used，即使另一轴 used 为零。
- **空值与有效区：** 空 batch/query 保留空 shape；非零容量中的零 KV 样本有效行 `O=0、LSE=-inf`；全局空 KV 初始化整个输出分配区为 `O=0、LSE=+inf`。used 不改变后续样本物理起点，其他未使用 query 槽位不承诺输出值。

**仍拒绝：** varlen `seqused_q/k`、内部混合 Q-used、Q-used 与显式 metadata、非连续外层 stride、PA、causal/window、自动或多 split、192/128 prefill，以及 KV 追加、内置 rotary 等范围外组合。内部通用混合入口仍拒绝 `max_seqlen_k`。fake/compile 已覆盖，完整 graph replay 留在 M16；目标包含 910C，生产平台检查仍只开放 910B。

入口对应 [Python 包装/fake](../../flash_attn_npu_3/flash_attn_npu_interface.py)、[C++ 通用入口](../../csrc/ascend910/flash_attn_npu_3/flash_api.cpp)、[专用前向](../../csrc/ascend910/flash_attn_npu_3/mla/mla_forward.cpp) 与 [静态校验](../../csrc/ascend910/flash_attn_npu_3/mla/mla_validation.h)。

## 本步实现边界

`BsndUsedForward` 只按静态容量预算 launch/workspace，复用既有 v5 AICPU 生成器、设备消费者和 `MlaUsed<InputT,false>`。新增 FP16/BF16 [生产 launcher](../../csrc/ascend910/flash_attn_npu_3/mla/mla_bsnd_used_launch_impl.hpp)，每个核在创建 TPipe、同步或写输出前完成 v5 校验；生成与 attention 使用同一组当次 used 描述符。数值 Kernel / Block / Tile 和 v5 协议未改动。

v5 的纯 context/输入描述符结构从 consumer 头移到 `mla_used_metadata.h`，字段与 ABI 保持不变，避免 host launcher 引入设备函数。B=0 的显式空 used 使用非空地址、零跨度描述符保留存在标记，不读取该地址。生成 worker 与当前 attention stream 都登记非空 used storage；没有新增公共参数、公开 v5 生成或显式复用入口。

三布局独立 attention 仍由 [used probe](../../tests/mla_used_attention_probe.cpp) 覆盖；本步生产只接 BSND。TND 与混合 used 的公开/内部接入留在下一步。

## 最近闭环证据

归档目录为 **`build/mla-m06d3a-bsnd-used/`**，总索引 `acceptance.json`；`*-command.json` / `*-exit.json` 保存命令、环境与退出码，`source-snapshots/` 和 `production-final/` 保存最终源码、对象及扩展。**本步 1,504 项去重 pytest 通过，无跳过项、无未解决失败。**

| 验收面 | 结果 | 证据（相对归档目录） |
| --- | --- | --- |
| 上一步最小复查 | 26 项；重建双 dtype probe，覆盖四数组更新、全局空 KV、alias/clone、跨流和 stale-used Trap | `baseline.xml`、`baseline-attention/`、`baseline-producer/` |
| v5 ABI / 消费者 | CPU 140 项、910B 8 项；固定源调度、字段/跨度/端点、四数组更新、生命周期和真实 Trap | `used-cpu.xml`、`used-device.xml`、`used-cpu-driver-exits.json`、`abi-device/` |
| 内部 BSND used | 291 项；真实/fake/C++、U01、M/S2 边界、16384 KV、poison/offset、默认 KV-used、空容量/零 used、scale、长度更新、alias/clone、compile、跨流 allocator | `production-tests.xml` 的 271 项通过、`production-completion.xml` 的 20 项通过；`attention/*/production-accuracy.jsonl` |
| 固定源调度 / 数值 | 7 组调度：非 FD 5 / 禁 split 映射 2；两 dtype 有效 O、非空行 LSE 与同调度源逐元素一致，再验独立双数学 reference；零 KV LSE 符号单独检查 | `production-tmp/test_production_source_schedul0/production-bsnd-used*.json`、`production-source/`、`attention/` |
| 生产真实 Trap（包含在 291 项内） | 两 dtype 各 8 项；Q/K 负数、超容量、另一轴 used 为零、全局空 Q/KV 时仍拒绝坏值 | `production/{fp16,bf16}/trap-*/runtime-error.json` 及命令/日志 |
| 普通 FA3 / 接口 | 833 项；普通 metadata、compile、签名和 MLA 契约 | `ordinary-regressions.xml`、`ordinary-nodes.json` |
| 既有长度入口 | 187 项；BSND/TND/混合返回、长度更新、跨流/allocator、compile 与仍未接入选项的拒绝 | `legacy-public.xml` |
| 构建 / 质量 | 19 项；生成实例、AICPU/header/设备依赖失效；生产加载核验、质量钩子和 diff 检查 | `build-tests.xml`、`production-build-manifest.json`、`quality-staged.log` |

CPU 四个持续驱动正常退出，ASan/UBSan/LSan 无报告。新入口首轮在 271 项通过后遇到 TorchDynamo 枚举 Tensor 方法与禁止主机读回 mock 的冲突；改为分别检查 eager enqueue 与 compiled 结果，20 项补齐通过。首轮 host 构建的设备头引用错误、性能驱动的路径类型错误均已修正并重跑；原始失败日志保留，不计作通过。新入口、入口回归及性能验收均在最终生产构建和加载核验后执行，后续生产源码哈希未变。

## 组件性能与限制

同卡同进程交替比较固定源非 FD Kernel 与生产 BSND used，FP16/BF16 各 **7 组固定用例**，使用相同输入、基本块、核数与端点；seed=20261010，CPU 亲和性 128..135，预热 5 次，五轮各 20 个 wall 样本和 10 个 profiler 样本。每 dtype 保存 1,400 个 wall 样本、350 个源 Kernel、350 个生产 Kernel 和 350 个 AICPU 生成样本。

| dtype | 源 Kernel 中位数范围 | 生产 Kernel 中位数范围 | 逐例生产/源比值范围 | v5 AICPU 中位数范围 |
| --- | --- | --- | --- | --- |
| FP16 | 20.78–949.52 μs | 46.55–963.12 μs | 1.014–2.240 | 23.93–37.94 μs |
| BF16 | 21.03–954.24 μs | 40.70–955.97 μs | 1.002–1.935 | 23.66–38.55 μs |

逐例结果为 `performance-{fp16,bf16}/summary.json`，原始数据为同目录 `wall.json`、`kernel-samples.json` 和 profiler CSV。生产内部 wall 范围为 FP16 428.03–1307.62 μs、BF16 426.54–1308.39 μs；包含 Python 内部入口、逐次 v5 生成、输出/workspace 分配、发射与完成同步。源 probe 预分配缓冲，wall 分别为 248.07–1188.93 / 250.11–1190.90 μs，不能将二者包装成公开 E2E 对照。

**本步存在组件开销，不声称不劣化。** 最大比值为全零 KV-used 的低计算量用例。BSND Q-used 无公开 FA3 等价入口；公开 varlen used 的适用 E2E、生成与 R=1/4 调用段留在 M06d3b。既有公开混合 E2E 证据仍为 `build/mla-m06c3c-kvcache/performance-acceptance.json`。

两 dtype profiler 的 framework DB / flow / step 关联导出失败，保存在 `profiler-diagnostics.json` 与 `performance-*/profiler/*/logs/`，未记为完整 framework trace 通过。计时使用独立调用顺序与原始 CANN `op_summary`：每 dtype 1,050 条任务在数量、名称、设备、时间戳和时长上与 `kernel_details.csv` 逐项匹配，包含 350 次生成与其后的生产 Kernel。

## 环境与构建

接手核验上一闭环 **2,269 个归档文件、242 个源码依赖、25 个对象和扩展哈希**。Python/NPU、sanitizer、编译与实机验证按既有方式在沙箱外执行；沙箱内 `npu-smi` 初始化失败，不作为实机结果。环境与源码工作区差异见 `precheck.json`。

| 项目 | 本次实测记录 |
| --- | --- |
| Python / PyTorch | `/usr/local/python3.11.15/bin/python3` 3.11.15；torch `2.7.1+cpu`、torch_npu `2.7.1` |
| CANN / 编译器 | `/usr/local/Ascend/cann-9.0.0-beta.2`；bisheng clang 15.0.5 |
| 芯片 | 八张 Ascend910B3；新入口精度/Trap 用 7，独立消费者/旧入口/性能用 4，普通 FA3 用 6 |
| CATLASS / 只读 CANN 源 | `769cd40a8716b28650b6bebb08db4834eea4462f` / `ca18aee610729ca5270a44e2d84c751bc687fbf0`；CATLASS 干净，CANN 原有 67 条删除/未跟踪记录保留 |
| 生产重建 | 27 个链接对象、248 个源码依赖；新增两个 used 实例，`flash_api`、`mla_forward`、v5 AICPU 对象变化，其余 22 个旧对象哈希不变 |
| 实际加载扩展 | `flash_attn_npu_3/flash_attn_npu_3.cpython-311-aarch64-linux-gnu.so`；SHA256 `2a32ecc7e8d4b17b16a583aacea2b042673e048b21f740079f6c6a84185bd596`；与本次链接产物一致 |

## 最小复现与交接

完整命令见 `production-tests-command.json`、`production-completion-command.json`、`ordinary-regressions-command.json`、`legacy-public-command.json` 和 `performance-{fp16-final,bf16}-command.json`。先检查健康空闲卡并设置 `MLA_DEVICE`，以下从源码重建独立 probe，检查 U01、空值、长度更新/生命周期和空 KV 坏长度 Trap：

```bash
ASCEND_RT_VISIBLE_DEVICES=${MLA_DEVICE:?set MLA_DEVICE} OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 MLA_USED_ATTENTION_DIR=build/mla-m06d3a-bsnd-used/recheck-attention MLA_USED_METADATA_DEVICE_DIR=build/mla-m06d3a-bsnd-used/recheck-producer MLA_BSND_USED_DIR=build/mla-m06d3a-bsnd-used/recheck-production python3 -m pytest -x -q tests/test_flash_attn_npu_v3_mla_bsnd_used.py -k '(production_accuracy and spec0) or shapes_and_empty or alias_clone or fullgraph_and_length_updates or cross_stream_allocator_lifetime or (runtime_trap and empty_k)' --basetemp=build/mla-m06d3a-bsnd-used/recheck-tmp
```

共享头或生产源码变化后先运行 `FLASH_ATTN_BUILD_VERSION=v3 FLASH_ATTN_BUILD_NPU=910 python3 setup.py build_ext --inplace`，核对加载路径、设备依赖及对象哈希，再补受影响回归。Kernel/输出变化须重跑相关源/数学精度与双 dtype 性能。质量命令为 `PYTHONPATH=/tmp/mla-m01-precommit:$PYTHONPATH PRE_COMMIT_HOME=/tmp/mla-m01-precommit-cache ci/code_quality.sh --staged`，保留 CANN PYTHONPATH。只暂存本步文件，保留 `kernel_meta/` 未跟踪产物。

唯一下一步为 M06d3b；公开 used、910C 实机和完整 graph replay 尚未验收。
