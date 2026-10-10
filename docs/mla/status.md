# MLA 当前状态

当前闭环为 **M06d3b1：公开 varlen used 接入**，2026-10-10 基于 `feat/mla-cann-migration` 的 `0a03e43` 完成。M06d3b 已按入口验收边界拆为 b1/b2，唯一下一步为 **[M06d3b2：内部混合 Q-used 接入](plan.md#下一步m06d3b2)**，尚未开始。本步提交用 `git log -1 -- docs/mla/status.md` 定位。

目标契约与实现结构见 [design](design.md)，步骤及完成条件见 [plan](plan.md)。本次开放公开 varlen 与内部 TND Q/KV 的可选 used；内部混合 Q-used 仍明确拒绝。

## 当前可调用能力

共同边界为 **910B C1V2、FP16/BF16、512+64 共享 KV、Hkv=1、连续布局、无梯度**；允许非零 storage offset。BSND 每样本 KV 容量与 TND 的 `max_seqlen_k` 均为 0..16384；TND 总 Q/KV 各不超过 `INT32_MAX`，总 KV 可以超过 16384。BSND 总容量与物理起点使用 64 位字段。完整界限见 [design 6.2](design.md#62-调度字段与边界)。

| 入口 | 输入与长度 | 调度记录 |
| --- | --- | --- |
| `flash_attn_func` | BSND 整容量 | 内部 v1：`int32[401]` |
| `flash_attn_varlen_func` | TND Q/KV；两组同设备连续 `int32[B+1]` 分配前缀，max 参数为每段分配上界；可选同设备连续 `seqused_q/k:int32[B]` | 任一 used 存在时每次内部 v5：`int32[410+8B]`，否则 v3：`int32[405+4B]`；无公开显式复用参数 |
| `flash_attn_with_kvcache`，BSND Q | Q 整容量，只读 BSND cache；KV 长度为 None、`0..Sk` 整数或同设备连续 `int32[B]` | 无长度且无记录走 v1；否则内部 v2 或显式 v2：`int32[401+B]` |
| `flash_attn_with_kvcache`，TND Q | Q 使用完整分配段，KV 为 BSND；必须给 Q 前缀和 `max_seqlen_q` 上界；KV 长度同上 | 内部 v4 或显式 v4：`int32[406+5B]` |
| `get_scheduler_metadata` | 必须给设备连续 `int32[B]` KV 长度，不接受 None/整数 | 无 Q 前缀生成 v2，有 Q 前缀生成混合 v4 |
| **内部 `_flash_attn_forward` / C++ `fwd`，BSND Q/KV** | 给出 `seqused_q:int32[B]`，可选 `seqused_k:int32[B]`；同设备连续，各值为 `0..本轴容量` | 每次内部 v5；拒绝显式 metadata；无 Q-used 沿用 v1/v2 |
| **内部 `_flash_attn_forward` / C++ `fwd`，TND Q/KV** | 前缀、max 和可选 used 与公开 varlen 相同 | 任一 used 存在时内部 v5，否则 v3；拒绝显式 metadata |

- **调用选项：** 仅 `num_splits=1`；func/varlen/内部前向默认 1，kvcache/metadata 默认 0，须显式传 1。无 mask，`softcap=0`、`pack_gqa=None`、`sm_margin=0`、`attention_chunk=0`；metadata 的 `has_softcap=False`。scale 默认 `1/sqrt(576)`，显式值须在 FP32 有限范围，记录不固化 scale。
- **输出：** O 与输入同 dtype，LSE 为 FP32。BSND O/LSE 为 `[B,Sq,Hq,512]` / `[B,Hq,Sq]`，TND 为 `[Tq,Hq,512]` / `[Hq,Tq]`。请求 LSE 时 func/varlen 返回二项，kvcache 返回四项；未 split 的两个 accumulator 为同设备空 tensor。
- **记录与长度：** 值留在设备检查，不做 D2H。显式 v2/v4 匹配当次长度，v4 还匹配 Q 前缀；损坏/过期记录由消费者 Trap 拒绝，空输出仍检查记录与长度。v5 逐次生成并完整检查分配前缀及两轴 used，即使另一轴 used 为零；缺省 used 使用整段分配。
- **空值与有效区：** 空 batch/query 保留空 shape；非零容量中的零 KV 样本有效行 `O=0、LSE=-inf`；全局空 KV 初始化整个输出分配区为 `O=0、LSE=+inf`。used 不改变后续样本物理起点，其他未使用 query 槽位不承诺输出值。

**仍拒绝：** 内部混合 Q-used、BSND Q-used 或 TND Q/KV 与显式 metadata 的组合、非连续外层 stride、PA、causal/window、自动或多 split、192/128 prefill，以及 KV 追加、内置 rotary 等范围外组合。内部通用混合入口仍拒绝 `max_seqlen_k`。fake/compile 已覆盖，完整 graph replay 留在 M16；目标包含 910C，生产平台检查仍只开放 910B。

入口对应 [Python 包装/fake](../../flash_attn_npu_3/flash_attn_npu_interface.py)、[C++ 通用入口](../../csrc/ascend910/flash_attn_npu_3/flash_api.cpp)、[专用前向](../../csrc/ascend910/flash_attn_npu_3/mla/mla_forward.cpp) 与 [静态校验](../../csrc/ascend910/flash_attn_npu_3/mla/mla_validation.h)。

## 本步实现边界

`UsedForward` 共用 BSND/TND 的 host 准备，只按静态容量预算 launch/workspace，复用既有 v5 AICPU 生成器、设备消费者与 `MlaUsed`。新增 FP16/BF16 [TND used launcher](../../csrc/ascend910/flash_attn_npu_3/mla/mla_tnd_used_launch_impl.hpp)，每个核在创建 TPipe、同步或写输出前完成 v5 校验，再执行 `MlaUsed<InputT,true>`。数值 Kernel / Block / Tile、v5 协议、生成器与消费者均未改动。

生成与 attention 使用同一组当次前缀/used 描述符。提交对象持有四组数组，worker/current stream 都登记其非空 storage。B=0 的显式空 used 沿用非空地址、零跨度描述符表示存在，不读取该地址。私有 `_mla_tnd_forward` 增加可选 used，公共签名不变；无公开 v5 生成或显式复用入口。

## 最近闭环证据

归档目录为 **`build/mla-m06d3b1-varlen-used/`**，总索引 `acceptance.json`；`*-command.json` / `*-exit.json` 保存命令、环境与退出码，`source-snapshots/` 和 `production-final/` 保存最终源码、对象及扩展。**最终 1,883 项去重 pytest 通过，无跳过项。** 首轮失败与后续复验分别保存，未将失败运行记为通过。

| 验收面 | 结果 | 证据（相对归档目录） |
| --- | --- | --- |
| 上一步最小复查 | 60 项；从源码重建双 dtype probe 与 v5 producer，复查 BSND used 精度、空值、更新和生命周期 | `baseline.xml`、`baseline-attention/`、`baseline-producer/` |
| 公开 varlen used | 468 项；真实/fake/私有 op/C++、T01、M/S2 边界、16384 KV、poison/offset、两轴缺省、空容量/零 used、scale、四数组更新、alias/clone、compile、跨流 allocator | `production-tests.xml`、`baseline-attention/*/production-varlen-accuracy.jsonl` |
| 固定源调度 / 数值 | 10 组调度：非 FD 映射 6 / 禁 split 映射 4；两 dtype 有效 O、非空行 LSE 与同调度源逐元素一致，再验独立双数学 reference；零 KV LSE 符号单独检查 | `production-tmp/test_production_source_schedul0/production-varlen-used*.json`、`production-source/` |
| 生产真实 Trap（包含在 468 项内） | 两 dtype 各 12 项；Q/K used 负数、超容量、另一轴 used 为零、全局空 Q/KV，以及前缀起点/尾值/递减/max 上界 | `production/{fp16,bf16}/trap-*/runtime-error.json` 及命令/日志 |
| 普通 FA3 / 接口 | 833 项；普通 metadata、compile、签名和 MLA 契约 | `ordinary-regressions.xml`、`ordinary-nodes.json` |
| 既有 MLA 入口 | 563 项；完整 BSND used 及 varlen/混合/kvcache 的返回、静态拒绝、更新、跨流/allocator、compile | `legacy-public-final.xml` |
| 构建 / 质量 | 19 项；生成实例、AICPU/header/设备依赖失效；生产加载核验、质量钩子和 diff 检查 | `build-tests.xml`、`production-build-manifest.json`、`quality-staged.log` |

旧 BSND BF16 长 KV 首轮在 4 号卡出现 AICore 超时；同卡单例复查及加载上一步扩展时，未改动的独立 CANN probe 也在生产调用前超时。5 号卡的性能准备同样在独立 FP16 源 probe 超时。根因尚未确定，不能据此认定硬件故障。6 号卡先分别同步源 probe 与当前生产，连续 20 轮共 60 次该例前向通过源/数学比较，再完整重跑上述 563 项通过；新入口与最终双 dtype 性能在 7 号卡通过。详见 `timeout-analysis.json`、`timeout-evidence/`、`timeout-*-command.json`；保留初始 `legacy-public.xml` 和单例失败日志。

首次性能采样后的解析器误把 torch 转换视为单个设备任务，已改为按完整任务序列归属并逐次核对；适配器同时去掉 LSE scatter 引入的多余转置。最终双 dtype 全矩阵重新采样，早期失败保存在 `performance-fp16.log` / `performance-fp16-final.log`，不计入最终验收。生产源码及扩展在全部最终精度、回归和性能验证期间哈希不变。

## 性能与限制

公开 E2E 使用固定源原生 FIA 与 [varlen used 对照](../../tests/mla_varlen_used_profile.py)，FP16/BF16 各 **10 组固定用例**。源侧每次所需的有效段压紧与输出恢复计入 E2E，used 等于分配长度的轴不复制。双方预热 5 次、五轮各 20 个 wall 样本，R=1/4；CPU 亲和性 128..135，同卡同进程交替测量。R=4 是连续四次前向，每次均含 v5 生成，没有显式记录复用。

实际 tiling key 与固定源调度确认每 dtype 为 **5 例非 FD、1 例空 KV、4 例 FD**。适用 6 例的两个调用段均满足生产/源中位数比值不超过 1：

| dtype | 适用 R=1 比值范围 | 适用 R=4 比值范围 | 适用用例索引 |
| --- | --- | --- | --- |
| FP16 | 0.505–0.966 | 0.387–0.808 | 0、2、6、7、8、9 |
| BF16 | 0.535–0.973 | 0.411–0.780 | 0、2、6、7、8、9 |

逐例生成、Kernel、转换、调用段总耗时与每次均摊值见 `performance-acceptance.json`；分派证明为 `native-dispatch-proof.json`。每 dtype 保存 4,000 个 E2E wall 样本、1,000 个转换 wall 样本、1,000 个 profiler 调用及 500 个 AICPU 生成任务，位于 `performance-{fp16,bf16}-card7/`。四例原生 FD 全部保留，单列为未迁移路径；长 KV 例 3 的 R=1/4 比值为 FP16 1.646/1.577、BF16 2.043/2.126，不计作同路径性能通过。

另用相同输入、基本块、核数和端点，对固定源非 FD Kernel 与生产 TND used 做双 dtype 各 10 例组件对照，seed=20261010。每 dtype 为 2,000 个 wall 样本、500 个源 Kernel、500 个生产 Kernel 和 500 个 v5 生成样本；驱动由既有 BSND profile 适配为 `component_profile.py`，逐例和原始数据在 `component-{fp16,bf16}/`。

| dtype | 源 Kernel 中位数范围 | 生产 Kernel 中位数范围 | 逐例生产/源比值范围 | v5 AICPU 中位数范围 |
| --- | --- | --- | --- | --- |
| FP16 | 16.63–965.18 μs | 31.31–980.58 μs | 1.016–1.883 | 24.74–38.88 μs |
| BF16 | 14.18–960.41 μs | 42.97–972.43 μs | 1.013–3.030 | 24.72–39.08 μs |

**组件仍有开销，不声称纯 Kernel 不劣化。** 最大比值是全零 KV-used 的低计算量用例。内部生产 wall 范围为 FP16 428.38–1332.67 μs、BF16 432.36–1326.73 μs；源 probe 预分配输出，与公开 E2E 口径不同。

profiler 出现 step_trace DB 警告，组件 profile 另有 framework DB 关联导出失败；见 `profiler-diagnostics.json` 及原始日志，未记为完整 framework trace 通过。计时使用独立调用顺序与原始 CANN `op_summary`：每 dtype 的公开 3,700 条、组件 1,500 条设备任务，均在数量、名称、设备、时间戳和时长上与 `kernel_details.csv` 逐项匹配。

## 环境与构建

接手核验上一闭环 **2,768 个归档文件、248 个源码依赖、27 个对象和扩展哈希**。Python/NPU、sanitizer、编译与实机验证按既有方式在沙箱外执行；环境与源码工作区差异见 `precheck.json`。

| 项目 | 本次实测记录 |
| --- | --- |
| Python / PyTorch | `/usr/local/python3.11.15/bin/python3` 3.11.15；torch `2.7.1+cpu`、torch_npu `2.7.1` |
| CANN / 编译器 | `/usr/local/Ascend/cann-9.0.0-beta.2`；bisheng clang 15.0.5 |
| 芯片 | 八张 Ascend910B3；新入口精度/Trap、公开及组件性能用 7，普通 FA3 和最终旧入口回归用 6；4/5 的超时另存证据 |
| CATLASS / 只读 CANN 源 | `769cd40a8716b28650b6bebb08db4834eea4462f` / `ca18aee610729ca5270a44e2d84c751bc687fbf0`；CATLASS 干净，CANN 原有 67 条删除/未跟踪记录保留 |
| 生产重建 | 29 个链接对象、251 个源码依赖；新增两个 TND used 实例，`flash_api` 与 `mla_forward` 对象变化，其余 25 个旧对象哈希不变，含既有 BSND 数值路径及 v5 AICPU |
| 实际加载扩展 | `flash_attn_npu_3/flash_attn_npu_3.cpython-311-aarch64-linux-gnu.so`；SHA256 `b88c02a9f9e0a70df9a38f014c12bb31bcd8e7406cac60d6adcb72bfc1462fa8`；与本次链接产物一致 |

## 最小复现与交接

完整命令见 `production-tests-command.json`、`ordinary-regressions-command.json`、`legacy-public-final-command.json`、`performance-{fp16,bf16}-card7-command.json` 和 `component-{fp16,bf16}-command.json`。先检查健康空闲卡并设置 `MLA_DEVICE`，加载上述 CANN `set_env.sh`，以下从源码重建独立 probe，检查 T01、空值、四数组更新/生命周期和空 KV 坏长度 Trap：

```bash
ASCEND_RT_VISIBLE_DEVICES=${MLA_DEVICE:?set MLA_DEVICE} OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 MLA_USED_ATTENTION_REUSE=0 MLA_USED_ATTENTION_DIR=build/mla-m06d3b1-varlen-used/recheck-attention MLA_USED_METADATA_DEVICE_DIR=build/mla-m06d3b1-varlen-used/recheck-producer MLA_VARLEN_USED_DIR=build/mla-m06d3b1-varlen-used/recheck-production python3 -m pytest -x -q tests/test_flash_attn_npu_v3_mla_varlen_used.py -k '(production_accuracy and spec0) or shapes_empty_and_defaults or four_array_updates or alias_clone or cross_stream_allocator_lifetime or (runtime_trap and empty_k)' --basetemp=build/mla-m06d3b1-varlen-used/recheck-tmp
```

共享头或生产源码变化后先运行 `FLASH_ATTN_BUILD_VERSION=v3 FLASH_ATTN_BUILD_NPU=910 python3 setup.py build_ext --inplace`，核对加载路径、设备依赖及对象哈希，再补受影响回归。Kernel/输出变化须重跑相关源/数学精度与双 dtype 性能。质量命令为 `PYTHONPATH=/tmp/mla-m01-precommit:$PYTHONPATH PRE_COMMIT_HOME=/tmp/mla-m01-precommit-cache ci/code_quality.sh --staged`，保留 CANN PYTHONPATH。只暂存本步文件，保留 `kernel_meta/` 未跟踪产物。

唯一下一步为 M06d3b2；内部混合 Q-used、910C 实机和完整 graph replay 尚未验收。
