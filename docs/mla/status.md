# MLA 当前状态

最近实现闭环为 **M06c2a2：TND v3 设备校验消费者**；分支 `feat/mla-cann-migration`。v3 的生成和校验已由独立驱动验收，公开能力仍为 **M06b4：BSND 实际 KV 长度**。

唯一下一步为 **M06c2b：TND 设备寻址与独立 attention 驱动**，本轮未开始。输入与完成条件见 [plan 的下一步](plan.md#下一步m06c2b)，继续在已确认范围内推进，无需重复确认。契约与迁移范围查 [design](design.md)。

## 当前可调用能力

| 入口或条件 | 当前行为 |
| --- | --- |
| 共同输入边界 | 910B C1V2、FP16/BF16、512+64 共享 KV、Hkv=1；连续 BSND，允许非零 storage offset；B/Sq/Sk 可为零，Hq>0，单样本 KV 容量 `Sk=0..16384`；Q 使用整容量，支持 S1G/S2 多块 |
| `flash_attn_func` | 整容量前向，内部生成 v1 `int32[401]`；没有公开长度或显式 metadata 参数 |
| `flash_attn_with_kvcache` | 只读 BSND；`cache_seqlens` 接受 None（整容量）、`0..Sk` 整数或同设备连续 `int32[B]`，支持短/非均匀/混合与全零长度；无长度、无显式记录时走 v1，否则走 v2 |
| `get_scheduler_metadata` | 64/512 分支生成 v2 `int32[401+B]`，必须传同设备连续 `int32[B]` 的 `cache_seqlens`；不接受 kvcache 的 None/整数便利输入。真实/fake/C++ 外形一致 |
| 显式记录复用 | 只读 kvcache 接受 v2；复核当次长度，省略长度表示整容量。旧 v1、损坏或过期记录拒绝；记录可克隆或用连续偏移视图复用，长度变化须重新生成 |
| 空行与返回 | 非零容量中的零 KV 样本 `O=0、LSE=-inf`，含端点外零尾部；全局空 KV 为 `O=0、LSE=+inf`。空 batch/query 保留空 shape，仍校验记录。kvcache 请求 LSE 返回四项，未 split 的两个 accumulator 为空 tensor |
| 调度选项 | 必须显式 `num_splits=1`；无 mask、`softcap=0`、`pack_gqa=None`、`sm_margin=0`。scale 由当次前向提供；静态界限见 [design 6.2.4](design.md#624-平台与溢出)，v2 另要求 `B<=2^32-1-401` |

尚未接通：TND 前向、Q-used、非连续外层 stride、PA、causal/window、自动或多 split、192/128 prefill 及其他调度选项。梯度、追加 KV、内置 rotary 等范围外组合继续明确拒绝。完整 graph replay 生命周期留在 M16；fake/compile 通过不代表 replay 已验收。目标仍包含 910C，当前生产平台只开放 910B；合成 C1V1 调度不代表 910C 实机验收。

v2 在 TPipe、attention 事件和输出写入前完成设备校验；非空容量使用平台全核 blockDim，工作核数来自设备记录，workspace 按最大 M 基块 256 预算。长度与记录无 D2H，沿用双事件与 allocator stream 生命周期。核验入口：[Python 校验与包装](../../flash_attn_npu_3/flash_attn_npu_interface.py)、[生产生成与发射](../../csrc/ascend910/flash_attn_npu_3/mla/mla_forward.cpp)、[设备消费](../../csrc/ascend910/flash_attn_npu_3/mla/mla_launch_impl.hpp)。

## 已完成的 TND 组件边界

[`mla_tnd_schedule.h/.cpp`](../../csrc/ascend910/flash_attn_npu_3/mla/mla_tnd_schedule.h) 解析紧凑 TND 前缀和并调度；[`mla_tnd_metadata.h/.cpp`](../../csrc/ascend910/flash_attn_npu_3/mla/mla_tnd_metadata.h) 与 [AICPU 生成器](../../csrc/ascend910/flash_attn_npu_3/mla/mla_tnd_metadata.aicpu) 生成 v3 `int32[405+4B]`。协议、边界与拒绝规则见 [design 5.4](design.md#54-v3紧凑-tnd-记录独立生成与设备校验)。

新增 [`mla_tnd_metadata_consumer.h`](../../csrc/ascend910/flash_attn_npu_3/mla/mla_tnd_metadata_consumer.h) 与 [设备适配](../../csrc/ascend910/flash_attn_npu_3/mla/kernel/mla_tnd_metadata.hpp)：逐项校验版本/类型/字数、dtype、总容量/max 上界、当前两组前缀和、起点/段长、变长 M 端点及零成本后缀；与 v1/v2 共用平台、空调度、workspace 和本地资源校验。输出使用独立 TND 调度类型；失败不改写调度，release 构建真实 Trap 在任何 attention 同步和输出/workspace 写入之前执行，无长度 D2H 或主机等待。

**已验收的是独立生成/校验链路，TND attention 寻址与公开 TND 尚未接通。** 本轮没有将 v3 发射接入生产前向；v1/v2 仍拒绝 v3。两组前缀均必须存在，包括 B=0 的 `[0]`；记录/输入允许 4 字节对齐的非零 offset。只读复用、别名输入、跨 stream 和释放/分配复用已验证；used、TND Q + BSND KV、实际设备寻址按后续步骤推进。

## 本轮闭环证据

证据目录 **`build/mla-m06c2a2-tnd-consumer/`**，总索引 `acceptance.json`。完整命令、环境、退出码在各 `*-command.json` / `*-exit.json`；JUnit、构建 manifest、原始异常、源快照和性能数据均保存在该目录，不覆盖上轮归档。

| 验收面 | 结果 | 相对证据路径 |
| --- | --- | --- |
| CPU 消费者与 sanitizer | TND 100、v1 121、v2 87 项通过；三套原生 ASan/UBSan/LeakSanitizer 边界/损坏记录驱动通过。TND 固定源差分 600 组：388 原生非 FD / 212 无 split 映射，含 6 组省略零成本尾部 | `sanitized.xml`、`sanitized/`、`sanitized-v1/`、`sanitized-v2/`、`sanitized-oracle/` |
| 910B TND 消费者 | 60 项通过；双 dtype、168 组固定源差分、28 个独立进程 release Trap（均小于 0.16 秒）、拒绝不写输出、空容量、前缀更新和生命周期 | `device.xml`、`device/`、`producer/`、`device-oracle/` |
| 构建与 v3 记录 | 9 + 63 项通过；新扩展实际加载路径、218 个源码依赖和 19 个链接对象已核对 | `build-tests-final.xml`、`wire-recheck/`、`production-build-manifest.json` |
| BSND 精度与旧 v2 设备消费 | 93 + 32 项通过；双 dtype 与固定 CANN 源/独立数学 reference，短/混合/全零/长 KV、offset、记录拒绝与生命周期 | `attention-legacy.xml`、`attention/`、`legacy-device/` |
| 普通 FA3 与公开 BSND | 923 项通过：普通 metadata/compile/签名 149、MLA 契约 582、公开 BSND 192 | `regressions.xml`、`regression-nodes.json`、`regression-artifacts/` |
| v1 前向、全局空容量与 TND 生成复核 | 20 + 234 + 17 项通过；含双 dtype 源精度、流生命周期、生产空记录真实 Trap 和 v3 AICPU | `v1-empty-producer.xml`、`v1-forward/`、`v1-empty-artifacts/`、`producer-recheck/` |

合计 **1,759 项去重 pytest 通过**，最终选入项无失败或跳过；CPU 首轮另有 100 项复验。`attention-legacy` 明确排除了 28 项既有运行时异常用例，未将它们计入通过数；本轮真实 Trap 证据来自新 v3 与生产空容量检查。

## 性能与证据限制

- **BSND 源组件对比**：FP16/BF16 各 12 个预定代表用例，精度后交替预热 5 次、5 轮各 20 个 wall 样本与 10 个 profiler 样本，CPU 亲和性 168..175、910B 卡 4。每 dtype 的 1,200 条 kernel 记录与原始 `op_summary` 逐项核对名称、时间和耗时，见 `performance-fp16/summary.json`、`performance-bf16/summary.json`。FP16 迁移/源 Kernel 比值 1.000–1.435；BF16 迁移/源 Kernel 比值 0.997–1.431；最大相对差异为单样本全零 KV 的微秒级校验开销。这是预分配组件对比，不宣称公开 E2E 不劣化。
- **TND 校验诊断**：双 dtype 各 3 个同调度用例交替比较 v3/v2，v3 中位数约 210–237 µs、比值 1.004–1.041；含 ctypes、stream 登记、诊断写回和同步，见 `validation-comparison.json`。它不是 TND attention 或公开 E2E。
- **Profiler 限制**：本轮 framework flow、step 与 framework DB 解析报错，完整 trace 不记为通过；上述 kernel CSV 与原始导出已独立核对，错误日志原样保留。最近公开 metadata E2E 闭环仍为 `276b7dc` 的 `build/mla-m06b4-public/performance-acceptance.json`，本轮未重测公开 E2E。

## 环境与本轮差异

Python `/usr/local/python3.11.15/bin/python3` 3.11.15、torch `2.7.1+cpu`、torch_npu `2.7.1`、CANN `/usr/local/Ascend/cann-9.0.0-beta.2`、bisheng clang 15.0.5、g++ 11.4.0。八张卡均为 Ascend910B3；开测前核验健康/占用，本轮 TND、v1/空容量在卡 7，普通 FA3/公开 BSND 在卡 5，BSND 组件在卡 4。详见 `precheck.json`、`npu-regression-precheck.log` 和各命令环境。

CATLASS 仍为 `769cd40a8716b28650b6bebb08db4834eea4462f` 且干净；CANN 源仓仍为 `ca18aee610729ca5270a44e2d84c751bc687fbf0`，原有 67 条删除/未跟踪记录保持不变。开始前复核上轮文档检查及 1,656 份归档证据通过；本轮生产构建只改变两个 MLA BSND 实例与 TND AICPU 对象，其余 16 个对象哈希不变，旧/新扩展与对象分别保存在 `production-before/`、`production-after/`；本轮生成的 `kernel_meta` 已归档到 `kernel-meta-artifacts/`。

新加载扩展仍为 `flash_attn_npu_3/flash_attn_npu_3.cpython-311-aarch64-linux-gnu.so`，SHA256 **`3f23e3d70dbed72b4e4e0ddbab35649ade20b193d90f84bedf3e5b850122d5d9`**。本轮独立消费者 release 库另有实际设备依赖与哈希，不把仅存在的 `.so` 当成构建/加载证据。

沙箱内 NPU 查询返回 `-8005`；首次构建测试的 LeakSanitizer 因 ptrace 限制失败，随后在沙箱外保留全部 sanitizer 检查重跑通过，见 `build-tests.log` / `build-tests-final.log`。首次 clang-format 自动改写后重新检查通过。上述受限/失败记录保留，不计为通过。

## 最小复现与交接

新证据使用新目录；设备先确认健康、空闲，再设置 `MLA_DEVICE`。完整选入用例及性能命令以本轮 `*-command.json` 为准。

```bash
MLA_TND_CONSUMER_CPU_DIR=build/mla-tnd-consumer-cpu-recheck MLA_TND_METADATA_CPU_DIR=build/mla-tnd-wire-recheck MLA_TND_SCHEDULE_ARTIFACT_DIR=build/mla-tnd-oracle-recheck python3 -m pytest -x -q tests/test_flash_attn_npu_v3_mla_tnd_metadata_consumer.py
MLA_CONSUMER_SANITIZE=1 MLA_TND_CONSUMER_CPU_DIR=build/mla-tnd-consumer-asan python3 -m pytest -x -q tests/test_flash_attn_npu_v3_mla_tnd_metadata_consumer.py tests/test_flash_attn_npu_v3_mla_metadata_consumer.py tests/test_flash_attn_npu_v3_mla_bsnd_metadata_consumer.py
ASCEND_RT_VISIBLE_DEVICES=${MLA_DEVICE:?set MLA_DEVICE} OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 MLA_TND_CONSUMER_DEVICE_DIR=build/mla-tnd-consumer-device-recheck MLA_TND_METADATA_DEVICE_DIR=build/mla-tnd-producer-recheck MLA_TND_SCHEDULE_ARTIFACT_DIR=build/mla-tnd-source-recheck python3 -m pytest -x -q tests/test_flash_attn_npu_v3_mla_tnd_metadata_device.py
ASCEND_RT_VISIBLE_DEVICES=${MLA_DEVICE:?set MLA_DEVICE} OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 python3 -m pytest -x -q tests/test_flash_attn_npu_v3_metadata_signature.py tests/test_flash_attn_npu_v3_mla_bsnd_public.py
```

sanitizer 须支持完整 ASan/UBSan/LeakSanitizer，不能关闭检查后记为通过。源码或构建变化时重跑受影响检查；历史验收脚本绑定本步基准和暂存列表，不在新提交上原样重跑。质量命令为 `PYTHONPATH=/tmp/mla-m01-precommit:$PYTHONPATH PRE_COMMIT_HOME=/tmp/mla-m01-precommit-cache ci/code_quality.sh --staged`，保留 CANN PYTHONPATH。本步提交可用 `git log -1 -- docs/mla/status.md` 定位。
