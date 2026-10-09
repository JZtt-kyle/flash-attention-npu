# MLA 当前状态

本轮为**文档核验与整理**，以分支 `feat/mla-cann-migration` 的 `fcc7ae1` 为代码基准；未修改实现、重建扩展或运行 NPU 测试。最近实现闭环仍为 **M06c2a1：TND v3 记录与独立 AICPU 生成**，公开能力停留在 **M06b4：BSND 实际 KV 长度**。

唯一下一步为 **M06c2a2：TND v3 设备校验消费者**，本轮未开始。输入、边界与完成条件集中在 [plan 的下一步](plan.md#下一步m06c2a2)，继续在已确认范围内推进，无需重复确认。目标契约与迁移范围查 [design](design.md)。

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

[`mla_tnd_schedule.h/.cpp`](../../csrc/ascend910/flash_attn_npu_3/mla/mla_tnd_schedule.h) 解析两组紧凑 TND 前缀和并计算调度；[`mla_tnd_metadata.h/.cpp`](../../csrc/ascend910/flash_attn_npu_3/mla/mla_tnd_metadata.h) 与 [`mla_tnd_metadata.aicpu`](../../csrc/ascend910/flash_attn_npu_3/mla/mla_tnd_metadata.aicpu) 生成 v3 `int32[405+4B]`。协议、错误优先级及缓冲规则见 [design 5.4](design.md#54-v3紧凑-tnd-记录与独立-aicpu-生成)，段长上界与总容量的区别见 [design 6.2.3](design.md#623-紧凑-tnd-解析与调度)。

**已验收的是 CPU/AICPU 生成；TND 设备消费者、真实 Trap、attention 寻址和公开 TND 均未接通。** 生成器已编入扩展，但生产前向没有调用它。v1/v2 CPU 消费者拒绝 v3；TND Q + BSND KV、used 与 stride 仍按后续步骤接通。

## 最近实现闭环的证据

实现提交 `fcc7ae1`，证据目录 **`build/mla-m06c2a1-tnd-metadata/`**，总索引 `acceptance.json`。下表是该提交的归档结果，本轮只核验文件、命令、退出码、JUnit 与哈希，没有重跑测试。

| 验收面 | 归档结果 | 相对证据路径 |
| --- | --- | --- |
| CPU v3 与 TND 调度 | 63 + 111 项通过；v3 固定源差分 1,206 组（782 原生非 FD / 424 无 split 映射） | `final.xml`、`final/`、`final-schedule/` |
| ASan/UBSan | 同一 174 项复验通过，未关闭 LeakSanitizer；完整差分输出哈希与常规运行一致 | `sanitized.xml`、`sanitized/`、`sanitized-schedule/` |
| 910B TND AICPU | 17 项通过；双 dtype、600 组固定源差分、非法前缀/失败复用/前缀更新/共享输入/跨流生命周期 | `device-final.xml`、`device-final/`、`device-final-oracle/` |
| 旧 AICPU 与构建 | v1/v2 91 项、构建 9 项通过 | `legacy-final.xml`、`legacy-final-*/`、`build-tests.xml` |
| 普通 FA3 与公开 MLA | 923 项通过：普通 metadata/compile/签名 149、MLA 契约 582、公开 BSND 192 | `regressions.xml`、`regression-nodes.json`、`regression-artifacts/` |

合计 **1,214 项去重 pytest 通过，另有 174 项 sanitizer 复验通过**，最终选入项无失败或跳过。早期过时文档哈希导致的 precheck 失败及 Ruff E731 已修复，原始失败与后续通过日志均保留在该目录。临时构建与回归产物已归档到 `build-tests-artifacts/`、`regression-artifacts/`；交接时须保留完整证据目录。

性能证据分两类：

- **公开 BSND E2E**：最近性能闭环为 `276b7dc`（M06b4），见 `build/mla-m06b4-public/acceptance.json` 与 `performance-acceptance.json`。固定 26 个整容量 + 11 个实际长度用例中，双 dtype 各 20 个原生同路径用例的复用、生成加 R=1/4 前向均达标；另 17 个原生 FD 用例单列。本轮未重新测量。
- **TND 生成诊断**：`device-final/comparison-*.json` / `timing-*.json`，均匀同调度 v3 中位数 237–248 µs、v1 188–198 µs，比值 1.254–1.271，含 ctypes、事件、前缀登记与逐次同步。这不是 TND attention 或公开 E2E 验收。

## 环境与本轮复核

上轮实测环境：Python `/usr/local/python3.11.15/bin/python3` 3.11.15、torch `2.7.1+cpu`、torch_npu `2.7.1`、CANN `/usr/local/Ascend/cann-9.0.0-beta.2`、bisheng clang 15.0.5、g++ 11.4.0；机器为八张 Ascend910B3。具体设备、CPU 亲和性及环境在 `precheck.json`、`*-command.json` 和 `*-exit.json`。上轮 NPU 与完整 sanitizer 验证在沙箱外执行；本轮未重新探测设备健康、占用或导入扩展，下次实机前必须重新检查。

本轮核实 CATLASS 仍为 `769cd40a8716b28650b6bebb08db4834eea4462f` 且干净；CANN 仍为 `ca18aee610729ca5270a44e2d84c751bc687fbf0`，原有 67 条删除/未跟踪记录未变。`production-build-manifest.json` 的 218 个源码依赖、19 个链接对象及扩展文件哈希均匹配；上轮记录的加载路径为 `flash_attn_npu_3/flash_attn_npu_3.cpython-311-aarch64-linux-gnu.so`，SHA256 为 `4c87d3124a51162d4f0bb7d9801c918d8811cca8c31716853f6eccbebceb90db`。本轮只核对磁盘产物，不将其记为新的加载验收。

文档核验结果在 **`build/mla-docs-review-20261009/`**：`evidence-check.json` 核对上轮 1,656 份归档文件、构建依赖、对象与固定源；`doc-checks.json` 核对链接/锚点、表格、52 个步骤状态、官方位置参数/默认值与当前代码边界；`quality-staged.log` 记录本轮质量检查。历史文档哈希按实现提交 `fcc7ae1` 的版本比较，避免把此次文档更新误判为实现漂移；历史 `audit.py` 绑定当时 HEAD 和暂存列表，不能在新提交上原样重跑。

## 复核与最小复现

仅核验本轮文档与历史证据：

```bash
python3 build/mla-docs-review-20261009/verify_evidence.py
python3 build/mla-docs-review-20261009/check_docs.py
```

下次实现或环境变化时，先按上轮完整命令重跑受影响检查；下面是 TND 生成组件与公开入口的最小入口。新证据须使用新目录，不能覆盖归档；NPU 命令中的 `MLA_DEVICE` 先设为已确认空闲的设备。

```bash
MLA_TND_METADATA_CPU_DIR=build/mla-tnd-wire-recheck MLA_TND_SCHEDULE_ARTIFACT_DIR=build/mla-tnd-wire-oracle-recheck python3 -m pytest -x -q tests/test_flash_attn_npu_v3_mla_tnd_metadata.py tests/test_flash_attn_npu_v3_mla_tnd_schedule.py
MLA_SCHEDULE_SANITIZE=1 MLA_TND_METADATA_CPU_DIR=build/mla-tnd-wire-asan MLA_TND_SCHEDULE_ARTIFACT_DIR=build/mla-tnd-wire-oracle-asan python3 -m pytest -x -q tests/test_flash_attn_npu_v3_mla_tnd_metadata.py tests/test_flash_attn_npu_v3_mla_tnd_schedule.py
ASCEND_RT_VISIBLE_DEVICES=${MLA_DEVICE:?set MLA_DEVICE} OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 MLA_TND_METADATA_DEVICE_DIR=build/mla-tnd-device-recheck MLA_TND_SCHEDULE_ARTIFACT_DIR=build/mla-tnd-device-oracle-recheck python3 -m pytest -x -q tests/test_flash_attn_npu_v3_mla_tnd_metadata_aicpu.py
ASCEND_RT_VISIBLE_DEVICES=${MLA_DEVICE:?set MLA_DEVICE} OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 python3 -m pytest -x -q tests/test_flash_attn_npu_v3_metadata_signature.py tests/test_flash_attn_npu_v3_mla_bsnd_public.py
```

sanitizer 须支持完整 ASan/UBSan/LeakSanitizer，不能关闭检查后记为通过。文档步只运行文档检查；质量命令为 `PYTHONPATH=/tmp/mla-m01-precommit:$PYTHONPATH PRE_COMMIT_HOME=/tmp/mla-m01-precommit-cache ci/code_quality.sh --staged`，保留 CANN PYTHONPATH。本次文档闭环提交可用 `git log -1 -- docs/mla/status.md` 定位。
