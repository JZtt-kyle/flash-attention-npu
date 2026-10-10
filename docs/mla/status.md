# MLA 当前状态

公开能力停在 **M06b4：BSND 实际 KV 长度**；最近实现闭环为 **M06c2b2c：混合布局记录设备校验消费者**（`a5fb50e`）。分支为 `feat/mla-cann-migration`。

本次为 **2026-10-10 文档核验**：对照源码、固定官方 FA3 快照和既有验收归档修正文档，没有修改实现、重建扩展或运行 NPU 测试。唯一下一步仍为 **[M06c2b2d：混合布局设备寻址与独立 attention 驱动](plan.md#下一步m06c2b2d)**，尚未开始，属于已确认范围。目标契约与迁移范围见 [design](design.md)。

## 当前可调用能力

以下均指公开 MLA 入口。共同边界为 **910B C1V2、FP16/BF16、512+64 共享 KV、Hkv=1、连续 BSND**，允许非零 storage offset。Q 使用整容量，支持 S1G/S2 多块；B/Sq/Sk 可为零，Hq>0，单样本 KV 容量 `Sk=0..16384`。其他字段、地址及资源上界见 [design 6.2.4](design.md#624-平台与溢出)。

| 入口或条件 | 当前行为 |
| --- | --- |
| `flash_attn_func` | 整容量前向，内部生成 v1 `int32[401]`；没有公开长度或显式 metadata 参数 |
| `flash_attn_with_kvcache` | 只读 BSND；`cache_seqlens` 接受 None（整容量）、`0..Sk` 整数或同设备连续 `int32[B]`，支持短/非均匀/全零长度。无长度且无显式记录时走 v1，否则走 v2 |
| `get_scheduler_metadata` | 64/512 分支生成 v2 `int32[401+B]`；必须传同设备连续 `int32[B]` 的 `cache_seqlens`，不接受 None/整数。真实/fake/C++ 外形一致 |
| 显式记录复用 | 只读 kvcache 只接受 v2，逐项复核当次长度；省略长度表示整容量。旧 v1、损坏或过期记录拒绝。可克隆或用连续偏移视图复用；长度变化须重新生成，v2 要求 `B<=2^32-1-401` |
| split 与选项 | 仅支持 `num_splits=1`；`flash_attn_func` 默认即为 1，kvcache/metadata 生成器默认是 0，须显式传 1。无 mask、`softcap=0`、`pack_gqa=None`、`sm_margin=0`、`attention_chunk=0` |
| scale | 当次前向提供，缺省 `1/sqrt(576)`，显式值须在 FP32 有限范围内；记录不固化 scale |
| 输出与返回 | O 与输入同 dtype，LSE 为 FP32 `[B,Hq,Sq]`；func 请求 LSE 返回二项，kvcache 请求 LSE 返回四项，未 split 的两个 accumulator 为同设备空 tensor |
| 空行与空容量 | 非零容量中的零 KV 样本 `O=0、LSE=-inf`，含端点外零尾部；全局空 KV 为 `O=0、LSE=+inf`。空 batch/query 保留空 shape，仍校验记录 |

**公开入口仍拒绝** TND 前向、Q-used、非连续外层 stride、PA、causal/window、自动或多 split、192/128 prefill 及其他未接通选项；梯度、追加 KV、内置 rotary 等范围外组合也明确拒绝。fake/compile 已有覆盖，完整 graph replay 生命周期留在 M16。目标仍包含 910C，生产平台当前只开放 910B；合成 C1V1 调度不代表 910C 实机验收。

核验入口：[Python 校验与包装](../../flash_attn_npu_3/flash_attn_npu_interface.py)、[C++ 输入边界](../../csrc/ascend910/flash_attn_npu_3/mla/mla_validation.h)、[生产生成与发射](../../csrc/ascend910/flash_attn_npu_3/mla/mla_forward.cpp)、[设备消费](../../csrc/ascend910/flash_attn_npu_3/mla/mla_launch_impl.hpp)。v2 在 TPipe、attention 事件与输出写入前校验；非空容量按平台全核发射，工作核数来自设备记录，workspace 按最大 M 基块 256 预算。长度与记录无 D2H，沿用双事件和 allocator stream 生命周期。

## 独立组件边界

“已验收”仅指该列的组件；生产构建包含生成器不等于公开入口能调用。

| 布局 / 协议 | CPU 调度 | AICPU 生成 | 设备校验 | 独立 attention | 公开入口 |
| --- | --- | --- | --- | --- | --- |
| BSND Q/KV，v1/v2 | 已验收 | 已验收 | 已验收 | 已验收 | 已开放上述子集 |
| 紧凑 TND Q/KV，v3 | 已验收 | 已验收 | 已验收 | 已验收（M06c2b1） | 拒绝，待 M06c3 |
| TND Q + BSND KV，v4 | 已验收 | 已验收 | 已验收（M06c2b2c） | 未实现，下一步 M06c2b2d | 拒绝，待 M06c3 |

紧凑 TND 与 BSND 复用 `MlaNonPaged` 的 Kernel / Block / Tile 数值路径，O 为 `[Tq,Hq,512]`，LSE 为 `[Hq,Tq]`，计算前校验当次前缀。混合布局目前只校验 v4、Q 前缀和 KV 实际长度：KV 起点固定为 64 位 `b*Sk`，短 KV 不重排后续 cache；非空物理容量中的全零 KV 与空容量分别处理。协议字段与失败行为见 [v3](design.md#54-v3紧凑-tnd-记录与设备消费)、[v4](design.md#55-v4tnd-q-与-bsnd-kv-混合记录)，源码和独立驱动见 [架构表](design.md#6-代码结构)。

## 本次文档核验

证据目录为 **`build/mla-docs-audit-20261010/`**。`evidence-check.json` 记录只读核验：上次实现归档的 **3,057 个文件**、123 个验证源码、223 个生产源码依赖、20 个链接对象、扩展文件和三个 release 消费者的依赖/产物哈希一致；固定官方 FA3 快照的 8 个文件与归档一致。重新汇总 JUnit 与 Trap 报告，确认下节历史统计。CANN/CATLASS 提交及工作区状态与上次验收一致。

文档核对修正了 AICPU 构建清单、TND 架构和性能矩阵遗漏，以及不同入口的 `num_splits` 默认值说明。协议按 v1→v4 排列，共同生命周期集中说明；计划先列剩余步骤，再列已完成步骤。文档链接/锚点、表格、步骤一致性及暂存质量检查见同目录 `docs-check.json`、`quality-staged.log`；`git diff --check` 通过。本次只核验归档和静态内容，下节测试通过数不是本次新运行结果。

## 最近实现闭环证据

以下为 **M06c2b2c** 的历史结果。本次已核对其索引、原始报告及哈希，未重新执行测试。归档目录 **`build/mla-m06c2b2c-mixed-consumer/`**，总索引 `acceptance.json`；`*-command.json` / `*-exit.json` 保存完整命令、环境与退出码。

| 验收面 | 历史结果 | 相对证据路径 |
| --- | --- | --- |
| CPU sanitizer 与回归 | 598 项通过；混合消费者/生成/调度与 v1/v2/v3 消费者，ASan/UBSan/LeakSanitizer 开启 | `cpu-sanitizer.xml`、`cpu-sanitizer-artifacts/` |
| 910B 混合消费者 | 72 项通过；双 dtype release、固定源记录、损坏/过期/缓冲/端点/资源、复用与跨流生命周期 | `mixed-device.xml`、`mixed-device/`、`mixed-producer/`、`device-oracle/` |
| 既有布局与生成回归 | 225 项通过；TND/BSND 消费者、双 dtype attention 源/数学精度及混合 AICPU | `layouts.xml`、`tnd-attention/`、`bsnd-attention/`、`layout-artifacts/` |
| 普通 FA3 与公开 BSND | 923 项通过；普通 metadata/compile/签名、MLA 契约及公开 BSND | `regressions.xml`、`regression-nodes.json` |
| 构建检查 | 13 项通过；AICPU 后端隔离、增量依赖与序列化 sanitizer；生产重建及加载核验 | `build-tests.xml`、`production-build-manifest.json` |

合计 **1,831 项去重 pytest 通过**，CPU 初验 111 项不重复计数。双 dtype 共 **34 个独立进程真实 Trap**，最长报错 **0.067 秒**；拒绝用例检查调度或 O/LSE/workspace 哨兵不变，Trap 前用同一错误的诊断调用核对缓冲，异常 stream 不复用。B=262,146 用例检查超过 uint32 的 KV 起点与总容量，仅分配 metadata/长度，不能视为大 cache attention 验收。

校验耗时诊断为双 dtype 各四组、每组 20 个样本，中位 **164.2..268.8 μs**，含诊断写入、ctypes、stream 登记与同步，不代表混合 attention 或公开 E2E。混合 attention 精度与组件性能留在下一步。已有 attention 的最近组件性能见 `build/mla-m06c2b1-tnd-attention/acceptance.json`；公开 BSND 的历史同路径 E2E 见 `build/mla-m06b4-public/performance-acceptance.json`。这些性能结果本次均未重测，也不证明 TND 公开 E2E 或混合 attention 已达标。

## 环境与构建

以下是上次实现验收环境，来源为 `build/mla-m06c2b2c-mixed-consumer/precheck.json`；本次未查询设备健康/占用，也未重新加载扩展。

| 项目 | 已归档环境 / 产物 |
| --- | --- |
| Python / PyTorch | `/usr/local/python3.11.15/bin/python3` 3.11.15；torch `2.7.1+cpu`、torch_npu `2.7.1` |
| CANN / 编译器 | `/usr/local/Ascend/cann-9.0.0-beta.2`；bisheng clang 15.0.5 |
| 芯片 | 八张 Ascend910B3；当时使用卡 7/4/6 分别验收混合消费者、普通/公开回归、既有布局；设备与完整 sanitizer 在沙箱外执行 |
| CATLASS | `769cd40a8716b28650b6bebb08db4834eea4462f`，工作区干净 |
| 只读 CANN 源仓 | `ca18aee610729ca5270a44e2d84c751bc687fbf0`，原有 67 条删除/未跟踪记录保持不变 |
| 生产构建 | 上次仅重建 `mla_tnd_bsnd_metadata.o`，其余 19 个对象哈希不变；产物在 `production-before/`、`production-after/` |
| 实际加载扩展 | 上次加载 `flash_attn_npu_3/flash_attn_npu_3.cpython-311-aarch64-linux-gnu.so`；本次磁盘文件 SHA256 仍为 `27dd0dcf42685e32f7344b6cbe308483fd872f1e9e9cacd166a15f8f6599631a` |

下一次开发或设备验证仍须重新检查环境、可用芯片、源码依赖和实际加载路径。910C 未实机验收；已有 `.so` 或 AICPU 对象不能替代接入验收。

## 最小复现与交接

本次文档检查可从仓根复现，不触发构建或 NPU：

```bash
python3 build/mla-docs-audit-20261010/audit_evidence.py
python3 build/mla-docs-audit-20261010/check_docs.py
git diff --check
```

下列是 M06c2b2c 的最小复查入口，**本次未运行**，不等于完整 1,831 项验收。新验收使用新目录；CPU 命令须支持完整 sanitizer，设备命令先确认健康空闲并设置 `MLA_DEVICE`。完整选择与环境以上次归档的 `*-command.json` 为准。

```bash
MLA_CONSUMER_SANITIZE=1 MLA_SCHEDULE_SANITIZE=1 ASAN_OPTIONS=detect_leaks=1 UBSAN_OPTIONS=halt_on_error=1 python3 -m pytest -x -q tests/test_flash_attn_npu_v3_mla_tnd_bsnd_metadata_consumer.py --basetemp=build/mla-mixed-consumer-cpu-recheck
FLASH_ATTN_BUILD_VERSION=v3 FLASH_ATTN_BUILD_NPU=910 python3 setup.py build_ext --inplace
ASCEND_RT_VISIBLE_DEVICES=${MLA_DEVICE:?set MLA_DEVICE} OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 MLA_TND_BSND_CONSUMER_DEVICE_DIR=build/mla-mixed-consumer-device-recheck MLA_TND_BSND_METADATA_DEVICE_DIR=build/mla-mixed-producer-recheck MLA_TND_BSND_SCHEDULE_ARTIFACT_DIR=build/mla-mixed-oracle-recheck python3 -m pytest -x -q tests/test_flash_attn_npu_v3_mla_tnd_bsnd_metadata_device.py tests/test_flash_attn_npu_v3_metadata_signature.py tests/test_flash_attn_npu_v3_mla_bsnd_public.py
```

质量命令为 `PYTHONPATH=/tmp/mla-m01-precommit:$PYTHONPATH PRE_COMMIT_HOME=/tmp/mla-m01-precommit-cache ci/code_quality.sh --staged`，保留 CANN PYTHONPATH。最新文档提交用 `git log -1 -- docs/mla/status.md` 定位，最近实现仍是开头的 M06c2b2c。历史验收脚本可能改写绑定目录的归档，不直接用于新一步；先读 plan 中唯一下一步的完成条件。
