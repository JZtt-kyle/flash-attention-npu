# MLA 当前状态

最近实现闭环为 **M06c2b2c：混合布局记录设备校验消费者**。分支 `feat/mla-cann-migration`，公开能力仍为 **M06b4：BSND 实际 KV 长度**。

唯一下一步为 **M06c2b2d：混合布局设备寻址与独立 attention 驱动**，本轮未开始。混合 attention 与公开接入继续分别验收，见 [plan 下一步](plan.md#下一步m06c2b2d)；仍在已确认范围内，无需重复确认。契约与迁移范围查 [design](design.md)。

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

尚未接通的公开能力：TND 前向、Q-used、非连续外层 stride、PA、causal/window、自动或多 split、192/128 prefill 及其他调度选项。梯度、追加 KV、内置 rotary 等范围外组合继续明确拒绝。完整 graph replay 生命周期留在 M16；fake/compile 通过不代表 replay 已验收。目标仍包含 910C，当前生产平台只开放 910B；合成 C1V1 调度不代表 910C 实机验收。

v2 在 TPipe、attention 事件和输出写入前完成设备校验；非空容量使用平台全核 blockDim，工作核数来自设备记录，workspace 按最大 M 基块 256 预算。长度与记录无 D2H，沿用双事件与 allocator stream 生命周期。核验入口：[Python 校验与包装](../../flash_attn_npu_3/flash_attn_npu_interface.py)、[生产生成与发射](../../csrc/ascend910/flash_attn_npu_3/mla/mla_forward.cpp)、[设备消费](../../csrc/ascend910/flash_attn_npu_3/mla/mla_launch_impl.hpp)。

## 独立组件边界

紧凑 TND v3 `int32[405+4B]` 的生成、设备校验和独立 attention 已验收；与 BSND 复用 `MlaNonPaged` 的 Kernel / Block / Tile 数值路径。O 为 `[Tq,Hq,512]`，LSE 为 `[Hq,Tq]`，计算前校验当次前缀，零 Q/KV 样本跳过计算且零成本尾部有定义输出。公开 varlen/kvcache TND 仍拒绝。

混合布局新增独立 **v4 `int32[406+5B]`**，共用已验收的 `BuildTndBsndSchedule`：Q 使用 `int32[B+1]` 前缀，KV 使用 `int32[B]` 实际长度。记录绑定 Q 总量/max 上界、物理 Sk、64 位 `B*Sk`，每个五字 span 保存 Q 起点、64 位 `b*Sk`、Q/KV 长度。实际 KV 变短不重排后续 cache；非空物理容量的全零 KV 与空容量分别调度。协议、静态界限及错误码见 [design 5.6](design.md#56-v4tnd-q-与-bsnd-kv-混合记录独立生成与设备校验)。

[`BuildTndBsndMetadata`](../../csrc/ascend910/flash_attn_npu_3/mla/mla_tnd_bsnd_metadata.cpp) 校验字数、对齐、地址及输入/输出重叠；不安全输出不写，安全输出的生成失败清零八字头之后的全部载荷。AICPU 从调度器拥有的快照序列化，输入期间只读，无长度 D2H，独立驱动沿用双事件与 allocator stream 登记。B=0 仍要求 Q=`[0]`，仅此时 KV 可为 null/count=0。

**混合布局的独立 v4 生成与设备校验已验收。** [`ReadTndBsndMetadata`](../../csrc/ascend910/flash_attn_npu_3/mla/mla_tnd_bsnd_metadata_consumer.h) 使用独立 context/调度类型，完整核对协议、当次 Q 前缀/KV 长度、64 位物理容量/起点、变长端点、逐样本零成本后缀、平台与资源。B=0 仍检查 Q=`[0]`；非空物理容量中的全零 KV 保留完整调度/workspace。失败不改写输出调度，v1/v2/v3 与 v4 相互拒绝。

[`LoadTndBsndMetadata`](../../csrc/ascend910/flash_attn_npu_3/mla/kernel/mla_tnd_bsnd_metadata.hpp) 在 TPipe、attention 事件及输出/workspace 写入前完成校验，release 遇错真实 Trap；无长度 D2H，独立驱动沿用双事件与 allocator stream 生命周期。**混合 attention 尚未实现，生产入口不发射 v4，公开签名与行为不变。** used、外层 stride 和后续功能按 plan 推进。

## 本轮闭环证据

证据目录 **`build/mla-m06c2b2c-mixed-consumer/`**；总索引 `acceptance.json`，`*-command.json` / `*-exit.json` 保存命令、环境及退出码。保留固定源 Git blob、逐字段输入/输出、编译依赖、产物哈希、JUnit 与真实 Trap 独立进程报告。

| 验收面 | 结果 | 相对证据路径 |
| --- | --- | --- |
| 交接核验 | 上轮 2,521 份归档、223 个源码依赖、20 个对象与实际扩展哈希一致；只读 CANN 状态不变 | `precheck.json` |
| CPU sanitizer 与回归 | 598 项通过；混合消费者/生成/调度与 v1/v2/v3 消费者，ASan/UBSan/LeakSanitizer 开启 | `cpu-sanitizer.xml`、`cpu-sanitizer-artifacts/` |
| 910B 混合消费者 | 72 项通过；双 dtype release、固定源完整记录、损坏/过期/缓冲/端点/资源、复用与跨流生命周期 | `mixed-device.xml`、`mixed-device/`、`mixed-producer/`、`device-oracle/` |
| 既有布局与生成回归 | 225 项通过；TND/BSND 消费者、双 dtype attention 源/数学精度及混合 AICPU | `layouts.xml`、`tnd-attention/`、`bsnd-attention/`、`layout-artifacts/` |
| 普通 FA3 与公开 BSND | 923 项通过；普通 metadata/compile/签名、MLA 契约及公开 BSND | `regressions.xml`、`regression-nodes.json` |
| 构建检查 | 13 项通过；AICPU 后端隔离、增量依赖与序列化 sanitizer；生产重建及实际加载核验通过 | `build-tests.xml`、`production-build-manifest.json` |

合计 **1,831 项去重 pytest 通过**。混合消费者的独立 CPU 初验 111 项不重复计数。CPU 的 1/15/20/26 核记录与设备 C1V2 记录均逐项比较固定源调度，原生非 FD 与无 split 映射分开计数；明细见 `acceptance.json`。所有拒绝用例检查输出调度或 O/LSE/workspace 哨兵不变。

双 dtype 共 **34 个独立进程真实 Trap**，覆盖 v1/v2/v3、精确字数、Q span、64 位 KV 起点/总容量、当次长度、空 batch/query、变长 M 端点、零成本后缀、workspace 与生成失败；device/stream/event 同步均有验证，最长报错耗时 **0.067 秒**。缓冲不写入由 Trap 前同一错误的诊断调用检查，异常 stream 不复用。

CPU 与 910B 双 dtype 的 B=262,146 用例验证最后 KV 起点 **4,294,983,680** 与总容量 **4,295,000,064**，并拒绝只截断高字的记录；仅分配 metadata/长度，不分配物理 cache。非空物理容量中的全零 KV、空物理容量和 B=0/null KV 分开验收。

校验耗时诊断保留双 dtype 各四组、每组 20 个原始样本，中位 **164.2..268.8 μs**，包含解码、诊断写入、ctypes、stream 登记和逐调用同步；不能作为混合 attention 或公开 E2E 性能结论。本步没有混合 attention 数值路径，代表性源组件性能留在 c2b2d；既有 attention 本轮完成精度回归，历史性能证据仍在 `build/mla-m06c2b1-tnd-attention/acceptance.json`。

## 环境与构建

Python `/usr/local/python3.11.15/bin/python3` 3.11.15、torch `2.7.1+cpu`、torch_npu `2.7.1`、CANN `/usr/local/Ascend/cann-9.0.0-beta.2`、bisheng clang 15.0.5；八张 Ascend910B3。设备与完整 sanitizer 在沙箱外执行；检查健康与占用后，卡 7 跑混合消费者，卡 4 跑普通/公开回归，卡 6 跑既有布局回归。其他卡上进程保持不动；环境核验见 `precheck.json`。

CATLASS 为 `769cd40a8716b28650b6bebb08db4834eea4462f` 且干净；只读 CANN 源仓为 `ca18aee610729ca5270a44e2d84c751bc687fbf0`，原有 67 条删除/未跟踪记录保持不变。生产定向构建仅重建 `mla_tnd_bsnd_metadata.o`，其余 19 个对象哈希不变，共 20 个对象、223 个源码依赖。实际加载扩展为 `flash_attn_npu_3/flash_attn_npu_3.cpython-311-aarch64-linux-gnu.so`，SHA256 **`27dd0dcf42685e32f7344b6cbe308483fd872f1e9e9cacd166a15f8f6599631a`**。前后产物保留在 `production-before/`、`production-after/`；独立消费者双 dtype 均为 `-O2 -DNDEBUG`，device 依赖与对象哈希在各自 `build-manifest.json`。

暂存质量检查、文档链接与 `git diff --check` 通过；验收结束时仓根没有 `kernel_meta` 遗留，检查记录为 `kernel-meta-manifest.json`。910C 仍未实机验收，独立消费者不代表混合 attention 或公开入口完成。

## 最小复现与交接

新验收使用新目录；CPU 命令须在支持完整 sanitizer 的环境执行，设备命令先确认健康空闲并设置 `MLA_DEVICE`。完整选择与环境以本轮 `*-command.json` 为准。

```bash
MLA_CONSUMER_SANITIZE=1 MLA_SCHEDULE_SANITIZE=1 ASAN_OPTIONS=detect_leaks=1 UBSAN_OPTIONS=halt_on_error=1 python3 -m pytest -x -q tests/test_flash_attn_npu_v3_mla_tnd_bsnd_metadata_consumer.py --basetemp=build/mla-mixed-consumer-cpu-recheck
FLASH_ATTN_BUILD_VERSION=v3 FLASH_ATTN_BUILD_NPU=910 python3 setup.py build_ext --inplace
ASCEND_RT_VISIBLE_DEVICES=${MLA_DEVICE:?set MLA_DEVICE} OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 MLA_TND_BSND_CONSUMER_DEVICE_DIR=build/mla-mixed-consumer-device-recheck MLA_TND_BSND_METADATA_DEVICE_DIR=build/mla-mixed-producer-recheck MLA_TND_BSND_SCHEDULE_ARTIFACT_DIR=build/mla-mixed-oracle-recheck python3 -m pytest -x -q tests/test_flash_attn_npu_v3_mla_tnd_bsnd_metadata_device.py tests/test_flash_attn_npu_v3_metadata_signature.py tests/test_flash_attn_npu_v3_mla_bsnd_public.py
```

质量命令为 `PYTHONPATH=/tmp/mla-m01-precommit:$PYTHONPATH PRE_COMMIT_HOME=/tmp/mla-m01-precommit-cache ci/code_quality.sh --staged`，保留 CANN PYTHONPATH。闭环提交用 `git log -1 -- docs/mla/status.md` 定位；归档脚本绑定本轮目录、基准和暂存列表，不直接用于下一步验收。
