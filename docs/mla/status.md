# MLA 当前状态

最近实现闭环为 **M06c2b2a：TND Q + BSND KV 的 CPU 解析与调度**。分支 `feat/mla-cann-migration`，公开能力仍为 **M06b4：BSND 实际 KV 长度**。

唯一下一步为 **M06c2b2b：混合布局 metadata 协议与 AICPU 生成**，本轮未开始。M06c2b2 已拆为 CPU 调度、AICPU 生成、设备校验和独立 attention 四步，见 [plan 下一步](plan.md#下一步m06c2b2b)；仍在已确认范围内，无需重复确认。契约与迁移范围查 [design](design.md)。

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

紧凑 TND v3 `int32[405+4B]` 的生成、设备校验和独立 attention 已验收；与 BSND 复用 `MlaNonPaged` 的 Kernel / Block / Tile 数值路径。O 为 `[Tq,Hq,512]`，LSE 为 `[Hq,Tq]`，计算前校验当次前缀，零 Q/KV 样本跳过计算且零成本尾部有定义输出。全局空 KV 为 `O=0、LSE=+inf`，非零容量中的零 KV 为 `O=0、LSE=-inf`。公开 varlen/kvcache TND 仍拒绝。

本轮新增 [`ParseTndBsndLengths` / `BuildTndBsndSchedule`](../../csrc/ascend910/flash_attn_npu_3/mla/mla_tnd_bsnd_schedule.h)：Q 使用 `int32[B+1]` 前缀和，KV 使用 `int32[B]` 实际长度；BSND KV 起点为 64 位 `b*Sk`，允许超过 uint32，短长度不重排后续 cache。调度保留物理 Sk，源等长判断使用非累计 KV 长度；与紧凑 TND 共用变长 M 块端点映射。零实际 KV 与空物理容量分别处理。边界见 [design 6.2.6](design.md#626-tnd-q-与-bsnd-kv-的混合调度)。

**混合布局目前仅有 CPU 解析与调度。** v1/v2/v3 协议和公开入口不变；没有混合 AICPU 生成、设备记录消费者或 attention，不能将 64 位混合起点写入紧凑 v3 span。used、外层 stride 和后续功能仍按 plan 推进。

## 本轮闭环证据

新证据目录 **`build/mla-m06c2b2a-mixed-schedule/`**。各 `*-command.json` / `*-exit.json` 保存命令、环境和退出码；保留独立固定源 Git blob、全字段输入/输出、编译日志、JUnit、依赖及对象哈希。总索引为 `acceptance.json`；下列最终选入项均无失败或跳过。

| 验收面 | 结果 | 相对证据路径 |
| --- | --- | --- |
| 交接核验 | 上轮 3,934 份归档、218 个源码依赖和 19 个对象哈希一致；CANN 源仓原有状态不变 | `precheck.json` |
| CPU 调度、记录与 sanitizer | 568 项通过，含混合布局 109 项；紧凑 TND/BSND/dense 调度、v3 序列化与消费者回归。调度/序列化可执行程序全套 ASan/UBSan，另跑消费者 sanitizer 驱动，LeakSanitizer 开启 | `cpu-parallel.xml`、`cpu-parallel/`、`cpu-parallel-artifacts/` |
| 910B 紧凑 TND / v3 AICPU | 90 项通过：73 项双 dtype attention（含 26 个真实 Trap 进程）及 17 项生成、源差分、生命周期与生成诊断 | `tnd-device.xml`、`attention/`、`producer/` |
| 普通 FA3 与公开 BSND | 923 项通过：普通 metadata/compile/签名、MLA 契约、公开 BSND | `regressions.xml`、`regression-nodes.json` |
| 构建检查 | 10 项通过，包含新头文件增量依赖与 sanitizer；生产定向重建及实际加载核验通过 | `build-tests.xml`、`production-build-manifest.json` |

合计 **1,591 项去重 pytest 通过**。混合布局的 5,624 组输入逐项比较完整调度、任务成本和物理起点，其中原生非 FD 3,582 组、无 split 映射 2,042 组；另有 1,296 组 BSND 等价检查。覆盖 1..26 核、C1V1/C1V2、变长/零 Q、短/混合/全零 KV、空容量、字段/物理块/workspace 边界及超过 uint32 的物理起点。26 个真实 Trap 最长约 40.0 ms。初验 109 项和串行中断的 88 项不重复计入。

上一轮完整证据仍保留在 `build/mla-m06c2b1-tnd-attention/acceptance.json`。本轮没有修改 attention kernel，未重测 attention 源组件性能或公开 E2E；v3 生成耗时诊断包含在 AICPU 回归中，不作为混合布局性能验收。

## 环境与构建

Python `/usr/local/python3.11.15/bin/python3` 3.11.15、torch `2.7.1+cpu`、torch_npu `2.7.1`、CANN `/usr/local/Ascend/cann-9.0.0-beta.2`、bisheng clang 15.0.5；八张 Ascend910B3。设备及完整 sanitizer 验收在沙箱外执行；本轮卡 7 跑紧凑 TND，卡 5 跑普通/公开回归，启动前均健康空闲。再次查询时卡 6 有进程，设备验收选择空闲的卡 5/7，见 `device-health.log` / `device-selection.json`。

CATLASS 为 `769cd40a8716b28650b6bebb08db4834eea4462f` 且干净；只读 CANN 源仓为 `ca18aee610729ca5270a44e2d84c751bc687fbf0`，原有 67 条删除/未跟踪记录保持不变。FA3/910 定向构建仅改变 `mla_tnd_metadata.o`，其余 18 个链接对象哈希不变；新依赖包含混合布局头文件，合计 219 个源码依赖。实际加载扩展为 `flash_attn_npu_3/flash_attn_npu_3.cpython-311-aarch64-linux-gnu.so`，SHA256 **`151113e0f636a747e06160779ddb879387200534bbb8c938c2b9e72aa0b5ff5d`**。旧/新产物在 `production-before/`、`production-after/`。本轮生成的 208 份 `kernel_meta` 文件已移入 `kernel-meta-artifacts/`，保留哈希清单。

串行 CPU sanitizer 因短进程退出扫描耗时，在 88 项通过后主动中断（退出码 2）；完整同套件改用已安装 pytest-xdist 的 8 个 worker，并隔离各 worker 的证据目录，未降低 sanitizer 配置。串行记录保留且不计为闭环，见 `cpu-serial-interruption.json` / `cpu-final-exit.json`。

首轮质量检查由 clang-format 自动改写文件，已复核；最终暂存质量检查、文档链接及 `git diff --check` 通过。CPU 初验的沙箱 NPU 日志查询提示不影响 CPU 结果；必要设备检查不依赖该沙箱查询。910C 仍未实机验收，合成 C1V1 调度不能替代实机。

## 最小复现与交接

新验收使用新目录；下面 CPU 命令需在支持完整 sanitizer 的环境执行，设备命令先确认健康空闲并设置 `MLA_DEVICE`。完整选择与环境以本轮 `*-command.json` 为准。

```bash
MLA_SCHEDULE_SANITIZE=1 ASAN_OPTIONS=detect_leaks=1 UBSAN_OPTIONS=halt_on_error=1 python3 -m pytest -n 8 -x -q tests/test_flash_attn_npu_v3_mla_tnd_bsnd_schedule.py tests/test_flash_attn_npu_v3_mla_tnd_schedule.py --basetemp=build/mla-mixed-schedule-recheck
FLASH_ATTN_BUILD_VERSION=v3 FLASH_ATTN_BUILD_NPU=910 python3 setup.py build_ext --inplace
ASCEND_RT_VISIBLE_DEVICES=${MLA_DEVICE:?set MLA_DEVICE} OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 MLA_TND_ATTENTION_DIR=build/mla-tnd-attention-recheck MLA_TND_METADATA_DEVICE_DIR=build/mla-tnd-producer-recheck MLA_TND_SCHEDULE_ARTIFACT_DIR=build/mla-tnd-oracle-recheck python3 -m pytest -x -q tests/test_flash_attn_npu_v3_mla_tnd_attention.py tests/test_flash_attn_npu_v3_metadata_signature.py tests/test_flash_attn_npu_v3_mla_bsnd_public.py
```

质量命令为 `PYTHONPATH=/tmp/mla-m01-precommit:$PYTHONPATH PRE_COMMIT_HOME=/tmp/mla-m01-precommit-cache ci/code_quality.sh --staged`，保留 CANN PYTHONPATH。闭环提交可用 `git log -1 -- docs/mla/status.md` 定位；旧归档脚本绑定自己的目录、基准和暂存列表，不直接作为新提交验收脚本。
