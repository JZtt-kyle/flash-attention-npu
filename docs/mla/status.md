# MLA 当前状态

最近实现闭环为 **M06c2b2b：混合布局 metadata 协议与 AICPU 生成**。分支 `feat/mla-cann-migration`，公开能力仍为 **M06b4：BSND 实际 KV 长度**。

唯一下一步为 **M06c2b2c：混合布局记录设备校验消费者**，本轮未开始。混合布局继续按生成、消费、独立 attention、公开接入分别验收，见 [plan 下一步](plan.md#下一步m06c2b2c)；仍在已确认范围内，无需重复确认。契约与迁移范围查 [design](design.md)。

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

混合布局新增独立 **v4 `int32[406+5B]`**，共用已验收的 `BuildTndBsndSchedule`：Q 使用 `int32[B+1]` 前缀，KV 使用 `int32[B]` 实际长度。记录绑定 Q 总量/max 上界、物理 Sk、64 位 `B*Sk`，每个五字 span 保存 Q 起点、64 位 `b*Sk`、Q/KV 长度。实际 KV 变短不重排后续 cache；非空物理容量的全零 KV 与空容量分别调度。协议、静态界限及错误码见 [design 5.6](design.md#56-v4tnd-q-与-bsnd-kv-混合记录独立生成)。

[`BuildTndBsndMetadata`](../../csrc/ascend910/flash_attn_npu_3/mla/mla_tnd_bsnd_metadata.cpp) 校验字数、对齐、地址及输入/输出重叠；不安全输出不写，安全输出的生成失败清零八字头之后的全部载荷。AICPU 从调度器拥有的快照序列化，输入期间只读，无长度 D2H，独立驱动沿用双事件与 allocator stream 登记。B=0 仍要求 Q=`[0]`，仅此时 KV 可为 null/count=0。

**混合布局目前仅有 CPU 调度与独立 AICPU 生成。** v1/v2/v3 消费者在读取载荷前拒绝 v4；没有混合设备消费者或 attention，生产入口不发射 v4，公开签名与行为不变。used、外层 stride 和后续功能按 plan 推进。

## 本轮闭环证据

新证据目录 **`build/mla-m06c2b2b-mixed-metadata/`**；总索引 `acceptance.json`，`*-command.json` / `*-exit.json` 保存命令、环境及退出码。保留固定源 Git blob、逐字段输入/输出、编译日志、JUnit、依赖及对象哈希。

| 验收面 | 结果 | 相对证据路径 |
| --- | --- | --- |
| 交接核验 | 上轮 3,276 份归档、219 个源码依赖、19 个链接对象及实际扩展哈希一致；只读 CANN 状态不变 | `precheck.json` |
| CPU 调度/记录与 sanitizer | 638 项通过，含新增混合记录 70 项与既有调度/v3 记录/消费者 568 项；完整调度/序列化 ASan/UBSan，消费者另有 sanitizer 驱动，LeakSanitizer 开启 | `cpu-parallel.xml`、`cpu-parallel/`、`cpu-parallel-artifacts/` |
| 910B AICPU | 59 项去重通过：混合 v4 21 项、紧凑 v3 17 项、BSND v2 21 项；涵盖双 dtype 记录、源差分、失败/缓冲、更新/复用、生命周期及耗时诊断 | `aicpu-device.xml`、`aicpu-final.xml`、`mixed-producer/`、`mixed-producer-final/`、`tnd-producer/`、`bsnd-producer/` |
| 普通 FA3 与公开 BSND | 923 项通过：普通 metadata/compile/签名、MLA 契约、公开 BSND | `regressions.xml`、`regression-nodes.json` |
| 构建检查 | 13 项通过，含新增 AICPU 后端隔离、header/参数 ABI 增量依赖及 sanitizer；生产定向重建与实际加载核验通过 | `build-tests.xml`、`production-build-manifest.json` |

合计 **1,633 项去重 pytest 通过**。混合记录 CPU 的 1,474 组、AICPU 的 732 组输入逐项比较完整源调度、workspace、任务成本及物理起点，原生非 FD 与无 split 映射分开计数，详见 `acceptance.json`。覆盖 C1V1/C1V2、1/15/20/26 核、变长/零 Q、短/混合/全零 KV、空物理容量、容量与 Q 上界、非法字段/长度/缓冲及复用。双 dtype 的 B=262,146 用例验证最后 KV 起点 4,294,983,680 和总容量 4,295,000,064，不分配物理 cache。

生成耗时诊断保留双 dtype 六组原始采样；同调度的三组 v4/v2 交替对比，中位耗时比为 **0.941..1.035**，包含 ctypes、事件及逐调用同步，不能作为 attention 或公开 E2E 性能结论。本轮没有修改 attention kernel，未重测 attention 源组件性能；其历史证据仍在 `build/mla-m06c2b1-tnd-attention/acceptance.json`。

首轮设备套件在 19 项通过后，v2 对比的测试桥接误用了 dense 的同名 ctypes `Args`，参数类型检查在设备发射前失败。已修正为 v2 类型，定向补跑双 dtype 对比和未运行的 v2/v3 回归共 40 项通过。原始失败保留，19 项成功记录纳入去重汇总，失败项以重跑成功结果闭环；CPU 初验 70 项不重复计入。

## 环境与构建

Python `/usr/local/python3.11.15/bin/python3` 3.11.15、torch `2.7.1+cpu`、torch_npu `2.7.1`、CANN `/usr/local/Ascend/cann-9.0.0-beta.2`、bisheng clang 15.0.5；八张 Ascend910B3。设备与完整 sanitizer 在沙箱外执行；设备验收前卡 2/5 有进程，选择健康空闲的卡 7 跑 AICPU、卡 4 跑普通/公开回归，见 `device-health.log`。沙箱内 NPU 查询失败不作为设备验收结果。

CATLASS 为 `769cd40a8716b28650b6bebb08db4834eea4462f` 且干净；只读 CANN 源仓为 `ca18aee610729ca5270a44e2d84c751bc687fbf0`，原有 67 条删除/未跟踪记录保持不变。生产定向构建新增 `mla_tnd_bsnd_metadata.o`，共享调度注释/头文件依赖触发 `mla_tnd_metadata.o` 重建，其余 18 个对象哈希不变；链接共 20 个对象、223 个源码依赖。实际加载扩展为 `flash_attn_npu_3/flash_attn_npu_3.cpython-311-aarch64-linux-gnu.so`，SHA256 **`5e6dbc62ab95e80a7ba756e58651a567363b00a4345b023dec2cd9e7e4f19511`**。旧/新产物在 `production-before/`、`production-after/`；验收结束时仓根没有 `kernel_meta` 遗留，检查记录为 `kernel-meta-manifest.json`。

首轮质量检查自动格式化文件，已复核；最终暂存质量检查、文档链接与 `git diff --check` 通过。910C 仍未实机验收，合成 C1V1 调度不能替代实机。

## 最小复现与交接

新验收使用新目录；CPU 命令须在支持完整 sanitizer 的环境执行，设备命令先确认健康空闲并设置 `MLA_DEVICE`。完整验收选择和环境以本轮 `*-command.json` 为准。

```bash
MLA_SCHEDULE_SANITIZE=1 ASAN_OPTIONS=detect_leaks=1 UBSAN_OPTIONS=halt_on_error=1 python3 -m pytest -n 8 -x -q tests/test_flash_attn_npu_v3_mla_tnd_bsnd_metadata.py tests/test_flash_attn_npu_v3_mla_tnd_bsnd_schedule.py --basetemp=build/mla-mixed-metadata-recheck
FLASH_ATTN_BUILD_VERSION=v3 FLASH_ATTN_BUILD_NPU=910 python3 setup.py build_ext --inplace
ASCEND_RT_VISIBLE_DEVICES=${MLA_DEVICE:?set MLA_DEVICE} OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 MLA_TND_BSND_METADATA_DEVICE_DIR=build/mla-mixed-producer-recheck MLA_TND_BSND_SCHEDULE_ARTIFACT_DIR=build/mla-mixed-oracle-recheck python3 -m pytest -x -q tests/test_flash_attn_npu_v3_mla_tnd_bsnd_metadata_aicpu.py tests/test_flash_attn_npu_v3_metadata_signature.py tests/test_flash_attn_npu_v3_mla_bsnd_public.py
```

质量命令为 `PYTHONPATH=/tmp/mla-m01-precommit:$PYTHONPATH PRE_COMMIT_HOME=/tmp/mla-m01-precommit-cache ci/code_quality.sh --staged`，保留 CANN PYTHONPATH。闭环提交用 `git log -1 -- docs/mla/status.md` 定位；归档脚本绑定本轮目录、基准和暂存列表，不直接用于下一步验收。
