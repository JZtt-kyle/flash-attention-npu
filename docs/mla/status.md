# MLA 当前状态

本轮完成 **M06c1：紧凑 TND 前缀和解析、起点与 CPU 调度**，新 CPU 套件 111 项、完整 ASan/UBSan 复验 111 项及受影响 dense/BSND 回归 185 项全部通过。分支 `feat/mla-cann-migration`，本轮起点 `276b7dcc7e282ffb0b9297ccbaf27cbc959d88ba`。公开能力仍停留在已闭环 M06b4。

唯一下一步为 **M06c2a：TND AICPU 记录与设备消费者**；本轮未开始该步。范围与完成条件见 [plan](plan.md#m06非分页长度tnd-与-stride)，后续在已确认范围内推进，无需重复确认。

## 当前支持边界

| 项目 | 当前代码行为 |
| --- | --- |
| 设备与表示 | 910B C1V2、FP16/BF16、512+64 共享 KV、Hkv=1。目标仍含 910C，合成 C1V1 调度不代表 910C 实机验收 |
| 输入 | 连续 BSND，允许非零 storage offset；B/Sq/Sk 可为零，Hq>0，KV 容量 `0..16384`；Q 使用整容量，支持 S1G/S2 多块 |
| 公开长度 | 只读 `flash_attn_with_kvcache` 接受 None（整容量）、`0..Sk` 整数或同设备连续 `int32[B]`；支持短/非均匀/混合与全零长度 |
| 公开记录 | `get_scheduler_metadata` 的 64/512 分支生成 v2 `int32[401+B]`，含实际长度快照；真实/fake/C++ 外形一致。消费时复核当前长度，省略长度表示整容量；旧 v1、损坏或过期记录拒绝 |
| 内部 dense | 无长度且无显式记录时仍使用 v1，包括 `flash_attn_func`。只读 kvcache 精简 dispatcher 复用已有 Kernel / Block / Tile，不调用原 CANN |
| 空行与返回 | 非零容量的零 KV 样本 `O=0、LSE=-inf`，含端点外零尾部；全局空 KV 为 `O=0、LSE=+inf`。空 batch/query 保留空 shape，仍校验记录。kvcache 的 LSE 返回四项，未 split 的 accumulator 为空 tensor |
| 调度限制 | 显式 `num_splits=1`、无 mask、`softcap=0`、`pack_gqa=None`、`sm_margin=0`；scale 由当次前向提供。静态字段上界见 [design 6.2](design.md#62-调度字段与边界)，v2 另要求 `B<=2^32-1-401` |
| 尚未接通 | TND、Q-used、非连续外层 stride、PA、causal/window、自动或多 split、192/128 prefill、其他调度选项、梯度；追加 KV、内置 rotary 等范围外选项继续明确拒绝。完整图 replay 生命周期留在 M16 |

生产入口在 TPipe、attention 事件或输出写入前完成 v2 校验。非空 v2 使用平台全核 blockDim，工作核数由设备记录决定；workspace 按源最大 M 基块 256 预算，覆盖非均匀长度使 M 增大的情形。长度与记录无 D2H，使用既有双事件和 allocator stream 生命周期。实现见 [生产生成/发射](../../csrc/ascend910/flash_attn_npu_3/mla/mla_forward.cpp)、[设备入口](../../csrc/ascend910/flash_attn_npu_3/mla/mla_launch_impl.hpp) 和 [接口校验](../../flash_attn_npu_3/flash_attn_npu_interface.py)。

## 本轮 CPU 组件与证据

新增 [`mla_tnd_schedule.h/.cpp`](../../csrc/ascend910/flash_attn_npu_3/mla/mla_tnd_schedule.h) 的 `ParseTndLengths` / `BuildTndSchedule`，仅供独立 CPU 驱动：紧凑 TND Q/KV、共享 512+64 KV、Hkv=1、无 mask、显式无 split。两组 int32 前缀和须含起始 0，计数 B+1、单调、尾值等于总容量；逐样本 token 起点与长度拥有独立存储，max 参数允许大于观测最大值。全部校验后数值转换为源 int64 累计尾值，复用既有 `SplitCore`。失败不改写调用方输出。

保留固定源 TND 的特殊等长判定：比较 KV 累计尾值，而非各段长度；源观察最大段长决定基块/workspace。变长 query 按逐样本 M 块映射端点，保留跨零 query 样本的 batch 坐标，尾部省略只允许 Q=0 或 KV=0。全局空总容量与非空总容量中的零成本样本分别处理。字段/地址/物理块数/平台/workspace 上界见 [design 6.2](design.md#62-调度字段与边界)。TND Q + BSND KV、used、AICPU/metadata、设备寻址与公开接入未在本步开放。

证据目录 **`build/mla-m06c1-tnd/`**，总索引 `acceptance.json`，由 `audit.py` 核验；原始命令、退出码、JUnit、完整输入/输出、固定源 blobs、编译命令均保存于该忽略目录，交接环境时须一并保留。

| 验收面 | 结果 | 证据（相对上述目录） |
| --- | --- | --- |
| TND CPU 与固定源 | 111 项通过；覆盖 1..26 核、C1V1/C1V2，4,853 组源差分，其中原生非 FD 2,585 组、无 split 映射 2,268 组；100 组保留端点外零成本后缀 | `final.xml`、`final/*-inputs.json` / `*-outputs.json` / `source-manifest.json` |
| 起点/边界/等价 | T01、混合/空样本、非均匀 Q、跨块尾部、max 上界、INT32_MAX 总 Q、非法计数/单调/尾值、静态/总物理块/workspace 溢出、失败结果不变；216 组与 BSND 全字段一致，源特有 M 基块差异另验覆盖和成本 | `final/bsnd-equivalence.json`、`final/t01-and-tail-equality*`、`final/m-base-*`、`final.xml` |
| ASan/UBSan | 同一完整 111 项通过，DUT 与固定源 oracle 均启用 address/undefined、不可恢复错误；完整输出与常规构建逐份哈希相同 | `sanitized.xml`、`sanitized/` |
| 受影响回归 | dense/BSND CPU 185 项通过，共用驱动三种编译分支均覆盖；不涉及公开接口或生产 metadata 修改 | `regressions.xml`、`dense/`、`bsnd/` |

合计 **296 项去重 pytest 通过，另有同一 TND 套件的 111 项 sanitizer 复验通过，最终选入项无失败/跳过**。

初始失败完整保留：`initial.xml` 的 oracle 将跨零 query 样本的不同 batch 端点合并为同一扁平行坐标，已修复 oracle、保留生产源端点；`cpu.xml` 的溢出测试构造实际未超过 uint32，已改为有效前缀且总物理块数恰为 `2^32` 的用例，确认进入源循环前拒绝。对应原始输出均保留，没有删除失败节点或放宽检查。

## 环境与上一步复核

Python `/usr/local/python3.11.15/bin/python3` 3.11.15、torch `2.7.1+cpu`、torch_npu `2.7.1`、CANN `/usr/local/Ascend/cann-9.0.0-beta.2`、bisheng clang 15.0.5、g++ 11.4.0。本轮 CPU / 回归 / sanitizer 分别使用 CPU 16–23 / 8–15 / 24–31。八张 Ascend910B3 健康，检查时设备 2–6 空闲；设备 6 用于核对实际加载路径。沙箱内 `npu-smi` 驱动初始化失败，环境核验与完整 sanitizer 在沙箱外执行；本步未修改 kernel，不要求新增 910B 精度或性能验收，合成 C1V1 仍不代表 910C 实机验证。

CATLASS 固定 `769cd40a8716b28650b6bebb08db4834eea4462f` 且干净；CANN 固定 `ca18aee610729ca5270a44e2d84c751bc687fbf0`，原有 67 条删除/未跟踪记录未变。本步不修改两者。TND 独立驱动直接编译本轮源码；没有接入 `setup.py` 或重建生产扩展。已逐项复核旧构建的 212 个源码依赖、18 个对象及实际加载的 `flash_attn_npu_3/flash_attn_npu_3.cpython-311-aarch64-linux-gnu.so`，SHA256 仍为 **`fa35de6c8b0b08e563fcb62ce4dbea9eb97f43618f6cf836b4b5a7ba11114099`**，详见 `precheck.json`。

上一步最小命令为 `ASCEND_RT_VISIBLE_DEVICES=6 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 taskset -c 16-23 python3 -m pytest -x -q tests/test_flash_attn_npu_v3_mla_bsnd_public.py`。已复核 `build/mla-m06b4-public/public-accuracy-command.json` 与退出码 0，以及后续 schedules/boundaries/empty-query 和最终 audit；该步 2,421 项与适用公开 E2E 的闭环证据保留在 `build/mla-m06b4-public/acceptance.json`。生产环境、源码依赖、对象与加载扩展均未改变，本步没有重跑该 NPU 命令，也不计为本轮通过。

质量工具通过 `PYTHONPATH=/tmp/mla-m01-precommit:$PYTHONPATH PRE_COMMIT_HOME=/tmp/mla-m01-precommit-cache` 使用固定 pre-commit 4.6.2、Ruff 0.16.5、clang-format 21.1.8，保留原 CANN PYTHONPATH。默认 Python 无 pre_commit，root 旧缓存的 Ruff/clang-format 版本不符；已切换到上述既有固定版本环境，未改全局环境。代码格式检查见 `format-final.log`；提交前 staged 质量检查、三份文档链接/锚点/步骤状态/公开签名检查分别见 `quality-staged.log`、`doc-checks.json`。`audit.py` 复核测试、固定源 blobs、两种构建的输出、源码快照、生产构建与源仓状态。

## 最小复现

新证据写到其他目录，不覆盖本轮原始结果；完整命令/环境/退出码见 `*-command.json` / `*-exit.json`。

```bash
MLA_TND_SCHEDULE_ARTIFACT_DIR=build/mla-m06c1-recheck python3 -m pytest -x -q tests/test_flash_attn_npu_v3_mla_tnd_schedule.py
MLA_SCHEDULE_SANITIZE=1 MLA_TND_SCHEDULE_ARTIFACT_DIR=build/mla-m06c1-recheck-asan python3 -m pytest -x -q tests/test_flash_attn_npu_v3_mla_tnd_schedule.py
python3 -m pytest -x -q tests/test_flash_attn_npu_v3_mla_schedule.py tests/test_flash_attn_npu_v3_mla_bsnd_schedule.py
```

sanitizer 须在支持完整 ASan/UBSan/LeakSanitizer 的环境执行，不能把关闭检查当作通过。当前闭环提交可用 `git log -1 -- docs/mla/status.md` 定位。
