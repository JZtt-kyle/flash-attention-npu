# MLA 当前状态

实现完成 **M06b3**，公开能力仍为 M04c4 加 M05 的整容量 dense 前向与 metadata 复用。本次基于分支 `feat/mla-cann-migration` 的 `0a44f24`，接手时工作区干净；独立驱动已接通 v2 校验和按样本实际 KV 长度计算，复用已有 Cube / Vector / Tile。

唯一下一步为 **M06b4：公开只读 BSND 实际 KV 长度与 metadata 生成/复用**。范围与完成条件见 [plan 的 M06 表](plan.md#m06非分页长度tnd-与-stride)：公开 None/整数/int32 tensor、有/无/过期记录、真实/fake/C++、跨流、compile、普通 FA3 回归及适用公开 E2E。后续在已确认范围内推进，无需重复确认。

## 当前支持边界

| 项目 | 代码实际行为 |
| --- | --- |
| 设备与表示 | 生产及本次独立 attention 为 **910B C1V2**；FP16/BF16、512+64 共享 KV、Hkv=1。目标仍含 910C，合成 C1V1 调度不代表 910C 前向验收 |
| 公开输入 | 连续 BSND，允许连续视图非零 storage offset；B/Sq/Sk 可为零，Hq>0，KV 容量 `0..16384`；支持 S1G 多块。字段/资源上界见 [design 6.2](design.md#62-调度字段与边界) |
| 公开前向 | `flash_attn_func` 内部生成 v1；`flash_attn_with_kvcache` 支持只读 dense 内部生成或显式复用。显式 `num_splits=1`、无 mask、`softcap=0`、`pack_gqa=None`、`sm_margin=0`；scale 由当次前向提供 |
| 公开长度与记录 | 64/512 的 `get_scheduler_metadata` 仍返回连续 `int32[401]` v1，设备长度须逐项等于 KV 容量。只读前向接受 None、等于容量的整数或同设备连续 int32 tensor；消费时重新检查当次长度 |
| 公开空容量与返回 | 空 batch/query 保留空 shape；全局空 KV 为 `O=0、LSE=+inf`，仍校验记录/长度。dense 请求 LSE 返回二元组，kvcache 返回四项，未 split 的两个 accumulator 为空 tensor |
| 独立实际长度计算 | v2 `int32[401+B]` AICPU 生成、设备校验、`MlaDense<InputT, true>` 已通过独立驱动验收；非零 BSND 容量、Q 用整容量、KV 长度逐样本 `0..Sk`，含短/非均匀/全零长度及端点外零尾部。零 KV 样本为 `O=0、LSE=-inf`；KV 寻址仍按分配容量 |
| 尚未公开 | 实际短长度/非均匀长度、TND、非连续外层 stride、PA、causal/window、自动或多 split、192/128 prefill、softcap/其他调度选项、梯度。追加 KV、内置 rotary 等范围外选项明确拒绝；完整图 replay 生命周期留在 M16 |

公开边界见 [Python 真实/fake 检查](../../flash_attn_npu_3/flash_attn_npu_interface.py)、[C++ 校验](../../csrc/ascend910/flash_attn_npu_3/mla/mla_validation.h) 和 [生产生成/发射](../../csrc/ascend910/flash_attn_npu_3/mla/mla_forward.cpp)。本次没有将公开前向切换到 v2；v1 生成非法设备长度时仍写失败记录，前向消费时真实 Trap。

## 最近闭环与证据

证据目录 **`build/mla-m06b3-bsnd/`**，索引 `acceptance.json`。原始日志、JUnit、确切命令/退出码、构建/源码清单与 profiler 数据保存在该忽略目录；交接到其他环境时需一并保留。历史结果从 Git 对应版本的 status 查找，不计作本次通过。

[`mla_dense.hpp`](../../csrc/ascend910/flash_attn_npu_3/mla/kernel/mla_dense.hpp) 新增长度特化：实际长度控制 S2 块数、尾块宽度和最后一块，容量控制 KV batch stride。零 KV 跳过任务但保留任务环计数和事件协议；输出初始化覆盖所有样本，包括计算端点外的零尾部。独立驱动先完成 v2 校验，再创建 TPipe 和启动 attention；普通 dense 特化保持原逻辑。架构与固定源 `needInit`/空行 LSE 差异见 [design 6.2](design.md#62-调度字段与边界)。

| 验收面 | 本次结果 | 证据（相对上述目录） |
| --- | --- | --- |
| 前置 CPU v2/v1 消费者 | 208 项通过，含 ASan/UBSan、固定源调度、损坏/过期记录、缓冲和资源边界；完整原始证据保存在固定目录 | `cpu-final.xml`、`cpu/`、`cpu-generator/`、`v1-cpu/`、`cpu-source/` |
| 前置 910B v2 消费者 | 双 dtype 22 项有效记录与复用通过，含全局空容量/混合与全零长度 | `precheck-device.xml`、`precheck-device/`、`aicpu/` |
| 新实际长度 attention | 99 项通过：92 项双 dtype 精度/scale/尾块/跨样本/克隆跨流/长度更新检查，1 项固定源调度矩阵，6 项真实 Trap。非空 O/LSE 与固定源逐元素一致；独立双 reference、NaN padding、只读输入及缓冲 guard 通过 | `attention-final.xml`、`device-final/{fp16,bf16}/accuracy.jsonl` |
| 固定源调度矩阵 | 18 组，其中原生非 FD 12 组、无 split 映射 6 组；含 1 组端点外零尾部。源设备代码未修改；源长度数值转换为 int64，迁移消费 int32 快照 | `aicpu/attention-schedules.json`、`source-schedules-final/`、`device-final/*/oracle-source-manifest.json` |
| 新 attention 真实错误 | 双 dtype 的过期长度、错误版本、非零尾部洞共 6 项独立进程 Trap，release 构建；每例先验有效计算，错误发射无 host readback | `device-final/*/error-*/runtime-error.json` |
| 旧 dense / 普通 FA3 | 1001 项通过，含普通 FA3 metadata/compile、公开 dense/kvcache、空容量、长 KV、S1G/S2 多块及生产 v1 消费者源精度 | `regressions.xml`、`regression-nodes.json`、`dense-regression/` |
| 公开短长度拒绝 | 双 dtype 的内部生成、显式生成后消费、复用前向共 6 项真实 Trap 通过 | `public-short.xml`、`public-short-errors/` |

合计 **1336 项去重 pytest 通过，选入项无失败/跳过**，含 **12 项真实 Trap**。首轮精度和重复 CPU 检查不重复计数；未选入的旧错误测试不计作本次覆盖。独立实际长度 attention 已验收，实际长度的公开接入与对应 E2E 尚未验收。

组件性能双 dtype 各 **12 组**，覆盖短 KV、跨块、长 KV、混合/全零、端点外零尾部及整容量退化。固定 CANN 非 FD Kernel 使用相同输入/基本块/核数/端点，按固定源 host 规则传 `needInit`；迁移调用包含 v2 设备校验。预分配缓冲，同轮交替 5 轮，分别保存调用墙钟与 Kernel 原始样本。Kernel 中位数比值 FP16 为 **0.992–1.427**，BF16 为 **0.999–1.414**；最小全零样本分别为 **5.11 / 3.58**、**4.99 / 3.53 微秒**（迁移 / 源）。非空用例最高比值为 **1.129 / 1.117**，出现在 16 样本零尾部矩阵；长 KV 接近固定源。保留这些实测开销，不宣称逐例不劣化或实际长度公开 E2E 达标。

性能证据为 `performance/{fp16,bf16}/summary.json`、`wall.json`、`kernel-samples.json` 与 `profiler/`；复现脚本为 [`tests/mla_bsnd_profile.py`](../../tests/mla_bsnd_profile.py)。profiler 的框架 ACL flow / `fwk_api_db` 解析报错，未记为通过；每 dtype 的 1,200 条原始 CANN `op_summary` 与 `kernel_details` 在名称、设备、起点、耗时上逐项一致，仅采用这些 Kernel 统计。

旧生产 v1 dense 路径另测双 dtype 各 7 组（原五组加 D01 `Sk=8193` 与 `Sk=16384,Hq=128`），当前扩展与固定源的 Kernel 中位数比值 FP16 为 **0.996–1.012**，BF16 为 **0.987–1.012**。O/LSE 源逐元素一致且独立 reference 通过；每 dtype 700 条原始 Kernel 记录匹配。证据为 `dense-performance/`，复现脚本 `dense_performance.py`；同样是组件调用。

最近整容量公开 E2E 闭环仍为 M04c4（`85ce13d`）的 `build/mla-m04c4-empty/acceptance.json`；本次未重测该公开 E2E。

初始失败保存在 `attention-errors.*` 和 `device/fp16/error-stale/process.log`（子进程模块路径缺少 `tests.`）、`build-audit.*`（沙箱驱动初始化失败）及 `format-*.log`（质量工具路径/格式修正）。修正驱动模块路径、使用已有 pre-commit 环境和沙箱外加载后，最终检查通过。源 probe 的 `needInit` 改为固定源 host 标志后，已在 `device-final/` 重建并完整重跑；首轮结果不替代最终验收。

## 环境与复现

Python `/usr/local/python3.11.15/bin/python3` 3.11.15、torch `2.7.1+cpu`、torch_npu `2.7.1`、CANN `/usr/local/Ascend/cann-9.0.0-beta.2`、bisheng clang 15.0.5、g++ 11.4.0。核实 8 张 Ascend910B3 健康；本次实际长度/性能/公开错误使用设备 6、CPU 16–23，生产回归使用设备 5、CPU 8–15，最终 CPU sanitizer 使用 24–31。保留原 CANN PYTHONPATH；实机与完整 sanitizer 在沙箱外执行。详情 `environment.json`、`npu-before-final.log`。

CATLASS 固定 `769cd40a8716b28650b6bebb08db4834eea4462f` 且子模块干净；CANN 固定 `ca18aee610729ca5270a44e2d84c751bc687fbf0`，原有 67 条删除/未跟踪记录未变且未修改。固定源 blob 核验见 `source-audit.json`。

生产增量构建重编双 dtype MLA 分派，其余 16 个链接对象哈希未变；18 个链接对象、210 个源码/依赖及设备 header 依赖已核对。实际加载扩展为 `flash_attn_npu_3/flash_attn_npu_3.cpython-311-aarch64-linux-gnu.so`，SHA256 **`459641efd18e3feeb6945b285ccf862c06d08bb64d167f37a98da4f854189713`**，详见 `production-build-manifest.json`。独立设备二进制均从当前源码构建；没有用旧扩展替代当前源码验证。

以下最小检查会从当前源码重建独立双 dtype 源/迁移驱动。新验收目录不要覆盖旧证据；设备号重跑前按占用调整。全部确切命令、环境与退出码见目录内 `*-command.json` / `*-exit.json`。

```bash
ASCEND_RT_VISIBLE_DEVICES=6 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 taskset -c 16-23 python3 -m pytest -x -q tests/test_flash_attn_npu_v3_mla_bsnd_attention.py
FLASH_ATTN_BUILD_VERSION=v3 FLASH_ATTN_BUILD_NPU=910 FLASH_ATTN_SKIP_SUBMODULE_INIT=1 python3 setup.py build_ext --inplace
```

用 `MLA_BSND_ATTENTION_DIR`、`MLA_BSND_METADATA_DEVICE_DIR` 和 `MLA_LENGTH_SCHEDULE_ARTIFACT_DIR` 保存独立驱动/AICPU/固定源证据；已有 manifest 复用必须设 `MLA_BSND_ATTENTION_REUSE=1` 并通过源码和二进制哈希检查。完整回归与证据核验脚本为目录内 `regressions.py`、`build_audit.py`、`audit.py`、`check_docs.py`。三份文档链接/锚点、表格、步骤与官方公开签名检查、`git diff --check`、仓库质量检查结果见 `doc-checks.json`、`quality-staged.log`。本次闭环提交可用 `git log -1 -- docs/mla/status.md` 定位。
