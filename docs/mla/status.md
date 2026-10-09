# MLA 当前状态

本轮完成 **M06b4：公开只读 BSND 实际 KV 长度与 metadata 生成/复用**，功能、普通 FA3 回归及适用公开 E2E 已闭环。分支 `feat/mla-cann-migration`，本轮起点为 `c7e6706a0bde438eb3c8bbf0b03827153efde636`。中断恢复后核对了源码、构建、进程和既有验证，继续同一步，未重复提交或启动后续开发。

唯一下一步为 **M06c1：紧凑 TND 前缀和解析、起点与 CPU 调度**；本轮未开始该步。范围与完成条件见 [plan](plan.md#m06非分页长度tnd-与-stride)，后续在已确认范围内推进，无需重复确认。

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

## 最近闭环与证据

证据目录 **`build/mla-m06b4-public/`**，总索引 `acceptance.json`，功能索引 `functional-acceptance.json`。原始日志、JUnit、确切命令/退出码、构建/源码清单、profiler 数据均保存在该忽略目录，交接环境时需一并保留。上轮最小闭环和旧构建已在 `precheck.json` 核验，不计作本轮通过。

| 验收面 | 已核实结果 | 证据（相对上述目录） |
| --- | --- | --- |
| 新公开 BSND 能力 | 192 项通过。含双 dtype 19 组 × 3 scale 的 114 项固定源精度：非空 O/LSE 源逐元素一致，独立双 reference、NaN padding、输入只读、快照克隆/奇数 offset 通过；另含 None/整数/tensor、真实/fake/C++、跨流、长度更新与 compile（eager/fullgraph） | `public-accuracy.xml`、`public-schedules.xml`、`public-boundaries.xml`、`empty-query-lengths.xml`、`attention/*/public-accuracy.jsonl` |
| 固定源调度 | 双 dtype 各 37 组公开记录与固定源无 FD 调度一致；原生分派 20 组非 FD、17 组 FD，全矩阵保留 | `schedule-aicpu/public-schedules-*-dispatch.json`、`schedule-source/` |
| 旧 dense 与普通 FA3 | 2124 项去重通过，含普通 FA3 metadata/compile/softcap、MLA contract/dispatch、空容量、长 KV、多块、v1 生产消费者 | `regressions-before-snapshot.xml` 中成功的 1727 项，加 `regressions-remaining.xml` 的 397 项；`regression-nodes.json`、`remaining-nodes.json` |
| 真实设备错误 | 98 项独立进程真实 Trap 通过，含非法/过期/省略长度、版本/类型/字数/快照、空容量错误；每例先验有效前向，错误发射禁止 host readback | `public-errors.xml`、`public-errors-extra.xml`、`public-errors*/**/runtime-error.json` |
| 构建与依赖 | 7 项通过，含 ACL 发射桥 ASan/UBSan 和增量依赖；18 个链接对象、212 个源码/依赖和设备 header 依赖匹配，当前加载扩展与本轮构建一致 | `build-tests.xml`、`production-build-manifest.json` |

合计 **2421 项去重 pytest 通过，最终选入项无失败/跳过**。`audit.py` 合并每个节点的最终结果，核验 98 份 Trap 与扩展哈希、256 个固定源 blob。保留初始失败：compile 测试 mock 干扰 Dynamo、ctypes Args 类型不一致、旧测试误把零长度当失败；分别修正测试夹具/预期后重跑成功。原失败在 `public-smoke.xml`、`regressions-initial.xml`、`regressions-before-snapshot.xml`，没有删除失败节点或更改精度阈值。

## 公开性能闭环

固定矩阵为双 dtype 各 **37 组**（原 26 组 + `BSND_CASES` 11 组）。固定源原生分派中 **20 组非 FD** 逐例验收，另 **17 组原生 FD** 保留诊断，不按耗时筛选。测试包含复用前向、单独生成、内部生成、生成加 R=1/4 前向及 CANN 对应调用段；双方同一进程/设备，输入 H2D 排除，入口校验、输出/workspace 分配、转换、发射和完成同步计入 E2E。

最终在空闲设备 6，按测量前两次五秒 CPU 空闲度采样固定 `[244,246,247,248,249,251,252,255]`，双 dtype / 双方使用同一亲和性。预热 5 次，每次采集五轮各 20 个 wall 样本，交替调用。FP16 使用一次完整同路径采集；BF16 首次有两例 R=1 超标，按补测前写入的方案再测全部 20 组，**合并两次全部 200 个样本/路径/用例**，包含首次失败样本，不选取较快轮次。严格门槛仍为每例 `<=1.0`。

| dtype | 每路径/用例 wall 样本数 | 复用前向 E2E 比值 | 生成加 R=1 比值 | 生成加 R=4 比值 |
| --- | --- | --- | --- | --- |
| FP16 | 100 | 0.5744–0.7646 | 0.8072–0.9052 | 0.4979–0.9031 |
| BF16 | 200 | 0.5734–0.7947 | 0.8269–0.9155 | 0.5177–0.9389 |

两种 dtype 的全部 20 组上述三个比值均达标。逐例数值、生成耗时、内部生成耗时、原始样本、每轮统计、原生 FD 差距见 `performance-acceptance.json` 和 `performance-controlled/{fp16,bf16,bf16-repeat}/`。采集前选择依据、持续主机负载、补测前约定分别在 `host-affinity-selection.json`、`host-load*.jsonl`、`repeat-protocol.json`。完整 37 组证据保留于 `performance-fp16/` 和 `performance-bf16-quiet/`。

Kernel 计时单独报告：FP16 比值 **0.990–4.564**，最小全零用例 FA3 / CANN 为 **15.6105 / 3.4200 微秒**；BF16 比值 **0.961–4.150**，最小全零用例 FA3 / CANN 为 **14.3600 / 3.4600 微秒**。保留 v2 平台全核发射及设备记录/长度校验的实测开销，不把公开 E2E 达标解释为 Kernel 逐例不劣化。原始 `op_summary` 与 `kernel_details` 的名称、设备、起点、耗时逐条匹配：完整矩阵每 dtype 3700 条，最终同路径 FP16 2000 条、BF16 两次共 4000 条；显式复用未出现 metadata 生成任务。`performance-device-proof.json` 将 namespace PID 映射到设备 host PID，核对测量前后仅本进程占用测试设备，并复核固定 CANN 库哈希。

早期失败完整保留：`performance-fp16/` 的索引 4/5/10/16 生成加 R=1 为 1.547/1.511/2.526/1.033，索引 10 的 R=4 为 1.250；`performance-bf16/` 缺少 NPU profiler 数据，未记为通过。设备 7 后来被外部进程占用，恢复时改用空闲设备 6；`performance-*-quiet/` 的 CPU 40–47 测量仍超标。主机负载约 230、该 CPU 组平均约 82% 忙碌的诊断见 `host-load-diagnostic.json`。固定亲和性后，BF16 首次索引 28/29 的 R=1 为 1.224/1.056，已按前述方式纳入最终合并数据。没有删除不利样本或放宽阈值。

## 环境与复现

Python `/usr/local/python3.11.15/bin/python3` 3.11.15、torch `2.7.1+cpu`、torch_npu `2.7.1`、CANN `/usr/local/Ascend/cann-9.0.0-beta.2`、bisheng clang 15.0.5、g++ 11.4.0。八张 Ascend910B3 健康。本轮精度设备 6 / CPU 16–23，回归设备 5 / CPU 8–15，错误设备 4 / CPU 32–39（补充设备 6 / CPU 16–23），首轮性能设备 7 / CPU 40–47，恢复后性能设备 6 / 先 CPU 40–47，最终使用上述预选 CPU 组，构建 sanitizer CPU 24–31。实机与完整 sanitizer 在沙箱外执行；保留原 CANN PYTHONPATH。见 `environment.json`、`environment-resume.json` 及性能目录的环境/设备快照。

CATLASS 固定 `769cd40a8716b28650b6bebb08db4834eea4462f` 且干净；CANN 固定 `ca18aee610729ca5270a44e2d84c751bc687fbf0`，原有 67 条删除/未跟踪记录未变，本轮未修改。生产增量构建重编双 dtype 分派、`mla_forward.o`、`flash_api.o` 四个对象，其余 14 个哈希未变。实际加载 `flash_attn_npu_3/flash_attn_npu_3.cpython-311-aarch64-linux-gnu.so`，SHA256 **`fa35de6c8b0b08e563fcb62ce4dbea9eb97f43618f6cf836b4b5a7ba11114099`**。

最小公开检查（复测前检查设备占用，证据写新目录，不覆盖本轮原始结果）：

```bash
ASCEND_RT_VISIBLE_DEVICES=6 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 taskset -c 16-23 python3 -m pytest -x -q tests/test_flash_attn_npu_v3_mla_bsnd_public.py
FLASH_ATTN_BUILD_VERSION=v3 FLASH_ATTN_BUILD_NPU=910 FLASH_ATTN_SKIP_SUBMODULE_INIT=1 python3 setup.py build_ext --inplace
```

用 `MLA_BSND_ATTENTION_DIR`、`MLA_BSND_METADATA_DEVICE_DIR` 和 `MLA_LENGTH_SCHEDULE_ARTIFACT_DIR` 保存独立驱动/AICPU/固定源证据。全部确切命令/环境/退出码见 `*-command.json` / `*-exit.json`；回归续跑按原选择集排除已通过节点，脚本为 `resume_regressions.py`。公开性能脚本为 [`tests/mla_forward_profile.py`](../../tests/mla_forward_profile.py)，本轮包装脚本为 `performance_isolated.py`。三份文档链接/锚点、表格、步骤状态与固定官方公开签名检查见 `doc-checks.json`，`git diff --check` 和仓库质量检查见 `quality-staged.log`，最终一致性核验为 `final_audit.py`。本轮运行生成的 `kernel_meta/` 已归入证据目录，不提交构建产物。当前闭环提交可用 `git log -1 -- docs/mla/status.md` 定位。
