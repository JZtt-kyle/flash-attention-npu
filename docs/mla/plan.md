# MLA 迭代计划

当前已将首步 FP16/BF16 dense MLA 的调度和 Kernel 接入 910B FA3 `flash_attn_func`，具备独立 reference 和固定源码 CANN 对照。M04b3 的两种 dtype 精度、舍入、入口边界及普通 FA3 回归已完成。MLA metadata 和范围外组合仍明确报未实现；无 metadata 的公开调用 E2E 仍慢于 CANN，数据仅作诊断。唯一下一步为 M05，将已验收范围接入 MLA metadata 并验收其 E2E。

实现原则见 [design](design.md)：以固定 CANN 源实现为起点，保持计算、精度、切块分核、缓冲和同步行为，按 CATLASS Kernel / Block / Tile 组织代码。FA3 参数映射与 metadata 接入遵循官方契约。性能验收使用 metadata 路径 E2E 与同轮 CANN E2E 的逐例比较。

## 提交粒度

每次完成一项可独立验收的小步，更新本表和 [status](status.md)，提交一次后停止。实现前将过大的步骤拆为有明确范围和验收条件的子步骤。每个入口只接通已验收组合，其余组合明确报错。

| 步骤 | 状态 | 产物 | 完成条件 |
| --- | --- | --- | --- |
| M00 | 已完成 | 范围、来源、FA3 契约和 CATLASS 结构约定 | 固定来源，明确迁移边界、组件等价条件及 metadata E2E 口径 |
| M01 | 已完成 | 两种 MLA 的独立 reference 和 CANN 输入映射 | 512+64 共享 KV、192/128 独立 V、scale、stride、右下 mask、窗口及空行行为有独立对照 |
| M02a | 已完成 | 910B CANN 执行器 | D01/D02/P01 的 FP16/BF16 O/LSE、重复执行、布局及实际加载路径可追溯 |
| M02b1 | 已完成 | 固定 CANN 源码隔离构建与分派溯源 | vendor、host tiling、device binary 与固定源码对应，运行分派可核实 |
| M02b2 | 已完成 | 固定源码 profiler 基线工具与代表用例 | kernel、转换和调用 E2E 分开记录，后续验收按同轮交替测量 |
| M03 | 已完成 | MLA 入口识别、参数校验和未实现拦截 | 构建只含普通 FA3 kernel；MLA 真实/fake/C++/公开入口明确拒绝；契约、普通 FA3 精度/metadata/编译图及 CPU reference 回归通过 |
| M04a | 已完成（纯文档） | CANN 源实现映射及 dense 实现拆分 | design 6.1..6.5 固定源函数、字段、数值顺序、资源/事件和组件复用条件；首步范围与验收子步骤明确，源码锚点及文档检查通过 |
| M04b1 | 已完成 | 首步 dense 的无 gert 调度模块及源输出对照 | 85 项 CPU 检查通过，49,251 组输入与固定源的基本块、端点、核数、无 FD 和 workspace 完全一致；字段/资源/溢出检查及 sanitizer 通过；尚不接生产前向或 metadata |
| M04b2a | 已完成 | FP16 QK/PV 的 Cube Block 与 Tile | 155 项 910B 检查通过，FP32 QK/PV 与固定源逐元素一致并满足独立 FP64 误差界；覆盖内容/RoPE 两半、M/N/K 尾块、多 query、非零 offset、缓冲复用；四组组件性能诊断完成；不接生产前向 |
| M04b2b | 已完成 | FP16 Vector Block 与 dense Kernel | 175 项独立 910B 检查通过，O/LSE 与固定源逐元素一致并通过独立数学对照；覆盖 D01/D02、尾块、多 batch、scale、任务环及复用；155 项 Cube 回归通过，五组同轮固定 CANN 性能诊断完成；不接公共入口 |
| M04b2c | 已完成 | 首步 FP16 的 FA3 接入与构建 | 强制构建、设备头增量依赖及加载产物已核查；339 项真实/fake/C++ 契约、101 项公开/内部入口精度与生命周期、176 项 CPU 和 11 项普通 FA3 回归通过；五组同轮 CANN 公开调用诊断完成，E2E 比值约 1.61–1.63，未达最终性能门槛；BF16 及范围外组合明确拒绝 |
| M04b3 | 已完成 | 同范围 BF16 实例化及验收 | 保留源的 BF16 中间/输出舍入及 FP16 计算逻辑；513 项组件、786 项公共入口/契约、176 项 CPU 和 11 项普通 FA3 回归通过；两种 dtype 边界一致，分别完成五组同轮 CANN 性能对照；BF16 无 metadata E2E 比值约 1.60–1.64，未达最终性能门槛 |
| M05 | 唯一下一步 | 已验收 dense 范围的 MLA metadata | 将源调度计算接入 AICPU，保持调度结果；官方参数、运行时 scale、有/无 metadata、连续复用和普通 FA 兼容正确；metadata E2E 逐例不劣于同轮 CANN |
| M04c1 | 待 M05 完成 | dense 的 S1G 多块，解除首步 `Sq*Hq<=256` 限制 | KV 仍为 1..512；按源 M 切块/分核接通，覆盖 257/511/512/513 行及跨核尾块；两种 dtype、metadata、重复执行、普通 FA3 回归和 E2E 通过 |
| M04c2 | 待 M04c1 完成 | dense 的 KV 513..1024、显式 num_splits=1 | 保留源 aMLA nUpdate、PV 原子加和事件依赖；单列禁止 FD 的必要调度映射；跨块最大值变化、O/LSE、重复执行、metadata、普通 FA3 回归和 E2E 通过 |
| M04c3 | 待 M04c2 完成 | dense 长 KV，含 8193 | 按源流程扩大范围，不改变切块/数值路径；单/多 query、多块尾部、两种 dtype、metadata、重复执行、普通 FA3 回归和 E2E 通过；显式 num_splits=1 |
| M04c4 | 待 M04c3 完成 | dense 全局空 Q/K/V 容量边界 | `Sq=0`、`Sk=0`、二者均零及空 batch 的 shape、O 和 FA3 的 LSE 符号正确；有/无 metadata、fake、真实及普通 FA3 回归通过；样本内零长度在 M06 验收 |
| M06 | 待开始 | 512+64 非分页 TND、实际长度和外层 stride | 不均匀 Q/KV、seqused、前缀和、零长度、尾块与 poison padding 通过；steady-state 无长度 D2H；相关 E2E 通过 |
| M07 | 待开始 | 512+64 的 ND paged cache | BSND/TND query、乱序页表、尾页、源有效页大小、K/V 独立首维 stride；精度及 PA E2E 通过 |
| M08 | 待开始 | 512+64 的右下 causal | Sq<Sk、Sq>Sk、多 query、空行、PA/TND 与 metadata 一致；精度及 E2E 通过 |
| M09 | 待开始 | 512+64 的局部窗口 | 有限/单侧窗口、causal 叠加、FA3 边界和页边界；精度及 E2E 通过 |
| M10 | 待开始 | 512+64 的 split-KV 与合并 | 按源分割和合并流程适配自动/1/显式 split；dense/TND/PA、空 split、O/LSE/accumulator 与 metadata 正确；长 KV E2E 通过 |
| M11 | 待开始 | 192/128 prefill BSND 前向与 metadata | 按源通用非量化路径适配 qv=None、独立 K/V、分组头和真实 stride；O/LSE 及 E2E 通过 |
| M12 | 待开始 | prefill TND 与实际长度 | 非均匀长度、seqused、边界和 GQA 与 reference 一致；精度及 E2E 通过 |
| M13 | 待开始 | prefill ND paged cache | 页寻址、内容/RoPE 真实 stride、精度及 PA E2E 通过 |
| M14 | 待开始 | prefill mask 与 split 组合 | 右下 causal/window、split 合并以及 metadata 组合通过；按组合拆成独立验收步骤 |
| M15 | 待开始 | MLA softcap 与调度选项适配 | 明确各项 FA3 契约映射；softcap、has_softcap、pack_gqa、sm_margin、资源校验及 E2E 通过 |
| M16 | 待开始 | 图执行与 metadata 生命周期 | eager/图执行一致，图内长度更新后 metadata 正确；跨层复用、多 stream、fake 与真实输出一致 |
| M17 | 待开始 | 全范围验收和用户示例 | 必迁能力均有用例，精度、逐例 metadata E2E 及普通 FA3 回归通过；README 给出调用方式和支持边界 |

M04b 的临时范围与验收切面见 [design 6.2](design.md#62-调度字段和首步边界) 和 [6.5](design.md#65-必要的-fa3-差异与验收切面)：连续 BSND、B>0、`1<=Sq*Hq<=256`、`1<=Sk<=512`、Hkv=1、无 mask、显式 `num_splits=1`；先接 `flash_attn_func`。FP16/BF16 分两次接通，各自验收前保持明确拒绝。M04c 的范围扩展不改变最终需求；TND、PA、mask、split 等仍按后续行推进。

M04b1 是有独立验收的调度实现，不创建未实现 kernel 或返回占位 metadata。源对照从固定 commit 读取原算法，覆盖 design 6.5 的边界；该调度尚未被生产构建引用时不要求重跑 NPU。M04b2a/b 的测试驱动只用于设备组件验收，不提供 MLA 前向接口；两步均须完成相关 910B 精度和性能诊断。M04b2c 接入生产时必须强制核查构建依赖、实际加载扩展和源组件对应关系。每个实现步骤保持源设备行为，组件替换按运算、数据布局、精度和资源/同步契约核对等价性。

E2E 优化集中于接口、调度计算位置、metadata 复用、重复准备与发射成本。设备计算耗时用于检查源行为一致性。新增能力及时验收相应精度和 E2E，普通 FA3 回归覆盖公共入口及 metadata 变化。

## 每步最小闭环

- 开始：核对分支、工作区、环境、实际加载扩展和上步证据，确定本次唯一范围。
- 实现：代码、构建接入、源对应关系及针对性检查一起完成。
- 验证：按影响范围运行 910B 精度、性能及普通 FA3 回归，记录实际设备条件、命令、原始数据和结果；未运行项目明确标注。
- 提交：检查差异和 git diff --check，只暂存本步文件，运行仓库质量检查，更新 status，提交一次后停止。

## 构建与检查

只构建目标后端：

```bash
FLASH_ATTN_BUILD_VERSION=v3 FLASH_ATTN_BUILD_NPU=910 FLASH_ATTN_SKIP_SUBMODULE_INIT=1 python setup.py build_ext --inplace
```

header、AICPU 或编译单元集合变化时核查增量依赖，必要时使用 `FLASH_ATTN_FORCE_REBUILD=TRUE`。测试记录扩展实际路径、哈希及参与链接的对象，确认与当前源码对应。设备选择依据实际健康和占用情况。

reference、来源校验、profiler 和契约检查分别使用现有 `tests/test_flash_attn_npu_v3_mla_reference.py`、`tests/test_flash_attn_npu_v3_mla_provenance.py`、`tests/test_flash_attn_npu_v3_mla_profile.py` 与 `tests/test_flash_attn_npu_v3_mla_contract.py`。精确节点和命令写入 status 指向的证据目录。纯文档步骤检查链接、范围、格式和状态一致性。
