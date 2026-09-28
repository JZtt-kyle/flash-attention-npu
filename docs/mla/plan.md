# MLA 迭代计划

当前具备 MLA 接口校验、独立 reference 和固定源码 CANN 基线。生产 MLA 前向与 MLA metadata 明确报未实现；普通 FA3 继续使用既有计算路径。M03 接口边界的构建与回归验收已完成，唯一下一步为 M04a。

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
| M04a | 唯一下一步 | CANN 源实现映射及 dense 实现拆分 | 固定源文件/函数对应 Kernel / Block / Tile；列明调度字段、资源布局、同步关系和必要 FA3 差异；确定首个有界范围及独立验收子步骤 |
| M04b | 待开始 | 512+64、连续 BSND、无 mask、显式 num_splits=1 的前向 | 按源执行流程实现已核实的有界范围；FP16/BF16 O/LSE、尾块、单/多 query、重复执行及相关 910B 性能对照通过 |
| M05 | 待开始 | 已验收 dense 范围的 MLA metadata | 将源调度计算接入 AICPU，保持调度结果；官方参数、运行时 scale、有/无 metadata、连续复用和普通 FA 兼容正确；metadata E2E 逐例不劣于同轮 CANN |
| M04c | 待 M05 完成 | dense 长 KV，含 8193 | 按源流程接通；FP16/BF16 O/LSE、单/多 query、重复执行及 metadata E2E 通过；显式 num_splits=1 |
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

M04a 的源码映射写入 design 对应架构部分，实现拆分写入本表，当前结果和证据写入 status。每个实现步骤保持源设备行为，组件替换按运算、数据布局、精度和资源/同步契约核对等价性。

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
