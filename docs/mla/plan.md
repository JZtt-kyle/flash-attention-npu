# MLA 迭代计划

本页只记录实施顺序、产物和完成条件。**R8 已完成：优化 MLA 普通调用并移除图模式；无待开发步骤。** R7 结构收敛及之前迁移步骤已完成；能力、限制和证据见 [status](status.md)，接口与来源见 [design](design.md)。

## R8：优化普通调用并移除 MLA 图模式（已完成）

基线 `36e8609`。用户先确认合并自动 split 发射、消除重复 metadata/分片扫描与未使用输出开销；2026-10-20 进一步明确“先删除图模式相关的实现和测试”。本轮按最新范围完成一次提交，不再要求 MLA 图功能或图性能验收。此前未提交的普通调用优化保留，普通 FA3 的图路径不变。

- 改动与影响：保留已实现的五布局自动 split 融合、分片二分查询及 O-only 输出准备优化；删除 `mla_forward.cpp` 中 metadata 的捕获切流分支及 INT8 冷捕获 mask 特例。所有 MLA real 前向/metadata 入口在 host 平台准备阶段明确拒绝捕获；保留 fake/compile、动态校验、普通多流依赖与已完成初始化的不可变 mask 复用。数值 Kernel、AICPU 记录协议、CANN/CATLASS 不因删除图模式而修改。
- 测试调整：删除 `tests/test_flash_attn_npu_v3_mla.py` 的图捕获/重放、图内动态长度/前缀、冷捕获和 compile 后重放测试；混合 stream/graph 生命周期用例改为普通多流调用，保留临时输入、独立输出及分配压力检查。`torch.compile(fullgraph=True)` 的 fullgraph 参数属于 Dynamo 跟踪边界，继续保留。不新增图测试目录或框架，捕获拒绝用一次性公开入口诊断核对。
- 环境及证据：工作根 `/tmp/flash-attention-npu-mla-46928a793876/mla-no-graph-20261020`；变更前源码/差异已保存到 `before/` 与 `before.diff`。核对当前源码、Python/torch/torch_npu/CANN、910B、CATLASS 与扩展 SHA，沿用已确认的环境启动器 `/tmp/mla-m16-schedule/run.py`、物理设备 3 和 CPU 48–51。所有构建、参考缓存和日志在仓库外。
- 必跑预检：既有 pre-commit 工具、pytest `--collect-only` 与新旧节点差分；正常 v3/910 构建。代表集合沿用前次 88 节点中仍保留的 80 个，再补普通多流生命周期节点，覆盖五布局/双 dtype、单片/显式/自动、最长 KV、INT8 与 fake/compile；捕获拒绝覆盖公开 func/varlen/cache/metadata、两种非量化表示及两种 INT8 布局。
- 正确性：实现固定后完整执行 `python3 -m pytest -q tests/test_flash_attn_npu_v3_mla.py`，加此前的 140 项普通 FA3 相关集合（含普通 FA3 图测试），保存 JUnit/原始日志及实际扩展身份。保留独立 reference 与 `assert_fa_close` 阈值，原始接口拒绝及数值边界不变。
- 性能：host 平台检查和 INT8 mask 路径改变，重测固定非量化 70 配置（调度 20、原 eager 40、默认 10）和 INT8 BSND 24/TND 15；沿用固定 CANN 包、输入、5 次预热、5×20 wall、入口/R 内及配置间等权、完整调用计时边界。先运行代表工具预检及成本边界，再采完整 eager 集合；图采样不再执行。通过均值 ≤1，保留每配置/分段退化及完整样本。
- 继承：仅继承未变的数值组件、参考方程、固定 native 包身份与原始覆盖目的。前次 1,719 通过、图性能失败及普通调用性能结果保存在 `mla-e2e-opt-20261020/`，不能计作本次修改后的通过数；前次分片查询等价诊断可作为未改算法的依据。修改/改名用例及新扩展重新实跑。

加载预检发现当前 torch_npu 未导出 `IsContextInitialized`，故捕获拒绝使用仓库原先已验证的 `currentStreamCaptureStatusMayInitCtx`。调用位于真实 NPU tensor 的 device guard / 平台准备路径，保持原上下文前提；初始加载失败日志保留，修正后重建再运行全部检查。

最终产物：删除 99 个图专用节点，24 个生命周期节点改为普通多流；1,480 MLA + 140 普通 FA3 共 1,620 项通过。固定 CANN 的非量化调度/原 eager/默认路径均值为 0.771881/0.802949/0.741942，INT8 39 配置为 0.597452；分段及相对 R7 的局部退化保留在 status。捕获拒绝与最终质量检查通过，本轮范围已验收完成。

完成条件：图模式支持分支与 MLA 图测试移除，公开捕获明确拒绝；当前范围内完整正确性与固定 CANN eager 门槛通过；三份文档和开发约定与新范围一致。完成既有 pre-commit 和 `git diff --check` 后，只提交本轮文件一次并立即停止。未完成时保留 R8 标记及实际结果，不把删除前的图失败写成通过。

## R7：按仓库已有范式继续收敛 MLA（已完成）

基线 `a48ba27`。用户进一步明确目录和 106 个文件仍过多，以当前库已有接口的组织范式为主。本轮在已授权结构整改内一次完成，不单独提交平铺中间态。参照普通 FA3 的 `fwd_dispatch.hpp` / `fwd_dispatch_impl.hpp`、`fa_metadata*`、`fa_split.h` 与 `fag_tiling.*`：入口、分派、metadata、tiling/split 集中，数值组件按实际职责保留，实例继续复用既有 `autogen/`。Kernel / Block / Tile 是逻辑分层，不为每个层次或布局建目录。

- 改动与影响：取消七个子目录；合并单片布局 launcher、显式/自动 split launcher 及薄 Kernel/metadata 适配；同一 PA 分派以编译期布局选择共用实现。基础形状/序列/mask 调度归入 tiling，split/GQA 计划归入 split，metadata 声明与生产集中，消费者按基础记录和 split/扩展记录组织。只被 forward 使用的分页静态准备回到 host 入口；vector 小操作放回同一 Tile 文件。保留独立的压缩、prefill 和 INT8 数值组件以及 CANN 源分核文件。更新 include、生成器、编译清单、来源说明和文档链接，不按历史开发阶段保留文件，也不新建通用框架。
- 边界：公开接口、dtype/布局组合、记录版本/字段/地址宽度、动态校验、算法顺序、分核/切块、发射属性及生命周期均不变。38 个 MLA ASC 编译实例和非量化 FIA / 量化 IFA 两个 AICPU 编译单元保留；host 调度文件归并会改变编译边界，纳入构建与性能验证。CANN/CATLASS 只读，保留来源和版权。
- 预检：已核对 Python/torch/torch_npu/CANN、CATLASS、910B 和 R6 实际源码/扩展，保存基线及旧扩展。每项合并核对前后函数/记录和 include 依赖，检查重名/循环与所有旧发射入口，数值 Kernel/Block/Tile 保持非注释 token 等价；共用分派逐布局核对展开分支。生成器 46 个实例（MLA 38）清单不增不减，输出幂等，先运行仓库质量工具及文档链接检查。
- 构建与正确性：`python3 -m pytest --collect-only -q tests/test_flash_attn_npu_v3_mla.py` 核对 1,578 项；使用原 v3/910 `setup.py build_ext --inplace`，构建与证据位于仓库外 `/tmp/flash-attention-npu-mla-46928a793876/mla-flat/`。固定最终实现后先复用 R6 的 88 项代表清单，覆盖双 dtype/表示、五布局、INT8、mask/split/softcap、长 KV 和 fake/compile/图成本边界；再完整运行 MLA 1,578 + 相关普通 FA3 140 项，保存原始日志、JUnit、实际扩展及源码身份。
- 性能：由于共享头与 host 编译边界变化，复用 R6 现有公开前后交替采样工具和固定 88 配置（非量化 80、INT8 8），对照本步保存的 R6 扩展。保留同设备/CPU、双 dtype 普通 FA3 对照、原独立 reference、输入/metadata/输出哈希、5 次预热和 5×20 采样、metadata implicit/generated/reuse 与 R=1/4；先检查代表及最长 KV 成本边界，再完整执行。记录配置等权耗时比及单例/分组变化，核对组织调整的影响，不将其称为重新完成对固定 CANN 的迁移性能验收。
- 继承：未修改测试/reference 的覆盖目的、固定源契约和 R6 已验证的原始算法。源码/构建发生变化，E2E 与结构性能对照本轮重跑；中间的纯路径构建和旧通过数不计作最终验收。现有公开用例足以验证行为，不新增 MLA 专用目录、私有验收格式或镜像实现的测试。相关实现改变后重新固定并重跑受影响组。

代码产物：MLA 106 → 49 文件，内部七个子目录取消；连同 38 个生成实例为 144 → 87 文件。源码 32,203 → 31,601 行，31 份数值组件保留，布局选择共享分派。验证结果统一记录于 status。

完成条件：MLA 无子目录，按仓库已有职责组织且文件实际减少；数值组件和独立来源保持可辨，所有受影响构建/公开 E2E 与必要性能检查完成，三份文档一致，仓库 pre-commit 和 `git diff --check` 通过后只提交一次并停止。

## R6：按 CANN 来源与 CATLASS 职责收敛生产代码（已完成）

基线 `87098f3`。一次完成适配层、构建、文档与验证并提交，不再按布局、协议版本或 Kernel / Block / Tile 分开发轮。固定 CANN 为 `ca18aee610729ca5270a44e2d84c751bc687fbf0`，CATLASS 为 `769cd40a8716b28650b6bebb08db4834eea4462f`；源工作树 HEAD 已不同，来源核对直接读取固定提交。

- 组织：保留 Kernel / Block / Tile 的算法与流水层次；host 保留公开 tensor 校验及生命周期，device 集中发射参数与 launcher，tiling 集中形状/长度、源分核与 split 计划，metadata 集中记录协议。布局变体在同一职责文件中组织；移除确认无生产引用的历史组件，合并纯转发头，实例化统一由现有 `autogen/generate_kernels.py` 生成。保留每份来源及版权。
- 收敛：按 dense、sequence、paged、mask、显式/自动 split、prefill/GQA 的实际职责合并 metadata 声明、生产者和消费者，抽取重复序列化；AICPU 按非量化 FIA 与量化 IFA 保留两个编译入口，继续复用同一 host 调度实现。保留独立动态输入校验、所有协议版本/字段/地址宽度/错误行为，以及现有图内发射和资源生命周期。
- 影响：所有 910 MLA 布局和 dtype，包括 INT8；共享头、AICPU 编译边界、ASC 实例生成、链接与依赖跟踪均纳入。算法、数值顺序、切块/分核策略、接口和测试阈值保持基线，CANN/CATLASS 源仓只读。无需新增数值算法或测试框架。
- 廉价预检：检查环境、设备、实际扩展及基线 SHA256，保存旧扩展；核对固定源与 CATLASS 对应实现。生成器先做语法、输出清单/幂等和旧入口完整性检查，host/AICPU 合并先检查依赖与重复定义；核对现有质量工具和 native 包路径。先 `python3 -m pytest --collect-only -q tests/test_flash_attn_npu_v3_mla.py`，收集应仍为 1,578 项，测试源码及 reference 不改。
- 构建与正确性：使用原 `setup.py build_ext --inplace`，`FLASH_ATTN_BUILD_VERSION=v3 FLASH_ATTN_BUILD_NPU=910`，产物放仓库外 R6 工作根。先双 dtype/表示、五布局、INT8、mask/split/softcap、长 KV、fake/compile/graph 代表；固定实现后运行完整 MLA 1,578 项与 R5 的普通 FA3 140 项相关集合，保存 pytest JUnit 和原始日志。原测试覆盖足以检查结构等价性，不新增镜像实现的单元测试；生成器与编译入口通过清单/构建实际验证。
- 等价与性能：检查重构前后注册/发射入口完整，设备数值函数仅移动或格式变化；复用原公开前后交替采样工具，同 CPU 48–51/910B 比较旧、新扩展。固定非量化五布局 × 两种表示 × 双 dtype × 四个场景（无 mask 单片、causal 显式 split、window+softcap 单片、无 mask 自动 split），共 80 个配置；INT8 两布局 × decode 单片、causal 单片、causal FD、长 KV 连续空尾四场景，共 8 个配置。另以两种 dtype 的普通 FA3 作对照，不计入 MLA 平均。先验证代表及最长 INT8 KV=16,385 的成本边界，再按原 5 次预热、5×20 采样、metadata implicit/generated/reuse 和 R=1/4 交替执行；原输入、metadata 与全部输出哈希须逐例一致，数值预检仍调用独立 reference。记录完整样本、逐项及等权均值，用于检查结构变化的影响，不改写 R5 已接受的既有 CANN 性能差距或历史图性能结论。
- 继承：测试与独立 reference 未变时继承其覆盖目的、成本边界和固定源来源契约；源码/构建已变，正确性重新实测，旧扩展通过数不计作本次通过。历史 native、组件和图性能只作背景，不累计为本次结果。

完成条件：职责清晰、重复包装和序列化实际减少、全部旧公开行为及必要检查通过，源码/扩展身份和受影响性能有可复核证据；更新 design 的组织/来源/构建链接和 status 的唯一交接，仓库 pre-commit 与 `git diff --check` 通过后提交一次并立即停止。必要验证失败或受限时保留 R6，不标记完成。

最终产物：MLA 239 → 106 文件，保留完整行为与数值层次；非量化 FIA 与量化 IFA 按源职责保留两个 AICPU 编译单元，实例由既有生成器管理。完整公开回归和固定 88 配置前后对照已完成，当时结果、环境、预检调整及限制见 `git show a48ba27:docs/mla/status.md`。

## 已完成步骤

这里只保留交付边界；具体历史命令和测试工具在对应 Git 提交中。各布局步均包含完整调用链和必要验收；非量化拒绝项见 design 第 2 节，INT8 边界见 3.4，旧步骤的命令、数量、耗时、失败修正及证据从对应提交的 status 查询。

| 步骤 | 已交付产物 |
| --- | --- |
| M00–M03 | 固定迁移交集/FA3 契约、独立双 reference 与固定源执行器、入口识别/校验/未实现拦截 |
| M04–M06 | 512+64 dense 数值与非 PA BSND/TND/混合生产前向，v1–v5、allocation/used、完整 storage/stride 与生命周期 |
| M07 | ND PA 地址/Copy/调度、v6、BSND/TND PA 公开 metadata/cache 及阶段收尾 |
| M08–M09 | 五布局右下 causal 与 window，v7–v16、公开接入及各阶段收尾 |
| M10 | partial/FD 归并与真实 accumulator；五布局显式/原生自动 split，v17–v26、公开入口及组合/性能阶段收尾 |
| M11–M14 | 192/128 独立 K/V、GQA 数值组件及五布局完整公开前向，v27–v31；三 mask、全部 split、真实 accumulator、allocation/used、stride、页安全与组合收尾 |
| M15a–f | softcap 独立数值组件及两种表示、五布局完整前向，v32–v46；三 mask、全部 split、allocation/used、分页安全与公开性能 |
| M15g1–g2 | packed/unpacked GQA 坐标与资源组件、pack_gqa/sm_margin 公开接入，v47；修复共享 softcap 比较尾部 |
| M16–M17 | 固定/动态图重放、发射优化、910B/C 配置兼容、双 dtype 用户示例及非量化阶段收尾；图性能失败按当次明确授权接受，默认门槛未变 |
| M18–M21 | INT8 独立数值与 ND PA 适配、BSND/TND 公开前向、metadata/real/fake/compile/图、示例与阶段收尾；graph 性能按授权暂缓 |
| R4 | 公开功能 E2E 合并为单个普通 pytest 文件，退役 MLA 私有单元、固定源/Trap、性能和验收工具，更新现有 CI 与文档入口 |
| R5 | 精简重复 E2E 并保留完整功能组合，修复零 PV 位修正及 softcap 归一化，补充确定性回归；既有默认路径性能差距按当次用户授权记录后收尾 |
| R6 | 按 CANN 来源与 CATLASS 职责归并 tiling/metadata/device，抽取重复逻辑、统一实例生成、保留量化编译边界；完成完整公开回归与前后性能检查 |
| R7 | 参照已有 FA3 范式平铺并归并为 49 个 MLA 文件，共用布局分派、集中 tiling/split/metadata；完成完整公开回归与固定 88 配置前后对照 |
| 工具整理、R1、R2 | 统一验收与缓存/证据管理、模块归属与共享 helper、共用前向分派、文档轮和无头交接；原 R3 的内核可读性核对合入 R2 |

非量化收尾为 `ebdf3eb`，M18 开发闭环为 `619787b`。组合映射见 [512+64](design.md#75-51264-组合覆盖)、[192/128](design.md#76-192128-组合覆盖) 和 [非量化收尾](design.md#77-非量化收尾映射)。历史查询：

```bash
git log --oneline -- docs/mla/status.md
git show ebdf3eb:docs/mla/status.md
git show 619787b:docs/mla/status.md
```

## 迭代与收尾规则

遵循 [AGENTS.md](../../AGENTS.md#一次工作只完成一个提交) 和 [design 第 7 节](design.md#7-验收口径)。实现前明确影响路径、必要用例、继承依据和完成条件；验证失败时保留交接，不提交为完成。测试整理后的入口为普通 pytest，只有一个 MLA E2E 文件，不恢复迁移阶段的独立测试框架。每成功完成五轮开发插入专门文档轮；文档轮不计开发轮数。一次提交完成后立即停止。
