# MLA 迭代计划

本页只记录实施顺序、产物和完成条件。**用户于 2026-10-19 授权的 R6 生产代码结构收敛已完成，无待开发步骤。** 非量化 M17、INT8 M18–M21、R4 测试整理与 R5 数值修复已完成；能力、限制和证据见 [status](status.md)，接口与来源见 [design](design.md)。

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

最终产物：MLA 239 → 106 文件，保留完整行为与数值层次；非量化 FIA 与量化 IFA 按源职责保留两个 AICPU 编译单元，实例由既有生成器管理。完整公开回归和固定 88 配置前后对照已完成，具体结果、环境、预检调整及限制只记录于 status。

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
| 工具整理、R1、R2 | 统一验收与缓存/证据管理、模块归属与共享 helper、共用前向分派、文档轮和无头交接；原 R3 的内核可读性核对合入 R2 |

非量化收尾为 `ebdf3eb`，M18 开发闭环为 `619787b`。组合映射见 [512+64](design.md#75-51264-组合覆盖)、[192/128](design.md#76-192128-组合覆盖) 和 [非量化收尾](design.md#77-非量化收尾映射)。历史查询：

```bash
git log --oneline -- docs/mla/status.md
git show ebdf3eb:docs/mla/status.md
git show 619787b:docs/mla/status.md
```

## 迭代与收尾规则

遵循 [AGENTS.md](../../AGENTS.md#一次工作只完成一个提交) 和 [design 第 7 节](design.md#7-验收口径)。实现前明确影响路径、必要用例、继承依据和完成条件；验证失败时保留交接，不提交为完成。测试整理后的入口为普通 pytest，只有一个 MLA E2E 文件，不恢复迁移阶段的独立测试框架。每成功完成五轮开发插入专门文档轮；文档轮不计开发轮数。一次提交完成后立即停止。
