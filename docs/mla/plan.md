# MLA 迭代计划

本页只记录实施顺序、产物和完成条件。**开发已完成到 M13b，M14 阶段收尾尚未完成。** 当前能力与最近证据见 [status](status.md)，契约与验收口径见 [design](design.md)。

## 唯一下一步 M14

完成 192/128 prefill 的组合收尾。M11–M13 已分别交付 BSND、紧凑 TND、TND Q + BSND cache、BSND PA、TND PA 的完整公开行为；下一轮核对阶段覆盖、补明确缺口并形成阶段验收，不提前开始 M15。

- 范围与产物：核对双 dtype、五布局、三 mask、自动/1/显式 split、GQA、allocation/used、合法 stride/页大小、空值、metadata 复用及 real/fake/compile/生命周期的已开放组合；以现有日常清单、专项和原始证据建立阶段覆盖说明。新增用例必须对应独立缺口，阶段结束不自动展开全组合，也不重复按布局或开关接入。
- 影响路径：优先复用 `tests/mla/regression_cases.py`、各 `e2e/prefill_*/`、现有公开性能清单/驱动及三份文档；实现前写明实际缺口、修改路径和必跑组。只有发现新增数值算法或共享协议的独立未覆盖风险，才另列组件产物和完成条件；不全面重构已验收生产路径。
- 正确性与继承：从 status 保留的 M13b summary、用例清单、依赖/构建和原始证据恢复；文档轮的离线校验不替代下一开发轮的环境与加载身份检查。仅身份匹配、原始证据完整的 E2E 正确性可由 `--resume` 继承，须指向原 `--output` 目录及相同选择。累计小输入以 `python3 -m tests.mla` 为主线，受影响普通回归用 `--suite ordinary`，缺口专项按实际影响选择；不累计历史模块单元。
- 阶段性能：在采样前固定五布局合计不超过 100 个配置（含 dtype）的清单及选择依据，复用 `prefill-bsnd-profile prefill-tnd-profile prefill-mixed-profile prefill-paged-bsnd-profile prefill-paged-tnd-profile`。先核对 native 包、工具、廉价夹具及 dry-run/collect；性能实际执行，从原始入口/R 样本按配置等权汇总，保留全部单例/分段、退化及原生 FD 单列项，门槛与计时协议不变。
- 完成条件：阶段约定的组合和未开放边界已核对，必要正确性、相关回归、阶段性能及证据有效性通过，更新 design/status/plan 并完成质量检查；提交一次后停止。验证受限或失败时如实交接，不提交为已闭环。

## 后续里程碑：M15–M17

每行是能力里程碑，不等于一次提交。细分须说明新增数值算法或共享协议的独立风险；其余按完整可调用行为交付，精度、安全、普通回归、生命周期和适用公开性能随入口验收。

| 里程碑 | 剩余产物与边界 | 完成条件 |
| --- | --- | --- |
| M15：softcap/调度选项 | softcap 数学独立闭环；pack_gqa、sm_margin 按共享调度影响划分 | 有效数学/调度语义、资源校验及适用 E2E，不能忽略参数 |
| M16：图执行 | 固定输入重放与图内长度更新/metadata 重生成分别闭环 | eager/图一致、跨层复用、多 stream 与 fake/真实输出一致 |
| M17：全范围收尾 | 冻结能力，按覆盖矩阵验收并补用户示例；实现问题先修复再验收 | 必迁组合、精度/平均公开 E2E/普通 FA3 回归；910B/C 实现与资源兼容，实机仍用 910B；README 调用与边界 |

## 已完成步骤

这里只保留交付边界，当前拒绝项以 status 和 design 第 2 节为准。每步的命令、数量、耗时、失败修正及原始证据从该提交的 status 查询；不把历史阶段的拒绝条件当作当前限制。

| 步骤 | 已交付产物 |
| --- | --- |
| M00–M03 | 固定迁移交集/FA3 契约、独立双 reference 与固定源执行器、入口识别/校验/未实现拦截 |
| M04–M06 | 512+64 dense 数值与非 PA BSND/TND/混合生产前向，v1–v5、allocation/used、完整 storage/stride 与生命周期 |
| M07 | ND PA 地址/Copy/调度、v6、BSND/TND PA 公开 metadata/cache 及阶段收尾 |
| M08–M09 | 五布局右下 causal 与 window，v7–v16、公开接入及各阶段收尾 |
| M10 | partial/FD 归并与真实 accumulator；五布局显式/原生自动 split，v17–v26、公开入口及组合/性能阶段收尾 |
| M11a–b | 192/128 独立 K/V、GQA 通用数值组件及 BSND 完整公开前向，v27 |
| M12a–b | 192/128 紧凑 TND、TND Q + BSND cache 完整公开前向，v28/v29；allocation/used 与独立 stride |
| M13a–b | 192/128 BSND/TND Q + ND PA KV 完整公开前向，v30/v31；页几何、allocation/used、实际页访问与隔离 Trap |
| 工具整理、R1、R2 | 统一验收与缓存/证据管理、模块归属与共享 helper、共用前向分派、文档轮和无头交接；原 R3 的内核可读性核对合入 R2 |

M11–M13 的布局步均包含 host/AICPU/Kernel、metadata、real/fake/compile、三 mask、自动/1/显式 split、真实 FP32 accumulator 及公开 E2E/性能验收。M13b 的五布局性能复测是各组结果，不能代替 M14 的阶段覆盖核对与汇总。

```bash
git log --oneline -- docs/mla/status.md
git show feb189b:docs/mla/status.md  # M10 阶段证据
git show 4033fe9:docs/mla/status.md  # M13b 开发闭环及失败修正记录
```

## 迭代与收尾规则

遵循 [AGENTS.md](../../AGENTS.md#一次工作只完成一个提交) 和 [design 第 7 节](design.md#7-验收口径)。实现前在当前步补齐实际影响路径、必跑组/命令及选择依据、拟继承证据与依赖、完成条件；验证受限或失败时保留交接，不提交为已闭环。每成功完成五轮开发插入一次专门文档修正轮；文档轮不计开发轮数，手动任务按当次用户要求停止。
