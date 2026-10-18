# MLA 迭代计划

本页只记录实施顺序、产物和完成条件。**M17 全范围收尾已完成，M16 后的专门文档核对合入同一提交；没有下一开发步。** 当前能力与实测证据见 [status](status.md)，契约与验收口径见 [design](design.md)。本轮图性能仍失败，按完整实测后用户对 M17 的明确授权接受为已知限制，默认性能门槛不变。

## M17 全范围收尾

2026-10-18 用户确认“完成 M17 全范围收尾，合并本次文档核对”。M16 后应做的专门文档核对合入本步，仅交付一个提交；不重启无头循环。

影响路径：`mla_forward.cpp::DensePlatform` 补齐 CANN `Ascend910_93`（910C）名称准入，沿用平台查询、C1V2、核数上界、内存与 blockDim 校验，不修改数值核、metadata 协议或 CATLASS。新增平台 CPU 检查，直接编译生产查询函数并使用安装 CANN 的 B/C 平台配置验证核数、存储与普通/全核发射；覆盖非目标芯片、空名称及错误资源拒绝。统一驱动注册该组，沿用已有依赖/证据格式。README 中英文同步两种表示、五布局、softcap、调度、图及拒绝边界，并添加可运行示例与其 E2E 验证。

必跑检查与选择依据：

- 先 `--dry-run`、`--collect-only` 核对下列集合；平台 CPU/ASan/UBSan、所属工具和示例静态检查先行。新用例只补平台兼容、示例调用及最终覆盖映射缺口。
- 正式构建并核对实际加载；`gqa-options-check gqa-options-source gqa-options-boundaries graph-check` 验证代表调用、固定源与双数学、拒绝与捕获。数值算法和 reference 策略未改，不重复历史最长 reference 成本检查。
- 冻结实现后执行 `e2e detail-gqa_options detail-graph ordinary graph-ordinary`，覆盖两种表示、五布局、三 mask、全部 split、softcap/调度组合、fake/compile、长度/stride/复用/生命周期与 Trap。核对 design 7.5/7.6 与实际节点的映射，缺失组合补入所属 E2E。
- 原固定性能集合 `graph-profile`（40 配置）与 `gqa-options-schedule-profile`（56 配置），合计 96，不按结果重新选样。先执行 `graph-profile-check gqa-options-profile-check` 确认 native 配方、全部廉价夹具及真实 profiler；eager、图与调度分别按原入口/R 和平均门槛汇总。图若仍失败须保留结论；M16 的单次例外不能自动写为 M17 通过。
- 收尾执行本次文档契约/链接核对、`git diff --check`、`ci/code_quality.sh --staged`，只暂存本步文件。

证据继承约定：M16 的完整 E2E 与性能用作历史基线，生产源码改变后由统一驱动判定，本步不预记继承通过；单元和性能实际执行。构建与数学缓存仅在内容身份有效时复用。证据根 `/tmp/mla-m17/`，环境使用已核验的 `/tmp/mla-m16-schedule/run.py`，实机仍为设备 7 的 910B3。910C 只作实现与资源配置兼容检查，不宣称已实测。

完成条件：上述必要检查和最终覆盖核对完成，公开示例可调用，三份文档职责清楚；失败、未运行和已接受限制分别记录。性能未达标时先完成其余收尾，再以具体结果处理验收边界，不改默认门槛或隐藏失败。

完成结论：约定检查与文档核对全部执行，正确性、eager 和调度性能通过；图性能实测失败，用户在看到本轮结果后确认“接受已知限制并提交”，按 design 第 7 节作为 M17 已知限制收尾。实际结果、耗时和原始证据位置统一保留在 status。

M17 是已确认迁移范围的最后一个里程碑；完成后没有自动启动的下一开发步，不把第 4 节排除项或图性能继续优化加入本次收尾。

## 已完成步骤

这里只保留交付边界。各布局步均包含完整调用链和必要验收；当前拒绝项以 design 第 2 节为准，旧步骤的命令、数量、耗时、失败修正及证据从对应提交的 status 查询。

| 步骤 | 已交付产物 |
| --- | --- |
| M00–M03 | 固定迁移交集/FA3 契约、独立双 reference 与固定源执行器、入口识别/校验/未实现拦截 |
| M04–M06 | 512+64 dense 数值与非 PA BSND/TND/混合生产前向，v1–v5、allocation/used、完整 storage/stride 与生命周期 |
| M07 | ND PA 地址/Copy/调度、v6、BSND/TND PA 公开 metadata/cache 及阶段收尾 |
| M08–M09 | 五布局右下 causal 与 window，v7–v16、公开接入及各阶段收尾 |
| M10 | partial/FD 归并与真实 accumulator；五布局显式/原生自动 split，v17–v26、公开入口及组合/性能阶段收尾 |
| M11–M14 | 192/128 独立 K/V、GQA 数值组件及五布局完整公开前向，v27–v31；三 mask、全部 split、真实 accumulator、allocation/used、stride、页安全与组合收尾 |
| M15a | 无同义 CANN attention 的 softcap 非线性独立组件：两种表示、双数学、舍入/零 scale 边界及真实 partial/FD |
| M15b–d | 两种表示的非 PA BSND、TND Q/KV、TND Q + BSND cache 正 softcap，v32–v40；完整公开入口、三 mask、全部 split、复用/Trap 与适用公开性能 |
| M15e | 两种表示的 BSND Q + ND PA KV 正 softcap，v41/v42/v43；完整公开入口、三 mask、全部 split、分页安全、复用/Trap 与适用公开性能 |
| M15f | 两种表示的 TND Q + ND PA KV 正 softcap，v44/v45/v46；完整公开入口、三 mask、全部 split、allocation/used、分页安全、复用/Trap 与适用公开性能 |
| M15g1 | 独立 packed/unpacked GQA 坐标、C1V2 剩余资源、源分核/显式片段及原 allocation O/LSE/accumulator 地址；仅 CPU 组件，公开接入留 M15g2 |
| M15g2 | 两种表示、五布局的 pack_gqa/sm_margin 完整公开 real/fake 接入，v47、host/AICPU/Kernel、三 mask/全部 split/softcap；修复共享 softcap 的 32-lane 比较尾部，完成公开专项、回归及固定性能集合 |
| M16 | 两种表示、五布局的固定/动态图重放、metadata 重生成、跨层/多 stream、生命周期、fake/compile 与 Trap；优化任务分配和融合发射，修复图采样与 wall 生命周期；正确性和 eager/调度性能通过，图性能按本次用户授权例外接受，失败结果保留 |
| M17 | 910B/C 平台资源兼容与 CPU/ASan/UBSan 检查、双 dtype 可运行用户示例、全范围正确性与固定性能集合收尾；同步中英文 README、覆盖映射及三份文档，图性能差距按本轮明确授权保留为已知限制 |
| 工具整理、R1、R2 | 统一验收与缓存/证据管理、模块归属与共享 helper、共用前向分派、文档轮和无头交接；原 R3 的内核可读性核对合入 R2 |

组合验收映射见 [512+64](design.md#75-51264-组合覆盖)、[192/128](design.md#76-192128-组合覆盖) 和 [全范围收尾](design.md#77-全范围收尾映射)。M14、M15a–d 五轮开发后的专门文档轮已完成，不计开发轮数。M15e、M15f、M15g1、M15g2、M16 是其后的五个已接受开发轮；本次按用户确认将随后文档核对合入 M17，不单独提交文档轮或重启无头循环。

```bash
git log --oneline -- docs/mla/status.md
git show feb189b:docs/mla/status.md  # M10 阶段证据
git show f7105ef:docs/mla/status.md  # M14 阶段证据
git show 26d7a70:docs/mla/status.md  # M15d 开发闭环与完整复现命令
```

## 迭代与收尾规则

遵循 [AGENTS.md](../../AGENTS.md#一次工作只完成一个提交) 和 [design 第 7 节](design.md#7-验收口径)。实现前在当前步补齐实际影响路径、必跑组/命令及选择依据、拟继承证据与依赖、完成条件；验证受限或失败时保留交接，不提交为已闭环。每成功完成五轮开发插入一次专门文档修正轮；文档轮不计开发轮数，手动任务按当次用户要求停止。
