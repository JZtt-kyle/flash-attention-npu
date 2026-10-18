# MLA 迭代计划

本页只记录实施顺序、产物和完成条件。**M16 按用户明确授权接受本次图性能例外并提交验收；正确性、eager 和调度专项性能通过，公开图性能仍未达标。** 当前能力与实测证据见 [status](status.md)，契约与验收口径见 [design](design.md)。

## M16 本次验收边界

2026-10-18 用户明确要求“图模式的性能先这样吧。先提交验收一次”。本次接受已实测的图性能差距，保留图平均比 1.455079 的失败结论及原始证据；eager 平均比 0.799189、调度专项平均比 0.726879 和必要正确性均通过。该例外仅用于本次 M16 提交，性能工具的默认平均门槛、输入、R、采样和失败传播不变。

本次收尾只同步三份文档的验收边界和下一步，不改变生产、测试或构建代码。复核统一驱动的用例清单、源码依赖、构建身份和原始结果，沿用同一步已实际完成的检查，不把证据核对记为新一轮实机通过。必跑收尾为相关文档契约/链接、`git diff --check` 和 `ci/code_quality.sh --staged`；完成后只提交本步 30 个文件并立即停止，不启动无头循环或下一开发步。

## 唯一下一步 专门文档修正轮

M16 是最近专门文档轮后的第五个已接受开发轮。下次接手先核对 design、plan、status 与代码及有效证据，修正错误、过期内容和冗余，保留本次图性能例外及后续默认门槛；仅作文档修正，不探测设备或重建扩展。文档轮完成后再推进 M17，本次提交不执行该轮。

## 后续里程碑：M17

每行是能力里程碑，不等于一次提交。细分须说明新增数值算法或共享协议的独立风险；其余按完整可调用行为交付，精度、安全、普通回归、生命周期和适用公开性能随入口验收。

| 里程碑 | 剩余产物与边界 | 完成条件 |
| --- | --- | --- |
| M17：全范围收尾 | 核对并补齐 910B/C 实现与资源兼容，再冻结能力、按覆盖矩阵验收并补用户示例 | 必迁组合、精度/平均公开 E2E/普通 FA3 回归；解决当前仅接受 910B 的生产平台限制，实机仍用 910B；README 调用与边界 |

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
| 工具整理、R1、R2 | 统一验收与缓存/证据管理、模块归属与共享 helper、共用前向分派、文档轮和无头交接；原 R3 的内核可读性核对合入 R2 |

组合验收映射见 [512+64](design.md#75-51264-组合覆盖) 和 [192/128](design.md#76-192128-组合覆盖)。M14、M15a–d 五轮开发后的专门文档轮已完成，不计开发轮数；该轮只修 Markdown 与公开 docstring，核对固定来源、原始证据和质量，未推进功能。M15e、M15f、M15g1、M15g2、M16 是该文档轮之后的第一至第五个已接受开发轮，M16 的图性能例外见上文。

```bash
git log --oneline -- docs/mla/status.md
git show feb189b:docs/mla/status.md  # M10 阶段证据
git show f7105ef:docs/mla/status.md  # M14 阶段证据
git show 26d7a70:docs/mla/status.md  # M15d 开发闭环与完整复现命令
```

## 迭代与收尾规则

遵循 [AGENTS.md](../../AGENTS.md#一次工作只完成一个提交) 和 [design 第 7 节](design.md#7-验收口径)。实现前在当前步补齐实际影响路径、必跑组/命令及选择依据、拟继承证据与依赖、完成条件；验证受限或失败时保留交接，不提交为已闭环。每成功完成五轮开发插入一次专门文档修正轮；文档轮不计开发轮数，手动任务按当次用户要求停止。
