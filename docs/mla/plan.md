# MLA 迭代计划

本页只记录实施顺序、产物和完成条件。**M15f 的完整公开接入与必要实机验收已完成；唯一下一步是 M15g：共享调度选项 `pack_gqa` / `sm_margin` 的完整公开接入。** 当前能力与最近证据见 [status](status.md)，契约与验收口径见 [design](design.md)。

## 唯一下一步 M15g 调度选项

为已开放的五种布局、两种表示实现有效的 `pack_gqa` / `sm_margin` 语义。依据固定 FA3 接口和现有 CANN 调度确定参数到打包、分核及资源预算的映射，完整接通 host/AICPU、metadata、消费者、Kernel 发射和公开 real/fake 入口；三 mask 与全部 split 随调用链验收。

- 产物与影响路径：核对共享平台/分核、显式与自动 split、prefill、公开 Python/C++ 和 metadata 复用身份。保留既有 allocation/used、stride、分页安全和生命周期边界；参数必须影响约定行为，不能忽略后放行。实现前在本节补齐实际文件、固定来源对应语义与 design 契约变更。
- 必跑验收：新增非默认选项的公开 E2E，覆盖默认值、有效组合、资源上下界、记录复用/拒绝、real/fake/compile 和适用 Trap；按依赖选择本模块 CPU 差分/边界/sanitizer、受影响设备双数学/固定源、累计小输入 E2E 及普通 FA3 回归。实现前列出具体用例、统一驱动组和命令，先 dry-run/collect-only、工具/native 包及代表检查，再固定实现运行最终集合。
- 性能与证据：采样前固定合计≤100 配置，覆盖新调度语义及受影响公开路径，沿用 R=1/4、计时边界和配置内等权平均。仅在驱动确认来源、依赖及原始证据完整有效时继承 E2E；单元和性能实际执行，数学 reference 按内容复用。最近闭环来源与复现命令见 status。
- 完成条件：完整调用、有效参数语义、资源安全、精度/固定源、复用/拒绝和适用平均公开性能通过，相关已开放能力无回归；更新文档、通过质量检查后提交一次即停。若发现新的数值算法或共享协议有独立未覆盖风险，先在 plan 写明风险、独立产物及完成条件，再调整步骤。

## 后续里程碑：M15–M17

每行是能力里程碑，不等于一次提交。细分须说明新增数值算法或共享协议的独立风险；其余按完整可调用行为交付，精度、安全、普通回归、生命周期和适用公开性能随入口验收。

| 里程碑 | 剩余产物与边界 | 完成条件 |
| --- | --- | --- |
| M15：softcap/调度选项 | M15g：pack_gqa、sm_margin 的有效语义与完整公开接入 | 有效数学/调度语义、资源校验及适用 E2E，不能忽略参数 |
| M16：图执行 | 固定输入重放与图内长度更新/metadata 重生成分别闭环 | eager/图一致、跨层复用、多 stream 与 fake/真实输出一致 |
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
| 工具整理、R1、R2 | 统一验收与缓存/证据管理、模块归属与共享 helper、共用前向分派、文档轮和无头交接；原 R3 的内核可读性核对合入 R2 |

组合验收映射见 [512+64](design.md#75-51264-组合覆盖) 和 [192/128](design.md#76-192128-组合覆盖)。M14、M15a–d 五轮开发后的专门文档轮已完成，不计开发轮数；该轮只修 Markdown 与公开 docstring，核对固定来源、原始证据和质量，未推进功能。M15e、M15f 是该文档轮之后的第一、第二个开发轮。

```bash
git log --oneline -- docs/mla/status.md
git show feb189b:docs/mla/status.md  # M10 阶段证据
git show f7105ef:docs/mla/status.md  # M14 阶段证据
git show 26d7a70:docs/mla/status.md  # M15d 开发闭环与完整复现命令
```

## 迭代与收尾规则

遵循 [AGENTS.md](../../AGENTS.md#一次工作只完成一个提交) 和 [design 第 7 节](design.md#7-验收口径)。实现前在当前步补齐实际影响路径、必跑组/命令及选择依据、拟继承证据与依赖、完成条件；验证受限或失败时保留交接，不提交为已闭环。每成功完成五轮开发插入一次专门文档修正轮；文档轮不计开发轮数，手动任务按当次用户要求停止。
