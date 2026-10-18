# MLA 迭代计划

本页只记录实施顺序、产物和完成条件。**M15e 的完整公开接入与必要实机验收已完成；唯一下一步是 M15f：TND Q + ND PA KV softcap 完整公开接入。** 当前能力与最近证据见 [status](status.md)，契约与验收口径见 [design](design.md)。

## 唯一下一步 M15f TND PA softcap

交付 512+64 / 192/128 的 TND Q + ND PA KV 正 softcap。复用已验收的非线性数值、PA 地址/页校验、TND allocation/used、mask 和 split 组件，将三 mask 与自动/1/显式 split 随完整入口一起接通，不按布局内层次、开关或版本号拆步。

- 产物与影响路径：910 MLA 的 `mla_split_paged_tnd_*`、`mla_auto_split_paged_tnd_*`、共享 prefill、partial/merge、host/AICPU、记录及消费者，与公开 metadata/kvcache、raw C++ 和 real/fake/compile 同步接入。保留只读 cache、独立 K/V/page-table stride、有效页校验和既有 Q-used 复用边界；确认共享发射与构建依赖的影响。
- 新增验收：两种表示、双 dtype、三 mask、单/显式/自动 split、双数学、cap=0 固定源、真实 accumulator；TND 非均匀/空 Q allocation、Q-used 原位置、max 声明、页尾/页大小、乱序/重复页、全零 Q-used 与全部 live 页校验；长度更新重生成、正 cap 改值复用、fake/compile、多 stream/生命周期与隔离 Trap。在 `regression_cases.py` 注册所属日常 E2E。
- 执行顺序：实现前在本节补齐实际文件、所属单元/专项组和命令及选择依据；先 dry-run/collect-only、工具/native 包和廉价代表检查，再固定实现运行本模块 CPU/sanitizer、设备专项、累计 E2E 与普通 FA3 回归。稳定组件不重复拆步，受影响共享路径仍按依赖验收。
- 性能与证据：采样前固定合计≤100 配置；正 cap 无同义 CANN attention，单列公开调用、metadata、kernel 和开销参照，受影响 cap=0 路径按适用配置平均验收。最近原始证据及复现命令见 status；只在驱动确认身份完整匹配时继承 E2E，不继承单元或性能。共享 Python/C++ 或构建变化须重跑受影响组，数学 reference 继续按内容复用。
- 完成条件：完整调用、精度/固定源、分页/TND 安全、复用/拒绝和适用公开性能通过，相关已开放能力及普通 FA3 无回归；更新文档并通过质量检查，提交一次即停。必要验证失败或受限时保留交接，不提交为闭环。

## 后续里程碑：M15–M17

每行是能力里程碑，不等于一次提交。细分须说明新增数值算法或共享协议的独立风险；其余按完整可调用行为交付，精度、安全、普通回归、生命周期和适用公开性能随入口验收。

| 里程碑 | 剩余产物与边界 | 完成条件 |
| --- | --- | --- |
| M15：softcap/调度选项 | 完成 M15f TND Q + ND PA KV softcap；pack_gqa、sm_margin 按共享调度影响划分 | 有效数学/调度语义、资源校验及适用 E2E，不能忽略参数 |
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
| 工具整理、R1、R2 | 统一验收与缓存/证据管理、模块归属与共享 helper、共用前向分派、文档轮和无头交接；原 R3 的内核可读性核对合入 R2 |

组合验收映射见 [512+64](design.md#75-51264-组合覆盖) 和 [192/128](design.md#76-192128-组合覆盖)。M14、M15a–d 五轮开发后的专门文档轮已完成，不计开发轮数；该轮只修 Markdown 与公开 docstring，核对固定来源、原始证据和质量，未推进功能。M15e 是该文档轮之后的第一个开发轮。

```bash
git log --oneline -- docs/mla/status.md
git show feb189b:docs/mla/status.md  # M10 阶段证据
git show f7105ef:docs/mla/status.md  # M14 阶段证据
git show 26d7a70:docs/mla/status.md  # M15d 开发闭环与完整复现命令
```

## 迭代与收尾规则

遵循 [AGENTS.md](../../AGENTS.md#一次工作只完成一个提交) 和 [design 第 7 节](design.md#7-验收口径)。实现前在当前步补齐实际影响路径、必跑组/命令及选择依据、拟继承证据与依赖、完成条件；验证受限或失败时保留交接，不提交为已闭环。每成功完成五轮开发插入一次专门文档修正轮；文档轮不计开发轮数，手动任务按当次用户要求停止。
