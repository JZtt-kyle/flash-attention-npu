# MLA 迭代计划

本页只记录实施顺序、产物和完成条件。**已完成 M15c 非 PA TND Q/KV softcap 完整公开接入；下一步是 M15d 非 PA 混合布局 softcap 完整公开接入。** 当前能力与最近证据见 [status](status.md)，契约与验收口径见 [design](design.md)。

## 唯一下一步 M15d

将已验证的 softcap 数值组件接入 **非 PA TND Q + BSND KV cache 的 512+64 和 192/128 完整公开行为**，稳定的三 mask、自动/1/显式 split 随同接入；范围仍为 [design 4.1/4.2](design.md#4-功能边界) 的 SC01、MD01。一个提交包含 host、AICPU、Kernel、metadata、公开 kvcache 与 real/fake 入口和必要验收，不按层或开关拆轮。

- 产物与影响路径：复用已有混合布局调度/地址、explicit/auto partial/merge 和共用 prefill 前向，传递真实 cap；沿用记录字段，以特征绑定隔离 cap=0 与正 cap，正值改值可复用。新增记录版本号不构成独立协议步。
- 接口边界：接通既有公开 kvcache、`get_scheduler_metadata(has_softcap=True)`、内部记录/前向和 raw C++；保留 Q allocation、KV used、声明 max、独立 stride 及现有 Q-used/复用边界，不增加官方接口之外的参数。PA、非默认调度和 graph replay 不混入本步。
- 必增验收：两种表示、双 dtype、三 mask、自动/单片/多片、默认/零/负 scale、非均匀/空 allocation、used/max、独立 stride、真实 accumulator；正 cap 独立双数学和 cap=0 固定源。增加 real/fake/compile、记录特征/长度冲突、改正 cap 复用、长度更新重生成、多 stream/生命周期及隔离 Trap，并注册所属日常小输入 E2E。
- 实现前在本节细化实际修改文件、必跑组/命令及选择依据，涵盖共享调度、地址、发射和构建影响；先 dry-run/collect-only、工具/native 包和代表检查，再固定实现运行本模块单元/sanitizer、公开专项、累计 E2E 与相关普通 FA3 回归。稳定数值组件不重复拆步。
- 公开性能按 design 第 7 节，在采样前固定≤100 配置及适用集合，保持 R 和计时边界。新路径与受影响已有路径本步实测；无同义 CANN softcap baseline 时单列公开调用、metadata、kernel 与开销参照，指定 split 组沿用既有汇总器。
- 证据与依赖：从 status 的 summary、实际清单、依赖/构建 manifest 和原始证据复核；只继承依赖未变且完整有效的 E2E，不累计历史单元或性能。沿用数学内容缓存和已验证的成本边界，新增入口仍须代表实测。
- 完成条件：上述混合布局完整调用、数学/固定源、accumulator、长度/复用/拒绝及适用平均公开性能通过；BSND/TND softcap 和相关普通 FA3 回归通过。更新三份文档、质量检查，提交一次即停；必要验证失败或受限时保留交接，不提交为闭环。

## 后续里程碑：M15–M17

每行是能力里程碑，不等于一次提交。细分须说明新增数值算法或共享协议的独立风险；其余按完整可调用行为交付，精度、安全、普通回归、生命周期和适用公开性能随入口验收。

| 里程碑 | 剩余产物与边界 | 完成条件 |
| --- | --- | --- |
| M15：softcap/调度选项 | 各布局 softcap 完整公开接入；pack_gqa、sm_margin 按共享调度影响划分 | 有效数学/调度语义、资源校验及适用 E2E，不能忽略参数 |
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
| M14 | 192/128 五布局组合收尾；修正过期拒绝测试，复用固定 60 配置并增加原样本阶段汇总，完成公开专项、累计 E2E、普通回归与性能验收 |
| M15a | softcap 无同义源的新非线性独立风险闭环：512+64 / 192/128 的 scale 后、mask 前计算、近零精度、输出中点舍入、零 scale 均匀权重与真实 partial/FD；组件双数学及分段性能，公开仍拒绝 |
| M15b | 非 PA BSND 的 512+64 / 192/128 正 softcap 完整公开接入，v32–v34；三 mask、单/显式/自动 split、real/fake/compile、复用与 Trap、双数学及公开性能；补共享单片计划边界和固定 split 组汇总 |
| M15c | 非 PA TND Q/KV 的 512+64 / 192/128 正 softcap 完整公开接入，v35–v37；三 mask、单/显式/自动 split、内部 metadata、公开 varlen real/fake/compile、长度更新/复用/Trap、双数学与固定源及公开性能 |
| 工具整理、R1、R2 | 统一验收与缓存/证据管理、模块归属与共享 helper、共用前向分派、文档轮和无头交接；原 R3 的内核可读性核对合入 R2 |

M11–M13 的布局步均包含 host/AICPU/Kernel、metadata、real/fake/compile、三 mask、自动/1/显式 split、真实 FP32 accumulator 及公开 E2E/性能验收。M14 的组合覆盖映射在 design 7.6，阶段证据与未开放边界在 status；不在 plan 重复用例数量、耗时或性能结果。

```bash
git log --oneline -- docs/mla/status.md
git show feb189b:docs/mla/status.md  # M10 阶段证据
git show 4033fe9:docs/mla/status.md  # M13b 开发闭环及失败修正记录
```

## 迭代与收尾规则

遵循 [AGENTS.md](../../AGENTS.md#一次工作只完成一个提交) 和 [design 第 7 节](design.md#7-验收口径)。实现前在当前步补齐实际影响路径、必跑组/命令及选择依据、拟继承证据与依赖、完成条件；验证受限或失败时保留交接，不提交为已闭环。每成功完成五轮开发插入一次专门文档修正轮；文档轮不计开发轮数，手动任务按当次用户要求停止。
