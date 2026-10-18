# MLA 迭代计划

本页只记录实施顺序、产物和完成条件。**已完成 M15d 非 PA 混合布局 softcap 完整公开接入；唯一下一步是五轮开发后的专门文档修正轮。** 当前能力与最近证据见 [status](status.md)，契约与验收口径见 [design](design.md)。

## 唯一下一步 文档修正轮

自上次专门文档轮起，M14、M15a、M15b、M15c、M15d 已完成五轮开发。按 [AGENTS.md](../../AGENTS.md#文档只保留三个职责) 单独进行一次文档修正优化，不混入下一个能力实现。

- 产物与影响范围：核对 `design.md`、`plan.md`、`status.md` 与当前代码、接口及最近闭环证据，修正错误、过期内容和冗余。design 保留契约，plan 保留顺序与完成条件，status 只保留当前状态与最近证据。
- 必跑检查：使用统一驱动的 summary、实际用例清单、依赖/构建 manifest 和原始证据校验；核对涉及的链接/表格，运行 `git diff --check` 和仓库文档质量检查。纯文档轮不探测 NPU、不重建扩展、不重复运行数值或性能集合；缺失证据如实标明，不推断通过。
- 完成条件：三份文档职责清晰且与已开放行为及证据一致，质量检查通过后单独提交。文档轮不计开发轮数；结束时将唯一下一步切换为 **M15e：BSND Q + ND PA KV 的 512+64 / 192/128 softcap 完整公开接入**。

M15e 复用已验收的数值、PA 地址/页校验、mask 和 split 组件，三 mask 与自动/1/显式 split 随完整 host/AICPU/Kernel、metadata、公开 real/fake 入口一起交付。实现前按实际共享影响细化本模块单元、双数学/固定源、分页边界/Trap、累计 E2E、普通回归及≤100 配置的适用公开性能；不以布局、开关或版本号本身拆出组件步。

## 后续里程碑：M15–M17

每行是能力里程碑，不等于一次提交。细分须说明新增数值算法或共享协议的独立风险；其余按完整可调用行为交付，精度、安全、普通回归、生命周期和适用公开性能随入口验收。

| 里程碑 | 剩余产物与边界 | 完成条件 |
| --- | --- | --- |
| M15：softcap/调度选项 | BSND/TND Q + ND PA KV 的 softcap 完整公开接入；pack_gqa、sm_margin 按共享调度影响划分 | 有效数学/调度语义、资源校验及适用 E2E，不能忽略参数 |
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
| M15d | 非 PA TND Q + BSND KV cache 的 512+64 / 192/128 正 softcap 完整公开接入，v38–v40；三 mask、单/显式/自动 split、公开 metadata/kvcache real/fake/compile、矩形 KV 与 Q allocation/used、复用/Trap、双数学与固定源及公开性能 |
| 工具整理、R1、R2 | 统一验收与缓存/证据管理、模块归属与共享 helper、共用前向分派、文档轮和无头交接；原 R3 的内核可读性核对合入 R2 |

M11–M13 的布局步均包含 host/AICPU/Kernel、metadata、real/fake/compile、三 mask、自动/1/显式 split、真实 FP32 accumulator 及公开 E2E/性能验收。M14 的组合覆盖映射在 design 7.6，阶段证据与未开放边界在 status；不在 plan 重复用例数量、耗时或性能结果。

```bash
git log --oneline -- docs/mla/status.md
git show feb189b:docs/mla/status.md  # M10 阶段证据
git show 4033fe9:docs/mla/status.md  # M13b 开发闭环及失败修正记录
```

## 迭代与收尾规则

遵循 [AGENTS.md](../../AGENTS.md#一次工作只完成一个提交) 和 [design 第 7 节](design.md#7-验收口径)。实现前在当前步补齐实际影响路径、必跑组/命令及选择依据、拟继承证据与依赖、完成条件；验证受限或失败时保留交接，不提交为已闭环。每成功完成五轮开发插入一次专门文档修正轮；文档轮不计开发轮数，手动任务按当次用户要求停止。
