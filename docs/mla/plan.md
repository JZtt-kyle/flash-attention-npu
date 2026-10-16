# MLA 迭代计划

本页只记录实施顺序、产物和完成条件。**已完成 M14 prefill 阶段收尾；下一步是 M15a softcap 独立数值闭环。** 当前能力与最近证据见 [status](status.md)，契约与验收口径见 [design](design.md)。

## 唯一下一步 M15a

完成 softcap 独立数值组件，再接入完整公开行为。范围以 [design 4.1/4.2](design.md#4-功能边界) 的 SC01、MD01 为准，不混入 pack_gqa、sm_margin 或图执行。

- 拆分依据：softcap 在两分量总分数乘 scale 后引入新非线性，影响饱和区、在线 softmax、partial 与归并精度；既有无 softcap 数值组件未覆盖该风险，CANN 没有同义参数，不能用源逐元素一致豁免数学失败。这是独立数值算法风险，不按布局、mask 或 split 开关拆轮。
- 独立产物：复用当前 Kernel / Block / Tile、双数学 reference 和 probe 驱动，覆盖 512+64 与 192/128 的 softcap 数值路径及其真实 partial/merge；组件测试归 `tests/mla/unit/softcap/`，共享夹具按现有 `support/components/` 组织。公开入口在数值闭环和完整接入前继续明确拒绝。
- 实现前补齐实际文件、必跑命令及继承依赖。应新增双 dtype、softcap=0/15、scale/饱和区、三 mask、单/多片、空行和重复执行的独立用例；复用现有输入和数学缓存，不机械展开布局全积。softcap=0 需保留固定源与原数学行为；非零值用双数学口径，CANN 无同义参数的性能明确单列。
- 完成条件：910B 双 dtype 的数学、O/LSE、真实 partial/归并、边界和适用组件性能通过；共享数值路径受影响时实跑相关累计 E2E。保存可复现组件产物与证据后，再将 metadata、host/AICPU/Kernel 和公开 real/fake/compile/生命周期作为完整调用行为接通，稳定 mask/split 一并验收。

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
| M14 | 192/128 五布局组合收尾；修正过期拒绝测试，复用固定 60 配置并增加原样本阶段汇总，完成公开专项、累计 E2E、普通回归与性能验收 |
| 工具整理、R1、R2 | 统一验收与缓存/证据管理、模块归属与共享 helper、共用前向分派、文档轮和无头交接；原 R3 的内核可读性核对合入 R2 |

M11–M13 的布局步均包含 host/AICPU/Kernel、metadata、real/fake/compile、三 mask、自动/1/显式 split、真实 FP32 accumulator 及公开 E2E/性能验收。M14 的组合覆盖映射在 design 7.6，阶段证据与未开放边界在 status；不在 plan 重复用例数量、耗时或性能结果。

```bash
git log --oneline -- docs/mla/status.md
git show feb189b:docs/mla/status.md  # M10 阶段证据
git show 4033fe9:docs/mla/status.md  # M13b 开发闭环及失败修正记录
```

## 迭代与收尾规则

遵循 [AGENTS.md](../../AGENTS.md#一次工作只完成一个提交) 和 [design 第 7 节](design.md#7-验收口径)。实现前在当前步补齐实际影响路径、必跑组/命令及选择依据、拟继承证据与依赖、完成条件；验证受限或失败时保留交接，不提交为已闭环。每成功完成五轮开发插入一次专门文档修正轮；文档轮不计开发轮数，手动任务按当次用户要求停止。
