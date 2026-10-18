# MLA 迭代计划

本页只记录实施顺序、产物和完成条件。**M15g2 的五布局调度选项完整公开接入已完成；唯一下一步是 M16：图执行。** 当前能力与最近证据见 [status](status.md)，契约与验收口径见 [design](design.md)。

## 唯一下一步 M16 图执行

为已开放的两种表示、五布局接通真实图捕获与重放，覆盖固定输入和图内长度更新/metadata 重生成。fake/compile 已通过不等于图重放完成；当前图执行仍未开放，范围见 [design](design.md#22-后端选项与返回值)。

- 首先核对本机 torch_npu 图 API、AICPU metadata 生成及多阶段发射的实际捕获行为，复用既有 stream/内存生命周期与统一驱动。实现前补齐实际改动路径、必跑命令及证据依赖；不能用主机读回长度或固定旧 metadata 掩盖图内更新。
- 按完整可调用行为接通 host/AICPU、记录持有与公开入口。固定重放和动态长度更新随同一图执行能力验收；只有新的共享协议存在独立未覆盖风险时，才按 AGENTS 写明风险、独立产物和完成条件后拆步。
- 新增所属图执行 E2E 与日常节点，覆盖两种表示、双 dtype、五布局的具名 mask/split 组合、重复重放、长度更新与重生成、跨层复用、多 stream、空值和生命周期。逐项比较 eager/图输出及独立数学；设备数值代码若变，补所属模块固定源与双数学。既有 E2E 只按驱动证据规则继承。
- 图性能按 design 单列，保留公开调用、metadata 生命周期和 R 的计时边界；同时运行受影响的 eager E2E 与普通 FA3。完成条件是图执行的完整行为、动态值和资源生命周期一致，必要正确性及适用性能通过，文档更新后一次提交。

M15g2 是最近专门文档轮后的第四个成功开发轮；第五轮成功后先执行一次专门文档修正轮，再推进后续开发。本次提交后停止，不在本轮开始 M16。

## 后续里程碑：M16–M17

每行是能力里程碑，不等于一次提交。细分须说明新增数值算法或共享协议的独立风险；其余按完整可调用行为交付，精度、安全、普通回归、生命周期和适用公开性能随入口验收。

| 里程碑 | 剩余产物与边界 | 完成条件 |
| --- | --- | --- |
| M16：图执行 | 固定输入重放与图内长度更新/metadata 重生成完整接通 | eager/图一致、跨层复用、多 stream 与 fake/真实输出一致 |
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
| 工具整理、R1、R2 | 统一验收与缓存/证据管理、模块归属与共享 helper、共用前向分派、文档轮和无头交接；原 R3 的内核可读性核对合入 R2 |

组合验收映射见 [512+64](design.md#75-51264-组合覆盖) 和 [192/128](design.md#76-192128-组合覆盖)。M14、M15a–d 五轮开发后的专门文档轮已完成，不计开发轮数；该轮只修 Markdown 与公开 docstring，核对固定来源、原始证据和质量，未推进功能。M15e、M15f、M15g1、M15g2 是该文档轮之后的第一至第四个开发轮。

```bash
git log --oneline -- docs/mla/status.md
git show feb189b:docs/mla/status.md  # M10 阶段证据
git show f7105ef:docs/mla/status.md  # M14 阶段证据
git show 26d7a70:docs/mla/status.md  # M15d 开发闭环与完整复现命令
```

## 迭代与收尾规则

遵循 [AGENTS.md](../../AGENTS.md#一次工作只完成一个提交) 和 [design 第 7 节](design.md#7-验收口径)。实现前在当前步补齐实际影响路径、必跑组/命令及选择依据、拟继承证据与依赖、完成条件；验证受限或失败时保留交接，不提交为已闭环。每成功完成五轮开发插入一次专门文档修正轮；文档轮不计开发轮数，手动任务按当次用户要求停止。
