# MLA 迭代计划

本页只记录实施顺序、产物和完成条件。**M10 的五种 512+64 布局及阶段收尾、M11 的 192/128 BSND 已完成。** 当前能力与最近证据见 [status](status.md)，契约与验收口径见 [design](design.md)。

## 唯一下一步 M12a

一次接通 192/128 的紧凑 TND Q/KV varlen 完整公开行为，复用 [M11 数值与 BSND 接入](design.md#651-192128-通用非量化数值组件) 和已有 TND allocation/used、mask/split 调度与校验。M11b 已完成，本轮提交后停止，不开始 M12a。

- 产物：同一提交完成 TND host、AICPU、Kernel、metadata、公开 real/fake/compile；支持 qv=None、独立 K/V、GQA、完整 storage/stride、分配前缀与实际 used、非均匀/空长度、O/LSE 和真实 FP32 accumulator。无 mask、causal、window 与自动/1/显式 split 随布局接入。
- 接口与影响：沿用固定 FA3 的 varlen 返回数量、max 长度和有效输出区域；复用源算法、既有 TND 寻址/调度与 M11 数值核，明确实际 TND token、KV head 和归并地址。新记录须说明 v27 无法表达的分配前缀/长度语义；不因新增布局或版本号拆组件轮。混合与 PA prefill 仍明确拒绝。
- 验收准备：实现前补齐本步具体影响文件、必跑组/命令及依据、拟继承 E2E 与依赖。新增模块所属 CPU/源差分/sanitizer 和双 dtype 设备数值，补 real/fake/compile、前缀/used/空值/stride/复用/生命周期/Trap；登记自己的日常 E2E，按共享路径选择累计 E2E 和普通 FA3 回归。
- 性能：复用 M11 的通用 template 3/4 分类与固定 native 包，在长测前补 TND 适配所需廉价夹具和实际分派检查；固定输入与原有计时协议，完成适用的公开平均性能。相同输入继续复用内容 reference 缓存，不继承历史单元或性能。
- 完成条件：完整可调用行为、必要验证及证据身份闭环，更新涉及的三份文档、通过质量检查，提交一次后停止；必要验证受限时如实交接，不提交为已闭环。

## 后续里程碑：M12–M17

每行是能力里程碑，不等于一次提交。以下布局步默认各交付一个完整行为，稳定的 mask/split 随布局接入，host/AICPU/Kernel/公开入口在同一轮完成。新数值组件可先独立闭环；进一步拆分须写明新增风险，不能以文件数、层数或开关数机械展开。精度、安全、普通回归、生命周期和适用公开性能均随入口交付。

| 里程碑 | 剩余产物与边界 | 完成条件 |
| --- | --- | --- |
| M12：prefill 非 PA TND/混合 | M12a TND varlen，M12b TND Q + BSND cache；各复用 M11 数值和已有调度，一次接通长度、metadata、mask/split 与公开行为 | allocation/used、非均匀/空长度、stride/GQA、三 mask、自动/1/显式 split、生产精度、生命周期及公开 E2E/性能 |
| M13：prefill PA | M13a BSND PA，M13b TND PA；各一次接通页访问、metadata、三 mask、自动/1/显式 split 和公开行为 | 合法 P 代表边界、独立 K/V 与内容/RoPE stride、长度/空值/复用/生命周期、数值及 PA 公开 E2E/性能 |
| M14：prefill 组合收尾 | 核对 M11–M13 的必迁组合、补明确缺口并完成阶段验收；不再按 mask × split × 布局重复开接入轮 | 前序入口已实测必要组合；新发现的算法缺口按风险单列，已有有效 E2E 按规则继承，阶段性能实际执行 |
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
| M11a | 192/128 通用数值组件，独立 K/V、GQA/stride、三 mask、direct/partial/128 维 merge |
| M11b | 完整 BSND prefill host/AICPU/Kernel/v27/real/fake/compile，三 mask、自动/1/显式 split、真实 accumulator、公开 E2E 与性能闭环 |
| 工具整理、R1、R2 | 统一验收与缓存/证据管理、模块归属与共享 helper、共用前向分派、文档轮和无头交接；原 R3 的内核可读性核对合入 R2 |

```bash
git log --oneline -- docs/mla/status.md
git show feb189b:docs/mla/status.md  # M10 阶段证据
git show fd7e133:docs/mla/status.md  # M11a 数值组件证据
```

## 迭代与收尾规则

遵循 [AGENTS.md](../../AGENTS.md#一次工作只完成一个提交) 和 [design 第 7 节](design.md#7-验收口径)。实现前在当前步补齐具体影响路径、必跑组/命令及选择依据、拟继承证据与依赖、完成条件；必要验证受限或失败时保留交接，不提交为已闭环。必跑检查通过后进入文档、质量检查和一次提交收尾；仅因新改动、失败或具体未解决风险扩测，提交后停止。

用户最新约定为每成功完成五轮开发插入一次专门文档修正轮；文档轮不计开发轮数，循环随后继续开发。手动文档任务仍按当次用户要求停止，不在本轮推进功能。循环行为由版本管理脚本和工具测试保证，本页不复制其历史实施日志。
