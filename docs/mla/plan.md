# MLA 迭代计划

本页只记录实施顺序、产物和完成条件。**M10 的五种 512+64 布局及阶段收尾已完成**；R1 入口收敛及 R2 整体整理已完成。当前能力与最近证据见 [status](status.md)，契约与验收口径见 [design](design.md)。

## 唯一下一步 M11a

实现 192/128 共享数值组件，范围按 [design 3.2](design.md#32-展开-mla-prefill192--128) 和固定 CANN 通用非量化路径。

M11a 独立组件步的依据是新数值算法风险：192/128 使用固定 CANN 的通用 `fia_kernel_nonquant.h` 路径，独立 K/V、GQA、192 行 stride 和输出 128 的矩阵乘/softmax/partial 行为未被现有 512+64 组件覆盖。该步不是按 Kernel、Block、Tile 或 mask/split 开关拆轮。

- 独立产物：910 目录中的通用非量化数值适配及可调用组件 probe，复用既有 mask/split 调度、归并和双数学 reference；保留固定源映射及版权。公开 prefill 入口在 M11b 完整接入前继续明确拒绝。
- 完成条件：FP16/BF16 在 910B 上通过固定源和双数学对照，覆盖独立 V、GQA、192 stride、尾块/空行、三 mask、直接/partial/merge 与适用组件性能；新增模块独占单元组，512+64 跨模块回归只走受影响累计 E2E。
- 实现前先确定通用源所需的最小 native 配方、组件用例与统一驱动命令，写入当前步；原生包必须能证明实际执行了通用路径。M11b 再一次交付 BSND host/AICPU/Kernel/metadata、real/fake、三 mask 和自动/单/显式 split 的公开行为及 E2E/性能。

## 后续里程碑：M11–M17

每行是能力里程碑，不等于一次提交。以下布局步默认各交付一个完整行为，稳定的 mask/split 随布局接入，host/AICPU/Kernel/公开入口在同一轮完成。新数值组件可先独立闭环；进一步拆分须写明新增风险，不能以文件数、层数或开关数机械展开。精度、安全、普通回归、生命周期和适用公开性能均随入口交付。

| 里程碑 | 剩余产物与边界 | 完成条件 |
| --- | --- | --- |
| M11：192/128 共享数值与 BSND | M11a 闭环固定源通用非量化数值组件及 mask/partial/merge 所需适配；M11b 一次接通 BSND 的 metadata/生产/公开调用 | qv=None、独立 K/V、GQA、192 stride、双 dtype、O/LSE/accumulator；BSND 三 mask 和自动/1/显式 split，新增算法单元及公开 E2E/性能 |
| M12：prefill 非 PA TND/混合 | M12a TND varlen，M12b TND Q + BSND cache；各复用 M11 数值和已有调度，一次接通长度、metadata、mask/split 与公开行为 | allocation/used、非均匀/空长度、stride/GQA、三 mask、自动/1/显式 split、生产精度、生命周期及公开 E2E/性能 |
| M13：prefill PA | M13a BSND PA，M13b TND PA；各一次接通页访问、metadata、三 mask、自动/1/显式 split 和公开行为 | 合法 P 代表边界、独立 K/V 与内容/RoPE stride、长度/空值/复用/生命周期、数值及 PA 公开 E2E/性能 |
| M14：prefill 组合收尾 | 核对 M11–M13 的必迁组合、补明确缺口并完成阶段验收；不再按 mask × split × 布局重复开接入轮 | 前序入口已实测必要组合；新发现的算法缺口按风险单列，已有有效 E2E 按规则继承，阶段性能实际执行 |
| M15：softcap/调度选项 | softcap 数学独立闭环；pack_gqa、sm_margin 按共享调度影响划分 | 有效数学/调度语义、资源校验及适用 E2E，不能忽略参数 |
| M16：图执行 | 固定输入重放与图内长度更新/metadata 重生成分别闭环 | eager/图一致、跨层复用、多 stream 与 fake/真实输出一致 |
| M17：全范围收尾 | 冻结能力，按覆盖矩阵验收并补用户示例；实现问题先修复再验收 | 必迁组合、精度/平均公开 E2E/普通 FA3 回归；910B/C 实现与资源兼容，实机仍用 910B；README 调用与边界 |

## 已完成步骤

这里只保留交付边界，当前拒绝项以 status 和 design 第 2 节为准。每步的命令、数量、耗时、失败修正及原始证据从该提交的 status 查询；不把历史阶段的拒绝条件当作当前限制。M04c 实际在 M05 后执行，编号保留。

| 步骤 | 已交付产物 |
| --- | --- |
| M00–M03 | 固定迁移交集/FA3 契约、独立双 reference 与固定源执行器、入口识别/校验/未实现拦截 |
| M04a–M04c4、M05a–M05c2 | dense 调度和 Kernel/Block/Tile、FP16/BF16 BSND 前向、v1 metadata 与公开只读 cache、S1G 多块/长 KV/空容量 |
| M06a–M06e2c2 | v2 实际 KV、v3 TND、v4 混合、v5 allocation/used；三种非 PA 布局的完整 storage/stride、生产/公开及生命周期 |
| M07a–M07e3c | ND PA 地址/Copy/调度、v6、BSND/TND PA 前向与公开 metadata/cache；M07 阶段完成 |
| M08a–M08e | 右下 causal 范围/数值、五种布局 v7–v11 及公开接入；M08 阶段完成 |
| M09a–M09e | window 归一化/数值、五种布局 v12–v16 及公开接入；M09 阶段完成 |
| M10a–M10b2 | split-KV 分区/workspace、partial attention、FD 归并与 FA3 accumulator 适配 |
| M10c1a–M10c2b2 | 非 PA BSND 显式/原生自动计划与 v17/v18、两/三阶段生产和公开调用 |
| M10c3a–M10c4b2 | 非 PA TND 显式/原生自动紧凑计划与 v19/v20、生产 varlen 和真实 accumulator |
| M10c5a–M10c6b | TND Q + BSND KV 显式/原生自动计划与 v21/v22、64 位 KV 起点、生产 cache/metadata |
| M10c7a–M10c8b | BSND Q + ND PA KV 显式/原生自动计划与 v23/v24、页校验、生产 cache/metadata 与真实 accumulator |
| M10c9a–M10c9b | TND Q + ND PA KV 显式紧凑计划、v25 安全协议、三阶段生产、公开 cache/metadata 和真实 TND accumulator |
| M10c10 | TND Q + ND PA KV 原生自动分区、v26、三阶段生产、公开 real/fake/cache/metadata、真实 26 片 accumulator；三 mask、源/双数学、协议/sanitizer、Trap、日常回归和平均公开性能闭环 |
| M10 阶段收尾 | 五布局单/显式/自动 split 与三 mask 的组合覆盖、累计 E2E/公开专项和固定 92 配置阶段公开性能；统一驱动选择、计时前适用集合检查与绑定原始证据的等权汇总 |
| 验收工具整理 | 统一入口、构建/reference 缓存和 E2E 证据恢复；模块单元/日常 E2E/专项归属、代表组合与显式穷举、仓库外产物及原生对照构建 |
| R1：前向入口收敛 | 三个公开 MLA 前向共用适配，real/fake split 共用分派；参数名错误提示、双语调用说明、日常接口回归和 100 配置公开性能验收 |
| R2：代码、测试与无头交接整理 | 共用 helper 下沉到 support、消除反向依赖，按实际 Python 依赖管理构建/证据；补齐工具归属并修正历史断言；集成五轮文档修正及干净收尾检查；原 R3 的内核可读性核对合入本步，保留源分层和数值流水 |

```bash
git log --oneline -- docs/mla/status.md
git log -G 'M10c8b' -- docs/mla/plan.md
git show 737d5ae:docs/mla/status.md
```

## 迭代与收尾规则

遵循 [AGENTS.md](../../AGENTS.md#一次工作只完成一个提交) 和 [design 第 7 节](design.md#7-验收口径)。实现前在当前步补齐具体影响路径、必跑组/命令及选择依据、拟继承证据与依赖、完成条件；必要验证受限或失败时保留交接，不提交为已闭环。必跑检查通过后进入文档、质量检查和一次提交收尾；仅因新改动、失败或具体未解决风险扩测，提交后停止。

用户最新约定为每成功完成五轮开发插入一次专门文档修正轮；文档轮不计开发轮数，循环随后继续开发。手动文档任务仍按当次用户要求停止，不在本轮推进功能。循环行为由版本管理脚本和工具测试保证，本页不复制其历史实施日志。
