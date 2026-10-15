# MLA 迭代计划

本页只记录实施顺序、产物和完成条件。功能基线为 **M10c8b BSND PA 原生自动 split 生产接入已完成**；M10 尚未阶段收尾。当前能力与最近证据见 [status](status.md)，契约与验收口径见 [design](design.md)。

## 本轮已完成：文档修正优化

仅修改 `docs/mla/design.md`、`plan.md`、`status.md`：核对当前 Python/C++ 分派、metadata/launcher、测试注册与固定 CANN/FA3 来源；复核最近闭环的原始证据，修正能力、状态、命令和引用，压缩已完成步骤的实施日志。既有未提交工作保留并排除在本轮提交之外。

必跑检查为文档与代码/固定来源的一致性、本地链接/锚点、命令注册及 dry-run、原始证据存在性/哈希和适用仓库质量检查。只读取历史运行记录，不探测设备、不重建扩展、不运行设备矩阵，也不把文档核对计为本轮 E2E 通过或继承。上述检查结果见 status；本轮以一次文档提交收尾并停止，不推进下列功能。

## 唯一下一步 M10c9a：TND PA 显式 split 计划与安全协议

独立闭环 TND Q + ND PA KV、512+64、显式 `S=2..256` 的 CPU 任务/持久布局及 CPU/AICore 共用安全协议。复用固定源 PA 几何、显式分片和三种 mask；Q allocation 前缀、used 与紧凑总 token 分开处理，不能以 `B*maxQ` 代替实际分配。

- 影响路径：在 FA3 910 的 `mla/` 新增 TND PA 显式 plan/metadata/consumer/query、所属 `tests/mla/unit/split_paged_tnd_protocol/` 和 `support/` 固定源/边界驱动。共享地址、页校验或 split 查询若需调整，实施前列明实际文件、影响路径、版本/字数、静态预算与任务/归约所有权。此步不接入生产 AICPU/launcher，不开放公开 TND PA split。
- 新增覆盖：非均匀/空 Q 分配、Q-used、实际 KV 与页容量、原 allocation 输出/accumulator 地址、页内/跨页、宽前缀/乘法溢出、失效/陈旧记录和坏页，包括 Q-used=0 时的有效页检查。协议参数两两覆盖，必要多轴组合具名，保留双 dtype。
- 必跑检查：固定 CANN PA/FA3 显式分片差分、独立行/token/page 枚举、边界/失败原子性和同集合 ASan/UBSan，以及新增工具的选择/身份/失败传播。**拟新增** `python3 -m tests.mla --unit split_paged_tnd_protocol` 及专属 sanitizer/工具组；当前尚未注册，实施时先补齐完整命令，再 dry-run/collect-only 核对集合。
- 继承依据：从 status 指向的最近功能证据复核原始 E2E、源码/构建依赖和实际加载身份；只继承完整且未失效的 E2E 正确性。共享生产依赖变化时实际运行受影响日常 E2E，不累计历史单元或继承性能。构建/reference 按内容复用，与测试通过证据分开。
- 完成条件：独立计划、预算、消费者与查询协议及上述必要检查通过，公开 TND PA split 继续明确拒绝；按影响更新三份文档、质量检查后提交一次。生产接入另立下一步，交付所属日常 E2E、实机数值/Trap/生命周期及公开性能。

## 后续里程碑：M10–M17

每行是能力里程碑，不等于一次提交。临近该阶段时按独立验收风险细化当前步；稳定组件的一种布局接入可整体完成 host、AICPU、Kernel、公开入口及必要验收。基础精度、普通回归与生命周期随入口交付。

| 里程碑 | 剩余产物与边界 | 完成条件 |
| --- | --- | --- |
| M10：512+64 split-KV | M10c9a 后完成 TND PA 显式生产，再闭环其原生自动计划/协议与生产；已有布局见下表 | 五种布局的自动/1/显式 split、三种 mask、空分片/全空、O/LSE/真实 accumulator/metadata 及长 KV E2E；阶段约定组合与性能 |
| M11：192/128 BSND prefill | 先验证独立 K/V、分组头及 192 stride 数值组件，再完成 BSND metadata/生产/公开接入 | qv=None、源通用非量化路径、GQA、真实 stride、O/LSE 和适用 E2E |
| M12：prefill TND/长度 | 复用 M11 数值，TND varlen 和混合 cache 按可调用行为接入 | allocation/used、非均匀/空长度、stride/GQA、生产精度及适用 E2E |
| M13：prefill PA | BSND PA、TND PA 各自接通 metadata/生产/入口 | 合法 P 代表边界、内容/RoPE 独立 stride、长度/复用/生命周期、精度及 PA E2E |
| M14：prefill mask/split | 分别闭环 causal、window、split 的受影响布局，再验收组合 | 右下 mask、自动/显式 split、metadata 和前序布局的必迁组合 |
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
| 验收工具整理 | 统一入口、构建/reference 缓存和 E2E 证据恢复；模块单元/日常 E2E/专项归属、代表组合与显式穷举、仓库外产物及原生对照构建 |

```bash
git log --oneline -- docs/mla/status.md
git log -G 'M10c8b' -- docs/mla/plan.md
git show 737d5ae:docs/mla/status.md
```

## 迭代与收尾规则

遵循 [AGENTS.md](../../AGENTS.md#一次工作只完成一个提交) 和 [design 第 7 节](design.md#7-验收口径)。实现前在当前步补齐具体影响路径、必跑组/命令及选择依据、拟继承证据与依赖、完成条件；必要验证受限或失败时保留交接，不提交为已闭环。必跑检查通过后进入文档、质量检查和一次提交收尾；仅因新改动、失败或具体未解决风险扩测，提交后停止。

用户最新约定为每成功完成五轮开发插入一次专门文档修正轮；文档轮不计开发轮数，循环随后继续开发。手动文档任务仍按当次用户要求停止，不在本轮推进功能。循环实现和接手时已有的未提交修复由 status 标明，本页不复制其历史实施日志。
