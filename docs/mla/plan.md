# MLA 迭代计划

本页只记录实施顺序、产物和完成条件。功能基线为 **M10c9b TND PA 显式 split 生产接入已完成**；M10 尚未阶段收尾。当前能力与最近证据见 [status](status.md)，契约与验收口径见 [design](design.md)。

## 本轮已完成 M10c9b：TND PA 显式 split 生产接入

在 v25 上交付 TND Q + ND PA KV、512+64、FP16/BF16、显式 S=2..256 的无 mask/右下 causal/window 公开行为，自动 0 继续拒绝。

- 影响路径：新增 v25 AICPU、TND PA init/partial/merge launcher 与双 dtype 实例；复用 PA ring 和 TND merge 数值组件，按原 allocation token 初始化、写回 O/LSE 和真实 accumulator。修改 host 紧凑预算、生产/公开 real/fake/metadata 分派及构建清单；共享路径通过累计 E2E 和普通 FA3 回归验证。
- 设备前缀适配：公开 metadata 在验证描述符后取得 totalQ，前向内部生成核对已知总量；空总量仍生成有效证书并校验有效页。改动所属 v25 CPU 单元普通/sanitizer 实跑，不继承旧单元结果。
- 新增独立验收：`unit/split_paged_tnd` 的固定源 partial/FD 与双数学；`e2e/split_paged_tnd` 的三 mask、非均匀/空 allocation、Q-used 间隙、页边界、stride/storage、metadata 复用、长度更新、fake/compile、生命周期、多流和 release Trap。在 `regression_cases.py` 注册本能力的日常小输入，累计清单增至 243 项。
- 实现前固定的命令全部用 `python3 -m tests.mla`，先 dry-run/collect-only：`--unit split_paged_tnd_protocol`、`--suite split-paged-tnd-protocol-sanitize split-paged-tnd-tools`、`--unit split_paged_tnd`、`--suite split-paged-tnd-check`，冻结后执行 `split-paged-tnd-public split-paged-tnd-trap e2e ordinary split-paged-tnd-profile`。代表组含最长 reference/最大 accumulator；性能固定 6 种形状×双 dtype，三入口、R=1/4，原生自动 FD 按既有口径单列，保留全部原始数据。
- 本步共享生产/构建改变，累计 E2E 实跑，继承为 0；数学内容缓存继续复用。必要组和平均公开性能通过，失败修正、环境身份及证据见 status；文档和质量检查后只提交本步并停止。

## 唯一下一步 M10c10a：TND PA 原生自动 split 计划与安全协议

独立闭环 TND Q + ND PA KV 的固定源原生自动任务/持久布局及 CPU/AICore 共用协议；自动生产接入随后作为 M10c10b 独立完成。本步不开放 `num_splits=0` 的公开行为。

- 复用已验收的 TND PA 几何、allocation/used、有效页校验以及 BSND PA/非 PA TND 的原生自动分区依据，保留原生 Direct/FD 端点和跨波次所有权。新增协议须与 v25 显式记录严格区分；host 静态预算覆盖动态任务与真实 TND accumulator，保持原 allocation token 地址、空行和未用槽语义。
- 实施前在本节列明具体新文件、共享头/地址/构建影响、必跑组及完成条件；新增所属 `unit/auto_split_paged_tnd_protocol/` 与工具归属。新协议先独立验收，不提前修改生产 launcher 或注册伪公开 E2E。
- 普通和 ASan/UBSan 使用同一集合：固定源三 mask/PA 几何/原生分核差分，独立 task/merge/row 枚举，非均匀/空前缀、Q-used、有效页和空成本，跨 M/片段/核边界、uint64 地址/预算溢出、陈旧或破坏记录及失败原子性；双 dtype、协议参数两两覆盖，具名高阶组合单列。
- 统一入口先 dry-run/collect-only，代表和成本边界后固定实现并执行所属完整组。已有公开路径通过累计日常 E2E；仅在依赖与原始证据完整有效时继承 E2E 正确性，不继承历史单元或性能。数学内容缓存与本步有效 native 包路径从 status 复核。
- 完成条件：独立计划/协议的源差分、边界、sanitizer、工具、适用 E2E 和文档/质量检查全部通过，状态更新后提交一次并停止。若触及共享数值或生产路径，先按影响范围补入本步验收；必要验证受限则保留交接且不提交为闭环。

## 后续里程碑：M10–M17

每行是能力里程碑，不等于一次提交。临近该阶段时按独立验收风险细化当前步；稳定组件的一种布局接入可整体完成 host、AICPU、Kernel、公开入口及必要验收。基础精度、普通回归与生命周期随入口交付。

| 里程碑 | 剩余产物与边界 | 完成条件 |
| --- | --- | --- |
| M10：512+64 split-KV | 闭环 TND PA 原生自动计划/协议与生产，再执行阶段收尾；已有布局见下表 | 五种布局的自动/1/显式 split、三种 mask、空分片/全空、O/LSE/真实 accumulator/metadata 及长 KV E2E；阶段约定组合与性能 |
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
| M10c9a–M10c9b | TND Q + ND PA KV 显式紧凑计划、v25 安全协议、三阶段生产、公开 cache/metadata 和真实 TND accumulator |
| 验收工具整理 | 统一入口、构建/reference 缓存和 E2E 证据恢复；模块单元/日常 E2E/专项归属、代表组合与显式穷举、仓库外产物及原生对照构建 |

```bash
git log --oneline -- docs/mla/status.md
git log -G 'M10c8b' -- docs/mla/plan.md
git show 737d5ae:docs/mla/status.md
```

## 迭代与收尾规则

遵循 [AGENTS.md](../../AGENTS.md#一次工作只完成一个提交) 和 [design 第 7 节](design.md#7-验收口径)。实现前在当前步补齐具体影响路径、必跑组/命令及选择依据、拟继承证据与依赖、完成条件；必要验证受限或失败时保留交接，不提交为已闭环。必跑检查通过后进入文档、质量检查和一次提交收尾；仅因新改动、失败或具体未解决风险扩测，提交后停止。

用户最新约定为每成功完成五轮开发插入一次专门文档修正轮；文档轮不计开发轮数，循环随后继续开发。手动文档任务仍按当次用户要求停止，不在本轮推进功能。循环实现和接手时已有的未提交修复由 status 标明，本页不复制其历史实施日志。
