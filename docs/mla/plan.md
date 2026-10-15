# MLA 迭代计划

本页只记录实施顺序、产物和完成条件。功能基线为 **M10c8b BSND PA 原生自动 split 生产接入已完成**；M10 尚未阶段收尾。当前能力与最近证据见 [status](status.md)，契约与验收口径见 [design](design.md)。

## 本轮已完成 M10c9a：TND PA 显式 split 计划与安全协议

独立交付 TND Q + ND PA KV、512+64、显式 S=2..256 的 CPU 任务/持久布局及 v25 CPU/AICore 共用安全协议。复用固定源 PA 几何、显式分片与三种 mask，分离 Q allocation 前缀、Q-used 和紧凑总 token；生产入口仍拒绝 TND PA split。

- 影响文件：新增 FA3 910 `mla_split_paged_tnd_plan.{h,cpp}`、`mla_split_paged_tnd_metadata.{h,cpp}`、`mla_split_paged_tnd_consumer.h`；新增所属 `tests/mla/unit/split_paged_tnd_protocol/`、`support/mla_split_paged_tnd_*` 及工具测试，在统一入口注册。已有 PA、mixed/split 组件只读复用，未改共享生产头、AICPU、launcher 或构建清单。
- 协议/预算：v25 为 `414+7B` 字，保存 v6 TND allocation/used、mask/S 和 uint64 M 前缀。静态 T=totalQ、A=min(B,T)，持久包络 `S*(T*H+255*A)` 覆盖 M=128..256 的分段 padding，scratch 取 M=256/容量 N；实际持久槽由紧凑 mixed 计划计算。partial 归属 `task%AIC`，归约归属 `group%(2*AIC)`。Row 保留原 allocation token，不暴露虚拟 KV 起点。
- 新增覆盖：固定 CANN PA/三 mask/workspace、FA3 分片与 TND accumulator 差分；独立行/token/page 枚举；非均匀/空分配、Q-used 间隙、坏页及 Q-used=0 时页检查、宽前缀/字节溢出、陈旧/破坏记录与失败原子性；双 dtype 与自洽 C1V1 记录拒绝。工具组检查归属、源身份、失败/中断传播及原有组命令不变。
- 必跑命令：`python3 -m tests.mla --unit split_paged_tnd_protocol`，专属 `--suite split-paged-tnd-protocol-check`、`--suite split-paged-tnd-protocol-sanitize split-paged-tnd-protocol-tools`；先 dry-run/collect-only，代表/成本边界后完整普通与 ASan/UBSan 同集合。日常累计清单为 `--suite e2e --device 6`；未截断、未运行历史单元组。数学 reference 内容缓存继续复用。
- 继承与完成条件：原始 E2E、6421 个构建依赖及实际加载身份已审计；本次 runner 的保守闭包包含新增协议/测试注册，按入口重建并实跑日常 E2E，继承为 0。必要组、文档/质量及差异核对全部通过后提交一次，结果与修正记录见 status。此 CPU 协议步不宣称设备新能力，不新增未开放入口的伪 E2E。

## 唯一下一步 M10c9b：TND PA 显式 split 生产接入

在已验收 v25 上闭环 TND Q + ND PA KV 的显式 S=2..256、512+64、FP16/BF16、无 mask/右下 causal/window 公开行为。复用稳定 PA partial/merge 数值组件，按紧凑 allocation token 接入 host 预算、AICPU、launcher、生产分派、公开只读 cache/metadata 和 real/fake/compile；自动 S=0 继续拒绝。

- 实施前列明共享 launch/address/构建依赖及影响能力，按实际接口细化所属单元/公开/Trap/性能组并注册命令；规划原 Q allocation 的 O/LSE/真实 FP32 `[S,H,totalQ,512]` / `[S,H,totalQ]` accumulator 地址、空值和初始化所有权。
- 新入口必须验证非均匀/空 Q 段、Q-used、实际 KV、四路 stride、合法页重排、页内/跨页、长度/metadata 复用、生命周期、多流与 release Trap。每阶段在写入或同步前完整校验 v25 和有效页，不读回动态长度，不转调 CANN attention。
- 固定源 PA partial/merge 与双数学在双 dtype、三 mask 上实测；新增所属日常小输入 E2E，运行累计受影响 E2E 与相关普通 FA3 回归。只继承完整且依赖未变的 E2E 正确性，不继承历史单元/性能；数学 reference 内容缓存复用。
- 实测预先固定、最多 100 配置的公开性能，覆盖隐式生成/显式生成/复用、R=1/4，保留单例/分段退化并按配置内/配置间等权平均比≤1验收。源表示域与尚未开放路径按 design 单列。
- 完成条件：必要工具/代表/边界后固定实现，完整本步集合、平均公开性能、文档与质量检查通过，更新状态后提交一次并停止。TND PA 原生自动计划/协议及生产随后另立步骤。

## 后续里程碑：M10–M17

每行是能力里程碑，不等于一次提交。临近该阶段时按独立验收风险细化当前步；稳定组件的一种布局接入可整体完成 host、AICPU、Kernel、公开入口及必要验收。基础精度、普通回归与生命周期随入口交付。

| 里程碑 | 剩余产物与边界 | 完成条件 |
| --- | --- | --- |
| M10：512+64 split-KV | M10c9b 完成 TND PA 显式生产，再闭环其原生自动计划/协议与生产；已有布局见下表 | 五种布局的自动/1/显式 split、三种 mask、空分片/全空、O/LSE/真实 accumulator/metadata 及长 KV E2E；阶段约定组合与性能 |
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
| M10c9a | TND Q + ND PA KV 显式紧凑计划、v25 生成/消费及 Task/Merge/Row 安全协议；生产待 M10c9b |
| 验收工具整理 | 统一入口、构建/reference 缓存和 E2E 证据恢复；模块单元/日常 E2E/专项归属、代表组合与显式穷举、仓库外产物及原生对照构建 |

```bash
git log --oneline -- docs/mla/status.md
git log -G 'M10c8b' -- docs/mla/plan.md
git show 737d5ae:docs/mla/status.md
```

## 迭代与收尾规则

遵循 [AGENTS.md](../../AGENTS.md#一次工作只完成一个提交) 和 [design 第 7 节](design.md#7-验收口径)。实现前在当前步补齐具体影响路径、必跑组/命令及选择依据、拟继承证据与依赖、完成条件；必要验证受限或失败时保留交接，不提交为已闭环。必跑检查通过后进入文档、质量检查和一次提交收尾；仅因新改动、失败或具体未解决风险扩测，提交后停止。

用户最新约定为每成功完成五轮开发插入一次专门文档修正轮；文档轮不计开发轮数，循环随后继续开发。手动文档任务仍按当次用户要求停止，不在本轮推进功能。循环实现和接手时已有的未提交修复由 status 标明，本页不复制其历史实施日志。
