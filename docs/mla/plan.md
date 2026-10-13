# MLA 迭代计划

本页记录实施顺序、每步产物与完成条件。**流程 W01–W05、功能 M00–M07、M08a、M08b1、M08b2、M08c1 及限量平均验收已完成；唯一下一步是 M08c2。** 当前能力和证据见 [status](status.md)，范围及验收契约见 [design](design.md)。功能开发按当前下一步执行，不重新启动流程优化任务。

## 提交边界

一个提交完成一个独立算法组件或一种布局的完整可调用行为。稳定组件的接入应把必要的 metadata/AICPU、生产分派、公开入口及验收一起完成；只有新算法或共享协议具有独立验收条件时才先单独闭环。不要按头文件、CPU、AICPU、Kernel、公开入口机械拆成多轮，也不以减少提交数量为由将多个尚不稳定的算法和入口塞进一轮。

边界依据是现有共享 `MlaTaskRing`/Vector、布局包装、metadata 路径及测试驱动。功能轮开始前，按下列当前步补齐实际改动路径、明确测试命令、继承依据和完成条件；进入远期里程碑时再细化，不另设审批或纯计划提交。

## 后续功能：M08

仅推进 **512+64 右下 causal、num_splits=1**。位置中心为 `i + Sk - Sq`，Sq/Sk 取逐样本有效长度；分配前缀、真实 stride 和 PA 页寻址沿用现有契约。范围保持不变，性能采用最多 100 个配置的平均门槛；精度按 [design 第 7 节](design.md#7-验收口径) 的源一致性优先级验收，数学阈值不变。

原 M08b 同时引入新的 mask/空行/任务环语义与首次公开接入，因此分为 b1、b2；原 M08c 同时覆盖 TND varlen 和混合 cache 两种调用行为，因此分为 c1、c2。M08a、M08d 保持独立，M08e 仍承接 TND PA 与阶段收尾。下表 **七行分别是计划提交**，b/c 不再作为额外步骤；不恢复原十二个逐层子步。原始草稿仍在 `build/mla-workflow-tools-20261012/preexisting.patch`；M08a 已复核并实现 `mla_causal.h/.cpp`，原头文件备份在 `build/mla-m08a/mla_causal.h.draft`。

| 子步 | 产物与影响路径 | 本步必跑与完成条件 | 可继承的证据 |
| --- | --- | --- | --- |
| M08a（已完成） | 独立 CPU causal 块范围、全遮罩行描述与分核计划；复用长度解析及固定源 SplitCore，不接生产分派 | 固定源范围/成本/完整端点差分和独立逐行任务枚举；有效长度、头跨块、尾块、空样本、溢出及 ASan/UBSan；边界见 [design 6.2.4](design.md#624-独立-cpu-causal-范围与分核) | 未变化的设备数值、公开入口与性能；共享 CPU 调度若改动则补相关无 mask 调度回归 |
| M08b1（已完成） | 可独立调用的 BSND causal 数值组件；共享任务环、Vector mask 和全遮罩 O/LSE；本步不开放公共 causal 入口 | 先接通固定源 causal probe、独立数学输入适配及成本边界；再跑 910B 双 dtype 源/双数学精度、任务环/同步/空行和代表性组件性能；覆盖共享修改影响的无 mask 布局 | M08a CPU 算法；依赖未变的地址和 Copy。现有无 mask 精度/E2E 不能替代 causal 数值验证 |
| M08b2（已完成） | 首个 BSND 非 PA causal 完整行为：func、只读 cache、metadata 复用及内部 BSND Q-used；必要的生成器、消费者、分派与入口同次接通 | 整容量、实际 KV、内部 Q-used、stride、空值、旧记录/参数不匹配与 Trap、真实/fake/compile/生命周期；生产精度及 `ordinary`，适用组件性能和公开 R=1/4 E2E | b1 中未变的数值组件；BSND 各实际调用路径及 metadata 传参仍须当次实测 |
| M08c1（已完成） | 非 PA TND Q/KV causal varlen：完整分配段与 seqused 一起接通 | 两组分配前缀、used/非均匀长度、地址间隙/poison、尾部/空行、真实/fake/compile/Trap/生命周期；生产精度、适用组件/公开 E2E、`ordinary` 和共享影响回归 | 已闭环的 causal 算法、BSND 与未变的长度解析；新 TND 行映射及参数传递必须实测 |
| M08c2 | 非 PA TND Q + BSND KV causal cache；公开只读 cache/metadata 复用和内部混合 Q-used 同次接通 | 分配起点与有效长度分离、KV 容量/stride、内部 used、记录复用、空值/Trap/compile/生命周期；生产精度、适用组件/公开 E2E、`ordinary` 和共享影响回归 | c1 中未变的 TND/used 算法和 b2 的 cache 调用契约；混合地址和原生对照转换不能沿用 TND 结果冒充 |
| M08d | BSND PA causal：现有页 Copy、metadata、生产、公开 cache 及内部 Q-used 的完整接入 | 全部合法 P、跨页 mask 边界、乱序/重复页、零 KV/空行、stride、记录复用/真实 Trap/compile/生命周期；双 dtype 精度、普通回归和适用组件/公开 R=1/4 E2E | 未变的页 Copy 组件与非 PA 算法；优先复用 Copy，不因接入 causal 预定重写；共享代码变化须覆盖受影响布局 |
| M08e | TND PA causal 完整行为及 M08 阶段验收；包含内部 Q-used | TND 前缀/used、全 P、stride/空值/复用/生命周期/Trap/compile、精度和适用组件/公开 E2E；在冻结实现上运行覆盖全部已开放 causal 组合、相关回归及性能的阶段矩阵，同一当次结果只运行/计数一次 | 可复用有效构建和数学 reference；阶段矩阵必须当次实测，不能拿之前提交的结果替代 |

BSND Q-used 明确归入 b2，TND/TND used 归入 c1，混合 Q-used 归入 c2，PA used 分别归入 d/e，避免将 used 留成没有入口归属的后置任务。它们按现有公共/内部接口边界验收，不给公开 cache 或 metadata 生成器新增 Q-used 参数。表内名称表示现有记录承担的行为，causal 如何扩展记录版本与签名在对应接入步确定；未开放布局仍须在真实与 fake 入口明确拒绝。

### 最近完成 M08c1：非 PA TND causal varlen

**产物：**完整 TND Q/KV 分配段与可选 `seqused_q/k` 已接通同一 causal 生产路径，保持公开 varlen 签名。v8 为独立 version/kind，沿用 `410+8B` allocation/used 编码；复用既有长度解析、causal 分核与数值组件。共享消费者由 launcher 模板固定 BSND v7/TND v8，四路 stride、前缀及 used 生命周期由 host/设备共同校验。混合和 PA causal 仍明确拒绝，不增加 TND 显式 metadata 接口。

- **验收范围：**新增双前缀/used CPU 源差分和 sanitizer，双 dtype 生产源/双数学、完整分配/NaN 间隙/stride/空值、真实/fake/compile/Trap/生命周期，以及固定 28 配置公开 E2E 和代表组件采样。原生对照使用固定版本 TND vendor，真实 mask、实际分派和必要 gather/scatter 均按 design 验收；FD 单列。
- **共享影响：**使用固定 `regression-nonpaged`、限量 `ordinary`；v7/v5 CPU/sanitizer 和 BSND causal 代表覆盖消费者/生成器的共享修改。首版 BSND Q-used Trap 修正后补限量 `causal-bsnd-public`，并以固定 `causal-tnd-boundary` 补齐清单未覆盖的 mixed/PA 双 dtype 真实/fake 拒绝。
- **统一入口：**`python3 -m tests.mla_validation --output build/mla-m08c1/<批次> --max-cases 100 --device 6 --suite <组>`；本步组为 `causal-tnd-cpu causal-tnd-sanitize causal-tnd-check causal-tnd-public causal-tnd-trap causal-tnd-boundary causal-tnd-profile causal-bsnd-cpu causal-bsnd-sanitize causal-bsnd-check causal-bsnd-public regression-nonpaged ordinary tools cpu`。先核对 dry-run 和集合；失败只续跑失效部分，数学 reference 复用内容缓存。
- **继承与收尾：**未变的 causal 算法、Block/Tile、任务环及最长 reference/最大内存证据经源码和完整产物核对后继承。新 TND 行映射、参数传递、公开 E2E 在本步实测；结果、耗时、修复和原始失败只在 status/忽略证据目录维护。更新文档和质量检查后提交一次即停止；M08 阶段组合留在 M08e 实测。

### 唯一下一步 M08c2：非 PA 混合 causal cache

在已确认范围内接通 **TND Q + BSND KV** 的 causal 只读 cache、公开 metadata 生成/复用和内部混合 Q-used；PA causal 仍拒绝。开始实现前按实际改动细化协议版本、静态签名、影响路径和统一组，不另设审批步骤。

- **独立行为与边界：**Q 分配起点与有效长度分离，KV 按 BSND 容量和实际长度访问；支持原合法四路 stride。覆盖 metadata 跨合法视图复用、长度更新、空 Q/KV、错误记录/Trap、真实/fake/compile 和流/allocator 生命周期。公开 cache/metadata 不增加 Q-used 参数。
- **验收选择：**新增混合 causal 专项源/双数学、协议及公开行为组；使用受影响的 `regression-nonpaged` 固定清单和限量 `ordinary`，清单未覆盖的新边界明确补例。变更共享协议/调度则补相应 CPU 差分与 sanitizer。公开原生混合适配、实际分派、R=1/4 平均 E2E 和代表组件性能在同一步完成，不能沿用 TND 实测结果冒充混合路径。
- **继承依据：**复核本步冻结源码/加载扩展及证据，继承未变的 c1 TND/used 算法、b2 cache 契约和 Block/Tile/成本边界；共享生产分派与新混合地址路径须实测。数学内容缓存、限量规则和失败恢复不变。
- **完成条件：**本步新增及受影响必跑组、适用平均 E2E、文档和质量检查全部通过，只提交一次后停止；无法完成验收则如实交接，不提交为闭环。

M08b1 已交付独立连续 BSND causal 包装、有效 KV、mask/全遮罩 O/LSE、固定源 probe、双 reference、成本代表及完整共享回归和组件性能。源精度优先比较保留原阈值和全部输入/输出证据；旧 dense 空容量测试已按 M04c4 契约修正，原输入保留。无头循环实现纳入版本管理，阻塞与中断分别处理。实现契约见 [design 6.2.5](design.md#625-独立-bsnd-causal-数值组件)，本次结果与复现命令只在 [status](status.md) 维护。

M08a 的 CPU 闭环入口仍为 `python3 -m tests.mla_validation --output build/mla-m08a --suite tools causal-cpu causal-sanitize`；CPU 证据不证明设备前向或公共入口已开放。

### M08 的验证顺序与复用边界

1. **先使本步对照可执行。**双数学核心已有 causal 语义，但现有原生 BSND 基线及 PA reference/profile 适配尚未接通 causal。b1 完成组件源 probe/数学适配；b2、c1、c2、d、e 在首次使用相应原生路径前补对照的 mask、有效长度、分派证明和固定性能用例。CANN sparse mode 不能代替合法 mask 输入；PA reference 缓存必须补 causal 语义键和命中/失效/一致性检查后才能启用。驱动与能力在对应提交内一起闭环，不另起一串工具提交。
2. **代表用例通过后冻结实现。**先检查不等长、首个/末个有效块、全遮罩、最长 reference/最大内存及跨页边界，再启动本步必跑矩阵。构建失败、源对照未接通或成本失控先解决，不在长测期间边改相关代码边收集结果。组件与公开调用的转换/计时边界沿用 [design 7.1/7.2](design.md#71-固定性能矩阵与阶段边界)。
3. **新增路径实测，共享影响按依赖覆盖。**所有设备数值步覆盖 Sq<Sk、Sq=Sk、Sq>Sk、多 query、部分/全部遮罩行、FP16/BF16。共享任务环和 Vector 被多布局复用，不能以“本步只开放 BSND”为由漏掉受影响的无 mask 路径。每个接入步必须实测该入口的精度、参数、复用、生命周期及适用 E2E；跨步未变的组件证据按 [design 7.3](design.md#73-增量验收与证据继承) 复核继承。
4. **阶段收尾取必跑项并集，去掉重复执行。**M08e 冻结实现后一次定义“本步 TND PA 必跑项 ∪ 阶段矩阵”。同一最终依赖和执行条件下，本次已运行的同一用例或性能段可同时满足两个完成条件，不先把 e 跑完再原样跑第二遍；历史提交的结果不能填充阶段矩阵。组件和公开 E2E 的计时边界不同，不能相互替代。之后若改实现，只使受影响结果失效。

新增组须在对应步骤注册或明确列出完整命令。原 `public`/`bsnd`/`device` 等仍为无 mask 组，不能拿来证明 causal 已实现；`--dry-run`/`--collect-only` 只用于命令和集合核对。阶段限量集合不使用 `--resume` 将旧运行结果充当本次实测；有效构建与数学缓存照常复用。

## 后续里程碑：M09–M17

**下表每行是能力里程碑，不是一个无头轮次或一个提交。**临近该阶段时根据已闭环组件和实际影响细化当前步，同时保留下一步；不现在为所有远期任务固定文件、测试数量或提交数。细化采用与 M08 相同的规则：新算法先独立验收，稳定布局接入一次闭环。实现和必要的测试/对照驱动一起交付，基础精度、普通回归与生命周期不推迟到阶段末。

| 里程碑 | 依赖与建议提交边界 | 阶段完成条件 |
| --- | --- | --- |
| M09：512+64 window | 复用 causal；先闭环窗口归一化/范围和首个 BSND 行为，再按 TND、混合、PA 接入；是否需要独立算法步取决于与 causal 的共享影响 | 有限/单侧窗口、causal 叠加、非等长与页边界；已开放组合精度和适用 E2E |
| M10：512+64 split-KV | 独立验证分区、accumulator 与合并数值；再按 BSND、TND/混合、PA 接入显式及自动调度，不把完整 split 迁移作为一轮 | 自动/1/显式 split、空分片/全空、O/LSE/accumulator/metadata 及已支持 mask 组合；长 KV E2E |
| M11：192/128 BSND prefill | 这是新数值表示；先验证独立 K/V、分组头及 192 stride 数值组件，再完成 BSND metadata/生产/公开接入 | qv=None、源通用非量化路径、GQA、真实 stride、O/LSE 和适用 E2E |
| M12：prefill TND/长度 | 复用 M11 数值；TND varlen 和混合 cache 按可调用行为接入，各自同时覆盖 allocation/used 与 stride | 非均匀长度、seqused、空值/边界/GQA、生产精度和适用 E2E |
| M13：prefill PA | 复用已有页寻址；BSND PA、TND PA 各自接通 metadata/生产/入口，避免按 Copy/host/API 重复拆层 | 全部合法 P、内容/RoPE 的真实 stride、长度/复用/生命周期、精度及 PA E2E |
| M14：prefill mask/split | causal、window、split 是不同扩展，分别闭环其受影响布局后验收组合，不一次承诺全部组合 | 右下 causal/window、自动/显式 split、metadata 和前序布局的必迁组合 |
| M15：softcap/调度选项 | softcap 数学独立闭环；pack_gqa、sm_margin 按共享调度影响确定各自提交边界，不与 softcap 混成一轮 | softcap/has_softcap 和调度参数的有效语义、资源校验及适用 E2E；不能忽略参数 |
| M16：图执行 | 分开闭环固定输入重放与长度更新/metadata 重生成等图内行为；基础多流/复用/生命周期在各入口接入时已验收 | eager/图一致，图内长度更新、跨层复用、多 stream、fake/真实输出一致 |
| M17：全范围收尾 | 冻结已实现能力，按覆盖矩阵统一验收并补用户示例；发现实现问题先完成对应修复与受影响验证，再继续收尾 | 必迁组合、精度/平均 E2E/普通 FA3 回归；核对 910B/C 实现与资源兼容，实机仍用 910B；README 说明调用及边界 |

## 每步完成条件

遵循根目录 [AGENTS.md](../../AGENTS.md#按影响范围验收) 和 [design 第 7 节](design.md#7-验收口径)。实现前在当前步列明实际改动与影响路径、必跑测试组/命令及选择依据、拟继承证据及依赖、完成条件；结果与耗时只写 status 和忽略目录的证据，不在 plan 复制运行日志。

必跑检查通过后完成文档、质量检查和一次提交，提交后停止。除新改动、失败或具体未解决风险外，不再扩大矩阵。未完成必要验证时保留工作和交接，不提交为闭环；未开放组合继续明确拒绝。无头开发循环在 `scripts/mla_codex_loop.py` 维护并由工具组验收；仓库外旧命令只是兼容入口，不属于 MLA 算子运行时。

## 已完成步骤

这里只保留已交付边界。完整子步拆分、测试数量、性能及排障证据从对应提交的 status 查询。M04c 实际在 M05 后执行，编号保留。

| 阶段 | 已交付产物 |
| --- | --- |
| W01–W05 | 按影响范围验收；统一测试入口、构建/reference 缓存和证据恢复；产物清理与外部无头脚本接续。流程已完成，不作为后续功能轮次的待办 |
| 增量验收提速 | dense/非分页/causal 双数学内容缓存、固定 `regression-*` 和 `causal-check`、按组 Python 依赖恢复；原完整组保留，新行为与阶段验收门槛不变。执行结果见 status，不作为后续待办 |
| M00–M03 | 固定迁移交集和 FA3 契约；独立双 reference、固定源执行器/性能基线；MLA 识别、校验及未实现拦截 |
| M04a–M04b3 | dense 来源与分层；无 gert CPU 调度；FP16/BF16 Cube、Vector、非分页 Kernel 与 BSND 生产接入 |
| M05a–M05c2 | v1 生成/设备校验及 release Trap；官方 metadata 签名、普通 FA byte 兼容；公开只读 cache/metadata 复用及 dispatcher 验收 |
| M04c1–M04c4 | S1G 多块、KV 上界 16384、无 split 端点映射及全局空容量 |
| M06a–M06b4 | BSND 实际 KV 长度；v2 CPU/AICPU/消费者/attention 及公开只读 cache |
| M06c1–M06c3c | 紧凑 TND v3、TND Q + BSND KV v4；公开 varlen/混合 cache 和 v4 复用 |
| M06d1–M06d3b2 | 分配/used 分离的 v5；三布局生成/消费/attention，公开 varlen used、内部 BSND/混合 Q-used |
| M06e1a–M06e2c2 | 三种非分页布局的完整 storage/stride、生产接入、真实/fake/compile/流生命周期及适用公开 E2E |
| M07a–M07c3 | ND PA 来源/契约、地址与源 Copy trace、调度、v6 CPU/AICPU 及设备消费者 |
| M07d1–M07d3 | PA Copy Tile、独立 BSND/TND PA attention；双 dtype 全页大小精度、真实 Trap、stride/空值及固定源组件性能 |
| M07e1 | 公开 PA metadata；BSND/TND Q 的 v6 生成、fake/compile、长度更新与生命周期 |
| M07e2a–M07e2c | BSND PA 静态计划、生产前向、只读公开 kvcache 及 R=1/4 公开 E2E |
| M07e3a–M07e3c | TND PA 静态计划、生产前向、只读公开 kvcache 及 R=1/4 公开 E2E；M07 完成 |
| M08a | 独立 CPU 右下 causal 块范围、全遮罩行与完整 M 分核；固定源/逐行差分、原始长度及 ASan/UBSan；公共 causal 仍拒绝 |
| M08b1 | 独立 BSND causal 数值组件、固定源/双数学对照与组件性能；共享无 mask 回归；源精度限制记录和无头接续修复 |
| M08b2 | BSND 非 PA causal func/cache、v7 metadata 生成/复用及内部 Q-used；固定限量与平均公开 E2E 验收 |
| M08c1 | 非 PA TND Q/KV causal varlen、v8 双前缀/used 与四路 stride；真实/fake/compile/Trap/生命周期、共享回归和平均公开 E2E |

M07 的页大小精度覆盖全部 16..1024 的 16 倍数；性能矩阵及原生非 FD/FD 的边界见 [design 7.1](design.md#71-固定性能矩阵与阶段边界)。公开接入包含 None/整数/tensor KV 长度、v6 复用及相关 MLA/普通 FA3 回归，结果只在 status 维护。

历史查询示例：

```bash
git log --oneline -- docs/mla/status.md
git log -G 'M06d3b2' -- docs/mla/plan.md
git show 29ed04f:docs/mla/status.md
```
