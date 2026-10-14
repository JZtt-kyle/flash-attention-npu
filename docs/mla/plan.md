# MLA 迭代计划

本页记录实施顺序、每步产物与完成条件。**M09e 与五布局窗口阶段验收已完成；唯一下一步是 M10a。** 当前能力和证据见 [status](status.md)，范围及验收契约见 [design](design.md)。

## 已完成：按覆盖目的精简测试

用户已确认直接实施：取消默认固定数量抽样，以合理的代表用例和开发反馈时间决定规模。

- 改动：精简 MLA 参数生成和重复入口组合，明确日常公开 E2E 清单与模块专项入口，取消默认 100 节点截断；同步统一驱动、选择/恢复证据、无头提示及验收说明。精度阈值、生产算法和原生性能计时协议不变。
- 日常 E2E 使用小输入覆盖所有已开放布局、dtype、无 mask/causal/window、长度、空值、stride 和 metadata 复用，避免每轮冷编译历史组件 probe。模块专项保留独立边界与错误类型，穷举矩阵须显式选择，不按固定数量填充。
- 验证：保存旧节点清单，核对新清单和代表参数的语义覆盖；新增生成规则、显式完整模式及恢复身份检查；运行相关工具单元测试、日常 E2E 全集和受影响代表专项。先 dry-run/collect-only，再在空闲 910B 卡实测记录耗时；本步不改变性能输入/计时逻辑，无需重跑原生性能矩阵。
- 继承：生产构建、固定源和数学 reference 缓存核对内容后复用；上一轮性能只保留历史记录，不计本次通过或继承，新增/调整的选择规则及公开回归本步实际执行。
- 完成：公布默认与专项的实际用例数及实测耗时，普通调用不收集再截断庞大参数全集；文档、质量检查通过后提交一次并停止，不开始 M09e。

## 已完成：单元测试 / E2E 整理与 build 清理

用户已确认：单元测试由开发模块拥有，E2E 随模块的可调用行为交付，后续仅累计 E2E 回归，失败再运行所属模块单元测试定位；`build/` 仅保留无头循环日志和恢复状态。

- 改动路径：将 MLA 测试按 `tests/mla/unit/<模块>`、`tests/mla/e2e/<能力>`、共享测试支持代码组织；修正导入、节点选择、源/探针构建路径、命令、无头提示和文档。保留原数学阈值与公开性能计时口径，不修改生产算子或 CATLASS。
- 清理前移除旧步骤目录执行依赖，把原生 CANN 对照构建配方纳入版本管理。临时产物、数学缓存和测试结果使用仓库外可配置目录；不搬运整个历史 build，不把已删除证据继续记为可继承。
- 验证：在 `/tmp/mla-test-reorg/` 记录重组前后的测试函数及参数清单；核对节点迁移、按模块选择、E2E 去重、单元不继承、失败/中断传播和路径失效。先执行组织/工具单元测试与代表性 CPU 编译用例；源和构建路径固定后，验证第 6 卡实际加载、双 dtype 公开代表 E2E、一个新路径原生性能组及相关 ordinary 回归。统一入口先 dry-run/collect-only，限量运行，不执行全部历史模块单元矩阵。
- 继承：仅将上一轮通过结论作为历史状态，不复用旧 build 中将删除的测试证据；生产源码、CANN 固定版本和 CATLASS 未改变，数值算法无需重新开发。重组后实际执行结果与未运行项分别记录。
- 本步设备实测固定为 `--e2e window_paged --max-cases 6`、`--suite window-paged-trap --max-cases 2`、`--suite ordinary` 和 `--suite window-paged-profile`。它们覆盖移动后 AICPU 头文件/生产加载、双 dtype 固定源/数学/公开调用、隔离错误子进程和新原生包。所有旧节点做全量收集对应核对；工具验收覆盖全模块选择及恢复策略。额外启动的全模块冷编译运行记为中断，不作为通过，也不作为本次纯测试整理的新增门槛。
- 完成：所有测试源码和必需执行器在仓库内，日常入口以 E2E 为主，单元按模块显式选择；清理后从新产物目录能构建和运行，`build/mla-codex-loop` 内容保留；同步 AGENTS/design/plan/status，质量检查通过后只提交一次并停止。

## 提交边界

一个提交完成一个独立算法组件或一种布局的完整可调用行为。稳定组件的接入应把必要的 metadata/AICPU、生产分派、公开入口及验收一起完成；只有新算法或共享协议具有独立验收条件时才先单独闭环。不要按头文件、CPU、AICPU、Kernel、公开入口机械拆成多轮，也不以减少提交数量为由将多个尚不稳定的算法和入口塞进一轮。

边界依据是现有共享 `MlaTaskRing`/Vector、布局包装、metadata 路径及测试驱动。功能轮开始前，按下列当前步补齐实际改动路径、明确测试命令、继承依据和完成条件；进入远期里程碑时再细化，不另设审批或纯计划提交。

### 已完成 M09e：TND PA window 与阶段收尾

在一个步骤内接通 TND Q + ND PA KV window 只读 cache：v16 metadata 的 CPU/AICPU 生成、共享消费者、TND launcher、生产分派和公开 real/fake/compile。复用现有 window 数值核、页 Copy 和四路 stride；公开签名不增加 Q-used，内部 used 与分配起点保持独立。CANN/CATLASS 未改。

本模块交付独占 CPU/sanitizer、固定源与双数学、公开/Trap/生命周期及成本边界测试，并在 `regression_cases.py` 新增 `window_paged_tnd` 的日常 E2E。M09 五种窗口布局的约定组合、累计日常 E2E、普通 FA3 和至多 100 配置阶段性能均已实测，阶段汇总核验原始证据、分派及身份后通过。具体命令、耗时、FD 和组件性能差距见 [status](status.md)；不继承历史单元或性能，不以无 mask 历史测试代替窗口验收。

### 唯一下一步 M10a：独立 CPU split-KV 分区与 workspace 布局

先闭环 512+64 split-KV 的独立调度/布局组件，再进入设备 partial attention 与合并数值。生产入口仍只接受 `num_splits=1`；本步不开放自动/显式 split，不返回占位 accumulator。

- 改动与影响路径：在 `csrc/ascend910/flash_attn_npu_3/mla/` 新增 CPU split-KV 适配，复用既有长度、块范围及固定源分核；逐项对齐 CANN FD 分区/workspace 与固定 FA3 的 0/1/显式 split、accumulator 外形。源代码保持只读，保留来源和版权。开始实现前在本节固定实际文件、共享头依赖和可独立验收的协议；若需变更既有共享调度，列明受影响生产路径。
- 本模块测试放在 `tests/mla/unit/split_kv/`，支持驱动放 `tests/mla/support/`；新增并注册 `python3 -m tests.mla --unit split_kv` 及对应 sanitizer 组，先 dry-run/collect-only。覆盖固定源分区/记账差分、独立覆盖与不重叠检查、空分片/全空、显式 1/2/8/256 及非法值、mask 范围、对齐/地址/尺寸溢出，保留 ASan/UBSan。CANN 自动策略与 FA3 请求语义分别验收，不能伪称源提供相同公开参数。
- 回归与继承：只复核本轮有效 E2E 的依赖和原始证据；独立组件未改变生产时可以继承其正确性，若修改共享生产依赖则通过统一入口实际运行受影响的累计小输入 E2E。单元测试和性能不继承；不把其他模块历史单元组加入门槛，不重复计算相同数学 reference。
- 完成条件：独立分区、accumulator/workspace 布局及失败边界有固定源和独立检查证据，sanitizer 与受影响回归通过；design 按实际协议更新，status 记录结果及未开放边界，质量检查后提交一次并停止。后续设备数值/合并及各布局生产接入另行闭环。

## 后续里程碑：M10–M17

**下表每行是能力里程碑，不是一个无头轮次或一个提交。**临近该阶段时根据已闭环组件和实际影响细化当前步，同时保留下一步；不现在为所有远期任务固定文件、测试数量或提交数。细化采用与 M08 相同的规则：新算法先独立验收，稳定布局接入一次闭环。实现和必要的测试/对照驱动一起交付，基础精度、普通回归与生命周期不推迟到阶段末。

| 里程碑 | 依赖与建议提交边界 | 阶段完成条件 |
| --- | --- | --- |
| M10：512+64 split-KV | 独立验证分区、accumulator 与合并数值；再按 BSND、TND/混合、PA 接入显式及自动调度，不把完整 split 迁移作为一轮 | 自动/1/显式 split、空分片/全空、O/LSE/accumulator/metadata 及已支持 mask 组合；长 KV E2E |
| M11：192/128 BSND prefill | 这是新数值表示；先验证独立 K/V、分组头及 192 stride 数值组件，再完成 BSND metadata/生产/公开接入 | qv=None、源通用非量化路径、GQA、真实 stride、O/LSE 和适用 E2E |
| M12：prefill TND/长度 | 复用 M11 数值；TND varlen 和混合 cache 按可调用行为接入，各自同时覆盖 allocation/used 与 stride | 非均匀长度、seqused、空值/边界/GQA、生产精度和适用 E2E |
| M13：prefill PA | 复用已有页寻址；BSND PA、TND PA 各自接通 metadata/生产/入口，避免按 Copy/host/API 重复拆层 | 合法 P 的代表边界、内容/RoPE 的真实 stride、长度/复用/生命周期、精度及 PA E2E |
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
| 增量验收提速 | dense/非分页/causal 双数学内容缓存、原固定组件诊断组与按组 Python 依赖恢复（组件组不再默认继承）；原完整组保留，新行为与阶段验收门槛不变。执行结果见 status，不作为后续待办 |
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
| M08c2 | 非 PA TND Q + BSND KV causal cache、v9 公开 metadata 生成/复用及内部 Q-used；混合地址/stride/空值/Trap/compile/生命周期、共享回归与平均公开 E2E |
| M08d | BSND Q + ND PA KV causal cache、v10 公开 metadata 生成/复用及内部 Q-used；全 P/跨页 mask/stride/空值/Trap/compile/生命周期、共享回归与平均公开 E2E |
| M08e | TND Q + ND PA KV causal cache、v11 前缀/used、metadata 生成/复用和全部入口；五种 causal 布局的阶段正确性、共享回归、组件及限量平均公开 E2E；M08 完成 |
| M09a | 独立 CPU window 归一化、band/全遮罩范围与完整 M 分核；固定 FA3/CANN、逐行/任务差分、五布局原始长度、溢出和 ASan/UBSan；生产窗口仍拒绝 |
| M09b1 | 独立连续 BSND window 数值组件；非零 S2 起点、双侧 mask、任务环/空值、固定源/双数学与源精度限制、共享回归及代表组件性能；公共窗口仍拒绝 |
| M09b2 | 非 PA BSND window func/只读 cache、v12 metadata 生成/复用及内部 Q-used；四路 stride、真实/fake/compile/Trap/生命周期、固定源/双数学、共享回归和平均公开 E2E |
| M09c1 | 非 PA TND Q/KV window varlen、v13 双前缀/used 与调用级窗口身份；四路 stride、真实/fake/compile/Trap/生命周期、固定源/双数学、共享回归及平均公开 E2E |
| M09c2 | 非 PA TND Q + BSND KV window cache、v14 公开 metadata 生成/复用及内部 Q-used；混合地址/四路 stride、真实/fake/compile/Trap/生命周期、固定源/双数学、共享回归及平均公开 E2E |
| M09d | BSND Q + ND PA KV window cache、v15 metadata 生成/复用与生产/公开接入；全 P、跨页窗口、stride/空值/Trap/compile/生命周期、固定源/双数学及平均公开 E2E |
| M09e | TND Q + ND PA KV window cache、v16 前缀/used、metadata 生成/复用与全部入口；五种窗口布局的阶段组合、累计 E2E、普通 FA3、源组件及平均公开 E2E；M09 完成 |
| 测试整理与精简 | 模块所属单元测试、明确的小输入日常 E2E 与共享支持目录；代表组合取代默认数量截断，穷举显式开启、仅继承 E2E；外部产物和可重建原生对照，build 仅保留无头日志与状态 |

M07 的页大小精度覆盖全部 16..1024 的 16 倍数；性能矩阵及原生非 FD/FD 的边界见 [design 7.1](design.md#71-固定性能矩阵与阶段边界)。公开接入包含 None/整数/tensor KV 长度、v6 复用及相关 MLA/普通 FA3 回归，结果只在 status 维护。

历史查询示例：

```bash
git log --oneline -- docs/mla/status.md
git log -G 'M06d3b2' -- docs/mla/plan.md
git show 29ed04f:docs/mla/status.md
```
