# MLA 迭代计划

本页记录实施顺序、每步产物与完成条件。**流程 W01–W05、功能 M00–M08（含阶段限量平均验收）、M09a/M09b1 已完成；唯一下一步是 M09b2。** 当前能力和证据见 [status](status.md)，范围及验收契约见 [design](design.md)。功能开发按当前下一步执行，不重新启动流程优化任务。

## 提交边界

一个提交完成一个独立算法组件或一种布局的完整可调用行为。稳定组件的接入应把必要的 metadata/AICPU、生产分派、公开入口及验收一起完成；只有新算法或共享协议具有独立验收条件时才先单独闭环。不要按头文件、CPU、AICPU、Kernel、公开入口机械拆成多轮，也不以减少提交数量为由将多个尚不稳定的算法和入口塞进一轮。

边界依据是现有共享 `MlaTaskRing`/Vector、布局包装、metadata 路径及测试驱动。功能轮开始前，按下列当前步补齐实际改动路径、明确测试命令、继承依据和完成条件；进入远期里程碑时再细化，不另设审批或纯计划提交。

## 已完成阶段：M08

仅推进 **512+64 右下 causal、num_splits=1**。位置中心为 `i + Sk - Sq`，Sq/Sk 取逐样本有效长度；分配前缀、真实 stride 和 PA 页寻址沿用现有契约。范围保持不变，性能采用最多 100 个配置的平均门槛；精度按 [design 第 7 节](design.md#7-验收口径) 的源一致性优先级验收，数学阈值不变。

原 M08b 同时引入新的 mask/空行/任务环语义与首次公开接入，因此分为 b1、b2；原 M08c 同时覆盖 TND varlen 和混合 cache 两种调用行为，因此分为 c1、c2。M08a、M08d 保持独立，M08e 仍承接 TND PA 与阶段收尾。下表 **七行分别是计划提交**，b/c 不再作为额外步骤；不恢复原十二个逐层子步。原始草稿仍在 `build/mla-workflow-tools-20261012/preexisting.patch`；M08a 已复核并实现 `mla_causal.h/.cpp`，原头文件备份在 `build/mla-m08a/mla_causal.h.draft`。

| 子步 | 产物与影响路径 | 本步必跑与完成条件 | 可继承的证据 |
| --- | --- | --- | --- |
| M08a（已完成） | 独立 CPU causal 块范围、全遮罩行描述与分核计划；复用长度解析及固定源 SplitCore，不接生产分派 | 固定源范围/成本/完整端点差分和独立逐行任务枚举；有效长度、头跨块、尾块、空样本、溢出及 ASan/UBSan；边界见 [design 6.2.4](design.md#624-独立-cpu-causal-范围与分核) | 未变化的设备数值、公开入口与性能；共享 CPU 调度若改动则补相关无 mask 调度回归 |
| M08b1（已完成） | 可独立调用的 BSND causal 数值组件；共享任务环、Vector mask 和全遮罩 O/LSE；本步不开放公共 causal 入口 | 先接通固定源 causal probe、独立数学输入适配及成本边界；再跑 910B 双 dtype 源/双数学精度、任务环/同步/空行和代表性组件性能；覆盖共享修改影响的无 mask 布局 | M08a CPU 算法；依赖未变的地址和 Copy。现有无 mask 精度/E2E 不能替代 causal 数值验证 |
| M08b2（已完成） | 首个 BSND 非 PA causal 完整行为：func、只读 cache、metadata 复用及内部 BSND Q-used；必要的生成器、消费者、分派与入口同次接通 | 整容量、实际 KV、内部 Q-used、stride、空值、旧记录/参数不匹配与 Trap、真实/fake/compile/生命周期；生产精度及 `ordinary`，适用组件性能和公开 R=1/4 E2E | b1 中未变的数值组件；BSND 各实际调用路径及 metadata 传参仍须当次实测 |
| M08c1（已完成） | 非 PA TND Q/KV causal varlen：完整分配段与 seqused 一起接通 | 两组分配前缀、used/非均匀长度、地址间隙/poison、尾部/空行、真实/fake/compile/Trap/生命周期；生产精度、适用组件/公开 E2E、`ordinary` 和共享影响回归 | 已闭环的 causal 算法、BSND 与未变的长度解析；新 TND 行映射及参数传递必须实测 |
| M08c2（已完成） | 非 PA TND Q + BSND KV causal cache；公开只读 cache/metadata 复用和内部混合 Q-used 同次接通 | 分配起点与有效长度分离、KV 容量/stride、内部 used、记录复用、空值/Trap/compile/生命周期；生产精度、适用组件/公开 E2E、`ordinary` 和共享影响回归 | c1 中未变的 TND/used 算法和 b2 的 cache 调用契约；混合地址和原生对照转换不能沿用 TND 结果冒充 |
| M08d（已完成） | BSND PA causal：现有页 Copy、metadata、生产、公开 cache 及内部 Q-used 的完整接入 | 全部合法 P、跨页 mask 边界、乱序/重复页、零 KV/空行、stride、记录复用/真实 Trap/compile/生命周期；双 dtype 精度、普通回归和适用组件/公开 R=1/4 E2E | 未变的页 Copy 组件与非 PA 算法；优先复用 Copy，不因接入 causal 预定重写；共享代码变化须覆盖受影响布局 |
| M08e（已完成） | TND PA causal 完整行为及 M08 阶段验收；包含内部 Q-used | TND 前缀/used、全 P、stride/空值/复用/生命周期/Trap/compile、精度和适用组件/公开 E2E；在冻结实现上运行覆盖全部已开放 causal 组合、相关回归及性能的阶段矩阵，同一当次结果只运行/计数一次 | 可复用有效构建和数学 reference；阶段矩阵必须当次实测，不能拿之前提交的结果替代 |

BSND Q-used 明确归入 b2，TND/TND used 归入 c1，混合 Q-used 归入 c2，PA used 分别归入 d/e，避免将 used 留成没有入口归属的后置任务。它们按现有公共/内部接口边界验收，不给公开 cache 或 metadata 生成器新增 Q-used 参数。表内名称表示现有记录承担的行为，causal 如何扩展记录版本与签名在对应接入步确定；未开放布局仍须在真实与 fake 入口明确拒绝。

### 已完成 M08e：TND PA causal 与阶段收尾

在已确认的 512+64、右下 causal、num_splits=1 范围内，已接通 **TND Q + ND PA KV** 的只读 cache、公开 metadata 生成/复用及内部 Q-used，并完成 M08 阶段验收。基线、结果与耗时见 [status](status.md)。

**本步实施边界与验收选择：**

- **协议与影响路径：**新增独立 `version=kind=11`，复用 `408+5B` 结构和既有 TND allocation/used 解析。CPU 显式总量与 AICPU 前缀推导分别检查；v6/v10/v11 互斥，消费时校验当次前缀/used/全部有效页。修改 causal PA 生成器、共享页消费者、TND launcher、生产提交及 C++/Python real/fake 分派；不修改页 Copy、Block/Tile、causal 任务环、CATLASS 或固定源。
- **新增专项：**注册 `causal-paged-tnd-cpu/sanitize/check/public/trap/profile`，新增 TND 非均匀前缀、空段、used 间隙/poison、全部合法 P、跨页 mask、Sq<Sk/=/>、合法 stride/offset、零 KV/全遮罩、三版本互斥、长度更新、记录复用、真实/fake/compile/生命周期与 release Trap。CPU 对照源 PA 几何和 causal 完整 M 分核；TND 原生 mask/长度及同调度固定源组件双 dtype 实测。共享原生性能驱动新增 TND 布局，补其计时转换和分派工具检查。
- **共享与阶段集合：**CPU 必跑新增组及 `causal-paged-cpu/sanitize`、`paged-shared-cpu/sanitize`；工具跑 `tools`、`causal-paged-tools`、`cpu`。设备先跑新 TND 代表，再冻结实现；正式集合取新路径与阶段验收的并集：五种布局的 `causal-*-public/trap/profile`、独立 `causal-attention/profile`、`regression-dense/nonpaged/paged`、`ordinary`，并补 `public/metadata` 的 TND 公开默认参数和 v11 生成边界。BSND 代表已包含于其 public，其他代表与 public 过滤互斥，避免重复执行。所有组默认最多 100 节点，性能双 dtype 按既有固定输入规则；不启用 full-matrix。
- **命令与继承：**统一 `python3 -m tests.mla_validation --output build/mla-m08e/<批次> --device 6 --max-cases 100 --suite <组>`，先 `--dry-run` 与 `--collect-only` 并核对清单。继承 M08d 已审计的未变数学组件及最长 reference/最大内存成本边界；阶段实测仍用本次结果。继续复用 `build/mla-validation-artifacts/golden`，相同 reference 不重复冷算；恢复只补未完成或失效组。环境、基线审计和原始结果均保存于 `build/mla-m08e/`。
- **阶段性能合并限量：**五布局原矩阵直接相加超过 100 个含 dtype 配置，因此增加仅用于性能组的 `--causal-stage-performance`。每布局固定选择至多 10 个输入、双 dtype 合计至多 100；PA 按页大小分层，其他布局按 used/stride/实际长度分层，再按既有内容哈希固定选择。普通正确性及单布局默认矩阵不变，保存全部候选/选中/未选中配置。阶段各布局保存均值与退化，不单独作为门槛；用 `python3 -m tests.mla_causal_stage_performance --evidence <本次性能 summary，可多次指定> --output <合并结果>` 按所有适用配置等权平均验收。新增全局上限、固定选择、默认矩阵不变、缺失/损坏/失败传播测试；跨批次汇总还须核对各布局共享源码身份一致，相关工具检查在性能采样前完成。
- **完成条件：**全部必要组通过，五种已开放 causal 布局的组合、普通回归和固定性能矩阵有本次证据；源精度限制及 FD 单列，适用公开 E2E 等权平均 ≤1。更新契约、阶段计划及状态，完成差异/链接/质量检查，只提交一次后停止；未完成必要验证则记录交接且不提交为闭环。

### 已完成 M09a：独立 CPU window 归一化与块范围

M09a 已完成独立 CPU 窗口归一化、逐样本范围和完整 M 分核；该步不包含设备与公开 window 前向。后续独立设备组件见下文 M09b1；继承证据见 [status](status.md)，CPU 组件契约见 [design 6.2.6](design.md#626-独立-cpu-window-归一化与范围)。

**M09a 实施边界与验收选择：**

- 新增独立 `mla_window.h/.cpp`；归一化使用调用级 Q/K 上界，逐样本几何使用已解析的有效长度。复用不变的 `BuildCausalBlock` 校验/causal 范围、`BuildCausalSchedule` 的 causal 特例及 `detail/cann_split_core.*` 的 band/无 mask 算法；window 的完整 M 端点适配单独实现。本步不改共享算法、生产构建列表、metadata、Kernel/Block/Tile、Python/C++ 公开分派。
- 固定 FA3 `mha_fwd` / metadata 的归一化片段从内容哈希校验的官方源码提取并独立编译；固定 CANN Git blob 独立编译 band 范围/成本/全部 FD 候选字段。Python 逐行 key 枚举与任务唯一归属另作独立检查。覆盖左右不同归一化阈值、单 token causal 优化、负数无限侧、有限窗口、causal 覆盖右窗、空值、Sq<Sk/=/>、上界与 used 分离、头跨块、尾块和 int64 极值；host 归一化后低于 INT32_MIN 的负值明确拒绝，避免 forward 的 int 窄化与 metadata 行为不一致，失败保持输出不变。
- 注册 `window-cpu/window-sanitize/window-tools`。必跑新窗口普通/ASan/UBSan 同集合，以及 `causal-cpu/causal-sanitize` 的复用契约回归；`tools` 与窗口工具专项检查新旧组选择、CPU 隔离、固定源身份、成功/失败/中断/恢复及损坏失效。无共享生产数值改动，因此不运行设备 `regression-*`；继承 M08e 中这些固定组、公开行为和性能的完整有效证据。allocation/used 解析器不变，本次 causal 与 window 原始输入测试覆盖五布局及 used 分离；不展开无关历史矩阵。
- 统一命令：`python3 -m tests.mla_validation --output build/mla-m09a/<批次> --suite window-cpu window-sanitize causal-cpu causal-sanitize window-tools tools`，先 dry-run/collect 核对集合，每组最多 100 节点。工具与小代表检查后固定源码，再执行正式集合；本步无设备数学计算，保留 `build/mla-validation-artifacts/golden`，不刷新 reference。证据、环境与依赖审计保存到 `build/mla-m09a/`。
- 完成条件：固定 FA3/CANN 差分、独立边界、完整 M 覆盖、失败原子性、sanitizer 和上述回归全部通过；新增文件不进入当前生产编译/调用依赖的核验完成。更新本步设计/计划/状态及唯一下一步，质量检查后只提交一次并停止。

### 最近完成 M09b1：独立 BSND window 数值组件

窗口新增非零 S2 起点及双侧 Vector mask，需要独立验收任务环与数值流水。M09b1 已完成独立连续 BSND 设备组件、固定源/双数学、同步/空值、共享回归与代表组件性能；结果及源精度限制见 [status](status.md)，契约见 [design 6.2.7](design.md#627-独立-bsnd-window-数值组件)。M09b2 再接 BSND metadata、生产分派与公开入口，不按每层另拆步骤。

- **产物与影响路径：**复用 M09a 归一化/范围及现有 causal Kernel / Block / Tile，接可独立调用的连续 BSND window 组件。先明确非零 S2 起点对发射计数、任务环槽位、正常/尾块、online softmax、全遮罩 O/LSE 与排空同步的影响；不修改固定 CANN 或 CATLASS，不提前开放公共 window。
- **应新增用例：**有限双侧、单侧无限、causal 叠加、Sq<Sk/=/>、调用级上界/逐样本有效 KV、跨 M/N 边界及非零起点、头跨块、KV 尾块、零 KV/全遮罩、scale、任务环复用。接通带真实 band mask/pre/next 参数的固定源 probe；独立数学参考按原始有效长度与归一化窗口切片，不从迁移调度反推参考。
- **必跑与选择：**注册 window 代表、双 dtype 固定源/双数学数值和代表组件性能组；先 dry-run/collect，检查代表及本步新增的成本边界后冻结实现。共享任务环/Vector 变更优先选择 `regression-dense/nonpaged/paged`，按实际影响补 causal 代表、空行/非零起点等清单缺口；每组最多 100 节点。影响 CPU 算法时补 `window-cpu/window-sanitize`，否则继承 M09a 完整证据。实现前按调用依赖在此补全实际组名/命令，不用旧无 mask 组替代窗口验收。
- **继承与完成条件：**复核不变的 M09a CPU 和既有地址/Copy/构建依赖，复用内容数学缓存；相同输入 reference 不重复计算。新组件在 910B FP16/BF16 的固定源与双数学精度、同步/空值及代表性能全部闭环，共享受影响路径通过；窗口公共行为仍明确拒绝。更新三份相关文档并提交一次后停止，必要验证未完成则如实交接，不提交为已闭环。

**M09b1 实施边界与验收选择：**

- 在 `kernel/mla_dense.hpp` 增加默认关闭的 window 特化和独立 BSND 包装；每个 M 块采用窗口 S2 起止范围，首块标志取实际起点，空块不增加发射计数。`block/mla_cube.hpp` 的共享任务描述增加左界，`block/mla_vector.hpp` / mask Tile 接双侧遮罩并复用既有 softmax/aMLA 与全遮罩 epilogue。所有既有模板保持默认行为；不改 CPU window/causal 调度、地址/Copy、metadata、公开分派、CATLASS 或 CANN。
- 增加 window 独立 schedule/probe、双 dtype 数值与 profile 驱动；固定源实际传 sparse 4、三角压缩 mask 和归一化 pre/next，另覆盖归一化后 causal/无 mask 的分派。数学侧直接按原始有效 KV 和独立 FA3 归一化参数切片，沿用 `mla_reference_pair` 内容缓存。新增有限/单侧/causal、上下界与 used、跨块非零起点、头尾块、空值、scale、连续任务/重复调用/stream 用例；代表与正式数值组过滤互斥，性能输入的 reference 可复用。
- 统一命令为 `python3 -m tests.mla_validation --output build/mla-m09b1/<批次> --device 6 --max-cases 100 --suite <组>`，各批次先 `--dry-run` / `--collect-only`。新增 `window-check/window-attention/window-profile/window-device-tools`；工具专项和 `tools/cpu` 先通过，再跑 window 小代表及新增长窗口成本边界并冻结实现，最后运行窗口正式组与性能。性能保持原组件的预热、wall/profiler 采样次数与同步边界，不将其记为公开 E2E。
- 共享头文件影响全部 MLA 实例，重建并核验实际扩展。回归采用 `regression-dense/nonpaged/paged`，覆盖双 dtype、M/N/长 KV、任务环、三种非分页 allocation/used、stride、页边界及生命周期；补 `causal-check` 和五布局 `causal-bsnd-check/causal-tnd-check/causal-mixed-check/causal-paged-check/causal-paged-tnd-check`，覆盖真实生产 masked 分派及空行；运行 `ordinary` 与 `causal-mixed-boundary` 检查普通入口及未开放窗口边界。没有新增公开能力或 host 调用开销改动，本步只要求代表组件性能。
- 继承 M09a 的 `window-cpu/window-sanitize/causal-cpu/causal-sanitize`：复核完整原始证据、CPU 源/驱动、固定 FA3/CANN 与工具链；共享设备头的变化不作为这些 CPU 组件失效的依据。既有地址/Copy 算法及最长无 mask/causal reference 成本边界按有效依赖继承；构建产物在共享头变更后必须重建，不继承旧设备结果替代上述必跑组。新增 reference 边界先实测并记录耗时/峰值，复用默认 golden 目录，不重复冷算。

### 唯一下一步 M09b2：BSND window 完整公开行为

以 M09b1 的稳定连续 BSND window 数值组件为依赖，在一个提交中接通 **非 PA BSND** 的 metadata、生产分派和公开 func/cache；本步尚未启动。

- 确定独立窗口记录的版本/字段与归一化身份，复用 M09a 长度/窗口/完整 M 分核和 M09b1 数值核。调用级上界与实际 Sq/Sk 分开；逐次核对长度、窗口和记录匹配，不能将 v7 causal 或无 mask 记录冒充 window。包含内部 Q-used 和四路真实 stride，保持公开签名及只读 cache 契约。
- 同步完成 CPU/AICPU 生成器、消费者、host/launcher、Python/C++ real/fake 入口和未开放布局拒绝。新增有限/单侧窗口、causal 叠加、归一化边界、长度更新、记录复用/互斥、空行/空容量、stride、compile、生命周期及真实 Trap 专项；固定源必须传实际 band 参数，数学从原始 allocation/used 与调用级窗口独立映射。
- 实现前补齐实际测试组/命令，按影响选择 `regression-*` 及 causal 代表；新公开行为、设备精度和适用 R=1/4 平均公开 E2E 当步闭环。复核后继承不变的 M09a CPU / M09b1 独立组件证据与数学缓存；新入口参数传递、分派和 E2E 必须当次实测。保留 M09b1 的源精度限制，不能记为数学通过。
- 完成条件是上述 BSND 行为及必要回归/平均性能通过，其他 window 布局仍明确拒绝；更新相关契约、状态和下一步，只提交一次并停止。

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
| M08c2 | 非 PA TND Q + BSND KV causal cache、v9 公开 metadata 生成/复用及内部 Q-used；混合地址/stride/空值/Trap/compile/生命周期、共享回归与平均公开 E2E |
| M08d | BSND Q + ND PA KV causal cache、v10 公开 metadata 生成/复用及内部 Q-used；全 P/跨页 mask/stride/空值/Trap/compile/生命周期、共享回归与平均公开 E2E |
| M08e | TND Q + ND PA KV causal cache、v11 前缀/used、metadata 生成/复用和全部入口；五种 causal 布局的阶段正确性、共享回归、组件及限量平均公开 E2E；M08 完成 |
| M09a | 独立 CPU window 归一化、band/全遮罩范围与完整 M 分核；固定 FA3/CANN、逐行/任务差分、五布局原始长度、溢出和 ASan/UBSan；生产窗口仍拒绝 |
| M09b1 | 独立连续 BSND window 数值组件；非零 S2 起点、双侧 mask、任务环/空值、固定源/双数学与源精度限制、共享回归及代表组件性能；公共窗口仍拒绝 |

M07 的页大小精度覆盖全部 16..1024 的 16 倍数；性能矩阵及原生非 FD/FD 的边界见 [design 7.1](design.md#71-固定性能矩阵与阶段边界)。公开接入包含 None/整数/tensor KV 长度、v6 复用及相关 MLA/普通 FA3 回归，结果只在 status 维护。

历史查询示例：

```bash
git log --oneline -- docs/mla/status.md
git log -G 'M06d3b2' -- docs/mla/plan.md
git show 29ed04f:docs/mla/status.md
```
