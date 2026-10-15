# MLA 迭代计划

本页记录实施顺序、每步产物与完成条件。**M10c8b BSND PA 原生自动生产接入已完成；唯一功能下一步为 M10c9a TND PA 显式 split 计划与安全协议。** 当前能力和证据见 [status](status.md)，范围及验收契约见 [design](design.md)。

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

### 已完成 M10a：独立 CPU split-KV 分区与 workspace 布局

M10a 已交付独立调度/布局组件，设备 partial attention 与合并数值由后续独立组件负责。生产入口仍只接受 `num_splits=1`，没有自动/显式 split 或占位 accumulator。

- 改动与影响路径：新增 `mla/mla_split_kv.h/.cpp`，只由独立 CPU 驱动编译；只读复用 `mla_window`、`mla_causal`、`mla_schedule`、`mla_tnd_schedule` 和 `detail/cann_split_core`，不修改共享实现、setup、metadata、生产分派或公开签名。输入为既有长度解析后的有效长度和逻辑 M/S2 基块，不重复实现五布局解析。
- 独立协议：固定 CANN 自动分区保留全部端点、FD 合并记账及成本；FA3 请求解析为 `<=0` 自动、1 单分片、2..256 显式；显式 S2 片限定于 mask 范围，空片为规范空区间。CANN 内部 workspace 与 FA3 FP32 accumulator shape/stride/地址分开描述。自动分区不能冒充均匀显式分片，源 sum/max 不能冒充 FA3 LSE；设备空片初始化留给下一步。
- 模块所有权：新增 `tests/mla/unit/split_kv/`、`support/mla_split_kv_build.py` 和两个 C++ 驱动；从固定 CANN Git blob 及哈希核验的 FA3 `flash_api.cpp`/`block.h` 提取 oracle。独立检查覆盖、无重叠、FD 记账、mask/空值与地址；保留 1/2/8/256、负值自动、非法宽值、C1V1/C1V2、尾块/全空及溢出边界。工具测试覆盖组选择、依赖身份、失败传播和单元不继承。
- 验收入口：`python3 -m tests.mla --unit split_kv`、`--suite split-kv-sanitize split-kv-tools`、`--suite e2e --device 6`，均先 dry-run/collect-only。本模块单元及 sanitizer、累计小输入 E2E 实跑，数学缓存复用，未继承历史单元/性能。没有公开 split 行为，本步不注册伪 split 数值 E2E，接通时再新增。
- 完成条件：上述固定源/独立边界、sanitizer、工具与累计 E2E 通过；actual load、design 协议、status 证据、链接和仓库质量核对后提交一次。具体结果、环境差异与产物路径只记 [status](status.md)。

### 已完成 M10b1：独立 BSND split-KV partial attention

在 M10a 协议上完成设备分片计算与独立验收，范围为共享 512+64、连续 BSND、FP16/BF16；公开 `num_splits` 和 metadata 仍不开放。产物为源 FD 的真实 FP32 部分 O 与 sum/max，归并和公开 accumulator 适配留到后续步骤。

- 改动与影响路径：在 FA3 的 910 MLA Kernel/Block/Tile 层复用现有矩阵乘、softmax、mask 和任务环，接入 M10a 的分片起止及 workspace。保留固定 CANN FD 的部分 O、sum/max、dtype、舍入和事件顺序；FA3 FP32 accumulator 需要的 LSE、空行与布局适配单列，不能将源 sum/max 直接当 LSE。实际文件及受影响共享依赖见下。
- 本模块测试归 `tests/mla/unit/split_kv_attention/`，probe/reference 放 `support/` 并注册统一入口。先用双 dtype 代表与成本边界固定实现，再跑约定集合；对照固定源的实际部分缓冲与独立双数学，覆盖无 mask/causal/window、非零 S2 起点、尾块、空分片/全遮罩/零 KV、多 M/核间边界和 1/2/8/256 分片的适用路径。做适用组件性能；不能用完整无 split 输出代替分片结果验收。
- 共享设备依赖变化时实际运行相关累计小输入 E2E，失败后再单点定位所属模块；不累计历史单元。数学 reference 内容缓存同一步复用，待测设备仍实际执行。未接通公开路径前不注册伪 split 公开 E2E，公开接入时再新增自己的日常清单。
- 完成条件：独立 partial attention 产生真实可验的部分结果，固定源/双数学、边界、组件性能和受影响 E2E 通过，文档及质量检查后只提交一次。FD 归并与各布局生产接入之后分别闭环，不在本步扩大到整个 M10。

本步实施与验收边界如下（结果和耗时见 status）：

- 新增 `mla/mla_split_kv_params.h` 和 `kernel/mla_split_kv.hpp`，通过默认关闭的模板特化复用 `kernel/mla_dense.hpp` 的三槽任务环及 `block/mla_vector.hpp`；Cube/Tile 算法、生产 launcher、metadata、setup 和公开签名不变。独立发射一核一个 M/S2 片，超过物理核数的 1/2/8/256 显式片由 probe 分波验收；每波使用 M10a 的真实 workspace 偏移。CPU 分片复用 `BuildSplitKvBlock`，不新增生产调度。
- 本步产物是源 FD 的 FP32 部分 O 与分开的八路广播 sum/max。源 `DealBmm2ResBaseBlock` 在 FD 写回前已执行 aMLA `RowDivs`，因此该 O 已归一化；已按源码和实测纠正 M10a 的“未归一化”描述。FA3 accumulator 的重排、空行适配及 LSE 转换在后续归并/接入中完成，不把原始 sum/max 宣称为 LSE。本步数学检查从原始输入独立定义每片的可见键，空片设备初始化，非空片直接比较源实际 FD 缓冲。
- 新增 `support/mla_split_kv_attention_{probe.cpp,schedule.cpp,cases.py,reference.py}` 和模块单元，复用固定 Git blob 构建驱动并注册 `split-kv-attention-check`、`split-kv-attention`、`split-kv-attention-profile`、`split-kv-attention-tools`。源 probe 调用未修改 CANN Kernel 的 partial 阶段，跳过 FD 合并；保存原始 workspace、实际加载与依赖哈希。测试涵盖双 dtype、三种 mask、非零起点、M/N 尾块、空片/全遮罩/零 KV、多 M/核边界、重复调用、guard 和 aMLA 跨块最大值变化。
- 必跑命令均为 `python3 -m tests.mla`，先 `--dry-run`、再 `--collect-only`：工具组；`--suite split-kv-attention-check --device 6` 的代表和最大成本边界；实现固定后运行 `--suite split-kv-attention split-kv-attention-profile e2e --device 6`。性能保留预热 5、五轮各 20 wall/10 profiler 样本，只比较同分片源与迁移的组件，不称公开 E2E。完整模块入口为 `--unit split_kv_attention`。
- 证据：M10a 累计 E2E 原始文件与源码已核对；本步共享 Kernel/Vector 和构建驱动改变，累计 119 项小输入 E2E 实跑，继承为 0。不重跑或继承历史单元/性能。构建和数学 reference 使用仓库外既有内容缓存；本模块相同原始输入的双数学结果在代表/验收之间复用；性能检查双方实际部分缓冲，设备每次执行。证据放 `/tmp/mla-m10b1/`，`build/` 只保留无头状态。

### 已完成 M10b2：独立 FD 归并与 FA3 accumulator 适配

在已验收的 BSND partial 数值之上，实现独立 Vector FD 归并；生产 `num_splits`、自动调度和 metadata 仍不开放。按源 FP32 部分 O 与 sum/max 定义输入，不能再次归一化已经执行过 RowDivs 的 O。

- 复用 CATLASS Block/Tile 与既有向量操作，迁移固定 `fia_block_vec_flashdecode_arch22.h` 的 max/sum 权重、合并顺序和最终 dtype 舍入；单列 FA3 FP32 O/LSE accumulator 的布局转换、`log(sum)+max` 及空行语义，使用 M10a 的 shape/stride/地址协议。
- 本模块单元放 `tests/mla/unit/split_kv_merge/`，probe/reference/性能与统一入口一起交付。开始前在本步固定实际文件、必跑命令、影响和证据依赖；以本步新生成的真实 partial 以及独立构造的极端统计量，对照固定源实际归并和双数学，不继承 M10b1 单元通过记录代替执行。
- 覆盖双 dtype、1/2/8/256 的适用组合、不同 M/head/尾行、分片内无效行、无 tile 空片、全空、不同片最大值/权重跨度、重复执行和 workspace 边界。特别区分源有限负值 max 哨兵与规范空片 `max=-inf`，确保最终 O/LSE 和 accumulator 无 NaN；不可忽略不支持的组合。
- 做适用组件性能并运行受影响累计小输入 E2E；数学 reference 仍按内容复用。未接通公开 split 前不注册伪公开 E2E，不把此独立归并验收称作公开性能或完整 M10 阶段收尾。
- 完成条件：独立归并与 accumulator 适配的源/双数学、空值/边界、组件性能及相关 E2E 通过，更新文档和质量检查后提交一次并停止。后续再按布局闭环生产显式及自动 split。

本步实施前固定如下边界：

- 新增 `mla/mla_split_kv_merge_params.h`、`tile/mla_fd_ops.hpp`、`block/mla_split_kv_merge.hpp` 与 `kernel/mla_split_kv_merge.hpp`。独立 BSND M 块归并消费源布局的 FP32 O/八路 sum/max；按原始片端点与逐行窗口判断空行，在设备完成 FP32 accumulator 重排及 LSE 转换，最终 O 使用源 FP16/BF16 舍入。1 片无 accumulator；2..256 片按资源容量缩小归并行块。生产 launcher、metadata、公开选项、CATLASS 和既有 partial 算法不改。
- 新增 `support/mla_split_kv_merge_{probe.cpp,build.py,cases.py,reference.py,profile.py}`、`unit/split_kv_merge/` 与模块工具测试，统一驱动注册 `split-kv-merge-check`、`split-kv-merge`、`split-kv-merge-profile`、`split-kv-merge-tools`。复用本次新编译并执行的 partial probe 产生输入，不继承其单元结果。源归并直接执行固定 FD Block；源 6 KiB 统计缓冲和 uint8 repeat 无法表示 256 片，oracle 明确拒绝；256 的扩展分段运算使用独立双数学验收，源逐元素差分覆盖其可表示域。源可见行与空行适配的对照边界分别记录。
- 新增用例覆盖双 dtype、1/2/8/256、M/head/尾行、causal/window 的片内无效行、无 tile 空片/全空、不同最大值及权重跨度、零/负 scale、重复调用、输入/输出 guard、accumulator shape/stride/每行地址和失败原子性。用原始 Q/K/V 双数学检查完整 O/LSE；用独立 FP64/FP32 合并方程检查合成统计量。代表与成本检查包括最大 split 缓冲和最长 KV；同输入 reference 内容缓存复用。
- 必跑命令：均先 `python3 -m tests.mla ... --dry-run`、再 `--collect-only`；工具 `--suite split-kv-merge-tools`，代表与成本 `--suite split-kv-merge-check --device 6`，固定实现后其余边界、组件性能及累计小输入回归 `--suite split-kv-merge split-kv-merge-profile e2e --device 6`；开发复核补齐源 O 双缓冲、按 query 分段写回及 probe 的 M10a 行宽校验后，最终集合加跑 `split-kv-merge-check`，覆盖修改后的完整发射路径且复用数学缓存。完整模块 `--unit split_kv_merge`。组件性能保持预热 5、五轮 20 wall/10 profiler 样本，明确归并与 accumulator 适配的计时边界，不称公开 E2E。
- 接手基线为 M10b1 提交 `d14eb49`，先核对其累计 E2E 原始证据、源码与实际加载。新增原生头文件和统一驱动会改变保守依赖身份，因此本步累计 E2E 实跑，继承 0；不累计历史单元/性能。构建和数学缓存继续使用仓库外既有 `MLA_WORK_DIR`，本步结果放 `/tmp/mla-m10b2/`；`build/` 只保留无头日志与状态。

### 已完成 M10c1：BSND 显式 split 生产接入

将已闭环的 M10a 分区/地址、M10b1 partial 和 M10b2 归并接入连续与既有合法 stride 的非 PA BSND 512+64 前向。在一个可调用行为内完成显式 `num_splits=2..256` 所需的 metadata、CPU/AICPU 生成/消费、workspace、生产分派和公开 func/只读 cache 返回；保留既有 `num_splits=1` 行为。自动 split、TND/混合和 PA split 继续明确拒绝，按后续布局步骤接入。

- 先复核本步实际调用路径与影响，在此补齐具体文件、必跑命令、继承依赖及完成条件，再实现。共享协议若引入尚未验证的新算法，按独立验收条件拆分，不机械按 host/Kernel/入口拆层。
- 新行为覆盖双 dtype、无 mask/causal/window、实际 KV 长度、原始 Q 几何、四路 stride、空片/全空/全局空容量、O/LSE/FP32 accumulator、重复调用和 metadata 长度更新。固定源可表示域与 193..256 扩展分开验收，不能将有限负 max 哨兵作为有效性标志，不能重新归一化 partial O。
- 模块单元与新的 `e2e/split_kv_bsnd/` 同步交付，在 `regression_cases.py` 注册日常小输入清单。新入口实际验证真实/fake/compile、拒绝路径、Trap、复用及生命周期，运行累计 E2E、相关普通 FA3 与适用的平均公开 E2E。仅按依赖与原始证据继承既有 E2E；本次独立组件性能不能替代公开验收。
- 完成条件：公开显式 split 真实执行迁移 partial/归并，完整返回 accumulator；本步所需源/双数学、接口/metadata/生命周期、相关回归和公开性能通过。更新三份职责文档及质量检查后只提交一次并停止，不在同一轮继续自动 split 或其他布局。

#### 已完成 M10c1a：显式 split 任务归属与持久 partial 布局

接手复核发现 M10b1 仅由测试驱动逐波提供每核一个片，写入按物理核复用的临时 FD 槽；M10b2 仅归并单个 M 块。直接接入生产还缺少跨波持久地址、partial/merge 两阶段所有权及 allocation/used 映射，属于新调度算法，不能用既有单块数值证据代替。本步先独立验收此协议；M10c1b 随后一次完成 metadata、CPU/AICPU、设备调度/stride、生产与公开入口，不再按技术层机械拆分。

- 改动：新增 `mla/mla_split_kv_plan.h/.cpp`，复用 M10a mask 分片、workspace 和 accumulator 布局。以 BSND 分配容量建立静态内存上界，以逐样本实际 Q/KV 长度建立紧凑 M 前缀；显式片按 `(有效 M 块, split)` 编号，partial/merge 分别按物理 AIC/AIV 循环归属。持久 O/sum/max 按 `(M 块, split, M 行)` 排列，独立于复用的每核双槽 scratch；单独提供原分配 O/LSE/accumulator 行地址。空片和全遮罩仍有任务，零 Q 样本不占任务；全局零 KV 容量和零任务分开表达。仅独立 CPU 组件，不开放公开 split、不改 CATLASS/既有 Kernel 或生产发射。
- 新增模块 `tests/mla/unit/split_kv_plan/`、`support/mla_split_kv_plan_{build.py,driver.cpp}` 及工具测试。固定源执行 CANN 块范围/workspace 与 FA3 显式片段/accumulator；独立枚举任务唯一归属、跨波保存、merge 覆盖、Q-used 原始几何、mask 空片、头/尾行地址，检查长度/参数/资源/整型溢出和失败原子性。覆盖 2/8/192/193/255/256、C1V1/C1V2、1/26 核及超核波数，不机械展开全部组合。新组件未形成公开可调用行为，不注册伪 E2E。
- 必跑命令：均先 `python3 -m tests.mla ... --dry-run`、再 `--collect-only`。工具 `--suite split-kv-plan-tools`；模块 `--unit split_kv_plan`；同一集合 ASan/UBSan `--suite split-kv-plan-sanitize`；累计日常小输入回归 `--suite e2e --device 6`。CPU 最大容量检查只构造描述符，不分配巨量 tensor；双 dtype 数值由累计 E2E 检查。本步无数值或公开调用变化，不重跑历史单元/性能。
- 证据：起点 `ff0a64d`，核对 M10b2 累计 E2E 原始文件、源码及实际加载扩展。新增源码/驱动改变保守依赖身份，本步累计 E2E 实跑、继承 0；保持既有数学 reference 内容缓存，相同输入不重复计算。产物放 `/tmp/mla-m10c1a/`，构建/reference 使用既有仓库外 `MLA_WORK_DIR`；`build/` 仅保留无头日志和状态。
- 完成条件：本模块源差分、独立任务/地址/边界、ASan/UBSan、工具和累计 E2E 全部通过，文档明确该协议尚未接入生产，质量检查后提交一次并停止。设备数值、metadata Trap/生命周期和适用平均公开性能仍是 M10c1b 的提交条件，不能以此 CPU 步骤宣称已开放 split。

#### 已完成 M10c1b：BSND 显式 split 生产接入

按上文 M10c1 的完整范围接通非 PA BSND。复用 M10c1a 的静态预算、64 位任务/紧凑 M 前缀和持久地址协议，一次完成 metadata 序列化/CPU/AICPU 生成与设备独立校验、partial 多任务与四路 stride、归并阶段同步、workspace/accumulator 分配和公开 func/只读 cache。不得直接将 CPU 计划对象当设备记录，也不能只接通连续整容量输入。

本步同步交付 `e2e/split_kv_bsnd/` 及日常小输入清单，实测双 dtype、mask/长度/空值/stride、真实/fake/compile、Trap/复用/生命周期、固定源可表示域与扩展双数学、累计 E2E、相关普通 FA3 和适用平均公开性能。CPU 所有权和旧单块数值证据不能替代本步设备分派、同步和公开验收。本步所需验收与正式原始证据已完成，见 status；只提交一次并停止，自动 split 和其他布局后续独立闭环。

本步实施与验收约定：

- 新增 `mla_split_bsnd_metadata.*`、独立 CPU/设备消费者和 FP16/BF16 launcher；v17 显式编码 allocation/used、归一化窗口、split 数及 64 位紧凑 M 前缀，复用 M10c1a 计划生成并独立校验。调整 `kernel/mla_split_kv.hpp`、共享任务环和归并参数以接入持久 partial、四路 stride 和有效 Q 长度；partial 与 merge 在同一 stream 分两次发射。共享数值顺序不变，CPU 对象不直接作为设备 ABI。
- `mla_forward.*`、`flash_api.cpp`、Python real/fake 与 `setup.py` 整体接通非 PA BSND func/cache、metadata、FP32 accumulator、静态预算和生命周期；显式 2..256，原 1 保留。其他布局与自动 split 仍拒绝。全局空容量先完成设备记录/长度校验，再按 FA3 初始化。
- 新增所属模块 `unit/split_kv_bsnd/`（CPU 序列化/独立消费者/溢出/ASan/UBSan及设备源差分、双数学），`e2e/split_kv_bsnd/`（接口、fake/compile、长度更新、四路 stride、空值、复用、Trap、流生命周期及公开性能）。日常清单在 `regression_cases.py` 注册；覆盖目的决定集合，不截断。固定源差分覆盖可表示域；193..256 单列扩展双数学，空片逐行验证，不把旧无 mask 结果算新能力。
- 必跑统一入口均先 `--dry-run` 再 `--collect-only`：`--suite split-bsnd-tools`、`--suite split-bsnd-cpu split-bsnd-sanitize`；双 dtype 代表/最大内存与最长 KV `--suite split-bsnd-check --device 6`；实现冻结后 `--suite split-bsnd split-bsnd-trap e2e ordinary split-bsnd-profile --device 6`。代表和完整约定集合复用内容数学缓存；性能使用既有预热、R=1/4、平均比值及原生 FD 单列口径。
- 接手 `84f36a1` 的 E2E 13 个原始证据、857 个源码及生产 6,333 个依赖/12 个产物核对完整，见 `/tmp/mla-m10c1b/takeover.json`。本步共享 Kernel、metadata、生产构建及公开入口改变，累计小输入 E2E 实跑，继承 0；不累计历史单元/性能。产物在 `/tmp/mla-m10c1b/`，构建/reference 沿用仓库外 `MLA_WORK_DIR`，`build/` 仅保留无头状态。
- 以上正确性、设备 Trap/生命周期、普通回归和适用平均公开性能全部通过，实际加载身份与原始证据核验、三份文档及质量检查完成后提交一次；必要验收失败或受限则保留工作和交接，不提交为闭环。

代表检查后补充的具体风险：性能解析须将 partial/merge 配对并保留两段原始时间，输入身份包含 S；工具测试拒绝缺段、乱序、错 dtype/设备和重叠。新协议另查宽整数/行数溢出，公开真实/fake 用小存储广播视图验证分配前预算拒绝；补窗口归一化及跨内容/scale 的 metadata 复用。代表证据文件名加入 shape，避免不同输入覆盖，完整 `split-bsnd` 因此保留代表节点并复用已计算 reference。普通契约的拒绝项改为尚未开放的自动 split，不能继续断言显式 256 被拒绝。上述为发现具体证据/覆盖风险后的补充，输入性能清单及数值、计时门槛不变。

### 已完成 M10c2a：BSND 自动 split 任务与归约计划

接手复核确认：固定 CANN 自动分区按成本切开 M/S2，只有跨核 M 块进入 FD，每块片数不同；Vector 归约按 8 行分块并使用独立端点。显式 `g*S+s` 和整 M 循环归并不能表达这一协议，直接用自动选出的最大 S 做均匀分片会改变源调度。本步独立闭环原生分区到任务/持久地址/归约所有权的新算法，M10c2b 再整体接通生产自动路径。

- 改动与影响：新增 `mla/mla_auto_split_kv_plan.h/.cpp`，复用 M10a 固定源 native 分区、window 几何及 workspace，保留原 AIC 端点和 FD Vector 端点。任务区分完整 M 直接输出、FD 部分输出和零成本初始化；只为 FD 片保留有界的紧凑持久槽，源 scratch 仍按物理核复用。独立计算分配容量下的 O/LSE/FP32 accumulator 地址，实际长度用于 mask；不改共享数值核、setup、metadata、生产发射、公开接口、CANN 或 CATLASS。
- 新增独占 `unit/auto_split_kv_plan/`、`support/mla_auto_split_kv_plan_{build.py,driver.cpp}` 与工具用例；注册 `--unit auto_split_kv_plan`、`auto-split-kv-plan-sanitize`、`auto-split-kv-plan-tools`。固定源原始 SplitCore/workspace/FA3 shape 对照；独立枚举可见 tile、AIC 唯一归属、跨多个 M/核的变长片、8 行归约、持久槽不覆盖、原分配行地址、空/全遮罩及未使用 accumulator 槽的初始化责任。覆盖 C1V1/C1V2、单核/最大核、非 FD/FD、边界/宽整数/溢出、失败原子性与输入生命周期。未接通公开能力，不注册伪自动 E2E。
- 必跑命令均先 `python3 -m tests.mla ... --dry-run` 再 `--collect-only`：`--suite auto-split-kv-plan-tools`；`--unit auto_split_kv_plan`；同一集合 `--suite auto-split-kv-plan-sanitize`；累计小输入 `--suite e2e --device 6`。代表源差分和最大逻辑容量用 `--suite auto-split-kv-plan-check`，固定实现后运行完整约定集合；CPU 不分配巨量 tensor，不计算数值 reference。既有 E2E 使用内容缓存，设备仍实际执行。
- 接手基线 `905d343`：上步 E2E 13 个原始文件、904 个源码及构建 6,349 个依赖/12 个产物逐项完整，见 `/tmp/mla-m10c2a/takeover.json`。本步新增源码/统一入口改变保守依赖身份，累计 E2E 实跑、继承 0；不累计历史单元/性能。结果在 `/tmp/mla-m10c2a/`，构建/reference 沿用仓库外 `MLA_WORK_DIR`，`build/` 仅保留无头日志与状态。
- 完成条件：新算法的源差分、独立边界/所有权/地址、ASan/UBSan、工具及累计 E2E 全部通过，实际加载身份和原始证据复核，更新本步相关文档、质量检查后提交一次并停止；必要验收受限或失败则更新交接，不提交为闭环。公开自动、设备数值/Trap/生命周期与原生 FD 平均公开性能属于 M10c2b，不能计为本步通过。

### 已完成 M10c2b1：自动 split metadata 安全协议

接手 `fb2086c` 发现新的共享协议风险：v17 的消费者要求无 FD 端点，不能表达原生每 M 不同片数、紧凑 FD 槽和独立 Vector 端点。M10c2a 的拥有存储 CPU 对象也不能直接下发。先独立闭环序列化及不信任记录的安全校验，再整体接通生产调用；此边界针对新协议的验收，不按 CPU/AICPU/Kernel 层逐层拆分公开接入。

- 改动与影响：新增 910 MLA 目录的 v18 BSND 自动记录构造器、无动态分配的 CPU/AICore 共用消费者和查询接口。复用 v5 allocation/used、源基本块及平台校验，另编码原生 AIC/FD 端点；独立验证可见 tile 覆盖、FD 片数/首片/持久槽、8 行归约所有权、64 位 M 前缀和静态预算。记录仅描述容量，不开放自动返回 shape；生产 setup、发射、数值核和公开接口不变。
- 新增独占 `tests/mla/unit/auto_split_bsnd_metadata/` 与支持驱动、固定源构建/工具用例。源差分覆盖三种 mask、FD/非 FD、不同基本块、变长/空值和 C1V2；沿用 v5 的设备平台契约，C1V1 明确拒绝。独立枚举检查任务/归约/地址。拒绝坏头、长度/窗口/平台不匹配、分区缺口、FD 数量/索引/归约损坏、预算不足、独立 accumulator 容量/stride 溢出及输入输出重叠，失败保持消费者输出不变、生成失败清空 payload。普通与 ASan/UBSan/LeakSanitizer 使用同一完整集合。未接通新公开行为，不注册伪自动 E2E。
- 必跑统一入口先 `--dry-run`、再 `--collect-only`：`--suite auto-split-metadata-tools`、`--suite auto-split-metadata-check` 代表、`--unit auto_split_bsnd_metadata`、`--suite auto-split-metadata-sanitize`、累计 `--suite e2e --device 6`。代表覆盖三种 mask、宽整数及独立伪造记录的 accumulator 溢出成本边界；完整集合不按数量截断。CPU 不计算 attention reference，累计 E2E 复用内容缓存，设备实现实际执行。
- 继承与依赖：先复核 M10c2a 的原始 E2E、源码、构建依赖/产物及实际加载库；本步新增共享头和验收驱动改变保守构建/证据身份，累计小输入 E2E 实跑、继承 0，不累计历史单元/性能。证据根 `/tmp/mla-m10c2b/`，构建/reference 沿用仓库外 `MLA_WORK_DIR`，`build/` 仅有无头循环状态。
- 完成条件：新协议源差分/独立消费者与所有边界、sanitizer、工具、累计 E2E 全部通过；核验实际加载和完整原始证据，更新相关文档并完成质量检查后只提交一次。必要验证失败或受限则更新交接，不提交为闭环。AICPU/设备记录消费、数值/Trap/生命周期、自动公开 shape 与平均公开性能均不计本步通过，留给下步整体接入。

### 已完成 M10c2b2：BSND 自动 split 生产接入

复用 M10c2a 的原生变长任务与归约协议及 M10c2b1 的记录安全协议，在同一个可调用行为中完成 metadata AICPU 生成/设备消费、partial/merge 设备循环、持久槽、自动 FA3 accumulator 与真实/fake/compile。保留显式 1/2..256，其他布局 split 继续拒绝。接通时交付自动能力的源/双数学、mask/长度/stride/空值、metadata/生命周期/Trap、日常 E2E 及适用公开性能；原生 FD 路径纳入对应性能验收。先复核本步完整证据和实际依赖，按实际影响补齐文件与必跑集合；明确静态 accumulator 上界与自动请求返回形状的适配，不按技术层机械拆分。

- 本步改动与影响：新增 v18 AICPU 生成和双 dtype 自动发射，在 `mla_forward.cpp`、`flash_api.cpp`、Python real/fake 及 setup 中接入 BSND `num_splits=0`。设备保留原生变长 AIC 分区和 8 行 Vector 归约；FD 使用紧凑持久槽，Direct 使用独立物理核临时槽并立即写回。复用 partial 与 merge 数值组件，merge 增加单片 FP32 accumulator 写回特化；显式路径与共享 Block 的累计 E2E 必须实跑。
- 自动返回采用静态后端容量 26（已验证最大 AIC 数），返回 FP32 `[26,B,H,Sq,512]` / `[26,B,H,Sq]`；实际分片数仍由设备长度与原生端点决定。Direct 写第 0 片，其余片及空行初始化 0/-inf，不读回设备决定 shape。此为固定 FA3 动态 split 的静态上界分配语义在本后端的资源适配，real/fake/compile 保持一致；不能把容量解释为 26 等分。另验分配/stride 溢出和全局空容量的 FA3 LSE 符号。
- 新增独占 `unit/auto_split_bsnd/` 固定源数值对照，`e2e/auto_split_bsnd/` 覆盖三种 mask、FD/Direct 混合与跨 M、长度更新/空值、独立 stride、metadata 跨层复用、真实/fake/compile、无 host readback、流生命周期和损坏记录 Trap。日常行为在 `regression_cases.py` 注册；新数值以原生分区为准，不用旧无 mask 或均匀显式分片替代。reference 以原始输入及固定源端点为身份内容缓存。
- 必跑命令均用 `python3 -m tests.mla`，先 `--dry-run`、再 `--collect-only`：`--suite auto-split-bsnd-tools`；双 dtype 代表/最长 KV 与内存边界 `--suite auto-split-bsnd-check --device 6`；固定实现后 `--suite auto-split-bsnd auto-split-bsnd-trap e2e ordinary auto-split-bsnd-profile --device 6`。源对照属于本模块单元，已有模块不累计重跑；公开性能预先固定覆盖输入，双 dtype、R=1/4、预热/重复/平均比值门槛保持不变，原生 FD 纳入本步自动路径平均并保留分类结果。
- 证据与继承：接手 `720fe2d`，上步 E2E 的 13 个原始文件、915 个源码及生产 6,349 个依赖/12 个产物、230 个实际加载库逐项核对完整，见 `/tmp/mla-m10c2b2/takeover.json`。本步生产构建/发射/公共入口和共享 Block 改变，累计小输入 E2E 实跑、继承 0；数学缓存沿用仓库外 `MLA_WORK_DIR`，结果放 `/tmp/mla-m10c2b2/`，build 仅保留无头状态。
- 完成条件：新自动路径固定源/双数学、公开接口/metadata/Trap/生命周期、累计 E2E、普通 FA3 和适用平均公开性能全部通过；核验实际加载与完整原始证据，更新相关契约与状态、质量检查后只提交一次并停止。任何必要验收失败或受限则保留工作及交接，不提交为闭环。

本步另交付三项验收保障：partial 的绝对偏移特化消除紧凑 FD 槽号小于物理核号时的中间减法下溢；片段 reference 只计算该片 key 区间并由独立标量对照验证；Trap 原始日志和加载身份进入正式证据清单。普通契约同步验证默认自动 metadata 和负 split 拒绝。源/双数学、公开行为、Trap、累计 E2E、普通 FA3 与含原生 FD 的平均公开性能均已闭环；具体失败处理、依赖失效后的重跑及最终证据见 status。

### 已完成 M10c3a：TND 显式 split 任务与持久布局

接手代码核对发现 `MlaSplitKvPlan` 的静态持久上界、最终 O/LSE 和 accumulator 地址都绑定 BSND 的 B×Sq；不能以 TND 的 maxQ 冒充分配步长。按原 M10c3 的新地址算法拆分规则，先独立闭环紧凑总量上界、双 allocation 前缀/used、任务与持久地址，后续在 M10c3b 整体接入设备和公开入口。

- 改动与影响路径：在 FA3 910 的 `mla_split_tnd_plan.h/.cpp` 新增独立 CPU 计划，复用 `ParseUsedLengths`、mask 显式分片、源 scratch 与 FA3 TND accumulator 组件。静态预算只依赖 totalQ/totalKV、maxQ/maxKV、B/H/M/N/S/资源；实际任务使用有效 Q 的紧凑 M 前缀，输入和最终输出使用原 allocation token。检查总量/段界、64 位任务、持久槽、字节溢出及失败原子性。不修改共享数值核、生产 setup/发射/公共入口、CANN 源仓或 CATLASS。
- 新增独占 `unit/split_tnd_plan/`、CPU driver/build 和统一驱动分组；源差分实际编译固定 CANN 块范围/workspace 与 FA3 split/accumulator 片段。独立枚举三种 mask、非均匀分配与 used 间隙、零段/全空、跨波/尾片/C1V1/C1V2、最大合法总量、上界反例及溢出；检查调用方输入销毁后的计划所有权、查询失败不修改输出。无新公开行为，本步不新增伪 TND split E2E。
- 必跑入口均为 `python3 -m tests.mla`，先 `--dry-run` 再 `--collect-only`：`--suite split-tnd-plan-tools`；代表与最大逻辑容量 `--suite split-tnd-plan-check`；固定实现后 `--unit split_tnd_plan`、`--suite split-tnd-plan-sanitize` 和累计 `--suite e2e --device 6`。普通与 ASan/UBSan 使用相同模块集合；CPU 边界不分配大 tensor、无需 attention reference。累计 E2E 继续用内容缓存，设备实现实际执行。
- 证据与继承：基线 `cba3b77`；前步 E2E 的 13 个原始文件、926 个源码、生产 6,362 个依赖/12 个产物和 230 个加载库共 7,225 个唯一文件已核对无差异，见 `/tmp/mla-m10c3a/takeover.json`。新 CPU 源及驱动改变保守源码/构建身份，本步累计 E2E 实跑、继承 0，不重跑历史单元或性能。构建/reference 继续使用仓库外默认 MLA_WORK_DIR，证据 `/tmp/mla-m10c3a/`，build 仅留无头状态。
- 完成条件：本模块固定源/独立边界、sanitizer、工具和累计 E2E 全通过；核对实际加载与完整证据，更新本节、相关算法契约和 status，质量检查后一次提交并立即停止。必要验收失败或受限则更新交接，不提交为闭环。

本步固定源差分、独立分配/used 与持久地址检查、普通及 ASan/UBSan 同集合、工具和累计小输入 E2E 均已闭环；具体结果、耗时、缓存命中、环境失败处理及完整证据见 status。CPU 计划不构成已开放的 TND split 行为，本轮提交后立即停止。

### 已完成 M10c3b：TND Q/KV 显式 split 生产接入

将已闭环的 partial/FD 数值和 M10c3a 任务地址接入非 PA TND Q/KV 的显式 `num_splits=2..256` varlen 前向，整体完成两组分配前缀、Q/KV-used、四路 stride、metadata 和真实 FP32 accumulator。保留 BSND 自动/显式与各布局单 split；TND 自动、混合和 PA split 在对应步骤前继续拒绝。

- 改动与影响路径：在 FA3 910 `mla/` 新增 v19 `mla_split_tnd_metadata` CPU/AICPU 生成器、独立消费者和双 dtype launcher，复用 M10c3a 的紧凑容量预算。`kernel/mla_split_kv.hpp` 接入 TND allocation/used 和四路 stride；共享 task ring 只放开相应模板组合，FD Block 增加 TND 原分配 token 的输出/accumulator 地址特化。`mla_forward.*`、`flash_api.cpp`、Python interface 和 `setup.py` 一起接通 varlen/内部 raw 入口、metadata 与构建/发射。BSND、单 split、混合/PA 未开放边界纳入回归。
- 新增 `unit/split_tnd/` 拥有协议/源分片、独立边界/损坏记录、ASan/UBSan 和设备源/双数学测试；新增 `e2e/split_tnd/` 覆盖非均匀双前缀、Q/KV-used 间隙、跨波/尾 M、S=2/8/193/256、三种 mask、四路独立 stride、空分配/零 used、真实/fake/compile、metadata 复用、无 host readback、流生命周期和 Trap。在 `regression_cases.py` 注册小输入 TND split 日常集合，不将旧无 mask 或 BSND 用例计为新增能力。
- 必跑入口均为 `python3 -m tests.mla`，各组先 `--dry-run`、再 `--collect-only`：工具 `--suite split-tnd-tools`；协议 `--suite split-tnd-cpu split-tnd-sanitize`；双 dtype 代表与最大内存/最长 reference 成本检查 `--suite split-tnd-check --device 6`。固定实现后运行 `--suite split-tnd split-tnd-trap e2e ordinary split-tnd-profile --device 6`。新设备组实际执行固定 CANN partial/可表示域 FD，193..256 按既定双数学扩展口径；公开性能输入在采样前固定，双 dtype、R=1/4、预热/重复、计时边界和平均门槛不变，保留原生 FD 分类。
- 证据与继承：接手基线 `e7f9f7b`，上步 E2E 的 13 个原始文件、931 个源码、生产 6,362 个依赖/12 个产物和 230 个加载库共 7,230 个唯一文件已逐项核对无差异，见 `/tmp/mla-m10c3b/takeover.json`。本步修改共享数值模板、生产构建/发射及公共入口，累计 E2E 实跑、继承 0，不累计历史单元组。构建/reference 沿用仓库外默认 MLA_WORK_DIR；本步证据 `/tmp/mla-m10c3b/`，相同输入 reference 内容缓存复用，设备实现实际执行。
- 完成条件：本模块协议/源/双数学/边界与 sanitizer、公开行为/Trap/生命周期、累计小输入 E2E、普通 FA3 和适用平均公开性能全部通过；核对实际加载、完整原始证据及本步相关契约，更新 status，质量检查后只提交一次并立即停止。必要验收失败或受限则保留工作和交接，不提交为闭环。CPU 地址证据和 BSND 数值证据不替代本步设备接入验收。

性能预检发现全部 KV-used 为零经原生 TND packing 后走固定源空张量模板，但既有对照包未编入该 key。补齐 `native_package.py` 双 dtype 的源空 KV key，保留原有 key 和全部固定性能输入；重建独立目录的固定源包并验证包身份/缺失/改变拒绝。保守源码身份因此变化，最终工具、CPU/ASan/UBSan、设备源/公开/Trap、累计 E2E、普通 FA3 和公开性能重新执行；reference 继续按内容命中，不重复计算。

本步协议/源/双数学、公开行为/真实 accumulator/生命周期、Trap、累计 E2E、普通 FA3 和适用平均公开性能均已闭环。性能工具同时验证实际 C++ 修饰符号中的 stage/dtype 与三阶段顺序；两次工具失败及修复后最终完整证据见 status。当前提交后立即停止，TND 自动路径仍拒绝。

### 已完成 M10c4a：TND 原生自动 split 任务与持久布局

现有 `MlaAutoSplitKvPlan` 的最终地址和静态布局仍绑定 BSND B×Sq。先独立闭环 TND 双 allocation/used 与原生自动 AIC/FD 任务的映射，再整体接入生产；不能用显式均匀分片或 B×maxQ 代替原生自动端点与紧凑总量。

- 改动与影响路径：在 FA3 910 `mla/mla_auto_split_tnd_plan.h/.cpp` 新增独立 TND 自动 CPU 计划，复用 `ParseUsedLengths`、固定 CANN 原生分区和已闭环 FD 槽/8 行 Vector 归约组件。沿用自动任务/归约/布局结构，保持共享 CPU 文件不变；独立适配原 allocation token 的 O/LSE/accumulator 地址、紧凑总量上界、Direct/Partial/Empty 任务、持久槽及所有权。此步不修改数值 Kernel、AICPU、生产发射/公共入口或 CATLASS。
- 新增独占 `unit/auto_split_tnd_plan/`、对应 CPU driver/build 与统一验收分组。固定源差分实际执行原生 AIC/FD 分区和 FA3 TND accumulator 片段；独立覆盖三种 mask、非均匀双前缀/used 间隙、零段/全空、Direct/FD 混合、跨 M/归约边界、64 位与溢出、失败原子性及计划所有权。普通和 ASan/UBSan 使用相同集合。
- 必跑入口均用 `python3 -m tests.mla`，先 dry-run/collect-only：`--suite auto-split-tnd-plan-tools`、代表/最大逻辑容量 `--suite auto-split-tnd-plan-check`，固定实现后 `--unit auto_split_tnd_plan`、`--suite auto-split-tnd-plan-sanitize` 和累计 `--suite e2e --device 6`。新旧集合与普通/sanitizer 节点须一致；本模块只查询 CPU 逻辑容量，无需 attention reference，不分配大 tensor。不累计历史单元，本步无新公开能力，不注册伪自动 TND E2E。
- 证据与完成条件：基线 `e62e7c1`；M10c3b 最终 E2E 的 13 个原始文件、952 个源码、生产 6,372 个依赖/12 个产物和 230 个加载库共 7,251 个唯一文件已核对无差异，见 `/tmp/mla-m10c4a/takeover.json`。新增 CPU 文件/驱动改变保守源码与构建身份，本步累计小输入 E2E 实跑、继承 0，不累计历史单元/性能。构建/reference 沿用仓库外默认 MLA_WORK_DIR，证据根 `/tmp/mla-m10c4a/`，build 仅留无头状态。固定源/独立边界、sanitizer、工具与适用累计 E2E 全通过，更新相关文档与质量检查后一次提交并停止；公开 TND 自动、混合和 PA split 在各自接入前仍明确拒绝。

本步固定源任务/FD 差分、独立 allocation 地址及容量边界、普通与 ASan/UBSan 同集合、工具和累计小输入 E2E 均已闭环；实际耗时、证据和环境失败处理见 status。CPU 计划不构成自动 TND 生产能力，当前提交后立即停止。

### 已完成 M10c4b1：TND 自动 split 的 v20 安全协议

接手复核发现 v18 消费者的空容量、accumulator 上界和 Row 地址绑定 BSND；TND 必须有独立记录身份并调整 CPU/AICore 共用校验。按原 M10c4b 允许的共享协议风险边界，先独立闭环 v20，再将设备发射和公开能力整体接通；不机械拆分 host/AICPU/Kernel。

- 改动与影响路径：新增 `mla_auto_split_tnd_metadata.h/.cpp`、`mla_auto_split_tnd_consumer.h`，复用 M10c4a 计划及 v5 双 allocation/used 检查。将 v18 原生 AIC/FD/Vector 校验提取为带明确布局/版本参数的共用实现，复用其布局无关的任务查询，保留 v18 wrapper 和原 BSND 地址行为；v20 独立验证紧凑总量、空容量、FP32 accumulator 上界及原 allocation token 地址。共享消费者头影响现有 BSND 自动设备编译，须重新构建并运行累计 E2E。此步不接通 AICPU、init/partial/merge 或公共 TND 自动入口，不修改 CATLASS 和 CANN 源仓。
- 新增独占 `unit/auto_split_tnd_metadata/` 及 CPU driver/build、工具组。固定源实际执行 AIC/FD 分区、workspace 与 FA3 TND accumulator 片段；独立覆盖三种 mask、非均匀双前缀/used 间隙、Direct/Partial/Empty、8 行 Vector/尾块、全空与零段、隐式 used、记录身份/损坏/描述符/预算、64 位前缀和失败原子性。普通与 ASan/UBSan 使用相同集合；增加同输入 v18/v20 共用逻辑与拒绝交叉记录检查，属于本步共享协议验收，不累计历史单元。
- 必跑命令均为 `python3 -m tests.mla`，先 dry-run/collect-only：`--suite auto-split-tnd-metadata-tools`、代表和最大逻辑容量 `--suite auto-split-tnd-metadata-check`；实现固定后 `--unit auto_split_tnd_metadata`、`--suite auto-split-tnd-metadata-sanitize` 和累计 `--suite e2e --device 6`。协议不执行 attention，无需新增数学 reference；E2E 沿用内容缓存，实际执行生产算子。无新公开行为，不注册伪 TND 自动 E2E，设备数值/Trap/性能留在 M10c4b2 并仍为该步必要门槛。
- 基线与完成条件：起点 `14ad721`；前步 E2E 的 13 个原始文件、1,104 个源码、6,372 个生产依赖/12 个产物及 230 个加载库共 7,400 个唯一文件已核验无差异，见 `/tmp/mla-m10c4b1/takeover.json`。共享头及保守构建身份改变，本轮 E2E 实跑、继承 0；不累计历史单元和性能。证据根 `/tmp/mla-m10c4b1/`，构建/reference 继续使用仓库外默认 MLA_WORK_DIR。新协议源/独立边界/sanitizer、工具和累计 E2E 全通过，相关文档/质量检查后提交一次并停止；必要验证不能完成则更新交接而不提交闭环。

本步 v20 固定源/独立边界、普通与 ASan/UBSan、共享 v18 兼容、工具和新构建上的累计 E2E 均已闭环。测试对照修正、准备阶段身份失效及最终证据保留在该步 status 的 Git 历史中；当时未开放 TND 自动请求，生产接入由 M10c4b2 完成。

### 已完成 M10c4b2：TND 原生自动 split 生产接入

复用 M10c4a 的原生任务/持久布局、M10c4b1 的 v20 协议、现有自动 init/partial/merge 和 TND 数值地址特化，在同一可调用行为中接通非 PA TND Q/KV 的 `num_splits=0`。整体完成双 allocation/used、三种 mask、四路 stride、设备 metadata 生成/独立消费及静态真实 FP32 accumulator；不能用显式等分替代原生每 M 的变长分区。接入时独立验收 Direct 临时槽和全局空容量行为。

- 改动与影响路径：新增 `mla_auto_split_tnd_metadata.aicpu`、TND 自动双 dtype launcher；将 BSND 自动 launcher 的三阶段实现抽为带布局特化的共享实现，复用 TND partial/merge 地址。修改 `mla_forward.cpp`、`flash_api.cpp`、Python real/fake 分派与 `setup.py`，使用现有 TND 静态计划计算紧凑预算，公开返回固定 26 片的真实 FP32 accumulator。共享 launcher 影响 BSND 自动路径，必须重建并实际回归；现有 CPU v20 协议/计划不改算法。混合/PA split 在各自步骤前继续拒绝。
- 新增独占 `unit/auto_split_tnd/`：以固定 CANN AIC/FD 端点驱动未修改的 partial/FD 源执行器，检查原 allocation 地址、部分结果和双数学，包含三种 mask、Direct/FD/Empty 混合、跨 M/8 行尾块和长 KV。新增 `e2e/auto_split_tnd/` 的 real/fake/compile、双 allocation/used、四路 stride、显式/隐式长度、全局空容量（含 max 非零）、静态真实 accumulator、复用/生命周期和 release Trap，并在 `regression_cases.py` 注册日常小输入。工具检查新组选择、失败传播、性能三阶段识别及 FD 纳入平均；调整 TND 显式测试原有自动拒绝边界。
- 必跑命令均为 `python3 -m tests.mla`，先 dry-run/collect-only：开发工具 `--suite auto-split-tnd-tools`，代表及成本边界 `--suite auto-split-tnd-check --device 6`；固定实现后 `--unit auto_split_tnd`、`--suite auto-split-tnd-public auto-split-tnd-trap e2e ordinary --device 6`，隔离执行 `--suite auto-split-tnd-profile --device 6`。本步复用已验收 CPU 计划/协议，不累计其历史 sanitizer；若修改 CPU 算法则增加该模块固定源/边界及 sanitizer。新源/双数学、公开入口与设备 Trap 本轮实际执行。
- 基线与完成条件：起点 `90d7dac`；前步 E2E 的 13 个原始文件、1,112 个源码、6,372 个生产依赖/12 个产物及 230 个加载库，共 7,408 个唯一文件逐项核验无差异，见 `/tmp/mla-m10c4b2/takeover.json`。本步生产与源码身份改变，累计 E2E 实跑、继承 0；不累计历史单元。证据根 `/tmp/mla-m10c4b2/`，构建/reference 沿用仓库外默认 MLA_WORK_DIR，原生包 `/tmp/mla-m10c3b/native`；同输入数学内容缓存复用，待测实现实际执行。该自动路径的原生 FD 纳入性能平均，固定输入及 R/阈值/计时边界不变。全部必要验证通过，更新相关文档与质量检查后一次提交并停止；无法验收则保留交接而不提交闭环。

本步 v20 AICPU/独立消费与共享三阶段 TND 特化、真实 FP32 accumulator、公开 real/fake/compile 均已接通。新模块固定源/双数学、公开/Trap/生命周期、累计 E2E、普通 FA3 与含原生 FD 的平均公开性能已完成；实际结果和原始证据只记 status。未修改 CPU 计划/协议算法，未累计历史单元或继承旧 E2E。

### 已完成 M10c5a：混合显式 split 计划与 v21 安全协议

接手 `d62d44f` 后确认，v19 的双 TND 静态界限要求 `totalKV<=INT32_MAX`，而混合 BSND KV 的原物理起点为 64 位 `b*Sk`；直接复用 v19 会拒绝合法大 cache，或混淆 allocation/used 与空容量。新记录还需要独立版本身份和设备侧重新校验。因此先闭环该共享协议，随后将稳定协议连同生产与公开入口整体接入；不是按实现层机械拆分。

- 改动与影响：新增 `mla_split_mixed_plan.h/.cpp`，以紧凑 Q 和矩形 KV 建立显式任务、持久预算与原 allocation 地址；新增 `mla_split_mixed_metadata.h/.cpp` 和 `mla_split_mixed_consumer.h`，提供 v21 构造和 CPU/AICore 共用的独立校验。复用既有 used 解析、mask 分片、workspace 和 accumulator 组件，不改变 v17/v19/v20、构建/发射参数、生产 dispatch 或公共边界。本步不编入 AICPU，不把 CPU 协议视为设备能力。
- 新增本模块 `unit/split_mixed_protocol/`：固定 CANN 范围/workspace 与 FA3 分片/accumulator 片段差分，独立三 mask 覆盖、Q allocation/used 间隙、KV 原 batch 地址、空容量/全零 used、跨波/尾行、64 位 KV/任务/前缀、精确溢出界和失败原子性/生命周期；v21 对几何/资源/长度/窗口/片数/前缀损坏和交叉版本明确拒绝。普通与 ASan/UBSan 使用同一集合。工具组核对选择、身份、失败传播和 E2E 清单未变。
- 必跑命令均为 `python3 -m tests.mla`，先 `--dry-run`、再 `--collect-only`：`--suite split-mixed-protocol-tools`，代表及成本边界 `--suite split-mixed-protocol-check`；固定实现后 `--unit split_mixed_protocol`、`--suite split-mixed-protocol-sanitize` 和累计 `--suite e2e --device 6`。CPU 新协议没有浮点执行或公开调用，不适用新设备数值/性能；后续 M10c5b 必须新增本模块公开 E2E、固定源/双数学、Trap 与性能，不能继承旧无 mask 验收替代。
- 证据与完成条件：先核验前步 `/tmp/mla-m10c4b2/` 的 E2E 原始文件、源码/生产依赖、产物和加载库；新增原生文件及驱动改变统一身份，本步累计 E2E 实跑、继承 0，不累计历史单元/性能。构建和数学 reference 沿用仓库外默认 `MLA_WORK_DIR`，本步证据放 `/tmp/mla-m10c5a/`。全部必要检查通过，更新相关文档并完成质量检查后只提交一次；无法验收则保留交接且不提交闭环。

本步固定源/独立边界、普通与宿主机 ASan/UBSan、工具和新构建上的累计 E2E 已完成，原始证据及环境差异见 status。未开放混合 split 的生产或公开入口，后续不累计本模块历史单元。

### 已完成 M10c5b：混合布局显式 split 生产接入

在稳定的显式 partial/merge 和 TND 原 allocation 地址上，接通非 PA TND Q + BSND KV 的 `num_splits=2..256`，作为一个完整可调用行为。覆盖三种 mask、Q allocation/used 与 KV 有效长度、四路 stride、真实 FP32 accumulator、公开只读 cache 及内部入口；混合自动和 PA split 保留各自后续步骤。

- 改动与影响：在 `mla_split_mixed_metadata` 增加复用既有安全 prefix 解析的公开生成模式，新增 v21 AICPU 单元、混合双 dtype launcher；将显式 TND launcher 抽为共享实现，数值 partial/merge 与四路 stride 组件保持复用。修改 `mla_forward.h/.cpp`、`flash_api.cpp`、Python real/fake 分派和 `setup.py`。紧凑 Q 静态预算使用 `BuildSplitMixedPlanLayout`，设备重算实际 M 前缀和持久预算，KV 地址始终为原 BSND batch/token stride；返回 `[S,H,totalQ,512]` / `[S,H,totalQ]` FP32 accumulator。共享 launcher 的 TND 路径和公开 metadata 分派均受影响，重建后实际回归。
- 新增独占 `unit/split_mixed/` 的固定源 partial/FD、双数学与 accumulator 对照；`e2e/split_mixed/` 覆盖三种 mask、allocation/used 间隙、四路 stride、空值/最大分片、真实/fake/compile、公开生成/隐式/复用、只读 cache、长度更新、流生命周期及 release Trap，并在 `regression_cases.py` 注册小输入日常集合。公开性能固定双 dtype、三种 mask、长 KV/stride/全零长度/分片边界，保留六个入口/R 段；原生 FD 按 design 单列，不按耗时筛选。
- 必跑命令均为 `python3 -m tests.mla`，先 `--dry-run`、再 `--collect-only`：`--suite split-mixed-tools`；代表及最大 reference/内存边界 `--suite split-mixed-check --device 6`。固定实现后运行 `--unit split_mixed`、`--suite split-mixed-public split-mixed-trap e2e ordinary --device 6`，隔离运行 `--suite split-mixed-profile --device 6`。v21 producer 增加 prefix 模式，补所属协议 CPU 差分/失败原子性用例并运行 `--unit split_mixed_protocol` 与 `--suite split-mixed-protocol-sanitize`；不累计其他历史单元。
- 基线与完成：起点 `42cb965`，核验 `/tmp/mla-m10c5a/` E2E 原始证据、源码/生产依赖、产物及加载库；生产和验收身份将改变，本步累计 E2E 实跑、继承 0。证据根 `/tmp/mla-m10c5b/`，构建/reference 使用仓库外默认 `MLA_WORK_DIR`，原生包使用既有内容核验缓存。同输入数学 reference 内容缓存复用，待测实现实际执行。必要验证、相关 design/plan/status 及质量检查通过后只提交一次并立即停止；验收受阻则如实交接，不提交为闭环。

本步 v21 AICPU、混合 partial/merge 分派、真实 accumulator 和公开 real/fake/compile 已接通。新增 prefix 模式的 CPU/ASan/UBSan、本模块固定源/双数学、公开/Trap/生命周期、累计 E2E、普通 FA3 和适用平均公开性能均已实测。日常清单新增本模块的三种 mask 与双 dtype；原生 FD/空 KV 模板性能按设计单列，完整结果和已处理的验收问题只记 status。混合自动及 PA split 继续拒绝。

### 已完成 M10c6a：混合原生自动 split 计划与安全协议

复用 BSND/TND 已迁移的原生 AIC/FD 分区及 8 行 Vector 归约，在一个独立 CPU/共享协议步骤中适配 TND Q + BSND KV。保留每 M 的原生变长分区、紧凑 Q 预算和原始 64 位 `b*Sk` 起点；不能将混合 KV 收窄成 TND int32 前缀，也不能用显式等分替代自动调度。

- 改动与影响：新增 `mla_auto_split_mixed_plan.h/.cpp`、独立 metadata 构造器和 CPU/AICore 共用消费者，复用既有 used、window 与原生自动组件。独立记录身份验证混合布局、几何/资源、长度、窗口、原生端点、持久槽及 accumulator 上界；共享紧凑计划及自动消费者的改动纳入已有 BSND/TND 自动路径的构建和累计 E2E。此步不接通 AICPU、设备发射或公开混合自动请求；生产接入留在后续一个完整可调用步骤。
- 新增独占 `unit/auto_split_mixed_protocol/` 与固定源驱动，检查 CANN AIC/FD/Vector 端点、workspace 及 FA3 accumulator 片段；独立覆盖三 mask、Q allocation/used、KV 实际长度、Direct/Partial/Empty、跨 M/8 行尾块、全空、64 位 KV/任务地址、溢出和失败原子性。生成器/消费者另验坏描述符、交叉版本、损坏端点与静态预算。普通和 ASan/UBSan 使用同一集合。
- 统一入口 `--unit auto_split_mixed_protocol` 与 `--suite auto-split-mixed-protocol-tools auto-split-mixed-protocol-check auto-split-mixed-protocol-sanitize`，先 dry-run/collect-only，再固定实现运行本模块及累计 `--suite e2e --device 6`。不累计其他模块历史单元；无新公开行为时不注册伪自动 E2E，数值/Trap/性能由生产接入步骤实际验收。
- 基线与完成：核验本步 status 所列的原始 E2E、源码、构建依赖/产物和实际加载，只有身份及证据完整有效才继承 E2E；变化则实跑。沿用仓库外构建/reference 缓存，不复算相同数学 reference。新计划/协议的固定源、独立边界、sanitizer、工具及受影响 E2E 全通过，更新相关文档和质量检查后只提交一次并停止。

本步实施基线 `94eedfe`，证据目录 `/tmp/mla-m10c6a/`。采用独立 v22 混合记录；从 TND 计划复用紧凑 Q 预算、原生任务/归约和地址查询，共享构造显式区分 KV 布局，BSND 总量仅以宽整数处理。自动消费者增加独立 KV 布局参数，既有 v18/v20 wrapper 保持原参数。影响 TND 计划与 BSND/TND 自动设备消费者，已重建并实跑累计 E2E，继承 0；不修改公开分派和 AICPU 注册。新增计划/协议驱动和独占测试，尤其保留 `b*Sk > UINT32_MAX`、紧凑 totalQ、跨 8 行 FD 尾部、精确 accumulator 字节边界及交叉版本拒绝。普通与 sanitizer 使用同一完整模块集合；先运行 tools、三 mask 代表和最大实际 CPU 输入，再冻结源码运行普通、sanitizer 与 910B 累计 E2E。此 CPU 步没有浮点 reference 新计算或公开性能要求；构建及 E2E 沿用仓库外内容缓存。

本步固定源/独立边界、普通与宿主机 ASan/UBSan、工具、完整重建及 910B 累计 E2E 均已通过；实际证据与环境差异见 status。混合自动生产入口仍拒绝，本次提交后停止，不累计本模块历史单元。

### 已完成 M10c6b：混合布局原生自动 split 生产接入

在 M10c6a 的原生变长分区、v22 协议和稳定的自动 init/partial/merge 上，整体接通非 PA TND Q + BSND KV 的 `num_splits=0`。覆盖三种 mask、Q allocation/used、实际 KV 长度、四路 stride、空容量和真实静态 FP32 accumulator；保留源每 M 的变长 split 数和 FD 归约，不能改为显式等分。

- 改动与影响：v22 增加公开安全 prefix 模式及 AICPU 注册；在共享自动 launcher/数值核适配 BSND KV batch/token stride，Q 与输出保持原 allocation 地址。接通 host 静态预算、双 dtype 实例、Python/C++ real/fake/compile 和公开 cache metadata。共享自动分派、消费者与构建参数的影响须纳入已有 BSND/TND 自动和混合显式 E2E；不修改 CATLASS/CANN 源仓。
- 新增独占 `unit/auto_split_mixed/` 固定源 partial/FD、双数学、Direct/Partial/Empty 和 accumulator 对照；`e2e/auto_split_mixed/` 覆盖三 mask、长度/stride/空值、metadata 复用和失败恢复、真实/fake/compile、公开/内部入口、流生命周期与 release Trap，并在 `regression_cases.py` 注册小输入日常节点。prefix 模式若改变协议生产逻辑，补所属协议单元/ASan/UBSan，其他模块仅累计 E2E。
- 必跑入口均为 `python3 -m tests.mla`，先 dry-run/collect-only。新增 `auto-split-mixed-tools`、代表/最大实际内存及最长 reference 的 `auto-split-mixed-check`，固定源码后运行 `--unit auto_split_mixed`、`--suite auto-split-mixed-public auto-split-mixed-trap e2e ordinary --device 6`。隔离运行 `auto-split-mixed-profile`，按固定输入、双 dtype、隐式/生成/复用各 R=1/4 平均公开 E2E；原生 FD 在本自动路径纳入门槛，空 KV 模板按原约定单列。
- 基线与完成：接手核验最新 status 的构建、实际加载与 E2E 原始证据；设备和入口变化本步实跑，只继承依赖未变且完整有效的 E2E。沿用仓库外内容缓存，复用同输入数学 reference。公开能力与适用平均性能闭环、文档/质量检查通过后提交一次并停止；无法完成必要验证时更新交接，不提交为闭环。

本步实施基线 `a02f194`，证据目录 `/tmp/mla-m10c6b/`。实际改动包括 `mla_auto_split_mixed_metadata.*` 的安全 prefix 模式/AICPU、共享 `mla_auto_split_launch_impl.hpp` 的独立 KV 布局特化、双 dtype 混合自动实例、`mla_forward.cpp`/`flash_api.cpp`/Python real/fake 分派和 `setup.py`；复用既有混合显式入口签名及原生 TND 自动数值组件，保持 BSND/TND 自动 wrapper 原语义。新增模块源对照、公开/Trap/性能驱动及日常 E2E，工具检查覆盖选择、原生 FD 计入门槛和失败传播。prefix 构造改变 v22 生产逻辑，运行所属 `unit-auto_split_mixed_protocol` 与同集合 `auto-split-mixed-protocol-sanitize`；共享发射和构建变化后累计 E2E 全部实际执行，继承 0，不累计其他模块单元。

先以统一入口 dry-run/collect-only 核对新增组及原有集合，再运行 tools、三 mask 代表和最大内存/最长 reference 成本用例；源码固定后运行本模块固定源/双数学、公开/Trap、累计 E2E 和 ordinary，最后隔离运行固定输入的平均公开性能。构建和数学 reference 继续使用 `/tmp/flash-attention-npu-mla-46928a793876/mla-validation-artifacts/`；同输入复用内容缓存而实际执行待测算子。完成条件为 v22 生成/消费与三阶段真实输出及 accumulator、real/fake/compile/生命周期/失败恢复全部通过，含原生 FD 的公开性能平均比值 ≤1，相关契约/状态及仓库质量检查通过后仅提交一次。复核发现共用混合入口缺少 Q-used 与显式 metadata 的 host/fake 组合拒绝，本步补齐并在新模块同时覆盖自动/显式模式、真实/fake 与 Python/C++ raw 入口；新增该边界的日常 E2E，旧实现中断构建不作为最终证据。

本步 prefix 协议/ASan/UBSan、固定源/双数学、真实 accumulator、公开/Trap/生命周期、累计小输入 E2E、普通 FA3 与含原生 FD 的平均公开性能均已通过。原生对照包只复用当前配方及完整内容有效的构建，性能实际重测，完整结果和开发失败记录见 status。提交后立即停止，PA split 保持拒绝。

### 已完成 M10c7a：BSND PA 显式 split 计划与安全协议

先独立闭环 BSND Q + ND PA KV、512+64、显式 `S=2..256` 的 CPU 任务/持久布局及 v23 安全协议。PA 的逻辑容量、按有效长度整页取整的调度上界和页大小相关基本块属于新的协议边界；本步不接通 AICPU、设备 partial/merge 或公开 PA split。已有非 PA split 与 PA 单 split 行为保持原契约。

- 改动与影响：在 910 MLA 目录新增 PA 显式计划、metadata 构造和独立 CPU/AICore 共用消费者；优先复用既有 PA 调度/地址、window 与显式 split 组件。明确页几何、Q allocation/used、实际 KV 长度、三种 mask、M 任务前缀、持久槽和 FP32 accumulator 的独立上界；保留原 Q 行地址和逻辑 token 范围。记录绑定范围与页表/物理 cache 的当次校验职责须在 design 对应段落实，不修改 CATLASS 或 CANN 源仓。
- 新增所属 `unit/split_paged_bsnd_protocol/` 的固定源差分、独立逐行/逐页枚举、边界/溢出及失败原子性。页大小保留 16/1024、非 512 因子及大于 512 的代表边界，覆盖尾页、跨页 mask、空分片/全空、坏长度/页几何、交叉版本、损坏任务端点和静态预算；双 dtype 协议参数两两覆盖，必要多轴组合具名。普通与 ASan/UBSan 使用同一集合，不展开无目的的笛卡尔积。
- 必跑：统一入口新增本模块工具、代表/最大实际 CPU 成本、普通及 sanitizer 组，先 `--dry-run`/`--collect-only`，冻结后实跑 `--unit split_paged_bsnd_protocol` 和对应 sanitizer；累计 `--suite e2e --device 6` 覆盖共享代码影响。只有新增设备可调用行为时才注册其日常 E2E，此 CPU 步不注册伪 PA split E2E，也不累计其他模块历史单元。
- 基线与完成：先核对本 status 的源码、构建、加载身份和完整 E2E 原始证据；只有未变化且完整有效的 E2E 可继承，变更则实际执行。构建/reference 缓存继续放仓库外，必要共享构建影响也须核验实际加载。固定源、独立边界、普通/sanitizer、工具及受影响 E2E 全通过，更新协议契约、plan/status 和质量检查后仅提交一次；设备数值、Trap 和公开性能由后续生产接入步骤独立验收。

本步实施前固定如下边界：

- 基线 `4f531a3`；新增 `mla/mla_split_paged_bsnd_{plan.h,plan.cpp,metadata.h,metadata.cpp,consumer.h}`。复用 `BuildPagedSchedule` 的 PA 源几何/长度快照、`MlaSplitKvPlan` 的显式分片与 BSND 原 allocation 地址、v6 页描述符/有效页检查；不修改共享数值实现、setup、生产分派或公开签名。静态资源上界独立于有效长度，v23 记录携带 window/S 与宽 M 前缀，明确全零 KV、空容量与逐次页表校验。
- 测试新增 `unit/split_paged_bsnd_protocol/`，构建及两个 CPU 驱动置于 `support/`；固定 CANN PA tiling、mask 范围/workspace 与 FA3 分片/accumulator 片段重新编译执行。独立逐行/逐页枚举任务与地址，覆盖 P=16/112/512/528/1024、尾页/跨页三 mask、used/空值/多波、S=2/8/256、输入/输出别名、坏描述符、页号、跨版本、前缀/预算/宽整数及失败原子性。
- 统一命令先 dry-run、collect-only：`--suite split-paged-bsnd-protocol-tools`；代表和最大实际 CPU 成本 `--suite split-paged-bsnd-protocol-check`；冻结后 `--unit split_paged_bsnd_protocol` 与 `--suite split-paged-bsnd-protocol-sanitize` 使用同一集合；`--suite e2e --device 6` 实跑累计清单。没有新增设备可调用行为，本步不增加伪 PA split E2E，公开性能和 Trap 不属于 CPU 协议验收。
- 接手核验前步 E2E 原始证据、源码、构建输入/依赖/产物和加载库；新增头文件/测试驱动会改变保守证据及构建身份，因此累计 E2E 本步实跑，不继承历史单元/性能。证据放 `/tmp/mla-m10c7a/`；构建与数学缓存沿用 `/tmp/flash-attention-npu-mla-46928a793876/mla-validation-artifacts/`，同输入数学缓存复用。完成上述固定源/独立边界、工具、普通/sanitizer 和重建加载后的 E2E，再更新 design/status、质量检查并提交一次。

本步固定源 PA 几何/mask/workspace/FA3 分片与 accumulator、独立行/页/任务地址、普通与 ASan/UBSan、工具及重建加载后的累计 E2E 均已通过。校验绑定逻辑容量与当前长度，不绑定页号/物理地址；消费者每次完整检查物理描述和所有有效页。开发失败、实际结果与耗时见 status。CPU 协议闭环不代表公开 PA split 已开放；提交后立即停止。

### 已完成 M10c7b：BSND PA 显式 split 生产接入

在已闭环 v23 与 PA Copy、显式 partial/merge 数值组件上，接通 BSND Q + ND PA KV、512+64、双 dtype、显式 S=2..256 的完整可调用行为。范围包含无 mask、右下 causal/window、Q allocation/used、实际 KV 长度及合法 Q/qv/cache/page-table stride；自动 split 和 TND PA split 继续拒绝。

- 改动与影响：在 FA3 910 MLA 目录接入 v23 AICPU、host 静态预算与调用描述、PA partial 和独立 merge/init 发射、双 dtype 实例及生产/Python/C++ real/fake/compile。优先复用已有页 Copy、任务环与显式归并；每次和每阶段在写入/同步前校验逻辑记录、当前长度、页表和物理 cache，不把原始页号、地址或 stride 固定到 metadata。保留原 Q allocation 行地址、跨波持久槽与真实 FP32 accumulator；共享 Kernel/Block/Tile、构建/调度/地址改动全部纳入影响范围。
- 新增本模块单元与 `e2e/split_paged_bsnd/` 专项，并在 `regression_cases.py` 注册公开能力的日常小输入 E2E。固定源 PA partial/归并和独立双数学覆盖双 dtype、三 mask、页大小代表边界、非 512 因子/大于 512、尾页/跨页、空片/全空、S=2..256、跨波及真实 accumulator；源不支持的表示域单列，不能以旧无 mask 或非 PA 用例替代新增能力。
- 实施前固定具体路径、测试组和命令；统一入口先 dry-run/collect-only，先工具、代表及最大内存/最长 reference 成本检查，固定实现后运行所属单元、必要协议/sanitizer、新公开专项、真实 release Trap、生命周期/失败恢复、累计 E2E、相关 ordinary 回归与限量平均公开性能。公开性能含页表/cache 转换和分配，保持既有 R/计时/数学阈值，适用配置平均比值 ≤1。
- 基线与完成：核对本 status 的完整源码、构建、加载与 E2E 原始证据；只继承依赖未变且完整有效的 E2E 正确性。共享路径变化则实际运行相关 E2E，数学 reference 按内容复用。v23 生成/消费、设备真实 O/LSE/accumulator、公开契约/拒绝边界及必要性能全部通过后更新 design/plan/status、质量检查并提交一次，不扩展到 PA 自动或 TND 布局。

本步实施前固定如下边界：

- 基线 `9e8e849`。在 `mla/mla_split_paged_bsnd_*` 增加 AICPU、双 dtype launcher 和设备消费者；`kernel/mla_split_kv.hpp` 复用 PA Copy 的 partial 特化，`kernel/mla_dense.hpp` 开放经验证的 PA/partial 组合。`mla_forward.cpp/.h`、`flash_api.cpp`、Python interface 和 `setup.py` 接通 v23 静态预算、生产分派、metadata、real/fake/compile；merge 复用已有 Block，空容量单独初始化且先完整校验。共享数值、地址、同步和构建影响通过累计 E2E 实跑验证。
- 新增 `e2e/split_paged_bsnd/`、本模块设备/协议用例及 `support/mla_split_paged_bsnd_*` 对照/Trap/性能支持；日常注册双 dtype 三 mask、空值、stride/复用与 fake/compile。新增用例覆盖 P=16/112/512/528/1024、尾页/跨页、S=2/8/193/255/256、M 尾块/跨波、Q-used、全零 KV、零容量、真实 accumulator、合法页重排及坏页/陈旧记录/长度失败恢复。固定源逐片 PA 和归并、独立双数学均须实际比较，源表示域限制单列。
- 统一入口先 `--dry-run`、`--collect-only`；依次 `--suite split-paged-bsnd-tools`、`--suite split-paged-bsnd-check --device 6`（含最长 reference/最大 accumulator 代表）、冻结后 `--unit split_paged_bsnd`（固定源 PA partial/FD 与双数学）、`--unit split_paged_bsnd_protocol`、`--suite split-paged-bsnd-protocol-sanitize`、`--suite split-paged-bsnd-public split-paged-bsnd-trap e2e ordinary --device 6`，最后 `--suite split-paged-bsnd-profile --device 6`。协议普通/sanitizer 集合相同；不累计其他历史单元。性能配置在执行前固定，保留原 R=1/4、5×20 计时及 FD 分类，适用平均比值 ≤1。
- 证据根 `/tmp/mla-m10c7b/`；构建与数学缓存沿用 `/tmp/flash-attention-npu-mla-46928a793876/mla-validation-artifacts/`。上步 E2E 原始证据、源码、构建和加载身份先核验；本步共享源码和扩展变化，累计小输入 E2E 实际重跑，不继承历史单元/性能。全部必要检查通过后更新本步相关契约、plan/status，质量检查并提交一次；未通过则更新交接，不提交为闭环。

### 已完成 M10c8a：BSND PA 原生自动计划与安全协议

独立闭环 BSND Q + ND PA KV 的原生自动任务、持久预算及 CPU/AICore 共用记录协议。保留固定 CANN 每 M 的变长 AIC 分区、FD 槽和 8 行 Vector 归约所有权，不能用均匀显式 split 代替；页大小对 N 的调整、实际 KV 与逻辑容量仍按固定 PA 源几何。该步完成前后公开 PA 自动与 TND PA split 仍明确拒绝，AICPU 和生产接入留到后续可调用行为步骤。

- 影响：优先复用现有原生自动任务映射、PA 长度/页校验、原 Q allocation 地址和静态布局校验；新增独立自动 PA 计划、预算、记录构造/消费/任务查询。记录须区分 v23 显式身份，不将物理页号、cache 地址或 stride 写入复用身份；设备实际页检查保留在消费契约内。
- 验证：本模块固定 CANN PA 调度/FD/Vector 差分、独立任务覆盖和地址检查；双 dtype、三 mask、页大小边界、Direct/Partial/Empty、空值/Q-used、原 allocation、宽任务号及 accumulator/持久槽溢出，普通与 ASan/UBSan 使用同一集合。新增协议损坏、陈旧长度及坏页用例；按共享依赖影响运行累计小输入 E2E，不累计历史单元。
- 实施前固定具体文件、所属测试组、命令和依赖，再按统一入口 dry-run/collect-only；数学 reference 与构建缓存按内容复用。只继承依赖未变且完整有效的 E2E 正确性；本步 CPU/协议测试实际执行。
- 完成条件：独立 CPU 源差分、边界/溢出、协议及 sanitizer 和受影响 E2E 全通过，记录表示域与拒绝边界；更新相关 design/plan/status、质量检查后提交一次。生产数值、真实 Trap 和公开性能须由后续接入步骤验收，本步不能提前宣称通过。

本步实施前固定如下边界：

- 基线 `f65820a`，工作树干净。新增 `mla/mla_auto_split_paged_bsnd_{plan,metadata}.{h,cpp}` 和 `_consumer.h`，复用 PA 长度/几何、原生自动计划及任务/归约查询；共享 `mla_auto_split_bsnd_consumer.h` 提取已校验几何之后的分区校验，保持 v18/v20/v22 行为。v24 使用 PA 逻辑快照、原生 AIC/FD/Vector tail 和 64 位 M prefix，不绑定物理页、地址或 stride。静态预算独立限制 accumulator、FD 持久槽、scratch、libapi；空容量及全零 used 仍校验。不上 AICPU/生产分派，不修改 CANN/CATLASS。
- 新增 `unit/auto_split_paged_bsnd_protocol/{test_plan,test_metadata}.py`、所属 CPU bridge/build 与工具组。固定 CANN PA 几何、AIC/FD/Vector、workspace 和 FA3 accumulator 差分；独立枚举三 mask、P=16/112/512/528/1024 及 M/N 尾界、Direct/Partial/Empty、Q-used/原 allocation、空值、宽 group、精确溢出。协议覆盖双 dtype、每类损坏/陈旧长度/当前坏页、页重排复用、查询失败原子性；普通和 sanitizer 集合相同。
- 全部通过 `python3 -m tests.mla`，各选择先 `--dry-run`、`--collect-only`。先 `--suite auto-split-paged-bsnd-protocol-tools auto-split-paged-bsnd-protocol-check` 验证工具、代表及容量成本边界；固定实现后 `--unit auto_split_paged_bsnd_protocol`、`--suite auto-split-paged-bsnd-protocol-sanitize`；共享消费者影响已接通自动路径，重建并核对实际加载后运行 `--suite e2e --device 6` 的累计小输入集合。不累计历史单元，不以旧无 mask/非 PA E2E 作为新协议验收，不声明生产数值/Trap/性能已完成。
- 证据根 `/tmp/mla-m10c8a/`，缓存沿用 `/tmp/flash-attention-npu-mla-46928a793876/mla-validation-artifacts/`。先核验上步完整 E2E 原始证据、源码/构建/加载身份；本步共享源变化使旧 E2E 失效，实跑累计集合并复用数学内容缓存。必要检查全通过后更新受影响 design、plan/status，执行仓库质量检查及一次提交；无法验收则只更新交接。

### M10c8b：BSND PA 原生自动生产接入（已完成）

在已验收的原生 PA 计划与 v24 协议上，独立闭环 BSND Q + ND PA KV 的 `num_splits=0` 可调用行为。保留固定 CANN 每 M 的变长 AIC 分区、FD 持久槽和 8 行 Vector 归约，不用均匀显式 split 替代；公开 TND PA split 继续拒绝。

- 影响：接通 v24 AICPU、双 dtype 自动 PA launcher、生产分派、公开 cache/metadata 与 real/fake/compile。复用稳定 PA Copy 和自动 partial/merge；原 allocation 地址、P 对 N 的调整、实际 KV/逻辑容量及每阶段当前页校验保持一致。生产预算另纳入每 AIC 的 Direct 临时槽、静态 accumulator 初始化和完整空容量行为，不能用本步 FD 预算直接覆盖新增 scratch。
- 新能力覆盖 FP16/BF16、三种 mask、P 边界/尾页/跨页、Direct/Partial/Empty、空值/Q-used、长度便利输入、独立 stride、metadata 复用与失败恢复、storage/流生命周期。固定源 PA+原生 FD 与独立双数学实际比较 O/LSE/真实 accumulator；共享组件证据不能代替新入口的参数传递、分派和公开 E2E。新增本模块日常 E2E 到 `regression_cases.py`，不以旧无 mask 或非 PA 用例作为新路径验收。
- 实施前固定具体路径、所属单元/公开/Trap/性能组、命令和依赖；各选择经统一入口 dry-run/collect-only。先工具、代表与适用成本边界，再冻结实现运行本模块必要检查、真实 release Trap、累计小输入 E2E 和相关 ordinary 回归；按实际改动运行所属协议/sanitizer，不累计历史单元。数学 reference 按内容复用。
- 本步实测固定输入平均公开 E2E，包含原生自动 FD，页表/cache 转换和分配按既有计时边界计入；保持 R=1/4、预热/重复次数和配置内/配置间等权平均，适用平均耗时比 ≤1，不按耗时筛选输入。源表示域和空模板按 design 单列。
- 完成条件：v24 生成/独立消费、自动 PA 真实数值/accumulator、公开契约/拒绝边界、生命周期及必要性能全部通过，更新相关 design/plan/status、质量检查并提交一次；未完成必要验收则保留交接，不提交为闭环。本步不开展 TND PA split 或阶段收尾。

实现包含生产静态 26 片 accumulator 与 Direct scratch 预算、v24 AICPU、独立双 dtype 三阶段 launcher、host/公开/fake 分派及本模块源数值/公开/Trap/性能和累计 E2E。实际平台 AIC 数与返回容量分别处理；CANN/CATLASS 未修改。所属 CPU/sanitizer、双 dtype 固定源与双数学、公开/Trap、累计 E2E、ordinary 和包含原生 FD 的完整性能矩阵均通过，命令、耗时、失败修正和原始证据见 [status](status.md)。本步未开展 TND PA split 或阶段收尾。

### 唯一下一步 M10c9a：TND PA 显式 split 计划与安全协议

先独立闭环 TND Q + ND PA KV、512+64、显式 `S=2..256` 的 CPU 任务/持久布局及 CPU/AICore 共用安全协议。复用固定源 PA 几何、现有显式分片和三种 mask；原 Q allocation 前缀、used 长度及紧凑总 token 分开处理，不能以 `B*maxQ` 代替实际分配。

- 影响路径：新增 TND PA 显式 plan/metadata/consumer/query 与所属 `unit/split_paged_tnd_protocol/`、固定源差分及边界驱动；尽量组合既有 TND/PA 组件。若必须调整共享地址、页校验或 split 查询，先列明具体受影响路径。此步不编入生产 AICPU/launcher，不开放公开 TND PA split。
- 实施前固定具体文件、协议字数/版本、静态预算、任务/归约所有权、拟继承 E2E 的依赖和完整必跑命令。新增独立覆盖目的：非均匀/空 Q 分配、Q-used、实际 KV 与页容量、原 allocation 输出/accumulator 地址、页内/跨页、宽前缀/乘法溢出、失效/陈旧记录及坏页，包括 Q-used=0 时当前页校验；协议参数两两覆盖与必要具名组合，双 dtype 保留。
- 所属检查通过 `python3 -m tests.mla --unit split_paged_tnd_protocol` 及专属 sanitizer/工具组，先 dry-run/collect-only；与固定 CANN PA/FA3 显式分片、独立行/token/page 枚举对照，普通与 ASan/UBSan 使用同一集合。仅继承依赖未变且原始证据完整的 E2E 正确性；共享生产依赖改变则重跑受影响累计小输入 E2E，不累计历史单元或继承性能。
- 完成条件：独立计划、完整失败原子性/预算/消费者与查询协议通过；公开 TND PA split 仍明确拒绝；按影响更新 design/plan/status 并提交一次。生产接入另立下一步，再新增所属公开 E2E、实机数值/Trap/生命周期和性能。本次不实现该步；无头循环按已确认规则另外安排到期文档修正轮。

## 后续里程碑：M10–M17

**下表每行是能力里程碑，不是一个无头轮次或一个提交。**临近该阶段时根据已闭环组件和实际影响细化当前步，同时保留下一步；不现在为所有远期任务固定文件、测试数量或提交数。细化采用与 M08 相同的规则：新算法先独立验收，稳定布局接入一次闭环。实现和必要的测试/对照驱动一起交付，基础精度、普通回归与生命周期不推迟到阶段末。

| 里程碑 | 依赖与建议提交边界 | 阶段完成条件 |
| --- | --- | --- |
| M10：512+64 split-KV | M10a CPU 分区/布局、M10b1 partial attention、M10b2 FD 归并、M10c1a 跨波任务/持久地址和 M10c1b BSND 显式生产、M10c2a 原生自动任务/归约计划、M10c2b1 v18 安全协议和 M10c2b2 BSND 自动生产、M10c3a TND 显式任务/持久布局、M10c3b TND 显式生产、M10c4a TND 原生自动任务/持久布局、M10c4b1 v20 安全协议、M10c4b2 TND 自动生产、M10c5a 混合显式计划/v21 协议、M10c5b 混合显式生产、M10c6a 混合原生自动计划/v22 协议、M10c6b 混合自动生产、M10c7a BSND PA 显式计划/v23 协议、M10c7b BSND PA 显式生产、M10c8a BSND PA 原生自动计划/v24 安全协议及 M10c8b BSND PA 原生自动生产已完成；接着 M10c9a TND PA 显式计划/安全协议，再按可调用行为完成 TND PA 布局，不把完整 split 迁移作为一轮 | 自动/1/显式 split、空分片/全空、O/LSE/accumulator/metadata 及已支持 mask 组合；长 KV E2E |
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
| M10a | 独立 CPU native FD 分区/归约记账、FA3 请求与显式 mask 分片、源 workspace 和 BSND/TND FP32 accumulator 布局/地址；固定源、独立覆盖/溢出及 ASan/UBSan，累计 E2E 实跑；生产 split 仍拒绝 |
| M10b1 | 独立连续 BSND FP16/BF16 partial attention；真实 FP32 部分 O/sum/max、mask/空片和分波、固定源与双数学、组件性能及累计 E2E；FD 合并和公开 split 仍未开放 |
| M10b2 | 独立连续 BSND FP16/BF16 FD 归并及 FP32 accumulator 重排/LSE；逐行空片适配、固定源可表示域/双数学、192/255/256 资源边界、组件性能及累计 E2E；公开 split 仍未开放 |
| M10c1a | 独立 BSND 显式 split CPU 任务归属、持久 partial 上界与原分配行地址；源范围/workspace/FA3 片段差分、跨波/空值/溢出、ASan/UBSan 及累计 E2E；未接入设备 metadata 或公开 split |
| M10c1b | 非 PA BSND 显式 2..256、v17 CPU/AICPU 与独立设备消费、跨波 partial/merge 同步、四路 stride/Q-used、真实 FP32 accumulator、公开 real/fake/compile；本模块源/双数学、Trap/生命周期、累计 E2E、普通 FA3 及适用平均公开性能 |
| M10c2a | 独立 BSND 原生自动任务映射、FD 持久槽与 8 行 Vector 归约所有权、分配/used 地址与静态 accumulator 上界；源差分、独立边界/溢出及 ASan/UBSan、累计 E2E；公开自动仍拒绝 |
| M10c2b1 | 独立 v18 自动 metadata 构造及 CPU/AICore 共用校验/查询协议；固定源/独立覆盖、FD/Vector/预算/宽整数损坏检查、ASan/UBSan 及累计 E2E；尚未编入 AICPU/设备或公开自动路径 |
| M10c2b2 | 非 PA BSND 自动 split、v18 AICPU 与 init/partial/merge、原生 AIC/8 行 Vector 归约、真实静态 FP32 accumulator、公开 real/fake/compile；本模块源/双数学、Trap/生命周期、累计 E2E、普通 FA3 与含原生 FD 的平均公开性能 |
| M10c3a | 独立 TND 显式 split CPU 任务、紧凑分配总量持久上界、双前缀/used 与原 allocation 地址；固定源/独立边界、64 位及溢出、ASan/UBSan 和累计 E2E；尚未接入设备或开放 TND split |
| M10c3b | 非 PA TND Q/KV 显式 2..256、v19 CPU/AICPU 与独立消费、双前缀/used/stride 和原 allocation token 地址、真实 FP32 accumulator、公开 real/fake/compile；本模块源/双数学、Trap/生命周期、累计 E2E、普通 FA3 与适用平均公开性能 |
| M10c4a | 独立 TND 原生自动 AIC/FD 任务、紧凑总量预算、双 allocation/used 与原 token O/LSE/accumulator 地址、8 行 Vector 归约；固定源/独立边界、64 位/溢出、普通与 ASan/UBSan、累计 E2E；尚未接通 TND 自动生产 |
| M10c4b1 | 独立 TND v20 构造及 CPU/AICore 共用校验/任务查询，紧凑总量与原 allocation 地址；固定源、三种 mask/损坏/溢出、ASan/UBSan、v18 兼容及重建后的累计 E2E；尚未接入 v20 AICPU/设备或公开自动路径 |
| M10c4b2 | 非 PA TND Q/KV 自动 split、v20 AICPU 与共享 init/partial/merge 的 TND 特化、紧凑预算与原 allocation 地址、真实静态 FP32 accumulator、公开 real/fake/compile；本模块固定源/双数学、Trap/生命周期、累计 E2E、普通 FA3 与含原生 FD 的平均公开性能 |
| M10c5a | 非 PA TND Q + BSND KV 显式 split CPU 计划与 v21 构造/独立消费；紧凑 Q 预算、64 位 KV 起点、三种 mask/空值/accumulator 溢出、固定源/独立边界、ASan/UBSan 与累计 E2E；尚未接入 AICPU/设备或开放混合 split |
| M10c5b | 非 PA TND Q + BSND KV 显式 2..256、v21 公开 prefix/AICPU、共享 partial/merge 的 BSND KV stride、紧凑预算与真实 FP32 accumulator、公开 real/fake/compile；本模块 CPU/sanitizer、固定源/双数学、公开/Trap/生命周期、累计 E2E、普通 FA3 与适用平均公开性能 |
| M10c6a | 非 PA TND Q + BSND KV 原生 AIC/FD/Vector 计划与 v22 构造/独立消费；复用紧凑 Q 与 64 位 KV 起点、三种 mask、Direct/Partial/Empty、宽任务号与 accumulator 溢出；本模块固定源/边界、普通/ASan/UBSan、工具、重建后的累计 E2E；尚未接通混合自动生产 |
| M10c6b | 非 PA TND Q + BSND KV 原生自动 split、v22 安全 prefix/AICPU、共享 init/partial/merge 的独立 KV 布局、紧凑预算与真实 FP32 accumulator、公开 real/fake/compile；本模块协议/sanitizer、固定源/双数学、公开/Trap/生命周期、累计 E2E、普通 FA3 与含原生 FD 的平均公开性能 |
| M10c7a | BSND Q + ND PA KV 显式 split CPU 计划、独立静态预算与 v23 构造/消费/查询；三 mask、页几何/有效页校验、原 allocation 地址、宽任务号及 accumulator/持久槽溢出；固定源/独立边界、普通/ASan/UBSan、工具与重建后的累计 E2E；尚未接入 AICPU/设备或公开 PA split |
| M10c7b | BSND Q + ND PA KV 显式 2..256、v23 AICPU、PA partial/merge、原 allocation 与真实 FP32 accumulator、公开 real/fake/compile；本模块协议/sanitizer、固定源/双数学、公开/Trap/生命周期、累计 E2E、普通 FA3 与适用平均公开性能 |
| M10c8a | BSND Q + ND PA KV 原生 AIC/FD/Vector CPU 计划、独立持久预算与 v24 构造/消费/查询；固定源/独立覆盖、三 mask、原 allocation、宽 group、记录/页校验及溢出、普通/ASan/UBSan、工具与重建后累计 E2E；尚未接通 PA 自动生产 |
| 测试整理与精简 | 模块所属单元测试、明确的小输入日常 E2E 与共享支持目录；代表组合取代默认数量截断，穷举显式开启、仅继承 E2E；外部产物和可重建原生对照，build 仅保留无头日志与状态 |

M07 的页大小精度覆盖全部 16..1024 的 16 倍数；性能矩阵及原生非 FD/FD 的边界见 [design 7.1](design.md#71-固定性能矩阵与阶段边界)。公开接入包含 None/整数/tensor KV 长度、v6 复用及相关 MLA/普通 FA3 回归，结果只在 status 维护。

历史查询示例：

```bash
git log --oneline -- docs/mla/status.md
git log -G 'M06d3b2' -- docs/mla/plan.md
git show 29ed04f:docs/mla/status.md
```
