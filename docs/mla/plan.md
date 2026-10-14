# MLA 迭代计划

本页记录实施顺序、每步产物与完成条件。**M10c3a TND 显式 split 任务与持久布局已完成；唯一下一步为 M10c3b TND Q/KV 显式 split 生产接入。** 当前能力和证据见 [status](status.md)，范围及验收契约见 [design](design.md)。

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

### 唯一下一步 M10c3b：TND Q/KV 显式 split 生产接入

将已闭环的 partial/FD 数值和 M10c3a 任务地址接入非 PA TND Q/KV 的显式 `num_splits=2..256` varlen 前向，整体完成两组分配前缀、Q/KV-used、四路 stride、metadata 和真实 FP32 accumulator。保留 BSND 自动/显式与各布局单 split；TND 自动、混合和 PA split 在对应步骤前继续拒绝。

实现前补齐实际文件和必跑命令。新模块独占源/双数学与边界测试，公开接入同步增加日常 E2E，完成真实/fake/compile、长度/stride/空值/生命周期/Trap、累计回归、普通 FA3 和适用平均公开性能。CPU 地址证据和 BSND 数值证据不能代替 TND 入口、设备寻址与同步验收。一次只完成一个独立闭环并提交。

## 后续里程碑：M10–M17

**下表每行是能力里程碑，不是一个无头轮次或一个提交。**临近该阶段时根据已闭环组件和实际影响细化当前步，同时保留下一步；不现在为所有远期任务固定文件、测试数量或提交数。细化采用与 M08 相同的规则：新算法先独立验收，稳定布局接入一次闭环。实现和必要的测试/对照驱动一起交付，基础精度、普通回归与生命周期不推迟到阶段末。

| 里程碑 | 依赖与建议提交边界 | 阶段完成条件 |
| --- | --- | --- |
| M10：512+64 split-KV | M10a CPU 分区/布局、M10b1 partial attention、M10b2 FD 归并、M10c1a 跨波任务/持久地址和 M10c1b BSND 显式生产、M10c2a 原生自动任务/归约计划、M10c2b1 v18 安全协议和 M10c2b2 BSND 自动生产、M10c3a TND 显式任务/持久布局已完成；接着按 TND/混合、PA 接入显式及自动调度，不把完整 split 迁移作为一轮 | 自动/1/显式 split、空分片/全空、O/LSE/accumulator/metadata 及已支持 mask 组合；长 KV E2E |
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
| 测试整理与精简 | 模块所属单元测试、明确的小输入日常 E2E 与共享支持目录；代表组合取代默认数量截断，穷举显式开启、仅继承 E2E；外部产物和可重建原生对照，build 仅保留无头日志与状态 |

M07 的页大小精度覆盖全部 16..1024 的 16 倍数；性能矩阵及原生非 FD/FD 的边界见 [design 7.1](design.md#71-固定性能矩阵与阶段边界)。公开接入包含 None/整数/tensor KV 长度、v6 复用及相关 MLA/普通 FA3 回归，结果只在 status 维护。

历史查询示例：

```bash
git log --oneline -- docs/mla/status.md
git log -G 'M06d3b2' -- docs/mla/plan.md
git show 29ed04f:docs/mla/status.md
```
