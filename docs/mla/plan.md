# MLA 迭代计划

当前已将 FP16/BF16 dense MLA 的源调度和 Kernel 接入 910B FA3。M05a/M05b1 完成 AICPU 生成、版本化 metadata 的设备读取与校验；M05b2a 对齐官方签名并保留普通 FA3 关键字兼容。M05b2b1/b2 完成设备长度读取和错误上报，b3a 完成生产内部生成/消费，b3b 已接通公开 `get_scheduler_metadata` 及只读 dense `flash_attn_with_kvcache` 复用。范围外组合仍明确拒绝。M05c1/M05c2 已完成精简 C++ 前向与 metadata dispatcher，并消除公开 dense qv 路径的重复张量检查；两种 dtype 各五组在相同固定 CPU 亲和性下的复用前向、一次生成加一次/四次前向均达到同轮 CANN E2E 中位数门槛，M05 已闭环。M04c1 已完成 dense S1G 多块、源 128..256 M 基块及跨核尾块验收；M04c2 已完成 KV 513..1024、禁止跨核 KV split 的端点映射及双 dtype 精度/性能验收，M04c3 已将 KV 容量扩展到 1..16384（含 8193），完成同路径源 Kernel、公开 E2E 及功能验收。M04c4 已完成 dense 全局空容量边界，2177 项检查通过；双 dtype 各 13 组同路径公开 E2E 达标。M06a 已完成实施拆分，唯一下一步为 M06b1：独立的 BSND 实际 KV 长度解析与 CPU 源调度对照。

实现原则见 [design](design.md)：以固定 CANN 源实现为起点，保持计算、精度、切块分核、缓冲和同步行为，按 CATLASS Kernel / Block / Tile 组织代码。FA3 参数映射与 metadata 接入遵循官方契约。性能只验收已迁移的执行路径：组件按固定源同路径对照，双方路径均已迁移的公开调用按 metadata E2E 逐例比较。

## 提交粒度

每次完成一项可独立验收的小步，更新本表和 [status](status.md)，提交一次后停止。实现前将过大的步骤拆为有明确范围和验收条件的子步骤。每个入口只接通已验收组合，其余组合明确报错。

| 步骤 | 状态 | 产物 | 完成条件 |
| --- | --- | --- | --- |
| M00 | 已完成 | 范围、来源、FA3 契约和 CATLASS 结构约定 | 固定来源，明确迁移边界、组件等价条件及 metadata E2E 口径 |
| M01 | 已完成 | 两种 MLA 的独立 reference 和 CANN 输入映射 | 512+64 共享 KV、192/128 独立 V、scale、stride、右下 mask、窗口及空行行为有独立对照 |
| M02a | 已完成 | 910B CANN 执行器 | D01/D02/P01 的 FP16/BF16 O/LSE、重复执行、布局及实际加载路径可追溯 |
| M02b1 | 已完成 | 固定 CANN 源码隔离构建与分派溯源 | vendor、host tiling、device binary 与固定源码对应，运行分派可核实 |
| M02b2 | 已完成 | 固定源码 profiler 基线工具与代表用例 | kernel、转换和调用 E2E 分开记录，后续验收按同轮交替测量 |
| M03 | 已完成 | MLA 入口识别、参数校验和未实现拦截 | 构建只含普通 FA3 kernel；MLA 真实/fake/C++/公开入口明确拒绝；契约、普通 FA3 精度/metadata/编译图及 CPU reference 回归通过 |
| M04a | 已完成（纯文档） | CANN 源实现映射及 dense 实现拆分 | design 6.1..6.5 固定源函数、字段、数值顺序、资源/事件和组件复用条件；首步范围与验收子步骤明确，源码锚点及文档检查通过 |
| M04b1 | 已完成 | 首步 dense 的无 gert 调度模块及源输出对照 | 85 项 CPU 检查通过，49,251 组输入与固定源的基本块、端点、核数、无 FD 和 workspace 完全一致；字段/资源/溢出检查及 sanitizer 通过；尚不接生产前向或 metadata |
| M04b2a | 已完成 | FP16 QK/PV 的 Cube Block 与 Tile | 155 项 910B 检查通过，FP32 QK/PV 与固定源逐元素一致并满足独立 FP64 误差界；覆盖内容/RoPE 两半、M/N/K 尾块、多 query、非零 offset、缓冲复用；四组组件性能诊断完成；不接生产前向 |
| M04b2b | 已完成 | FP16 Vector Block 与 dense Kernel | 175 项独立 910B 检查通过，O/LSE 与固定源逐元素一致并通过独立数学对照；覆盖 D01/D02、尾块、多 batch、scale、任务环及复用；155 项 Cube 回归通过，五组同轮固定 CANN 性能诊断完成；不接公共入口 |
| M04b2c | 已完成 | 首步 FP16 的 FA3 接入与构建 | 强制构建、设备头增量依赖及加载产物已核查；339 项真实/fake/C++ 契约、101 项公开/内部入口精度与生命周期、176 项 CPU 和 11 项普通 FA3 回归通过；五组同轮 CANN 公开调用诊断完成，E2E 比值约 1.61–1.63，未达最终性能门槛；BF16 及范围外组合明确拒绝 |
| M04b3 | 已完成 | 同范围 BF16 实例化及验收 | 保留源的 BF16 中间/输出舍入及 FP16 计算逻辑；513 项组件、786 项公共入口/契约、176 项 CPU 和 11 项普通 FA3 回归通过；两种 dtype 边界一致，分别完成五组同轮 CANN 性能对照；BF16 无 metadata E2E 比值约 1.60–1.64，未达最终性能门槛 |
| M05a | 已完成 | AICPU 调度组件及版本化 int32 metadata | 3,529 组设备记录与 host 一致，其中 3,494 组对照固定源；22 项设备、179 项 CPU、604 项 MLA 契约/精度和 11 项普通 FA3 回归通过；17 对象生产构建及增量依赖已核查；metadata 生成和双 dtype CANN 性能诊断完成；公开 MLA metadata 仍明确拒绝 |
| M05b1 | 已完成 | 版本化 metadata 的设备消费者及独立 dense 链路 | AICPU 记录经版本、类型、形状、平台、端点和 workspace 校验后驱动现有 Kernel；拒绝记录不触碰 O/LSE/workspace；两种 dtype 精度、复用、同轮性能诊断及普通 FA3 回归通过；公开入口继续拒绝 |
| M05b2a | 已完成 | 官方 metadata 签名与普通 FA3 兼容层 | 129 项签名/兼容/精度/编译图、622 项 MLA 与 20 项普通 FA3 回归通过；80 组旧协议逐字节一致；生产入口重建及 16 个未变对象哈希已核查；公开 MLA metadata 仍拒绝 |
| M05b2b1 | 已完成 | AICPU 读取 dense cache_seqlens 并拒绝未支持长度 | 175 项 CPU、52 项 AICPU、60 项双 dtype 消费精度/拒绝、129 项签名、622 项 MLA 与 20 项普通 FA3 回归通过；仅接受每个长度等于 KV 容量，失败记录清零 payload；生产 AICPU 重建和生成/复用性能诊断完成；公开入口仍拒绝 |
| M05b2b2 | 已完成 | 公开接入所需的设备校验错误上报 | 98 项双 dtype 设备检查通过，含 36 个独立进程的真实异常、device/stream/event 同步和成功精度；`-DNDEBUG` 仍触发 Trap，无长度 D2H 或 steady-state 主机同步；90 项 CPU、普通 FA3 回归、公开拒绝契约及同轮 CANN 性能诊断完成 |
| M05b2b3a | 已完成 | 现有 dense 前向的生产 AICPU 生成与校验消费链路 | `flash_attn_func` 内部生成 metadata 并由生产 launcher 调用 `LoadDenseMetadata`；两种 dtype 源对照精度、运行时 scale、连续调用与 allocator/stream 生命周期、生产构建、普通 FA3 回归及同轮性能诊断通过；显式 metadata 入口继续拒绝 |
| M05b2b3b | 已完成 | 已验收 dense 范围的公开 MLA metadata 生成与复用 | 开放官方 `get_scheduler_metadata` 和 `flash_attn_with_kvcache` 的只读连续 BSND 子集；真实/fake/C++ 边界、完整 dense 长度校验、运行时 scale、有/无 metadata、连续复用及 allocator/stream 生命周期正确；两种 dtype 精度、普通 FA3 回归及同轮性能诊断通过 |
| M05c1 | 已完成 | dense MLA kvcache 的精简 C++ dispatcher | 真实/fake/compile、参数拒绝、双 dtype 源对照精度、生命周期和普通 FA3 回归通过；1,595 个去重测试通过；十个代表用例的复用 E2E 比值为 0.746–0.775，生成加一次前向仍为 1.525–1.608，后续由 M05c2 完成完整调用段验收 |
| M05c2 | 已完成 | 精简 metadata dispatcher 与公开 dense qv 准备优化、完整 E2E 验收 | 1,789 项最终检查通过；相同固定 CPU 亲和性下，两种 dtype 各五组的复用前向、生成加一次/四次前向逐例均满足同轮 CANN E2E 比值 <= 1.0，metadata 生成和 Kernel 时间分别记录 |
| M04c1 | 已完成 | dense 的 S1G 多块，解除首步 `Sq*Hq<=256` 限制 | 2,008 项检查通过；51,129 组源调度一致；双 dtype 的 257/511/512/513 行、非整齐 M、跨核尾块、重复执行、metadata 和普通 FA3 回归通过；原五组加六组多块的复用及 R=1/4 E2E 逐例 <=1.0，KV 仍为 1..512 |
| M04c2 | 已完成 | dense 的 KV 513..1024、显式 num_splits=1 | 2,113 项检查通过；58,689 组源调度对照，双 dtype 各 31 组源 Kernel O/LSE 逐元素一致；保留 aMLA nUpdate、PV 原子加和事件依赖，完整 M 端点映射禁止 FD；原 11 组加 7 组新用例的复用及 R=1/4 E2E 逐例 <=1.0 |
| M04c3 | 已完成 | dense KV 1..16384，含 8193 | 2,217 项去重检查通过，78,345 组源调度对照；双 dtype 各 26 组非 FD 源 Kernel 对照完成，同路径各 13 组公开 metadata 复用及 R=1/4 E2E 逐例 <=1.0；显式 num_splits=1 |
| M04c4 | 已完成 | dense 全局空 Q/K/V 容量边界 | 2177 项检查通过；全部空容量组合的 O/LSE、metadata 校验、fake/compile、跨流及普通 FA3 正确；42 项真实 Trap 通过；双 dtype 各 13 组同路径公开 E2E 逐例 <=1.0，噪声用例补测闭环；样本内零长度留给 M06 |
| M06 | 已拆分，见下表 | 512+64 非分页 TND、实际长度和外层 stride | 按 M06a..M06e2 逐项完成；不均匀 Q/KV、seqused、前缀和、零长度、尾块与 poison padding 通过；steady-state 无长度 D2H；相关 E2E 通过 |
| M07 | 待开始 | 512+64 的 ND paged cache | BSND/TND query、乱序页表、尾页、源有效页大小、K/V 独立首维 stride；精度及 PA E2E 通过 |
| M08 | 待开始 | 512+64 的右下 causal | Sq<Sk、Sq>Sk、多 query、空行、PA/TND 与 metadata 一致；精度及 E2E 通过 |
| M09 | 待开始 | 512+64 的局部窗口 | 有限/单侧窗口、causal 叠加、FA3 边界和页边界；精度及 E2E 通过 |
| M10 | 待开始 | 512+64 的 split-KV 与合并 | 按源分割和合并流程适配自动/1/显式 split；dense/TND/PA、空 split、O/LSE/accumulator 与 metadata 正确；长 KV E2E 通过 |
| M11 | 待开始 | 192/128 prefill BSND 前向与 metadata | 按源通用非量化路径适配 qv=None、独立 K/V、分组头和真实 stride；O/LSE 及 E2E 通过 |
| M12 | 待开始 | prefill TND 与实际长度 | 非均匀长度、seqused、边界和 GQA 与 reference 一致；精度及 E2E 通过 |
| M13 | 待开始 | prefill ND paged cache | 页寻址、内容/RoPE 真实 stride、精度及 PA E2E 通过 |
| M14 | 待开始 | prefill mask 与 split 组合 | 右下 causal/window、split 合并以及 metadata 组合通过；按组合拆成独立验收步骤 |
| M15 | 待开始 | MLA softcap 与调度选项适配 | 明确各项 FA3 契约映射；softcap、has_softcap、pack_gqa、sm_margin、资源校验及 E2E 通过 |
| M16 | 待开始 | 图执行与 metadata 生命周期 | eager/图执行一致，图内长度更新后 metadata 正确；跨层复用、多 stream、fake 与真实输出一致 |
| M17 | 待开始 | 全范围验收和用户示例 | 必迁能力均有用例，精度、逐例 metadata E2E 及普通 FA3 回归通过；README 给出调用方式和支持边界 |

M04b 的临时范围与验收切面见 [design 6.2](design.md#62-调度字段和首步边界) 和 [6.5](design.md#65-必要的-fa3-差异与验收切面)：连续 BSND、B>0、`1<=Sq*Hq<=256`、`1<=Sk<=512`、Hkv=1、无 mask、显式 `num_splits=1`；先接 `flash_attn_func`。FP16/BF16 分两次接通，各自验收前保持明确拒绝。M04c 的范围扩展不改变最终需求；TND、PA、mask、split 等仍按后续行推进。

M04c3 的性能验收只覆盖已迁移路径：固定 26 组形状和双 dtype 均完成非 FD 源 Kernel 对照；按固定源调度判定的各 13 组同路径公开调用，metadata 复用和生成加 R=1/4 前向均达到 E2E 门槛。其余原生 CANN 自动 FD 用例的差距保留为诊断，不阻塞非 FD 迁移。M10 保留原顺序，当前 `num_splits=0/>1` 继续明确拒绝。测量范围、结果及证据见 [status](status.md)。

M04b1 是有独立验收的调度实现，不创建未实现 kernel 或返回占位 metadata。源对照从固定 commit 读取原算法，覆盖 design 6.5 的边界；该调度尚未被生产构建引用时不要求重跑 NPU。M04b2a/b 的测试驱动只用于设备组件验收，不提供 MLA 前向接口；两步均须完成相关 910B 精度和性能诊断。M04b2c 接入生产时必须强制核查构建依赖、实际加载扩展和源组件对应关系。每个实现步骤保持源设备行为，组件替换按运算、数据布局、精度和资源/同步契约核对等价性。

M05a 只提供内部 AICPU 生成组件和测试驱动，输出实际调度字段而非占位数据，不另造公开协议。调度和分核算法与 host 共用源码；不包含 scale、张量地址或跨调用可变 scratch。M05b 按设备消费（M05b1）、官方签名与旧调用兼容（M05b2a）、公开生成及消费（M05b2b）拆开验收。M05b2b 再分为设备长度读取（b1）、设备错误上报（b2）及公开接入（b3）；b3 进一步按生产内部链路（b3a）与公开生成/复用（b3b）拆分。官方 `flash_attn_func` 没有显式 metadata 参数，公开复用需要接通 `flash_attn_with_kvcache` 的只读 dense 子集及完整长度校验；b3b 已完成上述 dense 子集的公开生成与复用；生成和消费均校验全部 dense 长度，显式复用不再生成记录，其余组合仍明确拒绝。普通 FA 的 byte 协议仍保存 softcap 数值，单独 has_softcap=True 明确拒绝；具体兼容规则见 design 第 5 节。M05b2b 接通后仍以 M05c 完成作为整个 M05 的完成条件；M04c1 在此之前不开始。

E2E 优化集中于接口、调度计算位置、metadata 复用、重复准备与发射成本。设备计算耗时用于检查源行为一致性。新增能力及时验收相应精度和 E2E，普通 FA3 回归覆盖公共入口及 metadata 变化。

M05c1 依据公开入口/C++ 调用及 Python profile 定位通用 custom-op 的参数遍历、重复检查和返回准备开销；只为已支持的只读 dense kvcache 注册精简 C++ op，真实/fake 各自保留张量校验，公共入口仍拒绝全部未支持选项。它不改变调度、metadata 协议或设备 Kernel；验收包括独立入口边界、schema/fake、compile、双 dtype 精度/生命周期、普通 FA3 和固定五组的同轮性能。M05c2 通过精简 metadata C++ dispatcher 和公开 dense qv 路径的单次张量校验完成剩余 host 优化；真实/fake 各自保留全部边界，整数长度转换前仍检查几何。调度和设备计算保持原路径，十个代表用例的 R=1/4 完整调用段及复用 E2E 均已验收。

## M06：非分页长度、TND 与 stride

沿用已确认的 [design 3.3](design.md#33-共同契约)、[4.2](design.md#42-代表用例与参考实现边界) 和 [5](design.md#5-metadata-设计)。本轮拆分只安排实现顺序，不改变范围：512+64 共享 KV、FP16/BF16、Hkv=1、无 mask、显式 `num_splits=1`，非分页 KV 容量上界仍为 16384；PA、mask、FD、192/128 和调度选项仍在 M07 之后。先完成 BSND 实际 KV 长度，再接紧凑 TND，随后处理 `seqused` 与分配区间的区别，最后接外层 stride。下表按顺序推进，每行单独验收、提交；公开能力以接入行完成为准。

| 子步骤 | 状态 | 本步产物与接入边界 | 独立完成条件 |
| --- | --- | --- | --- |
| M06a | 已完成（纯文档） | 源依赖核对、实施拆分和下一步边界 | 固定源锚点、既有入口签名、文档链接及交接证据核查通过；生产能力保持 M04c4 |
| M06b1 | 唯一下一步 | 连续 BSND、Q 使用完整容量的实际 KV 长度解析与调度；仅独立 CPU 入口，不接生产 | 每样本 `0..Sk`、非均匀/全零长度、负值/超容量/计数/溢出拒绝、固定源切块分核对照及 sanitizer 通过；整容量退化与现有 dense 调度一致 |
| M06b2 | 待开始 | BSND 实际 KV 长度的 AICPU metadata 生成和设备校验消费者；公开短长度继续拒绝 | 真正记录调度与长度快照；CPU/AICPU 一致，损坏/过期记录设备拒绝、无长度 D2H、只读复用和生命周期检查通过；旧 dense 与普通 FA3 metadata 回归通过 |
| M06b3 | 待开始 | 按样本实际 KV 长度执行的 BSND Kernel / Block / Tile，独立设备驱动 | 双 dtype 的短 KV、跨 S2 块、混合零长度、全零实际长度及 poison padding 精度通过；源组件性能对照完成，现有 dense Kernel 回归通过 |
| M06b4 | 待开始 | 公开只读 BSND `cache_seqlens`、metadata 生成及复用 | None/整数/int32 tensor、真实/fake/C++、有/无及过期 metadata、跨 stream 与 compile 正确；普通 FA3 回归及同路径公开 E2E 达标 |
| M06c1 | 待开始 | 紧凑 TND 的 `cu_seqlens_q/k` 解析、样本起点和源调度；仅独立 CPU 入口 | T01 前缀和、空样本、总容量/max 上界分别校验；起始 0、单调性、尾值、计数及溢出边界通过；BSND 等价输入与固定源累计长度对照通过 |
| M06c2a | 待开始 | TND 的 AICPU 记录与设备消费者；公开 TND 继续拒绝 | 前缀和解析与 c1 一致，分配起点/有效长度/调度边界可校验；旧记录、篡改及前缀和变更真实拒绝，无 D2H；普通 FA3 metadata 回归通过 |
| M06c2b | 待开始 | 紧凑 TND Q/KV 及 TND Q + BSND KV 的设备寻址与写回，独立设备驱动 | 双 dtype 的 T01、非均匀多 query、跨样本尾块、零样本、O/LSE 布局和非零 offset 通过；同路径源精度/组件性能及 BSND 回归完成 |
| M06c3 | 待开始 | 公开 `flash_attn_varlen_func`，以及 TND query 的只读 kvcache / metadata | 前者内部生成，后者按现有签名生成/复用；真实/fake/C++、空总容量、生命周期、普通 FA3 回归及各入口适用 E2E 达标 |
| M06d1 | 待开始 | BSND/TND 实际 Q/KV 使用长度的独立 CPU 归一与调度，补齐 `seqused_q/k` | U01、T01 的使用长度小于分配长度变体通过；有效范围由 used 限定、下一样本起点仍由容量/前缀和限定；源调度对照和 sanitizer 通过 |
| M06d2 | 待开始 | used 长度的 AICPU、设备消费与独立 Kernel 链路；公开新增组合继续拒绝 | 双 dtype 有效行、零 Q/KV、poison padding、长度更新和错误 Trap 通过；仅对官方定义的输出区域断言数值；组件性能、旧 metadata 与普通 FA3 回归完成 |
| M06d3 | 待开始 | 既有 varlen `seqused_q/k` 和内部 BSND 前向使用长度接入，不增加公共参数 | 官方实际签名、真实/fake/C++、内部生成/复用范围、有效输出、普通 FA3 及适用公开 E2E 通过；无公开等价入口的 BSND Q-used 组合只验收内部入口 |
| M06e1 | 待开始 | BSND 最后一维连续的合法外层 stride | Q/qv/K/V 各自 batch/row/head stride、独立 offset、BNSD 归一视图及 U01 组合通过；无完整 cache 复制；双 dtype 精度、源组件性能、公开 E2E 和普通 FA3 回归通过 |
| M06e2 | 待开始 | TND 最后一维连续的合法外层 stride，完成 M06 | ST01 的 token/head stride、NTD 归一视图、Q/KV 独立 offset、T01/used 与 BSND cache 组合通过；双 dtype 精度、源组件性能、公开 E2E 和普通 FA3 回归通过 |

**M06b1 的最小实现边界。** 在 FA3 的 910 MLA 目录新增独立、无 gert/torch 依赖的调度入口，接收静态 shape/platform 与执行处理器可读的 `int32[B]` 长度。先做完整合法性校验，再数值转换为固定源 `BaseInfo::actualSeqS2Size`；不能把 int32 指针解释为 int64。复用 `detail/cann_split_core.*`，保持现有生产 `BuildDenseSchedule`、401 字 v1 metadata 和公开拒绝路径不变。没有长度或全部等于容量时须回到既有结果；非零容量内的零长度须按源任务与端点规则处理，不能直接当作 M04c4 全局空容量。

M06b1 的源对照从固定 CANN commit 提取 `attention/common/op_host/split_core.cpp` 的 `GetS1SeqSize`、`GetS2SeqSize`、`SplitCore`，以及 `fia_tiling_nonquant_mla.cpp` 的 `DealSameSeqEachBatch`、`CalcMBaseSize`、`CreateSplitInput`。特别覆盖 KV 实际长度不相等时不调整 M 基块的分支；对源原生非 FD 比较完整输出，需禁止跨核 KV split 的组合单独核对 FA3 `num_splits=1` 映射和任务覆盖，不能将改装结果称为原生源输出。矩阵包含 B=0/1/多 batch、核数两侧、M=128/256 及尾块、KV 使用长度 0/1/511/512/513/8193/16384、等长短序列、非均匀和全零序列；比较基本块、端点、核数、FD 字段及 workspace，并检查无任务遗漏/重复。此步不被生产调用，只运行相关 CPU 源差分、既有 dense 调度/metadata 消费回归与 ASan/UBSan，不需要 NPU 精度或 E2E。

**后续步骤的验收约束。** 设备长度路径以固定源 `fia_kernel_nonquant_mla.h` 的 `InitActualSeqLenQ/KV`、`GetTaskDealMode` 和 `DEAL_ZERO` 分支为起点，保留原计算和同步顺序。源累计长度与 FA3 带起始 0 的前缀和分开解析；`cu_seqlens` 给分配位置，`seqused` 给有效范围，不用缩短后的累计和重排后续样本。非零 KV 容量中的样本零长度写 `O=0、LSE=-inf`，全局空 KV 的 `+inf` 沿用 M04c4；未使用的 query 槽位是否写回以固定官方实现为准，不能把未定义区域当作零行要求。

新增 metadata 布局时更新版本/类型及 wordCount 校验，并在该实现步补充 design 的具体记录契约；不能让旧 401 字记录被新语义误读。记录保留长度快照及验证所需的调度信息，不存张量地址、scale 或可变 scratch；生成和消费均验证本次长度，变化后重算。host 只依据 shape/dtype/stride 和静态容量分配，AICPU/设备读取实际值，预留容量不足明确报错，不增加 steady-state D2H 或同步。每次协议或公共入口变化补普通 FA3 回归。

固定官方 `get_scheduler_metadata` 没有 `cu_seqlens_k` 或 `seqused_q` 公共参数，`flash_attn_func` / `flash_attn_varlen_func` 没有显式 metadata 参数；按原签名安排内部生成与只读 kvcache 复用，不为了 M06 增加公开参数。外层 stride 在当次前向传入，Q/qv/K/V 分别寻址；不把 stride 或地址固定在可跨层复用的 metadata 中。

每个设备步骤先固定源用例及分派，再做双 dtype 独立高精度/低精度 reference、源 Kernel 和代表性性能对照；新增解析不能与 oracle 共用。每个公开接入步骤固定原有代表形状与新增矩阵，分别报告 metadata 生成、复用前向、生成加 R=1/4 前向和 Kernel 时间，同路径逐例 E2E 仍须 `<=1.0`。TND varlen 的内部生成计入其完整调用段，不把内部 probe 标作公开复用。源原生自动 FD 或没有同义调用的组合按 design 第 7 节单列原因和诊断，不能按耗时筛选矩阵。

## 每步最小闭环

- 开始：核对分支、工作区、环境、实际加载扩展和上步证据，确定本次唯一范围。
- 实现：代码、构建接入、源对应关系及针对性检查一起完成。
- 验证：按影响范围运行 910B 精度、性能及普通 FA3 回归，记录实际设备条件、命令、原始数据和结果；未运行项目明确标注。
- 提交：检查差异和 git diff --check，只暂存本步文件，运行仓库质量检查，更新 status，提交一次后停止。

## 构建与检查

只构建目标后端：

```bash
FLASH_ATTN_BUILD_VERSION=v3 FLASH_ATTN_BUILD_NPU=910 FLASH_ATTN_SKIP_SUBMODULE_INIT=1 python setup.py build_ext --inplace
```

header、AICPU 或编译单元集合变化时核查增量依赖，必要时使用 `FLASH_ATTN_FORCE_REBUILD=TRUE`。测试记录扩展实际路径、哈希及参与链接的对象，确认与当前源码对应。设备选择依据实际健康和占用情况。

reference、来源校验、profiler 和契约检查分别使用现有 `tests/test_flash_attn_npu_v3_mla_reference.py`、`tests/test_flash_attn_npu_v3_mla_provenance.py`、`tests/test_flash_attn_npu_v3_mla_profile.py` 与 `tests/test_flash_attn_npu_v3_mla_contract.py`。精确节点和命令写入 status 指向的证据目录。纯文档步骤检查链接、范围、格式和状态一致性。
