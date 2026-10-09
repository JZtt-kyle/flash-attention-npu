# MLA 迭代计划

本文件只记录迭代顺序、每步产物与完成条件。当前公开能力、最近一次闭环和复现命令见 [status](status.md)，目标范围、协议与验收口径见 [design](design.md)。步骤编号沿用历史，M04c 在 M05 闭环后执行；不按编号重新排序。

已完成 M06c1，唯一下一步为 **M06c2a：TND AICPU 记录与设备消费者**。后续继续在已确认范围内按下表执行。

## 已完成步骤

表中完成条件描述该步闭环的范围，不是当前公共支持表。早期临时限制及诊断性能由后续步骤扩展/闭环；历史测试数量、耗时和原始失败从 Git 中对应的 status 查找，不在计划重复维护。

| 步骤 | 状态 | 产物 | 完成条件 |
| --- | --- | --- | --- |
| M00 | 已完成 | 固定范围、来源与 FA3 契约 | 交集、排除项、组件等价及 metadata E2E 口径明确 |
| M01 | 已完成 | 两种表示的独立 reference 与 CANN 映射 | 共享 KV/独立 V、scale、stride、mask、窗口与空行数学对照 |
| M02a | 已完成 | 910B CANN 执行器 | D01/D02/P01 双 dtype O/LSE、重复执行、布局和加载路径可追溯 |
| M02b1 | 已完成 | 固定 CANN 源隔离构建与分派溯源 | vendor、host tiling、device binary 对应固定源码 |
| M02b2 | 已完成 | profiler 基线与代表矩阵 | Kernel、转换与调用 E2E 分开计量，支持同轮交替测量 |
| M03 | 已完成 | MLA 识别、校验及未实现拦截 | 真实/fake/C++/公开入口拒绝一致，普通 FA3 回归通过 |
| M04a | 已完成 | 源映射与 dense 拆分（文档） | 固定函数、字段、数值/资源/事件及组件复用条件 |
| M04b1 | 已完成 | 无 gert 的 dense CPU 调度 | 源基本块/端点/核数/workspace 一致，边界及 sanitizer 通过 |
| M04b2a | 已完成 | FP16 QK/PV Cube Block 与 Tile | 独立 910B 精度、尾块/offset/复用与组件性能对照 |
| M04b2b | 已完成 | FP16 Vector Block 与 dense Kernel | O/LSE 源与数学对照、任务环及组件回归；性能诊断 |
| M04b2c | 已完成 | FP16 `flash_attn_func` 与构建接入 | 强制构建/增量依赖/加载产物、入口精度及普通 FA3 回归；性能诊断 |
| M04b3 | 已完成 | 同范围 BF16 实例 | 舍入/数值路径与源一致，双 dtype 同边界；性能诊断 |
| M05a | 已完成 | AICPU 调度生成与 v1 记录 | 设备与 CPU/固定源一致，生成生命周期、构建与回归通过 |
| M05b1 | 已完成 | v1 独立设备消费者 | 几何/平台/端点/workspace 校验，拒绝不写输出，精度与复用通过 |
| M05b2a | 已完成 | 官方 metadata 签名与普通 FA 兼容 | 官方位置参数、旧数值关键字、fake/compile 与 byte 协议回归 |
| M05b2b1 | 已完成 | AICPU 读取 dense 长度 | 仅接受整容量，失败清零 payload；设备及回归通过 |
| M05b2b2 | 已完成 | 消费者设备错误上报 | release 构建真实 Trap、拒绝不写输出，无长度 D2H/主机等待 |
| M05b2b3a | 已完成 | 生产内部生成/消费链路 | 前向内部生成 v1，launcher 校验消费；scale、精度和生命周期通过 |
| M05b2b3b | 已完成 | 公开生成与只读 kvcache 复用 | 官方入口、长度复核、真实/fake/C++、普通 FA3 和性能诊断 |
| M05c1 | 已完成 | 精简前向 dispatcher | 入口/compile/精度/生命周期通过；复用 E2E 达标 |
| M05c2 | 已完成 | 精简 metadata dispatcher 与重复准备优化 | 双 dtype 原五组复用及生成加 R=1/4 前向 E2E 逐例达标，M05 闭环 |
| M04c1 | 已完成 | S1G 多块 | 解除 256 行临时限制，源 M 基块/尾块、多块精度及适用 E2E 通过 |
| M04c2 | 已完成 | KV 513..1024 与无 split 端点映射 | 保留 nUpdate/PV 原子加/事件；同调度源精度及适用 E2E 通过 |
| M04c3 | 已完成 | KV 上界扩展至 16384 | 含 8193 尾块，固定 26 组同路径组件对照及适用公开 E2E 通过 |
| M04c4 | 已完成 | 全局空 batch/query/KV 容量 | 空 shape、O/LSE、记录校验、真实 Trap、普通 FA3 及适用 E2E 通过 |

## M06：非分页长度、TND 与 stride

512+64 共享 KV、FP16/BF16、Hkv=1、无 mask、显式 `num_splits=1`，KV 容量仍为 0..16384。先 BSND 实际 KV 长度，再紧凑 TND，再区分 used 与分配长度，最后外层 stride。PA、mask、FD、192/128 与调度选项仍按后续步骤推进。

每行单独验收、提交；独立组件完成不代表公开能力开放。b2 已拆为生成（b2a）与消费（b2b），b3 才接实际长度计算，b4 才接公开短长度。

| 子步骤 | 状态 | 本步产物与接入边界 | 独立完成条件 |
| --- | --- | --- | --- |
| M06a | 已完成（文档） | 源依赖核对与实施拆分 | 源锚点、官方签名、交接证据与步骤边界一致 |
| M06b1 | 已完成 | 独立 BSND 实际 KV 长度 CPU 调度，Q 使用整容量 | 原生非 FD/无 split 映射分开对照，零尾部、退化 dense、非法长度及 sanitizer 通过 |
| M06b2a | 已完成 | v2 序列化与独立 AICPU 生成，不接前向 | CPU/AICPU/固定源一致，失败清零、对齐/字数/重叠、生命周期、构建及旧路径回归；生成耗时诊断 |
| M06b2b | 已完成 | BSND 实际长度记录的设备校验消费者；公开短长度继续拒绝 | 版本/类型/字数、快照与当前长度、调度/资源均校验；省略尾部逐样本为零；损坏/过期记录真实 Trap，拒绝不写输出；只读复用、生命周期、旧 dense 与普通 FA3 metadata 回归通过 |
| M06b3 | 已完成 | 按样本实际 KV 长度执行的 BSND Kernel / Block / Tile，独立驱动 | 双 dtype 短 KV、跨 S2 块、混合/全零长度、端点外空行与 poison padding 精度；源组件性能及 dense 回归通过 |
| M06b4 | 已完成 | 公开只读 BSND `cache_seqlens`、metadata 生成/复用 | None/整数/int32 tensor、真实/fake/C++、有/无/过期记录、跨流与 compile；普通 FA3 及适用公开 E2E 达标 |
| M06c1 | 已完成 | 紧凑 TND 前缀和解析、起点与 CPU 调度 | T01 的起始 0、单调性、尾值、计数、空样本、总容量/max 上界/溢出；BSND 等价与固定源累计长度对照通过 |
| M06c2a | 唯一下一步 | TND AICPU 记录与设备消费者；公开 TND 仍拒绝 | 解析与 c1 一致，起点/有效长度/调度可校验；旧/篡改/变更记录真实拒绝，无 D2H，普通 FA3 metadata 回归通过 |
| M06c2b | 待开始 | 紧凑 TND Q/KV 和 TND Q + BSND KV 设备寻址，独立驱动 | 双 dtype T01、多 query/跨样本尾块/零样本/offset、O/LSE 布局；源精度/组件性能与 BSND 回归通过 |
| M06c3 | 待开始 | 公开 varlen 及 TND query 只读 kvcache / metadata | varlen 内部生成、kvcache 按原签名复用；真实/fake/C++、空总容量、生命周期、普通 FA3 与适用 E2E 达标 |
| M06d1 | 待开始 | BSND/TND 实际 Q/KV 使用长度 CPU 归一与调度 | U01/T01 used 小于分配长度；起点不随 used 重排；源对照与 sanitizer 通过 |
| M06d2 | 待开始 | used 的 AICPU、设备消费与独立 Kernel 链路 | 有效行/零 Q/KV/poison padding/长度更新/Trap；仅检查官方定义输出区，组件性能与 metadata/普通 FA3 回归通过 |
| M06d3 | 待开始 | varlen `seqused_q/k` 与内部 BSND Q-used 接入 | 不增加公共参数，签名/真实/fake/C++/有效输出/普通 FA3 与适用 E2E 通过；无公开等价的 BSND Q-used 只验收内部入口 |
| M06e1 | 待开始 | BSND 最后一维连续的合法外层 stride | Q/qv/K/V 独立 batch/row/head stride/offset、BNSD 视图与 U01；无完整 cache 复制，精度/性能/普通 FA3 通过 |
| M06e2 | 待开始 | TND 合法外层 stride，完成 M06 | ST01 token/head stride、NTD 视图、独立 offset、T01/used/BSND cache 组合；精度/性能/普通 FA3 通过 |

M06 的实现依据集中在 [metadata 协议](design.md#5-metadata-设计)、[源调度与零尾部](design.md#62-调度字段与边界)、[用例与 reference](design.md#42-代表用例与参考实现边界) 和 [验收矩阵](design.md#71-固定-dense-矩阵与阶段边界)。不能因 b1/b2a 已生成真实调度就绕过 b2b/b3，或把全部样本零长度当作全局空容量。

## M07 之后

以下为顺序和完成边界；进入每项前将过大的步骤拆为可独立验收的提交，不提前开放未验收组合。

| 步骤 | 状态 | 产物 | 完成条件 |
| --- | --- | --- | --- |
| M07 | 待开始 | 512+64 ND paged cache | BSND/TND query、乱序页表、尾页、源有效页大小、K/V 独立首维 stride；精度及 PA E2E 通过 |
| M08 | 待开始 | 512+64 右下 causal | Sq<Sk/Sq>Sk、多 query、空行、PA/TND/metadata；精度及 E2E 通过 |
| M09 | 待开始 | 512+64 局部窗口 | 有限/单侧窗口、causal 叠加、FA3 归一化和页边界；精度及 E2E 通过 |
| M10 | 待开始 | 512+64 split-KV 与合并 | 源分割/合并流程，自动/1/显式 split、dense/TND/PA/空 split、O/LSE/accumulator/metadata；长 KV E2E 通过 |
| M11 | 待开始 | 192/128 prefill BSND 前向与 metadata | 源通用非量化路径，qv=None、独立 K/V、分组头/真实 stride；O/LSE 及 E2E 通过 |
| M12 | 待开始 | prefill TND 与实际长度 | 非均匀长度、seqused、边界/GQA 与 reference 一致，精度及 E2E 通过 |
| M13 | 待开始 | prefill ND paged cache | 页寻址、内容/RoPE 真实 stride、精度及 PA E2E 通过 |
| M14 | 待开始 | prefill mask 与 split 组合 | 右下 causal/window、split 合并及 metadata 组合；拆分后逐项验收 |
| M15 | 待开始 | MLA softcap 与调度选项 | softcap/has_softcap、pack_gqa、sm_margin 的有效语义与资源校验；适用 E2E 通过 |
| M16 | 待开始 | 图执行与 metadata 生命周期 | eager/图一致，图内更新长度并重算，跨层复用、多 stream、fake/真实输出一致 |
| M17 | 待开始 | 全范围验收与用户示例 | 必迁能力均有用例，精度/逐例 metadata E2E/普通 FA3 回归通过；核对 910B/C 实现及资源兼容，实机验收仍只要求 910B；README 说明调用与边界 |

## 每步完成条件

遵循根目录 [AGENTS.md](../../AGENTS.md)：一次一个可独立验收的小步，必要检查完成后更新 status、只暂存本步文件、提交一次即停止。不能将失败、未运行、独立 probe 或仅已编译的对象记为公开能力。

- CPU 调度步骤运行独立固定源差分、字段/溢出边界、ASan/UBSan 和受影响回归；oracle 不调用待测实现。
- Kernel 步骤完成相关 910B 双 dtype 精度与代表性源组件性能；公共入口或 metadata 变化补普通 FA3 回归。公开接入另按 [design 第 7 节](design.md#7-验收口径) 完成同路径逐例 E2E，记录生成、复用与 R=1/4 调用段，FD 等未迁移路径单列。
- 构建变化核查 header/AICPU/实例依赖、实际加载扩展与对象哈希；设备验证前重新检查环境和占用。证据路径、复现命令及环境差异只写 status，历史由 Git 保存。
- 纯文档步骤核对源码、来源、链接/锚点、表格与三份文档的一致性，运行 `git diff --check` 和适用质量钩子；不要求重跑 NPU。
