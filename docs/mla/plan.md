# MLA 迭代计划

本文件只记录迭代顺序、每步产物与完成条件。当前公开能力、最近一次闭环和复现命令见 [status](status.md)，目标范围、协议与验收口径见 [design](design.md)。步骤编号沿用历史，M04c 在 M05 闭环后执行；不按编号重新排序。

唯一下一步为 **M06d2b：used 记录设备校验消费者**，完成条件见下。未完成工作按 M06 → M07..M17 排列，[已完成步骤](#已完成步骤) 集中放在末尾，保留原编号以便追溯提交。独立组件完成不代表公开入口开放。

## M06：非分页长度、TND 与 stride

本阶段限 512+64 共享 KV、FP16/BF16、Hkv=1、无 mask、`num_splits=1`。先 BSND 实际 KV 长度，再紧凑 TND 与混合布局，再区分 used 与分配长度，最后外层 stride；容量/字段/地址限制统一见 [design 6.2](design.md#62-调度字段与边界)，入口现状见 [status](status.md#当前可调用能力)。PA、mask、FD、192/128 与调度选项留在后续步骤。

每行单独验收、提交。长度和布局沿用“CPU 调度 → AICPU 生成 → 设备校验 → 独立 attention → 公开接入”的顺序；混合布局已接通公开只读 kvcache，used 继续按同一顺序推进。

### 下一步：M06d2b

M06d2a 已完成 v5 分配/used 记录和独立 AICPU 生成。M06d2 继续按生成、消费、独立 attention 分三个提交；下一步只完成 v5 的设备校验消费者，不接 attention 或公开入口。

- 使用独立 context 和返回类型，先校验 v5 字数、版本/类型、状态、dtype，再核对布局、容量、used 标记、完整调度和资源；拒绝旧协议。
- 逐项对照当次分配前缀与可选 used，检查两轴 64 位物理起点、分配/实际长度、变长端点和零成本后缀；无前缀/used D2H，空容量仍完整校验。
- 损坏、过期、错布局/标记/计数/对齐的记录由 release 设备真实 Trap 拒绝，拒绝前不写输出；覆盖 U01/T01、零 Q/KV、跨 stream 和 allocator 生命周期。
- CPU sanitizer、构建依赖、旧 metadata 和普通 FA3 回归通过。有效 attention 输出与公开签名分别留在 M06d2c、M06d3。

### M06 剩余步骤

| 子步骤 | 状态 | 本步产物与接入边界 | 独立完成条件 |
| --- | --- | --- | --- |
| M06d2b | 唯一下一步 | used 记录设备校验消费者 | 逐项复核分配前缀/used、端点/资源；损坏/过期记录真实 Trap、拒绝不写输出，生命周期及旧路径回归通过 |
| M06d2c | 待开始 | used 的独立 Kernel 链路 | 双 dtype 有效行/零 Q/KV/poison padding/长度更新/Trap；仅检查官方定义输出区，组件性能与 metadata/普通 FA3 回归通过 |
| M06d3 | 待开始 | varlen `seqused_q/k` 与内部 BSND Q-used 接入 | 不增加公共参数，签名/真实/fake/C++/有效输出/普通 FA3 与适用 E2E 通过；无公开等价的 BSND Q-used 只验收内部入口 |
| M06e1 | 待开始 | BSND 最后一维连续的合法外层 stride | Q/qv/K/V 独立 batch/row/head stride/offset、BNSD 视图与 U01；无完整 cache 复制，精度/性能/普通 FA3 通过 |
| M06e2 | 待开始 | TND 合法外层 stride，完成 M06 | ST01 token/head stride、NTD 视图、独立 offset、T01/used/BSND cache 组合；精度/性能/普通 FA3 通过 |

M06 的实现依据集中在 [metadata 协议](design.md#5-metadata-设计)、[源调度与零尾部](design.md#62-调度字段与边界)、[用例与 reference](design.md#42-代表用例与参考实现边界) 和 [验收矩阵](design.md#71-固定性能矩阵与阶段边界)。生成记录、校验记录和执行 attention 分别验收；全部样本实际长度为零也不等于全局空容量。

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

## 已完成步骤

以下只保留步骤编号、产物与完成边界，便于定位历史提交；早期临时限制已由后续步骤扩展。当前公共支持见 [status](status.md#当前可调用能力)，历史测试数量、性能和排障过程用 `git log -- docs/mla/status.md` 查找，不在计划重复维护。文档整理不改变任何功能步骤的状态。

### 范围、基线与 BSND 接入

| 步骤 | 已完成产物与边界 |
| --- | --- |
| M00 | 固定范围、来源、FA3 契约及 metadata E2E 口径 |
| M01 | 独立双 reference 与两种表示的 CANN 映射，覆盖数学/scale/stride/mask/空行 |
| M02a | 910B CANN 执行器；D01/D02/P01 双 dtype O/LSE 与运行溯源 |
| M02b1 | 固定 CANN 源隔离构建；vendor、host tiling、device binary 对应固定源码 |
| M02b2 | profiler 基线与代表矩阵；Kernel、转换、调用 E2E 分开计量 |
| M03 | MLA 识别、校验与未实现拦截；真实/fake/C++/公开入口及普通 FA3 回归 |
| M04a | dense 源映射与拆分设计；函数/字段、数值/资源/事件和组件复用条件 |
| M04b1 | 无 gert 的 dense CPU 调度；源字段差分、边界与 sanitizer |
| M04b2a | FP16 Cube Block / Tile；独立精度与源组件性能 |
| M04b2b | FP16 Vector Block / dense Kernel；源/数学精度、任务环及性能诊断 |
| M04b2c | FP16 flash_attn_func 与构建接入；依赖/加载核验、入口与普通 FA3 回归 |
| M04b3 | BF16 实例；源舍入/数值路径、双 dtype 边界及性能诊断 |
| M05a | AICPU v1 生成；CPU/固定源一致性、生命周期、构建与回归 |
| M05b1 | v1 独立设备消费者；几何/平台/端点/workspace 校验与拒绝不写输出 |
| M05b2a | 官方 metadata 签名与普通 FA 兼容；旧数值关键字、fake/compile、byte 协议 |
| M05b2b1 | AICPU 读取整容量长度；失败记录清零 payload |
| M05b2b2 | 消费者 release 真实 Trap；拒绝不写输出，无长度 D2H/主机等待 |
| M05b2b3a | 生产内部 v1 生成/消费；scale、精度和生命周期 |
| M05b2b3b | 公开 metadata 与只读 kvcache 复用；入口/普通 FA3 回归及性能诊断 |
| M05c1 | 精简前向 dispatcher；入口/compile/生命周期回归，复用 E2E 达标 |
| M05c2 | 精简 metadata dispatcher；双 dtype 原五组复用及生成加 R=1/4 E2E 逐例达标 |
| M04c1 | S1G 多块；解除 256 行临时限制，精度及适用 E2E 达标 |
| M04c2 | KV 513..1024 与无 split 端点映射；源数值/事件保留，精度及适用 E2E 达标 |
| M04c3 | KV 上界 16384；固定 26 组同路径组件及适用公开 E2E 达标 |
| M04c4 | 全局空 batch/query/KV；空 shape、O/LSE、记录校验、Trap 与适用 E2E |

### M06 已完成子步骤

| 步骤 | 已完成产物与边界 |
| --- | --- |
| M06a | 长度/布局源依赖与实施拆分；源锚点、官方签名和步骤边界核对 |
| M06b1 | BSND 实际 KV 长度 CPU 调度；Q 整容量，固定源差分、零尾部及 sanitizer |
| M06b2a | v2 独立 AICPU 生成；失败清零、缓冲/生命周期、构建及旧路径回归 |
| M06b2b | v2 设备校验；当前长度、零尾部、损坏/过期记录真实 Trap |
| M06b3 | BSND 实际长度独立 attention；双 dtype、零样本/poison、源组件性能 |
| M06b4 | 公开只读 BSND cache_seqlens 与 v2 复用；入口/普通 FA3 回归及适用 E2E 达标 |
| M06c1 | 紧凑 TND 前缀、起点和 CPU 调度；固定源差分、边界及 sanitizer |
| M06c2a1 | v3 独立 AICPU 生成；前缀/容量/max 上界快照、失败清零、生命周期 |
| M06c2a2 | v3 设备校验；前缀/端点/资源、真实 Trap 与拒绝不写输出 |
| M06c2b1 | 紧凑 TND 独立 attention；双 dtype、空容量/offset、O/LSE 与源组件性能 |
| M06c2b2a | 混合布局 CPU 调度；Q 前缀、64 位物理 KV 起点、容量/长度分离 |
| M06c2b2b | 混合 v4 独立 AICPU 生成；完整布局快照、失败清零与生命周期 |
| M06c2b2c | v4 设备校验；Q 前缀/KV 长度/容量/端点、真实 Trap 与旧路径回归 |
| M06c2b2d | 混合独立 attention；v4 校验与 64 位寻址、双 dtype/空行/poison/生命周期及组件性能 |
| M06c3a | 公开紧凑 TND varlen；内部 v3、入口/普通 FA3 回归，逐次生成的 R=1/4 E2E 达标 |
| M06c3b | 公开混合 v4 生成；AICPU 从 Q 前缀推导总量，入口/生命周期/普通 FA3 回归 |
| M06c3c | 公开只读混合 kvcache；内部 v4 与显式复用，适用七例各调用段 E2E 达标 |
| M06d1 | 分配/used CPU 归一与调度；64 位起点不重排、源差分/sanitizer/已有链路回归；未接 used 协议、设备或公开入口 |
| M06d2a | v5 分配/used 记录与独立 AICPU 生成；CPU/设备/固定源字段、失败清零、64 位跨度、生命周期、构建及旧路径回归；未接 used 消费者、attention 或公开入口 |
