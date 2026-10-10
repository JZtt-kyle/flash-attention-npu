# MLA 迭代计划

本文件只记录迭代顺序、每步产物与完成条件。当前可调用能力、最近一次闭环和复现命令见 [status](status.md)，目标范围、协议与验收口径见 [design](design.md)。步骤编号沿用历史，M04c 在 M05 闭环后执行；不按编号重新排序。

唯一下一步为 **M06e2b：TND 与混合布局独立 stride attention**，完成条件见下。未完成工作按 M06 → M07..M17 排列，[已完成步骤](#已完成步骤) 集中放在末尾，保留原编号以便追溯提交。独立组件完成不代表公开入口开放。

## M06：非分页长度、TND 与 stride

本阶段限 512+64 共享 KV、FP16/BF16、Hkv=1、无 mask、`num_splits=1`。先 BSND 实际 KV 长度，再紧凑 TND 与混合布局，再区分 used 与分配长度，最后外层 stride；容量/字段/地址限制统一见 [design 6.2](design.md#62-调度字段与边界)，入口现状见 [status](status.md#当前可调用能力)。PA、mask、FD、192/128 与调度选项留在后续步骤。

每行单独验收、提交。长度和布局沿用“CPU 调度 → AICPU 生成 → 设备校验 → 独立 attention → 公开接入”的顺序；混合布局已接通公开只读 kvcache，used 继续按同一顺序推进。

### 下一步：M06e2b

前置 M06e1a/b/c 已完成 BSND 四路 stride 的 CPU 校验、独立设备与生产接入；M06e2a 已完成 TND CPU 地址校验。下一步只处理 TND 与混合布局的独立 stride attention，覆盖 ST01 token/head stride、NTD 视图、四路独立 offset，以及 T01/used/BSND cache 组合；保持最后一维连续和既有功能范围。

M06e2 按“CPU 地址校验 → 独立设备 → 生产接入”拆分。已完成的 M06e2a 复用 BSND 完整 storage/字节边界校验，将 TND token/head stride 映射为单 batch 的 row/head stride；独立大整数 oracle、真实 PyTorch 视图、分配前缀/used/BSND cache 地址组合及 ASan/UBSan 已通过。该 CPU 组件未接入 Kernel、metadata 或生产入口。

M06e2b/c 继续复用 64 位寻址与现有 Kernel / Block / Tile，完成双 dtype 910B 源/数学精度、代表性性能、适用公开 E2E 和普通 FA3 回归；不提前开放未验收的组合。

### M06 剩余步骤

| 子步骤 | 状态 | 本步产物与接入边界 | 独立完成条件 |
| --- | --- | --- | --- |
| M06e2b | 唯一下一步 | TND 与混合布局独立 stride attention；v3/v4/v5 消费原视图 | 双 dtype 910B 源/数学精度，含 used poison、真实大地址和 65536 边界；固定源组件性能、连续及普通 FA3 回归；生产仍拒绝未接通组合 |
| M06e2c | 待开始 | TND stride 生产接入，完成 M06；公开 varlen、混合只读 kvcache、内部 used | 真实/fake/C++/compile、完整 storage 拒绝、长度/metadata 更新和生命周期；双 dtype 精度、生产 Kernel 对照、适用公开 E2E 及普通 FA3 回归通过 |

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

以下只保留步骤编号、产物与完成边界，便于定位历史提交。各行描述该步交付的部分，不能单独作为当前支持清单；早期组件步骤也不代表当时已经开放入口。共同验证要求见[每步完成条件](#每步完成条件)，历史数量、性能和排障用 `git log -- docs/mla/status.md` 查找。文档整理不改变任何功能步骤的状态。

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
| M06b2a | v2 独立 AICPU 生成；长度快照、失败清零与异步生命周期 |
| M06b2b | v2 设备校验；当前长度、零尾部、损坏/过期记录真实 Trap |
| M06b3 | BSND 实际长度独立 attention；零样本初始化与 poison padding |
| M06b4 | 公开只读 BSND cache_seqlens 与 v2 复用；适用 E2E 达标 |
| M06c1 | 紧凑 TND 前缀、起点和 CPU 调度；固定源差分、边界及 sanitizer |
| M06c2a1 | v3 独立 AICPU 生成；前缀/容量/max 上界快照、失败清零、生命周期 |
| M06c2a2 | v3 设备校验；前缀/端点/资源、真实 Trap 与拒绝不写输出 |
| M06c2b1 | 紧凑 TND 独立 attention；双 dtype、空容量/offset、O/LSE 与源组件性能 |
| M06c2b2a | 混合布局 CPU 调度；Q 前缀、64 位物理 KV 起点、容量/长度分离 |
| M06c2b2b | 混合 v4 独立 AICPU 生成；完整布局快照、失败清零与生命周期 |
| M06c2b2c | v4 设备校验；Q 前缀/KV 长度/容量/端点、真实 Trap 与旧路径回归 |
| M06c2b2d | 混合独立 attention；v4 校验、64 位寻址与空行初始化 |
| M06c3a | 公开紧凑 TND varlen；内部 v3，逐次生成的 R=1/4 E2E 达标 |
| M06c3b | 公开混合 v4 生成；AICPU 从 Q 前缀推导总量 |
| M06c3c | 公开只读混合 kvcache；内部 v4 与显式复用，适用七例各调用段 E2E 达标 |
| M06d1 | 分配/used CPU 归一与调度；独立保存两轴 64 位分配起点、容量和实际长度 |
| M06d2a | v5 协议与独立 AICPU 生成；四数组快照、失败清零与 64 位跨度 |
| M06d2b | v5 设备消费者；四数组、物理跨度、端点/资源校验与 release Trap |
| M06d2c | v5 三布局独立 attention；按分配起点和 used 执行，共用源数值路径并验收组件性能 |
| M06d3a | 内部 BSND Q-used 与可选 KV-used；每次生成/消费 v5，完成入口验收及生产 Kernel 对照；无公开等价入口 |
| M06d3b1 | 公开 varlen 与内部 TND Q/KV 可选 `seqused_q/k`；任一 used 存在时使用 v5，完成入口、四数组更新/生命周期、组件性能与适用公开 R=1/4 E2E 验收 |
| M06d3b2 | 内部 TND Q + BSND KV 的 Q-used 与可选 KV-used；每次内部生成/消费 v5，复用 TND used launcher，完成入口、前缀/used 更新与生命周期、设备拒绝、双 dtype 同调度源 Kernel 对照及普通 FA3 回归；公开混合 kvcache 仍为 v4 |
| M06e1a | BSND 独立 CPU stride 描述与地址校验；四路真实 stride/offset、整个分配视图的 storage/字节界限、空/单例/广播/重叠、大整数 oracle 与 ASan/UBSan；生产非连续输入仍拒绝 |
| M06e1b | BSND 独立 stride attention；四路 64 位地址接入 v1/v2/v5，保留源数值/任务环/同步，修复 65536 多矩阵边界，完成真实大地址/广播/重叠的双 dtype 910B 精度、连续与普通 FA3 回归及固定组件性能对照；生产非连续输入仍拒绝 |
| M06e1c | BSND stride 生产接入，完成 M06e1；整容量 v1、只读 cache 实际 KV 长度 v2 与内部 Q-used v5，真实/fake/C++/compile、完整 storage 拒绝、长度/记录更新与生命周期；双 dtype 源/数学精度、同调度生产 Kernel 对照、适用公开 metadata E2E 及普通 FA3 回归通过 |
| M06e2a | TND 独立 CPU stride 描述与地址校验；复用 BSND 完整 storage/字节边界检查，覆盖 ST01/NTD、四路独立 offset、T01/used/BSND cache 地址组合、空/单例/广播/重叠、大整数 oracle 与 ASan/UBSan；生产非连续 TND/混合输入仍拒绝 |
