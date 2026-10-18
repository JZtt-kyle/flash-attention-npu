# MLA 迭代计划

本页只记录实施顺序、产物和完成条件。**M17 非量化范围已收尾；用户于 2026-10-18 确认放开 INT8 契约、不迁外部 NZ，并授权拆分任务及准备无头开发。M18 数值组件已完成，唯一下一步为 M19。** 当前能力与证据见 [status](status.md)，新增契约见 [design 3.4](design.md#34-int8-压缩-mla-扩展)。这项授权覆盖 M18–M21，无需各步重复确认；尚未实现或未验收的能力不得写成已支持。M17 的图性能例外不适用于新 INT8 路径。

## INT8 后续步骤

只迁移 512+64 共享压缩 KV 的 INT8 前向，RoPE/输出为 BF16，公开缓存为 ND PA、页大小 128，query 为 BSND/TND。固定源旧 IFA 的 INT8 算法依赖 NZ PA，迁移时改为读取 ND 并在片上完成矩阵所需布局；不交付外部 NZ 格式、每次全 cache 转换或转调 CANN 的生产路径。无 mask/右下 causal、原生 FD 等实际组合先按 M18 固定源实测锁定；不把非量化的所有布局、窗口、softcap 或显式 split 自动复制成 INT8 需求。

| 步骤 | 单次提交的产物 | 完成条件 |
| --- | --- | --- |
| M18（已完成） | INT8 独立数值组件、ND 读取适配及固定源对照 | 固定源和双数学精度、INT32/舍入/缩放/FD/地址边界、适用组件性能实测；公开入口继续拒绝 INT8 |
| M19 | BSND Q + ND PA INT8 完整公开前向 | host/AICPU/Kernel/metadata/real/fake/compile/图同轮接通，必要正确性、回归及公开性能通过 |
| M20 | TND Q + ND PA INT8 完整公开前向 | 沿用 M18 数值组件和 M19 接入，完整 allocation/长度/scale 地址、公开图及性能验收 |
| M21 | INT8 组合覆盖与用户示例收尾 | 两布局及拒绝边界、累计 E2E、性能矩阵、示例与文档核对完成；唯一下一步改为 `none` |

### M18：INT8 数值与 ND 适配

已完成：INT8/BF16 数值组件、ND PA 读取、固定源和双数学对照、地址/舍入/空值/Trap、组件性能，以及共享改动的生产重建和累计回归。源语义与精度限制、组件性能差距及证据见 [status](status.md)，不把组件完成视为公开 INT8 已接通。

独立风险：既有 FP16/BF16 组件不覆盖 INT8 QK/PV 的 INT32 累加、逐 token/head Q 反量化、softmax 后动态 P 量化及舍入、量化 FD 归并；源 NZ 到目标 ND 的读取也不能靠 dtype 实例化复用。因此只此一步先交付可实机调用的组件和固定源 probe，布局、mask、split 或记录版本不再各拆一轮。

影响路径：`csrc/ascend910/flash_attn_npu_3/mla/{kernel,block,tile}/`、必要的现有数值/地址公共头；`tests/mla/support/` 的 fixed-source probe、数学 reference、native 包配方及身份依赖，新增 `tests/mla/unit/int8/`。保留源声明，CANN checkout 和 CATLASS 子模块只读。优先复用现有 PA/FD/调度组件，INT8 需要的区别按最小范围加入；生产前向和现有记录消费者不借此放开 dtype。

本轮执行细化（2026-10-18）：先在 `tests/mla/support/mla_int8_*` 和 `unit/int8/` 增加固定源 INT8 native 配方、ND/NZ 测试转换及原始输入夹具，沿用现有 native manifest/失败传播和统一驱动，不改旧非量化包。源有效性用具名用例核实 BSND/TND、Hq=1/2/4/8/16/32/64/128、Sq=1/2/4/16/32 及源实际上界、Sk=0/1/127/128/129/512/4097、sparse=0/3、直接/原生 FD；不机械展开形状轴，另保留 Hq=3、非法 Sq/mask、ND 源输入、scale dtype/shape/mode 的独立拒绝检查。源 host 与文档不一致时保存实际失败再确定迁移边界。数值组件随后增加 INT8 QK/RoPE/P 量化/PV/FD 的源 probe 与双数学，ND 页地址复用已有 PA 校验，新增首维间隔、页尾及乱序页用例。产物与数学缓存遵循 `MLA_WORK_DIR` 约定，实际路径由 status 记录。

本轮先 `int8-tools int8-source-check` 的 dry-run/collect/实测，再做组件代表值和边界成本检查，固定实现后运行 `--unit int8` 与 `int8-profile-check int8-component-profile`。新增源 native 配方须确认实际 INT8/BF16 二进制及 key 后才能进入长测。只有统一驱动核对身份与原始证据有效的既有 E2E 才可继承；本步共享头和布局校验有变，已用默认累计 E2E 加 `ordinary graph-ordinary` 实测，不累计历史单元。完成条件仍为本节全部数值、地址和组件性能闭环后一次提交；准备或源探测失败不能当成 M18 完成。

ND 实现复用 `mla_bsnd_layout`、`mla_paged_layout` 的地址描述与校验，增加 INT8 单字节存储入口；旧双字节入口及 ABI 保持。`tile/mla_copy.hpp` 的单行回退将 C0 从固定 16 改为 `32/sizeof(InputT)`，旧 FP16/BF16 结果不变。此影响需要本步 INT8 固定源地址差分、存储/溢出及 ASan/UBSan，并重建生产扩展、实测累计小输入 E2E 和普通 FA3；不能用旧扩展代替本次共享头的回归。新增 probe 直接回放固定 native 的原始 tiling，仅作为数值夹具；不新增生产 metadata 协议或手写一套 host 调度。

源探测已确认 BSND/TND 每段 Q 上界均为 32，以及普通 FD 与 `FdBalanceFlag=1` 两种归并。本步同时补齐源快速 FD 遗漏的 LSE 写回，复用已归并的 FP32 sum/max 和原有输出同步；BSND FD 的 LSE 统一为 `[B,Hq,Sq]`，空行按 FA3 归一化为负无穷。源快速 FD 的 LSE 对照使用同一分区/真实 partial 上的固定源普通归并，须证明 partial 一致后才可引用。CANN 9 的 `InitOutput` 在空 KV 连续初始化 O/LSE 时复用 stack UB，目标补写回完成同步；固定源 O 与 LSE 分别比对，并保留源空值初始化缺陷证据。非有限 scale 和当次非法页由设备 Trap 拒绝，独立进程验证；公开 real/fake/compile 继续拒绝 INT8，日常边界清单归属 `int8`。性能沿用既有组件采样器，保存全部固定输入的 kernel/probe wall 与源 NZ 准备耗时；按 design 第 7 节，它是组件对照，不是 M19/M20 的公开 E2E 门槛。

必跑检查与选择依据：

- 先注册 `--unit int8`、`int8-tools`、`int8-source-check`、`int8-profile-check`、`int8-component-profile`；在实现前用例列出源可运行的 BSND/TND、Sq/Hq/Sk 边界、sparse 0/3 组合及拒绝项。源文档与 host 上界有差异时以固定版本实测为依据，不能默默缩小集合。
- `python3 -m tests.mla --suite int8-tools int8-source-check --dry-run`，随后 `--collect-only` 和实际执行，验证原始 INT8/BF16/FP32 输入、源专用 NZ pack/unpack、真实 INT8 native 配方及其 hash/失败传播、驱动中断/依赖变化失效及单元/性能不继承；禁止先完整采样再发现 native 只有 FP16/BF16。
- `python3 -m tests.mla --unit int8`：910B 上 INT8 两次矩阵乘、BF16 RoPE/输出、FP32 scale、动态 P 量化及 FD 的源逐元素与 FP32/FP64 独立数学对照。保留源舍入、饱和、全零/极值、scale 零值与非法值处理、尾块、长序列、空行、非连续 ND 页和真实 partial。新增 CPU 地址/调度逻辑另含固定源差分、溢出与 ASan/UBSan。
- 先跑代表值、最大内存及最长 reference 成本边界，固定实现后跑完整约定集合。`python3 -m tests.mla --suite int8-profile-check int8-component-profile` 实测适用组件性能；组件结果不称公开 E2E。ND 适配成本全部计入待测路径。
- 原非量化相关 E2E 依赖未变且原始证据完整时可按统一驱动继承；共享头、构建配方或发射受影响则重跑对应 E2E/普通 FA3，核对实际加载扩展。不继承历史单元或性能。

完成条件：上述独立组件可调用、源有效组合和必要差异有证据，量化算法和 ND 地址风险闭环；未知、失败或源精度限制如实记录。更新 design 的实际 shape/组合及源适配表、plan 和 status 后单次提交，下一步 M19。数值未闭环不得先接公开入口。

### M19：BSND 公开 INT8

影响 Python 共享前向、C++ 注册/分派、host/AICPU、metadata 与 Kernel 启动、real/fake 输出，以及新增 `tests/mla/e2e/int8_bsnd/`、日常清单和公开 profile。复用 M18 完整算法、PA 安全检查及既有 metadata 生命周期；只有旧记录不能区分混合 dtype/量化模式或其安全边界时才扩协议，并写明原因。三个 INT8 scale 只在当次前向传递，不缓存到调度快照。

同步调整 M18 的 `e2e/int8` 拒绝边界，仅保留仍未接通的组合。新增 `int8-bsnd-check/source/boundaries/profile-check/profile`、`detail-int8_bsnd` 和 `--e2e int8_bsnd`。先 dry-run/collect 与工具/native 预检，再实测源与双数学、全部新行为、空值/stride/页表/长度/metadata 复用/生命周期/Trap；real/fake/compile 及真实固定/动态输入、长度和 scale 的图重放不可延后。验证不支持的选项明确拒绝，scale 更新确实改变结果。相关非量化 PA、图与普通 FA3 E2E 按依赖重跑或继承；源组件证据不替代公开调用。

本步固定并版本管理公开性能输入，包含 eager 与 graph，按配置等权平均 ≤1 分别验收，保留全部退化和不可同义比较项；与 M20 合计不超过 100 配置。先 `int8-bsnd-profile-check`，再正式 `int8-bsnd-profile`，不得沿用 M17 图性能豁免。完成完整 BSND 可调用行为、文档和质量检查后一次提交，下一步 M20。

### M20：TND 公开 INT8

同一提交完成 TND Q 的 host/AICPU/metadata/Kernel/real/fake 接入、`tests/mla/e2e/int8_tnd/`、日常清单及性能；保留原 allocation 地址，Q scale 使用 `[T,Hq]`，不能把分配前缀、实际长度和压紧后的源索引混用。复用 M18 数值组件与 M19 PA/量化参数协议，不因布局另拆数值或协议轮。

新增并按 M19 顺序运行 `int8-tnd-check/source/boundaries/profile-check/profile`、`detail-int8_tnd`、`--e2e int8_tnd`。覆盖非均匀/零长度、分配间隙 poison、scale 行/head stride、ND 页尾/乱序/越界、metadata 复用及 Trap、真实/fake/compile/动态图、跨层与 stream 生命周期。TND 公共生成器沿用 FA3 可表达的完整 Q 分配段，不凭空添加 Q-used 参数。BSND INT8、相关非量化和普通 FA3 E2E 按依赖继承或重跑，新数值/地址及公开 eager/graph 性能本步实测。通过后一次提交，下一步 M21。

### M21：INT8 收尾

按 M18 冻结的源有效组合与 M19/M20 开放边界核对两布局覆盖；补独立覆盖缺口，不自动穷举所有轴。实测 INT8 两布局专项、累计日常 E2E、相关普通 FA3 与图回归，复核实际加载、来源/版权和 910B/C 配置兼容；910C 未实测如实标记。沿用 M19/M20 固定输入重跑阶段 eager/graph 性能，最多 100 配置且平均门槛不变，所有单例、分段、布局转换成本与失败保留。

添加可运行 INT8 BSND/TND 示例及对应 E2E，中英文 README 和 design 的公开接口/拒绝项/验收映射同步；status 保存唯一当前结果和证据路径。必要检查通过才单次提交；将 status 的 `mla-loop-next` 标记设为 `none`，无头脚本正常结束。性能失败或环境受限时保留同一待办和失败结果，不能为了自动运行而提交或跳过。

## M17 全范围收尾

非量化范围已完成于 `ebdf3eb`，M16 后的专门文档核对合入同次提交。910B 正确性、eager 和调度性能通过；图平均比 1.484216023946693 未达到 ≤1，用户在完整实测后明确接受该次限制。详细结果和复现命令见 `git show ebdf3eb:docs/mla/status.md`，阶段覆盖见 [design 7.7](design.md#77-全范围收尾映射)。本次新增 INT8 授权不改变历史失败结论，也不重启 M16 待办。

## 已完成步骤

这里只保留交付边界。各布局步均包含完整调用链和必要验收；当前拒绝项以 design 第 2 节为准，旧步骤的命令、数量、耗时、失败修正及证据从对应提交的 status 查询。

| 步骤 | 已交付产物 |
| --- | --- |
| M00–M03 | 固定迁移交集/FA3 契约、独立双 reference 与固定源执行器、入口识别/校验/未实现拦截 |
| M04–M06 | 512+64 dense 数值与非 PA BSND/TND/混合生产前向，v1–v5、allocation/used、完整 storage/stride 与生命周期 |
| M07 | ND PA 地址/Copy/调度、v6、BSND/TND PA 公开 metadata/cache 及阶段收尾 |
| M08–M09 | 五布局右下 causal 与 window，v7–v16、公开接入及各阶段收尾 |
| M10 | partial/FD 归并与真实 accumulator；五布局显式/原生自动 split，v17–v26、公开入口及组合/性能阶段收尾 |
| M11–M14 | 192/128 独立 K/V、GQA 数值组件及五布局完整公开前向，v27–v31；三 mask、全部 split、真实 accumulator、allocation/used、stride、页安全与组合收尾 |
| M15a | 无同义 CANN attention 的 softcap 非线性独立组件：两种表示、双数学、舍入/零 scale 边界及真实 partial/FD |
| M15b–d | 两种表示的非 PA BSND、TND Q/KV、TND Q + BSND cache 正 softcap，v32–v40；完整公开入口、三 mask、全部 split、复用/Trap 与适用公开性能 |
| M15e | 两种表示的 BSND Q + ND PA KV 正 softcap，v41/v42/v43；完整公开入口、三 mask、全部 split、分页安全、复用/Trap 与适用公开性能 |
| M15f | 两种表示的 TND Q + ND PA KV 正 softcap，v44/v45/v46；完整公开入口、三 mask、全部 split、allocation/used、分页安全、复用/Trap 与适用公开性能 |
| M15g1 | 独立 packed/unpacked GQA 坐标、C1V2 剩余资源、源分核/显式片段及原 allocation O/LSE/accumulator 地址；仅 CPU 组件，公开接入留 M15g2 |
| M15g2 | 两种表示、五布局的 pack_gqa/sm_margin 完整公开 real/fake 接入，v47、host/AICPU/Kernel、三 mask/全部 split/softcap；修复共享 softcap 的 32-lane 比较尾部，完成公开专项、回归及固定性能集合 |
| M16 | 两种表示、五布局的固定/动态图重放、metadata 重生成、跨层/多 stream、生命周期、fake/compile 与 Trap；优化任务分配和融合发射，修复图采样与 wall 生命周期；正确性和 eager/调度性能通过，图性能按本次用户授权例外接受，失败结果保留 |
| M17 | 910B/C 平台资源兼容与 CPU/ASan/UBSan 检查、双 dtype 可运行用户示例、全范围正确性与固定性能集合收尾；同步中英文 README、覆盖映射及三份文档，图性能差距按本轮明确授权保留为已知限制 |
| 工具整理、R1、R2 | 统一验收与缓存/证据管理、模块归属与共享 helper、共用前向分派、文档轮和无头交接；原 R3 的内核可读性核对合入 R2 |

组合验收映射见 [512+64](design.md#75-51264-组合覆盖)、[192/128](design.md#76-192128-组合覆盖) 和 [全范围收尾](design.md#77-全范围收尾映射)。M14、M15a–d 五轮开发后的专门文档轮已完成，不计开发轮数。M15e、M15f、M15g1、M15g2、M16 是其后的五个已接受开发轮；当时按用户确认将随后文档核对合入 M17；本次无头循环仅继续新授权的 M18–M21。

```bash
git log --oneline -- docs/mla/status.md
git show feb189b:docs/mla/status.md  # M10 阶段证据
git show f7105ef:docs/mla/status.md  # M14 阶段证据
git show 26d7a70:docs/mla/status.md  # M15d 开发闭环与完整复现命令
```

## 迭代与收尾规则

遵循 [AGENTS.md](../../AGENTS.md#一次工作只完成一个提交) 和 [design 第 7 节](design.md#7-验收口径)。实现前在当前步补齐实际影响路径、必跑组/命令及选择依据、拟继承证据与依赖、完成条件；验证受限或失败时保留交接，不提交为已闭环。每成功完成五轮开发插入一次专门文档修正轮；文档轮不计开发轮数，手动任务按当次用户要求停止。
