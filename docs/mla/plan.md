# MLA 迭代计划

本文件只记录迭代顺序、产物与完成条件。当前能力与证据见 [status](status.md)，契约与验收口径见 [design](design.md)。接手只需先看 [M07e2b](#下一步m07e2b)；其余内容按 [M07 剩余步骤](#m07-剩余步骤)、[后续阶段](#m07-之后)、[共同完成条件](#每步完成条件)、[已完成里程碑](#已完成步骤) 查询。

M06 已完成，M07 完成到 M07e2a 的 BSND PA 静态发射计划；公开 v6 metadata 已接通，BSND/TND PA attention 仍为独立组件。M07e2 的静态参数、设备发射和公开 E2E 拆开验收；唯一下一步为 **[M07e2b：BSND PA 生产实例与内部前向](#下一步m07e2b)**，完成后依次推进 M07e2c、M07e3、M08..M17。

## M07：ND paged cache

本阶段限 FP16/BF16、512+64 共享 KV、Hkv=1、无 mask、`num_splits=1`，接入 BSND/TND query 和内部 Q-used。接口、页大小、长度、物理地址和来源见 [design 6.4.3](design.md#643-nd-paged-cache-契约与来源)，新增记录的职责见 [metadata PA 约束](design.md#51-签名与复用规则)。64 种源页大小、两路 cache 独立 stride、页表行 stride 均在范围内；CANN 原生只支持部分外层 stride 时，测试侧转换计入公开 E2E。

### 下一步：M07e2b

前置产物为 [BSND PA 静态计划](../../csrc/ascend910/flash_attn_npu_3/mla/mla_paged_plan.h)、[BSND/TND PA attention](../../csrc/ascend910/flash_attn_npu_3/mla/kernel/mla_paged.hpp)、[公开 v6 metadata](../../csrc/ascend910/flash_attn_npu_3/mla/mla_forward.cpp) 和 [v6 设备消费者](../../csrc/ascend910/flash_attn_npu_3/mla/kernel/mla_paged_metadata.hpp)，最近闭环证据见 [status](status.md#本轮闭环与证据)。M07e2b 只完成 BSND PA 的生产实例与内部前向：

- 将静态计划、PA Kernel 的 FP16/BF16 实例及完整依赖纳入生产构建，核对实际加载扩展与对象；复用现有 Kernel/Block/Tile，不修改 CANN 源或 CATLASS。
- 内部 BSND Q + PA KV 支持整容量、实际 KV 长度和 Q-used，生成或校验 v6 后发射。Q-used 每次生成记录，不开放显式复用；沿用静态单核/全核和 workspace 预算，不读回长度或 metadata，不 gather 完整 cache。
- host 校验真实 tensor 的 rank/dtype/device、四路完整 storage、页表、长度/metadata 描述符及地址；复用当前流/AICPU 的依赖和异步持有。每核检查当次长度及全部有效页；空表/池、KV 全零和未使用 Q 保持已定义行为。
- 完成全部 64 种 P、独立 stride、零值、非法页号/stale metadata 的真实 Trap、记录跨合法页表/物理池/stride 复用和生命周期验收；910B 双 dtype 源/双数学 O/LSE、生产 Kernel 代表性能，以及普通 FA3、非分页和 PA metadata 回归。
- 公开 kvcache 的 Python/dispatcher/raw C++/fake 入口继续拒绝 PA；TND PA 生产前向留在 M07e3。本步必要验证完成后更新状态，提交一次即停止。

M07e2b/M07e2c 合计完成原 M07e2 的 BSND Q 只读 PA 前向范围；M07e2c 再开放公开入口并完成公开 E2E：

- `flash_attn_with_kvcache(page_table=...)` 接通 BSND Q 的只读 PA，支持 None/整数/tensor KV 长度和显式 v6 复用；不新增公开 Q-used 参数或开放 Q-used 显式复用。TND PA 前向留在 M07e3。
- Python/dispatcher/raw C++/fake 接受范围一致；覆盖公开路径的全部页大小、storage/页表/空值、记录复用和协议互斥，补 compile、流生命周期与普通 FA3/非分页/PA metadata 回归。
- 910B 双 dtype 源/双数学 O/LSE、生产 Kernel 代表性能，以及公开生成/复用/内部生成的 R=1/4 E2E 逐例验收；CANN 转换计入公开 E2E，FD 原生分派单列。必要验证完成后更新状态，提交一次即停止。

### M07 剩余步骤

每行是一次独立提交；只有对应生产接入验收后才开放该组合。独立 CPU、生成器、设备消费者和 attention probe 不代表公开前向；公开生成器也不代表 PA attention 已接入。

| 子步骤 | 状态 | 本步产物与接入边界 | 独立完成条件 |
| --- | --- | --- | --- |
| M07e2b | 唯一下一步 | BSND PA 生产实例与内部前向，含 Q-used | 使用静态计划和现有 PA Kernel/消费者；910B 双 dtype 源/双数学 O/LSE、全部页大小、有效页/空值/真实 Trap、生命周期、metadata 跨页表/物理池复用、生产 Kernel 性能及普通 FA3 回归；公开 kvcache 继续拒绝 |
| M07e2c | 待开始 | BSND 只读 PA kvcache 公开接入 | 公开/dispatcher/raw C++/fake 一致；None/整数/tensor 长度、显式复用、compile/生命周期；双 dtype 公开生成/复用/内部生成 R=1/4 E2E、普通 FA3 和相关非分页/PA metadata 回归 |
| M07e3 | 待开始 | TND Q 只读 PA kvcache 与内部 PA Q-used 接入，完成 M07 | 在 M07e2b/M07e2c 条件上覆盖 Q 前缀、分配/used、逻辑 B 的整数长度物化、TND fake/compile/生命周期；双 dtype 精度、生产 Kernel、公开 PA E2E 与既有全布局/普通 FA3 回归 |

PA01/PA02 的地址、Copy 与精度覆盖全部 16..1024 的 16 倍数；代表性性能保留 P=16/128/256/1024，并加入 P=112 的基本块变化和 P=528 的跨页块。矩阵在采样前固化，含 BSND/TND、短/长 KV、单/多 query、乱序页/尾页、K/V 独立首维 padding 和零长度。固定源原生非 FD、FD 与不接受的空表等情况按分派/约束单列，不按耗时筛选；独立 probe 不能替代原生 PA E2E。

## M07 之后

以下为顺序和完成边界；进入每项前将过大的步骤拆为可独立验收的提交，不提前开放未验收组合。

| 步骤 | 状态 | 产物 | 完成条件 |
| --- | --- | --- | --- |
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

只保留里程碑和已交付边界；子提交的完整产物、测试数量、性能与排障由 Git 保存，不在本页重复维护。当前支持以 status 为准。M04c 实际在 M05 完成后执行，编号保留。

| 阶段 | 已交付产物 |
| --- | --- |
| M00–M03 | 固定迁移交集和 FA3 契约；独立双 reference、固定源构建/执行器/性能基线；MLA 识别、校验及未实现拦截 |
| M04a–M04b3 | dense 来源与分层；无 gert CPU 调度；FP16/BF16 Cube、Vector、非分页 Kernel 与 BSND 生产接入 |
| M05a–M05c2 | v1 生成/设备校验与 release Trap；官方 metadata 签名、普通 FA byte 兼容；公开只读 cache/metadata 复用及 dispatcher 开销验收 |
| M04c1–M04c4 | S1G 多块、KV 上界 16384、无 split 端点映射及全局空容量；保留源数值/事件路径 |
| M06a–M06b4 | BSND 实际 KV 长度，v2 CPU/AICPU/消费者/attention 与公开只读 cache 接入 |
| M06c1–M06c3c | 紧凑 TND v3、TND Q + BSND KV v4；完成各自调度、生成、校验、attention、公开 varlen/混合 cache 和 v4 复用 |
| M06d1–M06d3b2 | 分配/used 分离的 v5，三布局生成/消费/attention；公开 varlen used 与内部 BSND/混合 Q-used |
| M06e1a–M06e2c2 | 三种非分页布局的完整 storage/stride 校验、独立组件与生产接入；跨视图 metadata、真实/fake/compile/流生命周期和适用公开 E2E |

### M07 已完成子步骤

| 步骤 | 已交付产物 | 该步结束时尚未接入的下一层 |
| --- | --- | --- |
| M07a | 固定 PA 来源、页大小、容量/长度/stride/复用契约，拆分实施步骤 | 地址组件 |
| M07b1 | 独立 CPU 地址/页内分段；完整 storage、有效页校验及固定源 Copy 参数 trace | 调度；trace 不执行设备搬运 |
| M07b2 | 独立 CPU 调度；BSND/TND Q 分配/used、非累计 KV 长度、P/C 与源 S2 分离 | metadata 编码 |
| M07c1 | v6 CPU 序列化与独立 context，精确 408+5B 字、长度快照及失败清零 | AICPU 与设备消费 |
| M07c2 | 独立 v6 AICPU；三组设备长度、TND 总 Q 推导、异步生命周期和构建依赖 | 设备消费；编入扩展不代表生产发射 |
| M07c3 | 独立 v6 CPU/910B 消费者；当前长度、全部有效页、完整跨度/资源、release Trap 及流生命周期 | 设备 Copy、attention 和公开入口 |
| M07d1 | 独立 PA Copy Tile；QK 联合内容/RoPE 与 PV 搬运，全部 64 种 P 的双 dtype L1 NZ 精度、stride/大地址、双缓冲和固定源 Copy 性能 | PA attention 和公开入口 |
| M07d2 | 独立 BSND PA attention；v6/当次页表校验、Q-used、PA Copy 与共享任务环接通，全部页大小的源/双数学 O/LSE、空值/真实 Trap/stride/大地址、非分页回归及固定源 PA/non-FD 性能 | TND PA attention 和公开入口 |
| M07d3 | 独立 TND Q PA attention；分配前缀/used 分离、非累计 KV 长度、独立 stride 和 FA3 O/LSE 布局；全部页大小的双 dtype 源/双数学精度、空值/真实 Trap/大地址、相关回归及同调度 PA/non-FD 性能 | 公开 PA metadata 和前向 |
| M07e1 | 公开 `get_scheduler_metadata(page_size=P)`，BSND/TND Q；复用 v6 AICPU，全部页大小、固定源记录、真实/fake/C++/compile、长度更新与流生命周期、旧协议及普通 FA3 回归 | BSND/TND PA 前向与公开 PA E2E |
| M07e2a | 独立 BSND PA 静态计划；完整 storage/字段/资源、全部页大小与固定源预算、v6 context、sanitizer；动态零 KV 允许保留未用 workspace，补共享消费者的 BSND/TND 精度、真实 Trap、代表性能及普通 FA3/PA metadata 回归 | 生产实例、内部前向、公开 PA 前向与 E2E |

查某个子步骤的历史完成条件及证据，例如：

```bash
git log --oneline -- docs/mla/status.md
git log -G 'M06d3b2' -- docs/mla/plan.md
git show 9f9aeb9:docs/mla/status.md
```
