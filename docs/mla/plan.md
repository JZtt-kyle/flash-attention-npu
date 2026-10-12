# MLA 迭代计划

本页只记录实施顺序、每步产物与完成条件。**M00–M07 已完成；当前先完善开发流程，唯一下一步是 W02。** 当前可调用能力与最近证据见 [status](status.md)，范围、契约及验收口径见 [design](design.md)。流程改进完成后恢复 M08，不改变迁移范围。

## 下一步：W02

用户已确认先完善开发流程。按下表逐提交实施，提交一次即停止；W01 只落地规则，统一执行器及新增缓存/恢复机制尚未实现。

| 步骤 | 产物 | 完成条件 |
| --- | --- | --- |
| W01（已完成） | 按影响范围验收、证据继承及收尾规则 | AGENTS/design/plan/status 一致，文档与质量检查通过；原有 M08 工作保留 |
| W02（下一步） | 版本管理中的最小验收入口，先接通 TND PA 公开功能测试组 | 复用已有构建/测试工具；支持预览命令、执行及结构化结果；与 M07e3c 的 `public` 组（历史 2,145 项，不含 Trap）核对用例集合、参数、成功/失败传播和实际产物身份；保留旧证据且不减少测试集合 |
| W03 | 在已接通的驱动中消除昂贵重复计算 | 优先复用同输入的双数学 reference；进入时按可独立验收范围拆分缓存、证据选择与恢复；命中/失效/结果一致性有检查，待测路径仍实测，记录改进前后耗时 |
| W04 | 调整 M08 提交边界和当前步验收清单 | 核对保留的草稿，以独立组件或可调用行为组织步骤，优先 BSND 非 PA 闭环，再扩展 TND/used 和 PA；每步标明影响路径、测试组、继承依据及完成条件 |

W02 影响范围仅为测试执行与证据组织，不改生产接口或数值核。必跑项为清单/参数对照、执行器成功与失败路径、当前构建下的 TND PA `public` 实测及质量检查；M07e3c 的历史集合与命令只作为迁移基线，不计为本次通过。首次执行前核对源码、依赖及实际加载扩展；实现时把确切命令补入当前步。断点恢复及其他布局留待后续独立步骤。

## 后续功能：M08

下列功能分解由 W04 复核后恢复实施；工作树中已有的 M08 拆分和代码草稿不表示该阶段已验收。

仅推进 **512+64 右下 causal**。开始实现前先在本计划拆成可独立验收的提交，分别处理源 mask/调度与 metadata 表达、设备数值路径、生产和公开组合；每个子步的产物与完成条件写清后再实施。

| 产物 | 完成条件 |
| --- | --- |
| FA3 右下对齐 causal | 保留固定源 Kernel / Block / Tile 路径；位置中心为 `i + Sk - Sq`，按逐样本长度解释 |
| BSND/TND/PA 及 metadata 组合 | 覆盖 Sq<Sk、Sq>Sk、多 query、空行、生成/复用；尚未接通的组合明确拒绝 |
| 精度与性能证据 | Kernel 子步完成 910B 双 dtype 固定源/双数学精度与代表性能；公开接入补普通 FA3 回归及适用 E2E |

共同数学契约见 [design 3.3](design.md#33-共同契约)，代表输入见 [MS01/MS02](design.md#42-代表用例与参考实现边界)。window、split 和 prefill 仍按下表顺序实施。

## 后续顺序

进入每项前，将过大的步骤拆成可独立验收的提交，不提前开放未验收组合。

| 步骤 | 产物 | 完成条件 |
| --- | --- | --- |
| M09 | 512+64 局部窗口 | 有限/单侧窗口、causal 叠加、FA3 归一化和页边界；精度及 E2E 通过 |
| M10 | 512+64 split-KV 与合并 | 源分割/合并流程，自动/1/显式 split、dense/TND/PA/空 split、O/LSE/accumulator/metadata；长 KV E2E 通过 |
| M11 | 192/128 prefill BSND 前向与 metadata | 源通用非量化路径，qv=None、独立 K/V、分组头/真实 stride；O/LSE 及 E2E 通过 |
| M12 | prefill TND 与实际长度 | 非均匀长度、seqused、边界/GQA 与 reference 一致，精度及 E2E 通过 |
| M13 | prefill ND paged cache | 页寻址、内容/RoPE 真实 stride、精度及 PA E2E 通过 |
| M14 | prefill mask 与 split 组合 | 右下 causal/window、split 合并及 metadata 组合；拆分后逐项验收 |
| M15 | MLA softcap 与调度选项 | softcap/has_softcap、pack_gqa、sm_margin 的有效语义与资源校验；适用 E2E 通过 |
| M16 | 图执行与 metadata 生命周期 | eager/图一致，图内更新长度并重算，跨层复用、多 stream、fake/真实输出一致 |
| M17 | 全范围验收与用户示例 | 必迁能力均有用例，精度/逐例 metadata E2E/普通 FA3 回归通过；核对 910B/C 实现及资源兼容，实机验收仍只要求 910B；README 说明调用与边界 |

## 每步完成条件

遵循根目录 [AGENTS.md 的提交门槛](../../AGENTS.md#按影响范围验收) 和 [design 的证据继承规则](design.md#73-增量验收与证据继承)。实现前在当前步补齐改动与影响路径、必跑测试组/命令及选择依据、拟继承证据及依赖、完成条件；结果与耗时只在 status 和证据目录维护。

一次一个独立组件或可调用行为，必跑检查通过后完成文档、质量检查和一个提交。新改动、失败或具体未解决风险才触发补测；阶段结束实测约定完整组合。新增公开路径、设备数值变化及公共入口/metadata 的普通 FA3 回归不能用历史记录替代。失败、未运行、独立 probe 或仅已编译的对象不能记为已开放能力。

## 已完成步骤

这里只保留已交付边界。完整子步拆分、测试数量、性能及排障证据从对应提交的 status 查询。M04c 实际在 M05 后执行，编号保留。

| 阶段 | 已交付产物 |
| --- | --- |
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

M07 的页大小精度覆盖全部 16..1024 的 16 倍数；性能矩阵及原生非 FD/FD 的边界见 [design 7.1](design.md#71-固定性能矩阵与阶段边界)。公开接入包含 None/整数/tensor KV 长度、v6 复用及相关 MLA/普通 FA3 回归，结果只在 status 维护。

历史查询示例：

```bash
git log --oneline -- docs/mla/status.md
git log -G 'M06d3b2' -- docs/mla/plan.md
git show 29ed04f:docs/mla/status.md
```
