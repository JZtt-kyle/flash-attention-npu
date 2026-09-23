# MLA 迭代计划

状态：待确认的候选拆分；M00 仅完成分析与文档。用户确认 [design.md](design.md) 后启用实现步骤。
进度入口为 [status.md](status.md)。用户确认记录一次即可，后续不逐步重复审批。

## 提交粒度

每行是一项独立闭环，完成一行后一个 commit 并停止。后续实现若发现一行仍过大，在开始前拆成如 M07a/M07b，沿用同一张表。
新增能力只在本步验证通过的范围内接通，其余组合明确报错；不提交“测试全部 skip”或只有空壳分派的功能支持。

| 步骤 | 状态 | 单次产物 | 本步完成条件 |
| --- | --- | --- | --- |
| M00 | 文档完成，待确认 | 范围、接口适配、文档约定和候选计划 | 来源固定，关键冲突有处理结论，文档链接及空白检查通过 |
| M01 | 待开始 | 两种 MLA 的独立 reference、CANN↔FA3 输入映射及源能力用例清单 | CPU 小例核对两项点积、共享 KV 边界、192/128 表示、scale、右下 mask；每项源能力有来源和纳入/排除结论 |
| M02 | 待开始 | 910B CANN 基线执行器与可重复性能测量 | 核实实际 CANN 算子版本；两种表示、两种 dtype 的参考精度及代表性计时可复现；重建并通过现仓普通 FA3 最小回归 |
| M03 | 待开始 | MLA 入口识别、形状/stride 参数和明确的未支持拦截 | `qv` 不再被静默忽略；无效组合和未接通组合明确报错；fake/真实契约及普通 FA3 回归通过 |
| M04 | 待开始 | 512+64、BSND、无 mask、无 split 的 CATLASS 前向 | FP16/BF16，单/多 query，O/LSE 与 reference、CANN 一致；选定 dense 用例性能不退化；不走生产 CANN 回退 |
| M05 | 待开始 | 上述路径的 MLA metadata：AICPU 调度、官方参数入口和运行时 scale 分离 | 有/无 metadata 等价；默认与自定义 scale、连续多次复用正确；常规 FA byte metadata 兼容回归通过 |
| M06 | 待开始 | 512+64 非分页 TND、真实长度与外层 stride | 不均匀 Q/KV、`seqused_*`、前缀和、零长度、尾块与 poison padding 通过；steady-state 无长度 D2H；相应用例性能通过 |
| M07 | 待开始 | 512+64 的 ND paged cache | BSND/TND query，页表乱序/尾页、16..1024 的源有效页大小、K/V 独立首维 stride；精度、越界保护和 PA 性能通过 |
| M08 | 待开始 | 512+64 的右下 causal | `Sq<Sk`、`Sq>Sk`、多 query、空行、PA/TND 与 metadata 一致；精度及代表性性能通过 |
| M09 | 待开始 | 512+64 的局部窗口 | 有限/单侧窗口、causal 叠加、FA3 归一化边界、页边界；非等长用例精度及性能通过 |
| M10 | 待开始 | 512+64 的 split-KV 与合并 | 自动/1/显式 split；dense/TND/PA；空 split、O/LSE/accumulator 契约和 metadata 一致；长 KV 性能通过 |
| M11 | 待开始 | 192/128 prefill BSND 前向与 metadata | qv=None，独立 K/V，FP16/BF16、分组头、内容/RoPE stride；O/LSE 正确，prefill 性能通过 |
| M12 | 待开始 | prefill TND 与实际长度 | 非均匀长度、`seqused_*`、边界与 GQA；和 dense/reference 对齐，精度及性能通过 |
| M13 | 待开始 | prefill ND paged cache | 复用已经验收的页寻址契约，验证内容/RoPE 读取的真实 stride；PA 精度及性能通过 |
| M14 | 待开始 | prefill mask 与 split 组合 | 右下 causal/window、split 合并以及 metadata 组合通过；必要时拆开 mask 与 split 两步，不扩大单次提交 |
| M15 | 待开始 | MLA softcap 与调度选项适配 | softcap 对两个分量总分数生效；metadata 仅区分 has_softcap；pack_gqa/sm_margin 的有效语义、资源校验及性能通过 |
| M16 | 待开始 | MLA 图执行与 metadata 生命周期 | eager/图执行一致，长度在图内改变后重算 metadata 正确，跨层复用和多 stream 正确；fake shape 与实际一致 |
| M17 | 待开始 | 全范围验收和用户示例 | 清单无未处理必迁项，功能/精度全过，逐例性能不劣化，普通 FA3 回归通过；README 展示 FA3 接入和明确边界 |

M17 如果发现退化，按具体用例/瓶颈拆独立优化步并验证，不把“全量验收”变成一个大型改动。M04 之后每次引入新 kernel 能力就做相关性能对比，不能将性能风险留到最后一次。

## 每步最小闭环

- 开始：检查环境、上一步提交与证据；确定本次唯一的步骤编号。
- 实现：代码、针对性验证、必要的构建接入一起完成。测试命令选择本步用例，加受公共改动影响的最小回归。
- 结束：检查差异；更新本表状态及 status 的结果、命令、证据、下一步；一句话 commit；停止。

测试产物随功能逐步增加，拟使用 `tests/test_flash_attn_npu_v3_mla*.py`、`tests/common/mla_ref.py` 及独立 MLA 基线脚本；M00 不创建空脚手架。
原始性能数据保留用例 ID、shape、dtype、布局、mask、split、metadata 模式、软件版本和每轮计时。日志存放位置由执行步骤确定并写进 status，避免把大结果表加入代码仓。

## 构建与检查约束

需要重建时只选择目标后端；在核实当前 CATLASS 已就绪后可使用：

```bash
FLASH_ATTN_BUILD_VERSION=v3 FLASH_ATTN_BUILD_NPU=910 FLASH_ATTN_SKIP_SUBMODULE_INIT=1 python setup.py build_ext --inplace
```

涉及 AICPU/header 的改动要核查增量依赖，必要时加 `FLASH_ATTN_FORCE_REBUILD=TRUE`；不能用旧对象的成功运行作为新源码的证据。
执行验证前记录扩展 `__file__`，确认实际加载的是本次构建。设备选择每次检查后确定，不固定借用已占用或告警设备。

每步的精确 pytest 节点、设备和性能命令由该步交接记录给出，不提前编造尚不存在的脚本命令。纯文档步的验证仅限链接、范围、格式和状态一致性。
