# MLA 迭代计划

状态：方案已确认，M02a、M02b1、M02b2、M03、M04a、M04b1、M04b2a、M04b2b1 已完成。512+64 的连续 BSND 前向支持 KV 1..1024；双 reference、固定源码 CANN、逐例性能和普通 FA3 回归通过。M04b2b 在实现前拆为在线输出累积优化 M04b2b1 与更长 KV 调度 M04b2b2，下一步 M04b2b2。
进度、确认及证据入口为 [status.md](status.md)，源能力与代表用例见 [design.md](design.md) 4.1/4.2。后续在已确认范围内不逐步重复审批。

## 提交粒度

每行是一项独立闭环，完成一行后一个 commit 并停止。后续实现若发现一行仍过大，在开始前拆成如 M07a/M07b，沿用同一张表。
新增能力只在本步验证通过的范围内接通，其余组合明确报错；不提交“测试全部 skip”或只有空壳分派的功能支持。

| 步骤 | 状态 | 单次产物 | 本步完成条件 |
| --- | --- | --- | --- |
| M00 | 已完成，方案已确认 | 范围、接口适配、文档约定和候选计划 | 来源固定，关键冲突有处理结论，文档链接及空白检查通过 |
| M01 | 已完成 | 两种 MLA 的独立 reference、CANN↔FA3 输入映射及源能力用例清单 | CPU 71 例通过：两项点积、共享 KV 边界、192/128、GQA、scale、stride、右下 mask、窗口归一化和空行符号；源能力逐项给出来源及纳入/排除结论 |
| M02a | 已完成 | 910B CANN 执行器、安装包溯源与普通 FA3 重建回归 | 12 组 D01/D02/P01 的 FP16/BF16 O/LSE 双 reference 及重复执行通过；split/combined prefill 通过；两轮计时、加载库/哈希/安装版本已保存；强制重建普通 FA3 后 8 项回归通过。安装包结果仅验收执行器 |
| M02b1 | 已完成 | 固定 CANN 源码的隔离构建与加载溯源 | 固定 commit 导出及原生 host/tiling 构建通过；FP16/BF16 各编译 12 个 dense keys，实际 48 次 FIA launch 均来自配套 vendor 二进制；12 组精度/重复执行、6 项包校验 CPU 回归及 8 项普通 FA3 回归通过。JIT-only 混用旧 device binary 的问题已定位并拦截；计时仍非正式性能基线 |
| M02b2 | 已完成 | 固定源码的 profiler 性能基线 | 18 个输入 × FP16/BF16 精度、重复执行通过；144 次 launch 匹配配套二进制；四次独立采集保存逐例 kernel、Slice 转换和同步调用数据，含两次 CPU 亲和性对照；17 项 CPU 检查通过。主机噪声如实保留，后续性能验收仍须同轮交替测量 |
| M03 | 已完成 | MLA 入口识别、形状/stride 校验和明确的未支持拦截 | `qv` 不再被静默忽略；两种表示及异维 metadata 在 Python/fake/C++ 入口明确拒绝；910B 契约 233 项、普通 FA3 精度/metadata/编译图 11 项和既有 CPU 88 项通过 |
| M04a | 已完成 | 512+64、连续 BSND、短 KV（1..128）、无 mask、显式 `num_splits=1` 的 CATLASS 前向 | 910B MLA 374 项、普通 FA3 11 项、CPU 88 项通过；24 组 CANN/新实现精度与分派溯源通过；D02 及同形单 query × 两种 dtype 的 kernel 比值 0.652..0.672、调用比值 0.885..0.923，逐例不退化 |
| M04b1 | 已完成 | 上述路径 KV 129..512 的在线 softmax、FP32 输出累积及紧凑 dense MLA 调用入口 | 910B MLA 476 项、普通 FA3 11 项、CPU 88 项通过；30 组 CANN/新实现精度与分派通过；D01 默认/显式 scale、511 单/多 query 和短路径共 12 例 kernel 比值 0.651..0.912、调用比值 0.822..0.871，逐例不退化；更长 KV 明确拒绝 |
| M04b2a | 已完成 | 上述路径双缓冲输出累积、归约及空闲核 head 分工，支持 KV 513..1024 | 910B MLA 564 项（完整 552 + 归约尾块 12）、普通 FA3 11 项、CPU 88 项通过；72 组 CANN/新实现精度与分派通过；24 例 kernel 比值 0.589..0.988、调用比值 0.780..0.864，逐例不退化；保持显式 num_splits=1，>1024 明确拒绝 |
| M04b2b1 | 已完成 | 缩短在线输出累积的逐 tile 向量指令序列，保持 KV 1..1024 | 910B MLA 586 项、普通 FA3 11 项通过；94 组 CANN/新实现精度与分派通过；32 例 kernel 比值 0.603..0.975、调用比值 0.833..0.882，逐例不退化；旧新版 32 例 O/LSE 逐位一致，少行任务获得小幅 kernel 收益 |
| M04b2b2 | 待开始 | 上述路径 KV >1024 的流水与长 KV 调度优化 | KV 尾块、单/多 query，含 8193 的 FP16/BF16 O/LSE、重复执行及同轮 CANN 性能通过；保持显式 num_splits=1，不走生产 CANN 回退 |
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

测试产物随功能逐步增加。M01 提供 `tests/common/mla_ref.py` 与 `tests/test_flash_attn_npu_v3_mla_reference.py`；M02a 增加 `tests/mla_cann_baseline.py` 与 `tests/test_flash_attn_npu_v3_mla_cann.py`，覆盖安装包的 BSND 执行与计时。M02b1 增加执行器 `--expected-vendor` 检查与 `tests/test_flash_attn_npu_v3_mla_provenance.py`。M02b2 扩展到 P02、长 KV 和较长 prefill，新增 `--profile-dir`、`tests/mla_cann_profile.py` 及 CPU 数据完整性检查；全部实际 keys 已在 M02b1 包内，无需补编译。源码包、逐例分派和原始 profiler 证据见 status。M03 增加 `tests/test_flash_attn_npu_v3_mla_contract.py`，验证两种表示在 dense/TND/PA 的拒绝契约、非法输入、合法外层 stride、空输入及普通 fake shape。M04a 新增 `tests/test_flash_attn_npu_v3_mla_short.py`，覆盖短 KV 的精度、两种 dtype、尾块、连续非零 storage offset、重复执行与真实/fake/C++ 参数边界。M04b1 新增 `tests/test_flash_attn_npu_v3_mla_online.py`，覆盖 129..512 的尾块、跨 tile 最大值变化、FP32/零/负 scale、状态复用及紧凑公开入口的真实/fake 边界；静态 dense 调度随 kernel 参数传递，尚无 MLA metadata 协议。后续按步骤增加更长 KV、metadata 与 TND/PA 设备解析，不提前创建空脚手架。
M04b2a 将在线用例扩至 1024，覆盖两组缓冲的跨任务复用、异长调用连续入队、8/16-head 分工及 16/64 列归约边界，保留 1025/8193 的拒绝检查。M04b2b1 先优化每个 tile 的 FP32 输出重缩放与最终归一化，复用现有数学和跨核流水；更长 KV 与 8193 性能验收留在 M04b2b2。
原始性能数据保留用例 ID、shape、dtype、布局、mask、split、metadata 模式、软件版本和每轮计时。日志存放位置由执行步骤确定并写进 status，避免把大结果表加入代码仓。

## 构建与检查约束

需要重建时只选择目标后端；在核实当前 CATLASS 已就绪后可使用：

```bash
FLASH_ATTN_BUILD_VERSION=v3 FLASH_ATTN_BUILD_NPU=910 FLASH_ATTN_SKIP_SUBMODULE_INIT=1 python setup.py build_ext --inplace
```

涉及 AICPU/header 的改动要核查增量依赖，必要时加 `FLASH_ATTN_FORCE_REBUILD=TRUE`；不能用旧对象的成功运行作为新源码的证据。
执行验证前记录扩展 `__file__`，确认实际加载的是本次构建。设备选择每次检查后确定，不固定借用已占用或告警设备。

每步的精确 pytest 节点、设备和性能命令由该步交接记录给出，不提前编造尚不存在的脚本命令。纯文档步的验证仅限链接、范围、格式和状态一致性。
