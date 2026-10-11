# MLA 当前状态

核对日期：2026-10-11；分支：`feat/mla-cann-migration`；接手基线：`0813a17`。
本轮完成 **M07e2b：BSND PA 生产实例与内部前向**，6,392 项去重 pytest 通过，910B 双 dtype 代表精度和生产 Kernel 性能对照完成。公开 PA 前向仍明确拒绝；唯一下一步为 [M07e2c：BSND 只读 PA kvcache 公开接入](plan.md#下一步m07e2c)。

本页记录当前能力、最近闭环证据和唯一下一步；契约见 [design](design.md)，迭代边界见 [plan](plan.md)。本轮闭环提交用 `git log -1 -- docs/mla/status.md` 定位。

## 当前可调用能力

共同边界：**910B C1V2、FP16/BF16、512+64 共享 KV、Hkv=1、无梯度、无 mask、`num_splits=1`**。Q/K 末维为 64，qv/V 为 512，四路同 dtype、同 NPU 设备。非分页 BSND 单样本 KV 容量及 TND 的 `max_seqlen_k` 为 0..16384；非分页 TND 总 Q/KV 各不超过 `INT32_MAX`，总 KV 可超过 16384。完整边界见 [design 6.2–6.4](design.md#62-调度字段与边界)。

| 层次 | 已接通能力 | 记录与边界 |
| --- | --- | --- |
| 公开前向 | `flash_attn_func`，BSND Q/KV | 整容量，内部 v1 |
| 公开前向 | `flash_attn_varlen_func`，TND Q/KV | 两轴分配前缀；无 used 时内部 v3，任一 `seqused_q/k` 存在时内部 v5 |
| 公开只读 cache | `flash_attn_with_kvcache`，BSND Q + BSND KV | 整容量或实际 KV 长度；无长度且无记录走 v1，否则 v2；支持显式 v2 复用 |
| 公开只读 cache | `flash_attn_with_kvcache`，TND Q + BSND KV | Q 使用完整分配段；内部生成或显式复用 v4 |
| 公开非分页 metadata | `get_scheduler_metadata(page_size=None)` | BSND Q 生成 v2，TND Q 生成 v4 |
| **公开 PA metadata** | `get_scheduler_metadata(page_size=P)`，BSND/TND Q | **v6 `int32[408+5B]`**；生成器只接收逻辑几何和长度，公开 PA 前向尚未接通 |
| 内部非分页前向 | BSND、TND、混合布局的 Q-used | 内部 v5；BSND/混合 Q-used 无公开等价入口，不开放显式 v5 复用 |
| **内部 BSND PA 前向** | `_mla_paged_bsnd_forward`，整容量/实际 KV/Q-used，独立 stride | 生产双 dtype 实例；无 Q-used 可消费公开 v6，Q-used 每次生成新记录 |
| 独立 PA 组件/attention | 地址、调度、v6 生成/消费、Copy；TND Q attention 含 Q-used 和独立 stride | TND PA 尚无生产前向；独立 probe 不代表公开 PA 能力 |

公开 PA 生成器要求 P 为 **16..1024 的 16 倍数**，`max_seqlen_k=C` 在 **0..16384 且可被 P 整除**。C 是每样本逻辑容量；生成器没有页表、物理页数、cache 或 stride 参数。KV 长度须为同设备连续 `int32[B]` tensor，始终非累计；可选 Q 前缀为同设备连续 `int32[B+1]`，B=0 时仍需 `[0]`。无前缀时 Q 为 BSND 整容量；有前缀时 TND 总 Q 由 AICPU 推导，不按矩形 `B*max_seqlen_q` 限制实际分配。公开生成器没有 Q-used 参数。协议和资源规则见 [design 5.7](design.md#57-v6nd-pa-记录)。

调用时须注意：

- kvcache/metadata 的 `num_splits` 默认 0，MLA 须显式传 **1**；生成器的 `qkv_dtype` 默认 BF16，须匹配前向输入。func/varlen 默认 `num_splits=1`。
- 所有公开 MLA metadata 生成器都要求长度 tensor；公开入口的 None/整数便利长度只属于 kvcache。仅接受默认 mask/调度选项；生成器使用 `has_softcap=False`，拒绝数值 `softcap/softmax_scale` 兼容关键字。前向 scale 默认 `1/sqrt(576)`，显式值须在 FP32 有限范围。
- metadata 不绑定 cache 地址、stride 或 scale。生成器在设备检查动态值，非法值写失败记录并清零载荷；消费者才触发 Trap。PA 与非 PA 协议互斥，v6 不能传给当前非分页前向。
- 三种非分页布局及内部 BSND PA 支持四路末维连续、外层非负 stride，包括切片、广播、重叠及 offset；发射前检查完整 storage。显式 v2/v4 仍须匹配当次长度和 v4 Q 前缀；PA v6 仍须匹配当次长度及全部有效页。
- 请求 LSE 时 func/varlen 返回二项，kvcache 返回四项。非分页非零容量中的零 KV 有效行为 `O=0、LSE=-inf`；全局空 KV 初始化分配区为 `O=0、LSE=+inf`。内部 BSND PA 返回二项，所有有效零 KV 行均为 `O=0、LSE=-inf`，包括 C=0/空物理池；其他未使用 query 槽位不承诺值。详见 [design 6.7](design.md#67-生产接入与构建)。
- 内部 BSND PA 的长度只接受 None 或同设备连续 `int32[B]`；None 按逻辑容量 C 物化。内部无整数便利输入、fake 或 compile 接入；公开 PA kvcache 仍拒绝调用。

计划内尚未开放：公开 PA 前向、causal/window、自动或多 split、192/128 prefill、非零 softcap 与非默认调度选项。已开放公开路径的 fake/compile 已有覆盖，完整 graph replay 留在 M16；910C 尚未开放或实机验收。KV 追加、内置 rotary、量化、backward 等排除项见 [design 第 4 节](design.md#4-功能边界)。

## 本轮闭环与证据

本轮将既有 BSND PA 静态计划、布局适配器和双 dtype PA attention 实例编入生产扩展，新增内部 `_mla_paged_bsnd_forward`。host 校验 rank/dtype/device、完整 storage/跨度/地址、长度和 v6 描述符，复用当前流与 AICPU worker 的 event 依赖和异步持有；每核在 TPipe/同步/写出之前检查记录、当前长度及全部有效页，再进入既有 `MlaPagedBsnd`。无动态值读回、完整 cache gather 或 CANN attention 转调。共享调用参数拆入 host/device 可共用的头文件，消费者算法及 Kernel/Block/Tile 数值路径未改。

证据保存在忽略目录 `build/mla-m07e2b-bsnd-paged-production/`，以下路径相对此目录。保留命令、日志、XML、源码/固定源快照、实际加载库及性能原始数据；本机证据不随 Git 分发，迁机须复制目录。

| 核验 | 结果与依据 |
| --- | --- |
| 前置闭环 | 编辑前核验 M07e2a 的 5,991 项历史证据和最小质量检查通过，不计本轮新测试。`prerequisite.json` |
| 本轮功能与回归 | **6,392 项去重 pytest 通过，无失败/跳过**：CPU 893、生产功能 713、真实 Trap 26、普通 FA3/入口 822、非分页及 TND PA 共享组件 1,098、公开 PA metadata 2,840。各模式 XML、日志及 `*-command.json` |
| 固定源记录与精度 | 双 dtype 共 **152 组**完整 v6 固定源对照，每种 64 种 P 与 12 个代表形状。源 O/非空行 LSE 逐元素一致，独立双数学 O/LSE 通过；含独立 stride、真实大地址、scale/半路扰动、任务环与头数边界。`probes/*/production-schedules.json`、`probes/*/production-accuracy/` |
| 空值、动态值与生命周期 | 整容量/实际 KV/Q-used、空表/池/全零 KV、完整 storage、记录跨合法物理池/页表/stride 复用、当前流长度更新与多流释放后复用均通过。非法页、过期长度、损坏协议/空记录等双 dtype **26 项真实 Trap**；每例独立进程，正常调用预热后触发，最长约 0.131 秒。`device.xml`、`trap.xml`、`production/*/trap-*/` |
| 构建与加载 | 最终生产扩展 **36 个对象、271 个仓库源、6,197 个依赖**及双 dtype 设备依赖核验通过；静态计划、布局和 PA Kernel 均在实际链接/加载路径中。双 dtype PA probe 和 v6 AICPU probe 重编。`production-build-command.json`、`production-load.json`、`production-final/`、`probes/*/build-manifest.json`、`producer/build-manifest.json` |
| 代表性能 | 双 dtype **24 个代表用例**先通过源/双数学精度，再交替采样生产/固定源 Kernel、v6 AICPU 和内部 wall；这 24 例在性能驱动中运行，不计 pytest 总数。`profile-*/summary.json`、逐样本 JSON 及 profiler CSV |
| 源码一致性与重复覆盖 | CPU 893 项消费的编译源码与最终版本一致；其广泛快照另含两处未消费文件变更，由实际依赖 manifest 核对。生产功能运行后仅将空值断言缩至契约定义的 Q-used 区域，保存旧测试快照并核对其余 AST 一致，最终 **24 项重跑通过**，去重后不增加总数。`device-test-snapshot/`、`empty-final.xml`、`acceptance.json` |
| 提交质量 | 文档链接/锚点、表格/命令语法、差异及适用质量钩子通过；只暂存本步 **17 个文件**。`docs-check.json`、`quality-staged.log`、`source-snapshots.json`；`check_acceptance.py` 汇总为 `acceptance.json` |

两次头文件依赖编译失败已修复，原日志保存在 `failed-initial-build/`、`failed-context-include/`。后续发现 Q-used 描述符不能复用 v5 的字数上界，已改用静态计划的 v6 上界并重编、重跑最终生产检查；此前 40 项 smoke、706 项生产检查及早期 CPU 运行分别保留在对应日志和 `pre-v6-bound/`、`prebuild-cpu/`，均不重复计数。格式修正前的质量日志保留，最终 staged 检查为准。

## 性能范围与保留限制

本轮固定 BSND × FP16/BF16 各 12 例，P=16/112/128/256/528/1024 × 短/长 KV；每例预热 5 次，5 轮交替采样，每轮每实现 20 个 wall、10 个 Kernel 样本。同进程固定同一 CPU，seed 为 20261011；矩阵和原生 FD 分派在采样前保存。每种 dtype 各有 6 个原生非 FD、6 个原生 FD 用例；FD 按已确认的无 split 映射对照，不作为原生 E2E。

下表为逐例中位耗时范围，Kernel 比值为“生产 / 固定源”，其余单位为微秒。**24 例中 24 例 Kernel 慢于固定源，不能宣称 Kernel 逐例不劣化。**

| dtype | Kernel 比值 | 较慢例数 | v6 AICPU | 源 probe wall | 生产内部 wall |
| --- | --- | --- | --- | --- | --- |
| FP16 | 1.0058–1.1242 | 12/12 | 24.83–38.85 | 336.32–990.82 | 343.41–1008.21 |
| BF16 | 1.0142–1.1651 | 12/12 | 25.22–39.22 | 332.69–984.03 | 357.74–999.42 |

两侧使用相同逻辑输入、调度和核数；生产 Kernel 包含 v6/当次长度/全部有效页校验，源侧视图物化在计时外。生产内部 wall 包含 v6 生成、输出/workspace 分配、发射与同步，源 probe 使用预分配缓冲。**这些 wall 数据不能作为公开 PA E2E 或“不劣化”结论**；公开生成/复用/内部生成 R=1/4 调用段留在 M07e2c。逐例输入视图、源记录/原生 FD、误差和原始样本均保存在 `profile-fp16/`、`profile-bf16/`。

两种 dtype 的 profiler **框架 API 数据库导出失败**，错误为 `Failed to find node launch str id`；数据库关联表不作为本次证据。计时使用的文本 Kernel CSV 直接读取 CANN `op_summary`，不依赖该表；每种 dtype 的 1,800 条设备任务逐行核对名称、设备、起点及耗时，1,200 个 Kernel 样本和 2,400 个 wall 样本完整，逐例中位数重算一致。失败日志、解析器源码和核验见 `profile-*/profiler/*/logs/`、`profiler-parser-sources/`、`profile-export-audit.json`。

未重跑普通 FA3 全套、完整 TND PA attention 矩阵、独立 Copy 性能、全部历史 Trap、sanitizer 或全布局生产 MLA E2E；静态计划及消费者算法未改，相关 CPU、TND PA 共享组件和普通 FA3 回归已补。最近非分页公开性能仍为 M06e2c2 的历史证据：`build/mla-m06e2c2-mixed-stride-production/performance-acceptance.json` 中 96 个适用公开调用段逐例达标，原生 FD 留待 M10；当时组件 Kernel 的 FP16 0.898–1.888、BF16 0.901–2.458 比值及各 6 例中 4 例较慢的限制保留。

## 环境与构建

Python 3.11.15，torch 2.7.1+cpu、torch_npu 2.7.1，CANN 9.0.0-beta.2；8 张 910B3，使用物理卡 6、进程内设备 0，设备任务串行。CATLASS `769cd40a8716b28650b6bebb08db4834eea4462f` 干净；只读 CANN `ca18aee610729ca5270a44e2d84c751bc687fbf0` 原有 67 项差异和本仓未跟踪 `kernel_meta/` 保留。环境记录为 `environment.json`。

最终生产扩展实际路径为 `flash_attn_npu_3/flash_attn_npu_3.cpython-311-aarch64-linux-gnu.so`，SHA256 `0811e7bd54a214b057715982cbda625dc33c98ef684e2f8b6b9c96f3cb92fdd3`。36 个对象、271 个仓库源、6,197 个依赖核验见 `production-load.json`；双 dtype PA 独立探针及 v6 AICPU 探针也按本轮依赖重新构建。

## 唯一下一步与复现

下一步仅为 **[M07e2c](plan.md#下一步m07e2c)**：复用内部 BSND PA 路径，接通公开只读 kvcache，完成 None/整数/tensor 长度、显式 v6 复用、Python/dispatcher/raw C++/fake/compile 一致性和公开 R=1/4 E2E。TND PA 生产前向留在 M07e3。本轮只提交一次，成功后停止。

本轮最小闭环复核（仓库根目录，非 login shell，不发射 NPU）：

```bash
python3 build/mla-m07e2b-bsnd-paged-production/check_acceptance.py
PYTHONPATH=/tmp/mla-m01-precommit PRE_COMMIT_HOME=/tmp/mla-m01-precommit-cache \
bash ci/code_quality.sh --staged
git diff --check
git diff --cached --check
```

证据脚本 `build/mla-m07e2b-bsnd-paged-production/run_checks.py` 支持 `cpu`、`probe-build`、`device`、`trap`、`ordinary`、`nonpaged`、`metadata`、`profile-fp16`、`profile-bf16` 和 `empty-final`；构建脚本为同目录 `build_production.py`，构建与加载审计为 `audit_production.py`。设备任务在沙箱外串行执行，CPU 禁用 torch 自动加载 NPU，使用非 login shell；重跑复制脚本到新证据目录，不能覆盖已有数据。
