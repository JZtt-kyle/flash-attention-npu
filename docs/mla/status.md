# MLA 当前状态

核对日期：2026-10-11；分支：`feat/mla-cann-migration`；接手基线：`2f969b2`。
本轮完成 **M07e2a：BSND PA 静态发射计划与零 KV 预算兼容**，5,991 项去重 pytest 通过，sanitizer 和双布局代表性能对照完成。公开 PA 前向仍明确拒绝；唯一下一步为 [M07e2b：BSND PA 生产实例与内部前向](plan.md#下一步m07e2b)。

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
| 独立 PA 组件/attention | 地址、调度、v6 生成/消费、Copy，BSND/TND Q attention 含内部 Q-used 和独立 stride | 仅独立 probe；不能据此调用公开 PA 前向 |

公开 PA 生成器要求 P 为 **16..1024 的 16 倍数**，`max_seqlen_k=C` 在 **0..16384 且可被 P 整除**。C 是每样本逻辑容量；生成器没有页表、物理页数、cache 或 stride 参数。KV 长度须为同设备连续 `int32[B]` tensor，始终非累计；可选 Q 前缀为同设备连续 `int32[B+1]`，B=0 时仍需 `[0]`。无前缀时 Q 为 BSND 整容量；有前缀时 TND 总 Q 由 AICPU 推导，不按矩形 `B*max_seqlen_q` 限制实际分配。公开生成器没有 Q-used 参数。协议和资源规则见 [design 5.7](design.md#57-v6nd-pa-记录)。

调用时须注意：

- kvcache/metadata 的 `num_splits` 默认 0，MLA 须显式传 **1**；生成器的 `qkv_dtype` 默认 BF16，须匹配前向输入。func/varlen 默认 `num_splits=1`。
- 所有公开 MLA metadata 生成器都要求长度 tensor；None/整数便利输入只属于 kvcache。仅接受默认 mask/调度选项；生成器使用 `has_softcap=False`，拒绝数值 `softcap/softmax_scale` 兼容关键字。前向 scale 默认 `1/sqrt(576)`，显式值须在 FP32 有限范围。
- metadata 不绑定 cache 地址、stride 或 scale。生成器在设备检查动态值，非法值写失败记录并清零载荷；消费者才触发 Trap。PA 与非 PA 协议互斥，v6 不能传给当前非分页前向。
- 三种非分页布局支持四路末维连续、外层非负 stride，包括切片、广播、重叠及 offset；发射前检查完整 storage。显式 v2/v4 仍须匹配当次长度和 v4 Q 前缀。
- 请求 LSE 时 func/varlen 返回二项，kvcache 返回四项。非分页非零容量中的零 KV 有效行为 `O=0、LSE=-inf`；全局空 KV 初始化分配区为 `O=0、LSE=+inf`；其他未使用 query 槽位不承诺值。详见 [design 6.7](design.md#67-生产接入与构建)。

计划内尚未开放：公开 PA 前向、causal/window、自动或多 split、192/128 prefill、非零 softcap 与非默认调度选项。fake/compile 已有覆盖，完整 graph replay 留在 M16；910C 尚未开放或实机验收。KV 追加、内置 rotary、量化、backward 等排除项见 [design 第 4 节](design.md#4-功能边界)。

## 本轮闭环与证据

本轮只完成独立静态计划及其所需的共享 PA 消费者修正，未接入生产前向。

`BuildPagedBsndPlan` 只读静态视图和平台描述，复用完整 storage 校验，输出逻辑 shape、四路独立布局、cache 描述、v6 字数、AICPU 平台参数和消费者 context。B/Sq/C 决定单核或全核，workspace 覆盖动态长度/Q-used；不读 tensor 数据或地址，不运行 SplitCore，也不分配逐 batch 数组。适配器未编入生产扩展，尚无生产发射。

首次计划测试在非空逻辑容量、空物理池、KV 全零时以 InvalidTiling 失败（157 通过、1 失败）。已修正 PA 消费者：动态 KV 全零且当前长度、有效页和平台已完整校验后，允许 host 保留未使用的 workspace；记录的空切块、端点、FD/workspace 载荷仍须全零。静态空容量保持零预算要求。数值和 Copy 未改，公共入口和协议未改。详见 [design 6.7.4](design.md#674-bsnd-pa-静态发射计划)。

证据保存在忽略目录 `build/mla-m07e2a-bsnd-paged-plan/`，以下路径相对此目录。保留命令、日志、XML、源码/固定源快照、实际加载库和原始性能数据；本机证据不随 Git 分发，迁机须复制目录。

| 核验 | 结果与依据 |
| --- | --- |
| 前置闭环 | 编辑前核验 M07e1 的 4,791 项历史证据、差异及最小质量检查通过，不计本轮新测试。`prerequisite.json` |
| 本轮功能 | **5,991 项去重 pytest 通过，无失败/跳过**：新计划 294、CPU 回归 597、BSND/TND PA attention 1,422、真实 Trap 16、公开 PA metadata 2,840、普通 FA3/入口 822。各模式 XML、日志和 `*-command.json` |
| 固定源与预算 | **3,668 组**完整 v6 记录：原生非 FD 1,953、已确认 FA3 无 split 映射 1,715。全部 64 种 P、1..26 核、短容量、M=128→256、Q-used；新计划 context/cache 经既有消费者检查。`plan/metadata/`、`plan/source/` |
| 空值与边界 | 计划覆盖完整 storage、独立 stride/offset、字段/字节溢出、局部资源和 workspace 上界。实机包含全核零 KV/空物理池/预留 workspace 不写入的 16 项，损坏空记录分别通过诊断拒绝和 release Trap。`plan.xml`、`device.xml`、`trap.xml`、`probes/` |
| Sanitizer | ASan/UBSan/LeakSanitizer 下新计划 **294 项通过**；共享消费者插桩 **2,622,720 次检查通过**。未变化的固定源/序列化器保持普通构建；sanitizer 和 16 项 smoke 均为重复覆盖，不计入去重总数。`sanitizer.xml`、`sanitizer/consumer/build.log` |
| 构建与加载 | 双 dtype PA 探针按最终源码重编并核对加载库及全部依赖；生产扩展及其 32 个对象、260 个仓库源、6,186 个依赖不变。`probes/*/build-manifest.json`、`loaded.json`、`production-unchanged.json` |
| 代表精度与性能 | BSND/TND 双 dtype 共 **48 个代表用例**的源/双数学精度通过，完成固定源 PA/non-FD Kernel 与 probe wall 对照；这 48 例在性能驱动中执行，不计入 pytest 总数。逐例数据与限制见下节 |
| 提交质量 | 文档链接/锚点、表格/命令语法、差异、源码/加载库哈希及适用质量钩子的记录为 `docs-check.json`、`quality-staged.log`、`source-snapshots.json`；仅本步 10 个文件，`check_acceptance.py` 汇总为 `acceptance.json` |

**已处理与未计数的运行：** 首次零 KV 预算失败保存在 `failed-empty-budget/`。沙箱 ptrace 导致的 LeakSanitizer 退出失败在 `failed-sanitizer-sandbox/`；最终在沙箱外保留泄漏检查并通过。被取消的重复插桩和重复长 KV 精度计算分别在 `stopped-source-sanitizer/`、`stopped-duplicate-performance-accuracy/`，不计通过；48 个代表用例由性能驱动同时做源/双数学精度和性能验收。早期及重复运行不计入上表。

## 性能范围与保留限制

本轮采样矩阵固定为 BSND/TND × FP16/BF16，各 12 例，P=16/112/128/256/528/1024 × 短/长 KV；每例预热 5 次，5 轮交替采样，每轮每实现 20 个 wall、10 个 Kernel 样本。矩阵与源原生 FD 分派在采样前保存；独立 PA/non-FD 源和迁移侧使用相同逻辑输入、基本块、核数和端点。每套 12 例中原生非 FD/FD 各 6 例，FD 按已确认的无 split 映射作组件对照，不作为原生 E2E。

下表为逐例中位耗时的“迁移 / 固定源”范围。**48 例 Kernel 均慢于固定源，不能宣称 Kernel 逐例不劣化。**

| 布局 / dtype | Kernel 比值 | 较慢例数 | 独立 probe wall 比值 |
| --- | --- | --- | --- |
| BSND FP16 | 1.0110–1.1557 | 12/12 | 1.0080–1.0374 |
| BSND BF16 | 1.0102–1.1490 | 12/12 | 0.9986–1.0224 |
| TND FP16 | 1.0109–1.1827 | 12/12 | 1.0043–1.0243 |
| TND BF16 | 1.0091–1.1959 | 12/12 | 1.0061–1.0293 |

组件计时在 metadata 生成和分配之后；源 ABI 之外的视图转换、TND 有效 Q 压紧在源计时前完成，迁移 Kernel 包含 v6/当前长度/全部有效页校验。probe wall 包含 ctypes、发射与同步，**不能作为公开 PA E2E**。结果保存在 `profile-bsnd/`、`profile-tnd/` 的 `comparison.json`、逐 dtype `summary.json`、wall/Kernel 样本和 profiler 原始 CSV。

未重跑普通 FA3 全套、独立 PA Copy 性能、全部历史 Trap 或全布局生产 MLA E2E。最近非分页公开性能仍是 M06e2c2 的历史验收：`build/mla-m06e2c2-mixed-stride-production/performance-acceptance.json` 中 96 个适用公开调用段逐例达标，原生 FD 留待 M10；组件 Kernel 的 FP16 0.898–1.888、BF16 0.901–2.458 比值及各 6 例中 4 例慢于固定源的限制仍保留。

## 环境与构建

Python 3.11.15，torch 2.7.1+cpu、torch_npu 2.7.1，CANN 9.0.0-beta.2；8 张 910B3，使用物理卡 6、进程内设备 0，设备任务串行。CATLASS `769cd40a8716b28650b6bebb08db4834eea4462f` 干净；只读 CANN `ca18aee610729ca5270a44e2d84c751bc687fbf0` 原有 67 项差异和本仓未跟踪 `kernel_meta/` 保留。环境记录为 `environment.json`。

生产扩展实际路径仍是 `flash_attn_npu_3/flash_attn_npu_3.cpython-311-aarch64-linux-gnu.so`，SHA256 `177e3e147ebbd56380a1e0eee60f5912e7fb709abad0551efa86b78fa5731977`。修改的 PA 消费者和新计划不在生产依赖中；32 个对象、260 个仓库源、6,186 个依赖的核验见 `production-unchanged.json`。独立 PA 探针已按本轮源码重编双 dtype；不能从独立验收推断生产前向可用。

## 唯一下一步与复现

下一步仅为 **[M07e2b](plan.md#下一步m07e2b)**：将静态计划和 PA Kernel 接入生产构建及内部 BSND 前向，含 Q-used、长度/页表校验、异步生命周期、双 dtype 精度/Kernel 性能和相关回归。公开 BSND PA kvcache 与 R=1/4 E2E 留在 M07e2c，TND PA 生产前向留在 M07e3。本轮只提交一次，成功后停止。

本轮最小闭环复核（仓库根目录，非 login shell，不发射 NPU）：

```bash
python3 build/mla-m07e2a-bsnd-paged-plan/check_acceptance.py
PYTHONPATH=/tmp/mla-m01-precommit PRE_COMMIT_HOME=/tmp/mla-m01-precommit-cache \
bash ci/code_quality.sh --staged
git diff --check
git diff --cached --check
```

CPU 复现脚本为 `build/mla-m07e2a-bsnd-paged-plan/run_checks.py`，模式 `plan`、`cpu-regression`、`sanitizer`；设备脚本为同目录 `run_npu_checks.py`，先 `probe-build`，然后串行 `device`、`trap`、`ordinary-regression`、`public-metadata-regression`、`profile-bsnd`、`profile-tnd`。重跑复制脚本到新证据目录，不能覆盖已有数据。使用非 login shell；CPU 禁用 torch 自动加载 NPU，NPU 和最终 sanitizer 在沙箱外运行。历史验收用 Git 中的上一步 status 查询。
