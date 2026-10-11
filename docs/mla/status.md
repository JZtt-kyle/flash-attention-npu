# MLA 当前状态

核对日期：2026-10-11；分支：`feat/mla-cann-migration`；接手基线：`193390e`。本轮完成 **M07e1：公开 PA metadata**，4,791 项去重 pytest 通过。公开 PA 前向仍明确拒绝；唯一下一步为 [M07e2：BSND PA 前向生产接入](plan.md#下一步m07e2)。

本页回答“现在能用什么、证据在哪里、下一步是什么”。接口、协议和验收方法查 [design](design.md)，步骤与完成条件查 [plan](plan.md)。本轮闭环提交用 `git log -1 -- docs/mla/status.md` 定位。

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

[`PagedMetadataOp`](../../csrc/ascend910/flash_attn_npu_3/mla/mla_forward.cpp) 将公开 PA metadata 接入既有 v6 AICPU 生成器，复用当前流与 worker 流的双事件握手、tensor 持有和 allocator stream 登记。Python 公开入口、通用 custom op、专用 dispatcher、raw C++ 与 fake 保持相同接受范围。没有长度 D2H、主机同步、页表/cache 读取或 CANN attention 转调；attention Kernel、序列化器和 AICPU 源码均未修改。

B、Q 上界和 C 均非零时，公开记录 blockDim 固定为平台全核；任一静态容量为零时使用单核。设备判定的零 Q/KV 仍保存单工作核、零 workspace 和完整快照，blockDim 沿用静态选择，避免未来前向为选择核数读回记录。PA 前向仍须在 M07e2/M07e3 完成独立生产接入及资源验收。

证据保存在忽略目录 `build/mla-m07e1-paged-metadata/`，以下路径相对此目录。保留命令、日志、XML、源码快照、固定源、实际加载库、链接对象、完整记录与生成耗时样本；`check_acceptance.py` 汇总为 `acceptance.json`。这些本机证据不随 Git 分发，迁机须复制目录。

| 核验 | 最终结果与依据 |
| --- | --- |
| 前置闭环 | 编辑前核验 M07d3 的 3,832 项历史证据、差异及最小质量检查通过；不计作本轮新测试。`prerequisite.json` |
| 本轮功能 | **4,791 项去重 pytest 通过，无失败/跳过**：公开 PA metadata 2,840、非分页 metadata/cache 1,030、普通 FA3/入口 822、旧协议/CPU 消费者 99；四套 XML、日志及 `*-command.json` |
| 全页大小与固定源 | 双 dtype、BSND/TND 各 **446 组**完整记录，合计 **1,784 组**；全部 64 种 P、页前/页上/页后长度、短尾、逻辑容量上界。逐字匹配 CPU 序列化器和独立固定源 PA/non-FD 调度；每组矩阵原生非 FD 190 例，其余 256 例保留源 FD 分派并使用已确认的 FA3 无 split 映射。`public/source-*.json`、`source/source-manifest.json` |
| 接口与边界 | 真实/fake/C++/dispatcher 的 shape、dtype、device 一致；非法页大小/容量、Q 与字节界限、缺失/错误长度描述符、空 batch/Q/KV、非均匀 TND、偏移及只读别名通过；BSND 总 Q/起点超过 uint32，TND fake 不套矩形上界。`device.xml` |
| 动态长度与拒绝 | 设备非法值写失败头并清零完整载荷，同流修正后恢复；空 Q 仍检查 KV。新公开记录经公共 CPU 消费者验证，过期 Q/KV 拒绝；v2/v4 与填充到 v6 长度的旧头拒绝，v1–v5 消费者拒绝 v6。公开/内部/C++/fake PA 前向以及非分页前向传 v6 均拒绝 |
| 生命周期与 compile | 两条 producer stream、consumer stream、长度更新后重算、输入只读别名、释放/回收与记录快照通过；`opcheck` 的 schema/fake/autograd 注册和 `torch.compile(backend="eager", fullgraph=True)` 通过 |
| 构建与实际加载 | 最终扩展重新链接并核对 **32 个对象、260 个仓库源文件、6,186 个依赖**；最终三套 NPU pytest 均记录同一扩展哈希及 Python 源码哈希。`production-load.json`、`production-final/`、`*-command.json` |
| 生成耗时 | 双 dtype × 两种 Q 布局 × P=16/112/128/256/528/1024，共 24 例；每例预热 5 次、20 个 wall 样本，中位数范围 **186.47–216.53 μs**。含公开分配、AICPU、事件及同步；**不是 PA attention 或公开 PA E2E**。`public/timing-*.json` |
| 回归边界 | 非分页 v2/v4 生成、cache 输出/复用/生命周期/compile；普通 FA3 metadata/func/varlen/cache/FD/compile 相关用例。未重跑普通 FA3 全套、独立 PA attention/Copy 精度性能、release Trap、全布局生产精度或公开 MLA E2E；本步未修改 Kernel，不用历史性能冒充新采样 |
| 提交质量 | 文档链接/锚点、表格/命令语法、源码/对象/加载库哈希、差异及适用质量钩子通过；仅本步 11 个文件。`docs-check.json`、`quality-staged.log`、`source-snapshots.json` |

**已处理的测试失败：** 固定静态核数后，新加的测试断言误把记录索引 22 当作 blockDim（实际索引为 21），首次重跑失败；按独立 v6 编码修正测试后重跑最终三套 NPU 验收。失败日志、XML、命令和错误测试快照保留于 `failed-field-assertion/`；上表仅统计最终结果。此前可变核数版本的初轮检查留在 `initial-variable-blocks/`，不重复计数。文档检查脚本首次误用了下一步的输出目录，修正后重新通过；原错误记录于 `docs-check-initial-error.json`。

## 最近功能闭环与限制

以下为历史验收，本轮没有重新执行其全部内容。路径相对仓库根目录。

| 功能闭环 | 结果与证据 |
| --- | --- |
| M07d3 BSND/TND 独立 PA attention，`193390e` | `build/mla-m07d3-tnd-paged-attention/acceptance.json`：3,832 项去重 pytest；TND 双 dtype 各 12 例 Kernel 中位比 FP16 1.0085–1.1534、BF16 1.0052–1.1500，均慢于固定源 |
| M07d2 BSND 独立 PA attention，`82d3809` | `build/mla-m07d2-paged-attention/acceptance.json`：3,122 项去重 pytest；双 dtype 各 12 例 Kernel 中位比 FP16 1.0114–1.1386、BF16 1.0097–1.1515，均慢于固定源 |
| M07d1 PA Copy，`139b992` | `build/mla-m07d1-paged-copy/acceptance.json`：1,401 项去重 pytest；全部页大小、64 位地址与同步，双 dtype 代表源 Copy 性能 |
| M07c3 独立 v6 消费者，`9f9aeb9` | `build/mla-m07c3-paged-consumer/acceptance.json`：2,273 项去重 pytest；CPU/910B 各 2,552 组源差分、58 次 release Trap、sanitizer |
| 最近前向生产接入 M06e2c2，`538b8e5` | `build/mla-m06e2c2-mixed-stride-production/acceptance.json`：4,659 项去重 pytest，三种非分页布局 stride 接入 |
| M06e2c2 公开性能 | 同目录 `performance-acceptance.json`：双 dtype 各 8 个适用公开用例，R=1/4 复用、生成加前向、内部生成共 96 段逐例达标；FA3/CANN E2E 为 FP16 0.284–0.623、BF16 0.284–0.675。各 3 个原生 FD 用例留待 M10 |

**仍保留的限制：** M06e2c2 组件 Kernel 比值 FP16 0.898–1.888、BF16 0.901–2.458，两种 dtype 各有 6 例中的 4 例慢于固定源。独立 PA attention 同样不能宣称 Kernel 逐例不劣化；其 probe wall 含 ctypes、发射和同步，不代表公开 E2E。公开 PA 前向及相应 E2E 尚待验收。

## 环境与构建

Python `/usr/local/python3.11.15/bin/python3` 3.11.15，torch `2.7.1+cpu`、torch_npu `2.7.1`，CANN `/usr/local/Ascend/cann-9.0.0-beta.2`；8 张 910B3，使用物理卡 6，进程内设备 0。环境与芯片记录见 `environment.json`、`npu-before.log`、`npu-final-before.log`、`npu-after.log`；设备任务串行。

CATLASS `769cd40a8716b28650b6bebb08db4834eea4462f` 且干净；只读 CANN `ca18aee610729ca5270a44e2d84c751bc687fbf0` 原有 67 项差异、本仓原有未跟踪 `kernel_meta/` 均保留。接手与提交前状态见 `*-status-before.txt` / `*-status-final.txt`。

生产扩展实际路径为 `flash_attn_npu_3/flash_attn_npu_3.cpython-311-aarch64-linux-gnu.so`。接手 SHA256 `cd44a4c5caae26478a1a8407250bc2d60219018714111c15f2a36d985cf49404`；最终构建/加载 SHA256 `177e3e147ebbd56380a1e0eee60f5912e7fb709abad0551efa86b78fa5731977`。本轮重编 host 入口，attention/AICPU 对象按未变化的依赖复用，构建仍包含七个 AICPU 单元；v6 从公开 metadata 发射，生产前向仍只实例化非分页 Kernel。

继续使用非 login shell；CPU pytest 设置 `TORCH_DEVICE_BACKEND_AUTOLOAD=0`，NPU 在沙箱外执行。质量工具使用 `/tmp/mla-m01-precommit` 及其 cache；版本/tiling 注册警告、构建时的 `DrvMngGetConsoleLogLevel` 警告保留在日志中。本轮未运行 profiler。

## 唯一下一步与复现

下一步仅为 **[M07e2](plan.md#下一步m07e2)**：BSND Q 只读 PA kvcache 与内部 PA Q-used 生产接入，完成 storage/页表/长度校验、双 dtype 精度、Kernel 性能、公开生成/复用/内部生成 R=1/4 E2E 及普通 FA3 回归。TND PA 前向留在 M07e3；本轮提交后停止。

本轮最小闭环复核（仓库根目录，非 login shell，不发射 NPU）：

```bash
python3 build/mla-m07e1-paged-metadata/check_acceptance.py
PYTHONPATH=/tmp/mla-m01-precommit PRE_COMMIT_HOME=/tmp/mla-m01-precommit-cache \
bash ci/code_quality.sh --staged
git diff --check
git diff --cached --check
```

完整构建/功能命令保存在 `run_checks.py` 和 `*-command.json`；实际加载核验用 `audit_production.py`。重跑时换新的证据目录并相应修改脚本的输出根目录；依次运行 `production-build`、加载核验、`device`、`nonpaged-regression`、`ordinary-regression`，CPU 独立运行 `cpu-regression`。每次设备运行前核对占用；不要覆盖既有验收证据。源码、环境或构建变化后重跑受影响检查，历史证据核验不能替代新运行。
