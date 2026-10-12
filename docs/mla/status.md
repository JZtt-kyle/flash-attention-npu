# MLA 当前状态

核对日期：2026-10-12；分支：`feat/mla-cann-migration`；接手提交：`7cb4ef8`；最近前向闭环：`29ed04f`（M07e3c）。
**本轮完成 M08a：独立 CPU 右下 causal 范围、全遮罩行描述和完整 M 分核；公共 causal 仍拒绝。唯一下一步为 [M08b1](plan.md#当前步-m08b1bsnd-独立数值组件)，本次提交后停止。**

本页保存当前能力、最近闭环证据和环境差异；接口与架构见 [design](design.md)，实施顺序见 [plan](plan.md)。当前闭环提交可用 `git log -1 -- docs/mla/status.md` 定位。

## 当前可调用能力

共同边界：**910B C1V2、FP16/BF16、512+64 共享 KV、Hkv=1、无梯度、无 mask、`num_splits=1`**。Q/K 末维为 64，qv/V 为 512，四路同 dtype、同 NPU 设备；末维连续，合法外层 stride 按完整 storage 校验。每样本 KV 容量或 TND 段长上界为 16384，TND 总 KV 和 PA 物理页池不套用这个上限。

| 入口 | 已接通输入 | metadata 与长度 |
| --- | --- | --- |
| `flash_attn_func` | BSND Q/KV | 整容量，内部 v1 |
| `flash_attn_varlen_func` | TND Q/KV | 两组分配前缀；无 used 时内部 v3，任一 `seqused_q/k` 存在时内部 v5 |
| `flash_attn_with_kvcache` | BSND Q + BSND KV | 无长度且无记录走 v1，否则内部生成或显式复用 v2 |
| `flash_attn_with_kvcache` | TND Q + BSND KV | Q 完整分配段，内部生成或显式复用 v4 |
| `flash_attn_with_kvcache(page_table=...)` | BSND/TND Q + ND PA KV | Q 完整分配段，内部生成或显式复用 v6 |
| `get_scheduler_metadata` | BSND/TND Q 的只读 cache 调度 | 非 PA 生成 v2/v4；指定 `page_size` 生成 v6 |
| 内部 `fwd` / PA 专用算子 | 非 PA 三布局及 BSND/TND PA 的 Q-used | 分别内部生成 v5/v6；拒绝 Q-used 与显式记录复用 |

调用要点：

- kvcache 和 metadata 的 `num_splits` 默认 0，MLA 须显式传 **1**。生成器须指定 `headdim=64, headdim_v=512`，并让 `qkv_dtype` 匹配输入（默认 BF16）。完整选项见 [design 2.2](design.md#22-后端选项与返回值)。
- kvcache 的 KV 长度接受 None、整数或同设备连续 `int32[B]`；生成器只接受 tensor。TND cache 前向须给 `cu_seqlens_q` 和 `max_seqlen_q`。**公开 varlen 支持 Q-used；公开 kvcache 和 metadata 生成器没有 Q-used 参数。**
- PA 的 P 为 16..1024 的 16 倍数，逻辑容量 `C=页表列数×P` 在 0..16384；生成器的 `max_seqlen_k` 须等于 C。None 表示 C，整数须在 0..C，按逻辑 B 物化，不能用 Tq 或物理页数代替 B。
- 分配前缀决定物理起点，used 只限制有效范围。记录不绑定地址、stride、页表或 scale；每次前向核对当次长度、前缀和全部有效页。长度或前缀改变须重生成；输入与记录在异步执行期间须保持只读和存活。详见 [metadata 复用](design.md#51-签名与复用规则)。
- func/varlen 请求 LSE 返回二项，kvcache 返回四项，未 split 的两个 accumulator 为空 tensor。PA 的有效零 KV 行一律 `O=0、LSE=-inf`；非 PA 全局空 KV 为 `+inf`。完整 shape 与空值契约见 [design 3.3](design.md#33-共同契约)。

真实/fake 入口及公开 fullgraph compile 已有验收；完整图重放仍在 M16。尚未开放：causal/window、自动或多 split、192/128 prefill、非零 softcap、非默认调度选项和 910C。追加 KV、内置 rotary、量化、backward 等范围排除项见 [design 第 4 节](design.md#4-功能边界)。

## 本次闭环：M08a

独立实现为 `csrc/ascend910/flash_attn_npu_3/mla/mla_causal.h/.cpp`，来源和算法契约见 [design 6.2.4](design.md#624-独立-cpu-causal-范围与分核)。保留原草稿的接口含义和版权，原文件备份为 `build/mla-m08a/mla_causal.h.draft`。

- 块范围使用逐样本有效 Sq/Sk，给出 S1G 行/query/key/S2 半开区间、前导全遮罩行和源成本；覆盖跨头、尾块、空 Q/KV 与宽整数边界，失败保持输出不变。
- 复用未修改的固定源 SplitCore，将候选端点映射到完整 M 块并逐块重算 causal 成本。最终端点为 `(B,0,0)`，所有有效 Q/head 行都有唯一所属核，包括零成本尾部；不把首块成本乘块数用于 causal。
- 测试从原始 BSND/TND/混合/PA allocation 与 used 构造长度，迁移侧复用既有解析器，独立对照自行解读原始输入。固定源 oracle 不链接迁移适配器；另比较迁移 SplitCore 的全部原生 FD 字段，再以逐行可见 key 和每核任务枚举检查完整分区。
- 本步未修改共享调度、长度解析、Kernel/Block/Tile、metadata/AICPU、公共入口或生产构建清单。新增组件没有被生产路径引用，见 `production-isolation.json`；不据此宣称 causal 前向已开放。

证据根目录为 **`build/mla-m08a/`**；最终完整实测在 `attempt-1791780656927232530/`，汇总为 `acceptance.json` 和 `summary.json`。保存统一入口命令/环境、完整节点清单、XML、固定源 blob、18 个本步/直接依赖源码快照、编译命令、全部输入输出及进程退出记录。

| 本次检查 | 结果 | 耗时 |
| --- | --- | --- |
| `tools` | 57 项通过；原执行器/缓存检查及新组选择、依赖失效、进程协议、异常退出与 stderr 传播 | 8.45 秒 |
| `causal-cpu` | 68 项通过；1,713 组块几何、1,450 组分核输入（含重复调用）、32 组拒绝输入 | 21.95 秒 |
| `causal-sanitize` | 同一 68 项完整通过；适配器、固定源、迁移 SplitCore 均启用 ASan/UBSan，退出时 LeakSanitizer 无错误 | 61.40 秒 |

本次共 **193 次 pytest 通过，对应 125 个去重节点**；最终失败/跳过均为 0。前置 smoke 属于完整集合，不重复计数。普通/sanitizer 两组的节点与输入输出逐项一致，见 `selection-equivalence.json` 和 `acceptance.json`。三个驱动在每组内分别复用独立进程，输入/输出逐项落盘，完整集合结束后仍检查退出码和 stderr；工具测试包含 EOF、错误 JSON、额外输出、非零退出及退出时 sanitizer 诊断。

保留的失败和中断不计为通过：

- `smoke.log`：驱动缺少 `<algorithm>` 导致编译失败，补头文件后代表用例通过，后续完整组包含这些用例。
- `attempt-1791780126583958760/`：对照与枚举结果一致，但覆盖断言误要求固定源必须遗漏零成本尾部；改为断言测试包含真实零成本尾部，继续严格验证全部行的唯一归属。
- `attempt-1791780231653664210/`：普通组通过，sanitizer 被沙箱的 LeakSanitizer 进程检查限制阻断，未关闭 sanitizer；转到宿主执行。
- `attempt-1791780323894592770/`：原驱动反复启动进程使 LeakSanitizer 扫描成本过高，181.8 秒时主动中断；调整为每组复用进程。新旧 causal 集合各 68 项完全一致，代表用例和协议失败传播检查后再次冻结实现，最终完整组全部重跑通过。

## 继承证据与环境

本轮继承 **4,100 项历史 pytest**，与本次 193 次执行分别计数。核对脚本、结果和适用边界在 `audit_inheritance.py` / `inheritance.json`，核对耗时约 5.7 秒：

| 继承项 | 历史结果与依据 | 本步匹配范围 |
| --- | --- | --- |
| 旧 CPU 调度 | M07b2 `8fb46d7`；`build/mla-m07b2-paged-schedule/`，1,131 项，另有 31,836 组 sanitizer replay 输入 | 20 个调度/driver/oracle 依赖；唯一历史差异为 PA 声明头的生产使用说明注释，去注释后相同；3,120 个原始证据产物哈希一致 |
| 原构建基础、TND PA 公开及普通入口 | W02–W04 `build/mla-workflow-tools-20261012/cold-summary.json`；分别 2、2,145、822 项 | 原组选择、用例和生产依赖未变，原始证据完整；执行器新增 CPU 组的影响由本次 `tools` 实测覆盖 |

生产构建记录的 **6,317 个依赖、11 个产物**哈希一致；现有输入中只有未接入生产的 causal 草稿发生改变，新文件未进入 setup 显式源码清单。实际加载路径仍为本仓 `flash_attn_npu_3/flash_attn_npu_3.cpython-311-aarch64-linux-gnu.so`，SHA256 为 `4e2cf66ad8a507b1c3bad54bf3ab5f440ab928ffbd95041104e1c54470e3782c`。本轮 `runtime.json` 与上轮实测记录逐字段相同，含 230 个实际加载库；不能把新增 CPU 文件误认为已经编入此扩展。

本次实查环境为 Python 3.11.15、torch 2.7.1+cpu、torch_npu 2.7.1、CANN 9.0.0-beta.2、910B3；CATLASS `769cd40a`，只读固定 CANN 源 `ca18aee6`。开始时第 6 卡无运行进程；本步只探测/加载环境，未重建扩展或运行 NPU 数值/性能。进程检查没有发现需恢复的旧验收任务；本轮中断进程已收尾。

旧前向闭环的完整 9,614 项及双 dtype 各 36 个适用非 FD 公开 E2E 段仍保存在 `build/mla-m07e3c-tnd-paged-public/acceptance.json`。其历史比值 FP16 0.348..0.683、BF16 0.353..0.677；原生 FD 各 6 例仍单列。**本轮没有重测这些完整矩阵或性能，也不再将该旧整组加入上述继承计数。** 本步未改设备/公开行为，NPU 数值、组件性能、公开 E2E、普通入口额外实测和完整图重放不属于本步必跑项。

证据、缓存和本轮源码快照都在 Git 忽略目录，不随提交分发，迁机须复制依赖或重新验收。流程 W01–W05 的其他记录仍在 `build/mla-workflow-tools-20261012/`、`build/mla-cleanup-20261012/`；外部无头脚本 `/home/zhangqijia/repo/mla_codex_loop.py` 本步未改，不纳入 Git。

## 唯一下一步与复现

下一步仅为 [M08b1：BSND 独立 causal 数值组件](plan.md#当前步-m08b1bsnd-独立数值组件)：补固定源 causal probe/数学适配，再接任务环、Vector mask 与全遮罩 O/LSE，完成受影响设备路径和代表组件性能。不得用本次 CPU 结果或旧无 mask 矩阵代替设备验收；首次公开 BSND 行为在 M08b2。

本步最小闭环命令：

```bash
python3 -m tests.mla_validation --output build/mla-m08a --suite tools causal-cpu causal-sanitize --dry-run
python3 -m tests.mla_validation --output build/mla-m08a --suite tools causal-cpu causal-sanitize --collect-only
python3 -m tests.mla_validation --output build/mla-m08a --suite tools causal-cpu causal-sanitize --resume
PYTHONPATH=/tmp/mla-m01-precommit PRE_COMMIT_HOME=/tmp/mla-m01-precommit-cache \
bash ci/code_quality.sh --staged
git diff --check
git diff --cached --check
```

`--resume` 只继承依赖和证据仍匹配的成功组；新实测使用新输出目录或去掉 `--resume`。sanitizer 须在允许 LeakSanitizer 进程检查的环境执行，本机最终组使用宿主环境。文档按新增 CPU 契约及统一入口的影响核对，本轮不是 M08 阶段结束，不启动全面文档审计或下一功能步。
