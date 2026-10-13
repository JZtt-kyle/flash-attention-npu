# MLA 当前状态

**M08d：BSND PA causal 已接通并通过必要验收；唯一下一步是 M08e。** 分支 `feat/mla-cann-migration`，本步基线 `527fd6891b52ca6aa05f56e55882a1ee05c0a3c5`。本次只提交 M08d，提交成功后立即停止；当前闭环提交用 `git log -1 -- docs/mla/status.md` 定位。

## 当前可调用能力

- 512+64、FP16/BF16、Hkv=1、`num_splits=1` 下，BSND Q + ND PA KV 已支持右下 causal 的只读 cache、公开 metadata 生成/复用和内部 Q-used；全部合法 P 为 16..1024 的 16 倍数。TND PA causal 仍明确拒绝。
- 独立 v10（`version=kind=10`、`408+5B` 个 int32）复用 PA 切块/页 Copy 和既有 causal 完整 M 分核/数值任务环；v6/v10 双向拒绝。mask 按有效 Sq/Sk 定位，消费时校验所有有效页，包括 Q/used 为零的样本；padding 页不读取。
- 公开 cache 支持 None/整数/tensor KV 长度、合法 stride/offset 和 v10 复用。内部 Q-used 每次生成新记录且拒绝显式复用，不给公开接口新增 Q-used 参数。记录不绑定页 ID、物理池大小、地址、合法 stride 或 scale；长度/P/C 改变须重建。全遮罩及 PA 零 KV 均为 `O=0、LSE=-inf`。

契约见 [design 5.12](design.md#512-v10bsnd-pa-causal-记录)，本步产物与影响路径见 [plan](plan.md#最近完成-m08dbsnd-pa-causal)。CATLASS、Block/Tile、causal 数值任务环及固定 CANN 源未改。

## 最近一次闭环证据

证据根 `build/mla-m08d/`，数学缓存 `build/mla-validation-artifacts/golden`。本次有效结果 **20 组、1188 项 pytest 通过**（组之间可能有重复节点）；另继承历史组件证据 326 项。未解决失败 0，必要未运行 0；失效/失败/中断尝试单列于下节，不计通过。

统一复现：`python3 -m tests.mla_validation --output build/mla-m08d/<新批次> --device 6 --max-cases 100 --suite <组>`。先 dry-run/collect；恢复前核对进程、依赖和完成记录。最终汇总 `final-evidence-audit.json` 核验 851 个源码身份及 4683 个原始结果文件（5.6 秒），保留每组确切 summary/命令/源码差异；`final-selection-audit.json` 保存全部组候选、选中与未选中数量，各组目录保留完整节点清单。

| 本次检查 | 通过数 | 执行耗时（秒） | 根目录下的确切批次 |
| --- | --- | --- | --- |
| `causal-paged-cpu` | 36 | 32.9 | `cpu-final/attempt-1791891883525338020` |
| `causal-paged-sanitize` | 36 | 112.7 | `cpu-resume/attempt-1791892305321400320` |
| `paged-shared-cpu` / `paged-shared-sanitize` | 100 / 100 | 55.5 / 786.4 | 同上 |
| `tools` / `causal-paged-tools` / `cpu` | 100 / 55 / 2 | 19.9 / 6.1 / 5.5 | `dispatch-tools/attempt-1791897082693314130` |
| `causal-paged-check` | 6 | 139.1 | `representative-resume/attempt-1791892302006253970` |
| `causal-paged-public` / `causal-paged-trap` | 70 / 26 | 1272.0 / 593.6 | `final-devices/attempt-1791892503150876240` |
| `regression-paged` / `regression-nonpaged` | 100 / 86 | 106.9 / 771.7 | 同上 |
| `causal-bsnd-check` / `causal-tnd-check` / `causal-mixed-check` | 56 / 6 / 6 | 107.3 / 102.3 / 103.8 | 同上 |
| `ordinary` | 100 | 28.1 | 同上 |
| `bsnd` / `public` | 100 / 100 | 59.1 / 48.9 | `shared-public/attempt-1791895707934293410` |
| `metadata` | 100 | 58.0 | `metadata-resume/attempt-1791896104453069170` |
| `causal-paged-profile` | 3 | 648.3 | `profile-retry/attempt-1791897195026249460` |

PA 固定回归 120 选 100，非 PA 86 全选，ordinary 816 选 100；补充 `bsnd/public/metadata` 分别从 1872/2145/2844 选 100。metadata 的四个新增 v10 节点全部入选。新公开 70、Trap 26、代表 6 全选；全部 64 个合法 P 在每 dtype 的固定循环节点中实测，逐例输入/输出留存。覆盖跨页 mask、乱序/重复页、Sq<Sk/=/>、Q-used/poison、零 KV/空行、记录更新与新页池/stride/scale 复用、真实/fake/compile、流/allocator 生命周期及设备 Trap。

新增 PA causal 精度记录中，每 dtype 的 O、LSE 各 709 次比较均为固定源逐元素一致且双数学通过，另各有 27 次空值检查；本步新路径源精度限制计数为 0。原生公开对照的 15 个配置/每 dtype 另与双数学核对，FD 不要求与迁移核逐元素一致。源全遮罩 LSE 的 `+inf` 先单独断言，再按 FA3 `-inf` 契约归一。原始记录见代表、公开和性能组的 `causal-paged-accuracy/results.jsonl`、`public-profile/*-dispatch/accuracy-*.pt`。

数学缓存的数值组共命中 287 次、冷算并写入 322 次；性能重跑 66 次全部命中。工具组的关闭/刷新/损坏故障注入另计。代表、正式验收和性能使用同一缓存，同输入 reference 不重复冷算，双方设备实现仍实际执行。

## 公开与组件性能

`profile-retry/attempt-1791897195026249460/causal-paged-profile/public-profile/average.json`：固定 30 个配置含双 dtype，18 个原生非 FD 配置参与验收，12 个原生 FD 配置单列；适用配置的等权平均耗时比 **0.495594 ≤ 1**，退化配置 **0/18**、退化分段 **0/108**。不按耗时筛选；原始单例、分段、metadata 生成和 Kernel 时间均保留。

| 公开调用段 | R=1 平均耗时比 | R=4 平均耗时比 |
| --- | --- | --- |
| 隐式生成 | 0.625614 | 0.486543 |
| 公开生成一次再前向 | 0.714017 | 0.387471 |
| 预生成记录复用 | 0.449020 | 0.310901 |

双方同一 CPU 亲和性、5 次预热、5 轮×20 次 wall；必要 Q/cache/页表物化、输出分配和完成同步计入原生 E2E，真实 causal mask 一次性准备在外。固定源原生 PA 包及实际二进制/tiling key 核验见 `native-vendor-audit.json` 和性能组 `*.dispatch.json`。第六位编码为 `PA*2+FD`，仅低位表示 FD。

同调度源代表组件另采 5 轮×10 次，生产 attention 含当次 metadata/页表校验，AICPU 单列。三例顺序为 P112 跨页/部分遮罩、P528 多 query、P16 内部 Q-used/重复页；FP16 Kernel 耗时比分别 **1.338/1.147/1.818**，BF16 为 **1.170/1.112/1.391**。六个组件单例均较源核慢，完整耗时及 profiler 在 `component/{fp16,bf16}/summary.json`；按 design，第 7 节的不劣化门槛为上述公开 E2E 平均值，不把组件退化记成提速。

## 证据继承与失败记录

- 接手时工作树干净；基线 844 个源码身份与 18 组 3579 个原始产物全部核验，见 `baseline-audit.json`。`component-inheritance.json` 再核验 13 个未变算法依赖和 M08b2 数值/组件证据 326 项、1577 个原始产物（6.5 秒），仅继承未变组件及最长 reference/最大内存边界；新 PA causal 协议、分派、mask/页地址和公开性能均本次实测。
- 首轮代表集合在沙箱因 NPU 不可见而导入失败，主机环境重新收集成功。首轮生产编译成功，但执行器因期间新增性能驱动而拒绝整树身份；第一次 sanitizer 的 36 项也实际通过，但因驱动源码变化失效。原尝试不计通过，源码固定后构建复用核验及 sanitizer 重跑通过。
- shared metadata 首轮 78 项通过后，旧 `opcheck` 参数驱动遗漏新增可选 `causal` 的缺省行为而失败。两处签名参数提取均已修正；候选/选中/未选中清单不变，整个失败组 100 项重新通过，见 `metadata-test-change.json`。
- 第一轮性能在 605.1 秒主动中断：旧分派解析器将 PA 非 FD 误判为 FD。按固定源编码修复并补四种组合/证据失效测试后，工具及原固定性能组重新通过。原始日志保留在 `profile-final/`，不计通过；来源、改动及不变计时口径见 `dispatch-parser-fix.json`。
- 最终实现身份在 `final-sources.json`。早期 CPU 36 项跨无关 docstring/性能驱动变化的有效性见 `cpu-inheritance.json`；后续 metadata 测试与性能分类工具改动未触及数值实现。最终审计按实际调用依赖保留已完成组，仅重跑受影响工具、metadata 和性能，未重复整个长矩阵。

## 环境、构建与提交检查

Python 3.11.15、torch 2.7.1+cpu、torch_npu 2.7.1、CANN 9.0.0-beta.2，第 6 卡 Ascend910B3，20 AIC/40 AIV。CATLASS `769cd40a8716b28650b6bebb08db4834eea4462f`；只读固定 CANN `ca18aee610729ca5270a44e2d84c751bc687fbf0`。环境和进程快照在证据根；无上步遗留验证任务。性能前设备健康且 AICore 为 0，存在三个各约 136 MiB 的外部常驻上下文，未终止它们；快照见 `before-profile-retry-device.log` 及中断后的进程记录。

实际加载本仓 `flash_attn_npu_3/flash_attn_npu_3.cpython-311-aarch64-linux-gnu.so`，SHA256 `30821e39c0bfe5a379074f6f7419d13bde40875efe69d4a4a172bcbba7ffa46b`。首次重建/probe 准备 498.9 秒；最终复用核验 1.4 秒、实际加载核验 34.7 秒。构建 manifest 在 `build/mla-validation-artifacts/prepare.json`，最终运行加载身份随性能组保存。设备与 LSan 使用主机可见性，shell 为 `login=false`。

提交前检查本步契约、链接/表格和 `git diff --check`，仅暂存本步文件，质量入口为 `PYTHONPATH=/tmp/mla-m01-precommit:$PYTHONPATH ci/code_quality.sh --staged`；记录在 `document-links-final.json` 与 `quality-final.log`。原始结果、构建和缓存均在忽略目录；迁机须复制实际依赖与证据或重新验收。

## 唯一下一步

[M08e：TND PA causal 与阶段收尾](plan.md#唯一下一步-m08etnd-pa-causal-与阶段收尾)。先复核本步状态/源码/进程和证据，再细化 TND PA 协议、前缀/used、生成/消费及统一专项；冻结后按 design 限量规则实测全部已开放 causal 组合及阶段性能。M08e 尚未开始，本次提交后停止。
