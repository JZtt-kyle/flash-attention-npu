# MLA 当前状态

<!-- mla-loop-next: none -->

**R7 代码归并、构建与实机验证完成。** 基线 `a48ba27`，分支 `feat/mla-cann-migration`。本轮按用户要求，以当前库已有接口范式继续收敛 MLA，一次提交完成目录、文件、构建与文档整改。产物与完成条件见 [plan](plan.md)，组织与来源见 [design 第 6 节](design.md#6-代码结构)。本次闭环提交可用 `git log -1 -- docs/mla/status.md` 定位。

## 当前交付

MLA **106 → 49 文件**、内部子目录 **7 → 0**、源码 **32,203 → 31,601 行**。连同外层 `autogen/` 的 38 个生成实例，总数 **144 → 87**；没有将移动文件算作删除。

- 参照普通 FA3 的 dispatch / metadata / tiling / split 组织：公开 host 和仅由它调用的 PA 静态准备集中于 forward；基础形状/长度/mask 放在 tiling，split/GQA 计划放在 split，记录声明与生产集中。
- 单片、显式 split、自动 split 各共用一份分派头；PA BSND/TND 共用布局选择。薄 Kernel 包装合入压缩 Kernel，vector 小操作合入同一 Tile。压缩、prefill、INT8 及 Cube / Vector / Tile 的实际职责仍独立可辨。
- 38 个 ASC 实例、非量化 FIA / 量化 IFA 两个 AICPU 编译入口保留。协议版本/字段/宽度、动态校验、发射属性与生命周期不变；公开 Python、测试/reference、阈值、CANN 源仓及 CATLASS 子模块均未修改。

## 最近一次验收

证据根：`/tmp/flash-attention-npu-mla-46928a793876/mla-flat/`。旧源码保存为 `baseline/`，旧扩展为 `baseline-extension.so`。源码和构建变化后重新实测，旧通过数不计作本次结果。

| 检查 | 本次结果与证据 |
| --- | --- |
| 收集清单 | **1,578 项 MLA**，`collect.log`；测试与独立 reference 未修改 |
| 最终构建 | 原 v3/910 配方，**356.94 s**；`build-command.json`、`build-final.log` |
| 代表用例 | **88 passed，44.25 s**；`representative.log` / `.xml`，覆盖双 dtype/表示、五布局、INT8、mask/split/softcap、长 KV、fake/compile/graph |
| 完整 MLA + 普通 FA3 | **1,718 passed，0 failed/error/skipped，309.82 s**；MLA 1,578、普通 140；`acceptance.log` / `.xml` / `acceptance-summary.json` |
| 前后性能对照 | **88 配置，新/旧配置等权平均 1.007228，53/88 变慢，414.21 s**；两侧各 47,200 条 wall 样本；`performance.log`、`interleaved-timing/comparison.json` 和原始 JSON/log |
| 数值与协议等价 | 31 份 Kernel/Block/Tile 数值 token 一致；92 份原文件的声明/函数 token 保留；38 个分派实例展开等价；性能逐例独立 reference 通过，输入/metadata/全部输出哈希相同；`numerical-equivalence.json`、`consolidation-audit.json`、`interleaved-timing/{baseline,fixed}.json` |
| 构建入口与依赖 | 生成器 46 实例（MLA 38）清单一致且幂等；39 个 host MLA 发射符号保留；49 文件的 99 条本地 include 无缺失或循环；`launch-symbols.json`、`include-audit.json` |
| 预检 | host g++ 语法、Bisheng 六类代表编译单元、性能 6 个 MLA 配置与 2 个普通 FA3 对照通过；`asc-syntax-commands.json` / `asc-syntax-results.json`、`performance-preflight.log`、`timing-preflight/` |
| 质量与文档 | 仓库 pre-commit、最终差异、文档链接和 `git diff --check` 通过；`quality-final.log`、`markdown-links.json` |

统一 MLA 命令仍为加载 CANN 环境后的 `python3 -m pytest -q tests/test_flash_attn_npu_v3_mla.py`。本轮使用 `ASCEND_RT_VISIBLE_DEVICES=4 OMP_NUM_THREADS=1`。普通 FA3 140 项为相关集合（metadata 签名/兼容 129、数值 6、compile 3、图 2），不代表全部普通 FA3。精确命令见 `representative-command.json`、`acceptance-command.json`、`performance-command.json`；可用 `subprocess.run(json.loads(path.read_text()), check=True)` 重放。构建与证据在仓库外，不恢复 MLA 专用测试目录或框架。

性能采用与 R6 相同的公开输入、reference、采样与统计口径：同设备/CPU 48–51 两个常驻进程交替，5 次预热、5×20 采样、适用 metadata implicit/generated/reuse 与 R=1/4，先配置内等权再配置间等权。压缩 40、prefill 40、INT8 8 配置的新/旧平均比分别为 **1.003596 / 1.007874 / 1.022152**，变慢数 **21/40、27/40、5/8**；逐配置范围 **0.945963–1.106188**，最大退化为 INT8 BSND causal FD。普通 FA3 双 dtype 的前后对照范围 **0.959612–1.030118**，不计入 MLA 均值。单例退化保留，不归因于单一因素，也不宣称零开销；该数据用于检查组织变化的影响，不是重新执行对固定 CANN 的迁移阶段性能验收。性能预检用时 83.92 s，其结果没有混入完整统计。

首轮归并构建发现 include 顺序问题：host/AICPU 的架构宏处理晚于标量消费者，共用 metadata 的设备内联声明早于 `kernel_operator.h`，vector Tile 合并暴露了对 CATLASS `Min` 的隐式 include 依赖。已保留 host 宏处理、恢复设备头的条件包含并显式声明依赖；停止已失败的构建后通过 Bisheng 预检，再重新冻结源码并完成最终构建/验证。失败日志为 `build-before-dependency-fix.log`；纯平铺构建及失败中间态均不计作最终通过。暂存 pre-commit 首次因 `.git` 只读沙箱无法执行 `git write-tree`（`quality-staged-sandbox.log`）；使用正常索引写权限重试后全部通过，未跳过钩子。

## 环境与历史边界

Python 3.11.15、torch 2.7.1+cpu、torch_npu 2.7.1、CANN 9.0.0-beta.2；CATLASS `769cd40a8716b28650b6bebb08db4834eea4462f`。固定 CANN 来源 `ca18aee610729ca5270a44e2d84c751bc687fbf0`，源工作树 HEAD 为 `22a2355cfa57d1ff60e7013ea32068ef5597e6d7`；继承固定提交的来源与未变算法核对，不使用当前源 HEAD 冒充固定版本。

本轮实机为物理设备 **4（Ascend910B3）/逻辑 0**。实际加载 `flash_attn_npu_3/flash_attn_npu_3.cpython-311-aarch64-linux-gnu.so`，SHA256 **`92ca4037ef8a6e08632090b12bb675ac62ca3b924a332d6897d4d32006bea233`**；基线扩展 SHA256 `4097b068b5a2451ed1801107ffb013418e3f1a89c70fed5ca9c96a41146abce6`。环境、设备与实际路径见 `environment-final.json` 和 `npu-before-tests.log`；140 个生产/测试依赖的身份见 `source-final-sha256.json`，完整回归及性能结束后核对未变。

R5 既有 CANN 性能差距及当次收尾例外见 `git show 87098f3:docs/mla/status.md`，本轮不借用该例外。INT8 M21 图性能仍按既有授权暂缓，一般随机 INT8 的固定源/独立数学限制保持。本轮未重跑固定原生 CANN、Trap、完整迁移阶段或图性能，也未做 910C 实机验收。R6 历史闭环见 `git show a48ba27:docs/mla/status.md`。

唯一下一步：`none`，当前已授权的迁移与结构收敛范围已完成。
