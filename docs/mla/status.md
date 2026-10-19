# MLA 当前状态

<!-- mla-loop-next: none -->

**R6 生产结构收敛及验收完成，无待开发步骤。** 基线 `87098f3`，分支 `feat/mla-cann-migration`。用户于 2026-10-19 授权按固定 CANN 来源与 CATLASS 职责整改；本轮一次完成适配层、构建、文档与必要验证，不再按布局、协议或组件拆开发轮。产物与完成条件见 [plan](plan.md)，架构见 [design 6.7](design.md#67-生产接入与构建)。本次闭环提交可用 `git log -1 -- docs/mla/status.md` 定位。

## 当前交付

MLA 从 **239 → 106 文件**、顶层 **205 → 9**，源码行数 **33,432 → 32,203**。按 `tiling/`、`metadata/`、`device/` 与原 Kernel/Block/Tile 组织；保留独立来源算法和版权，不修改 CANN 或 CATLASS。

- 24 个手写 dtype 实例交给既有生成器；连同 `autogen/` 的 MLA 实例计算，生产文件从 **253 → 144**，净减少 **109 个**。实际 38 个 MLA 编译实例保持齐全，没有把移动文件算作删除。
- 合并布局参数、发射声明、序列调度、显式/自动 split 计划和八类 metadata；抽取五布局重复序列化、页容量规则和预算计算，保留布局专属校验及地址语义。
- 删除两个无生产引用组件和三个宏转发头。非量化 FIA 与 INT8 IFA 各保留一个 AICPU 编译单元，替代原 21 个入口文件；32 个函数入口及参数、记录协议不变。
- 公开 Python、E2E/reference、阈值和功能范围未变。31 份保留的 Kernel/Block/Tile、host 前向的数值/调用 token 相同；32 个 AICPU 入口函数体相同。清理过时开发阶段注释，文档按当前职责收敛，历史由 Git 保存。

## 最近一次验收

证据根：`/tmp/flash-attention-npu-mla-46928a793876/mla-structure/`。本轮重建并实测，不继承旧扩展的通过数；最终源码、测试/reference 与扩展身份为 `implementation-final-sha256.json`，采样结束再次核对未变。

| 检查 | 结果与证据 |
| --- | --- |
| 收集清单 | **1,578 项 MLA**；`collect-baseline.log`，测试源码未变，最终实际节点以 JUnit 为准 |
| 完整 MLA + 普通 FA3 | **1,718 passed，0 failed/error/skipped，278.34 s**；MLA 1,578、普通 140；`acceptance.log` / `.xml` / `acceptance-summary.json` |
| 最终代表用例 | **88 passed，43.79 s**；双 dtype/表示、五布局、mask/split/softcap、长 KV、INT8、fake/compile/graph；`representative-final.log` / `.xml` |
| 前后公开性能对照 | **88 配置，新/旧平均比 1.017194，60/88 变慢**；两侧各 47,200 条 wall 样本；`interleaved-timing/comparison.json` 和原始 JSON/log |
| 输入及行为等价 | 性能集合逐例独立数值检查通过，输入、metadata、全部输出哈希与基线相同；`interleaved-timing/{baseline,fixed}.json` |
| 结构与构建入口 | 31 份数值实现、32 个 AICPU 入口、生成器 46 实例（MLA 38）、五布局序列化、导出符号和本地 include 核对通过；`structure-audit.json` / `serializer-equivalence.json` / `export-audit.json` |
| 质量与文档 | 仓库 pre-commit、`git diff --check`、文档路径及 230 个本地锚点核对通过；`quality-final.log` / `markdown-links.json` / `final-checks.json` |

普通 FA3 的 140 项仍是 R5 的相关子集（metadata 签名/兼容 129、数值 6、compile 3、图 2），不是全部普通 FA3。统一 MLA 入口仍为加载 CANN 环境后的 `python3 -m pytest -q tests/test_flash_attn_npu_v3_mla.py`，设备选择 `ASCEND_RT_VISIBLE_DEVICES=4 OMP_NUM_THREADS=1`。完整集合与性能的精确命令分别在 `acceptance-command.json`、`performance-command.json`；可直接以 `subprocess.run(json.loads(path.read_text()), check=True)` 重放。构建与证据均在仓库外，不恢复 MLA 专用测试目录或框架。

性能固定为压缩 40、prefill 40、INT8 8 配置，同设备 4/CPU 48–51 的两个常驻进程交替执行；5 次预热、5×20 wall、适用 metadata implicit/generated/reuse 与 R=1/4 均保留，先配置内等权再配置间等权。分组新/旧平均比分别为 **1.009898 / 1.035109 / 0.964103**，变慢数为 **23/40、37/40、0/8**。普通 FA3 双 dtype 在采样前后作对照，平均比范围 **0.982468–1.033900**，不计入 MLA 均值。不能把小幅耗时差异全归因于重构，也不宣称零开销或确定加速；该对照用于评估结构影响，不冒充重新完成固定 CANN 的迁移阶段性能验收。

首版 88 项代表检查通过，但合并 INT8 与非量化 AICPU 后，性能预检发现显式 metadata 开销上升，遂按固定源独立职责恢复 INT8 编译边界；首版扩展与 `timing-preflight/`、`timing-same-binary/`、`timing-aicpu-isolation/` 均保留，不计最终通过或混入最终均值。最终短 profiler 的 BSND metadata 中位数为旧/新 30.690/31.521 μs，attention 约 44–45 μs，见 `profile-boundary-summary.json`。ACL/NPU flow 关联与 FWK API DB 导出报错，仅采用实际生成的 kernel CSV 作诊断，不称完整关联导出成功。

## 环境与历史边界

Python 3.11.15、torch 2.7.1+cpu、torch_npu 2.7.1、CANN 9.0.0-beta.2；CATLASS `769cd40a8716b28650b6bebb08db4834eea4462f`。固定 CANN 为 `ca18aee610729ca5270a44e2d84c751bc687fbf0`；源工作树 HEAD 为 `22a2355cfa57d1ff60e7013ea32068ef5597e6d7`，来源核对直接读取固定提交，源仓只读。

最终验收使用物理设备 **4（Ascend910B3）/逻辑 0**。设备 1 初始化报 507033，首版代表检查改在设备 3 完成；设备 3 后被其他作业占用，短 profiler 也在初始化阶段失败，未产生可用样本。设备 4 最小张量检查通过后，最终正确性和性能均在此完成；失败日志及设备占用查询保留，未干预其他作业。

生产使用原 `setup.py build_ext --inplace`、v3/910 配方，日志 `build-verified.log`。实际加载 `flash_attn_npu_3/flash_attn_npu_3.cpython-311-aarch64-linux-gnu.so`，SHA256 **`4097b068b5a2451ed1801107ffb013418e3f1a89c70fed5ca9c96a41146abce6`**；旧扩展保留为 `baseline-extension.so`，SHA256 `ee17fa9fda69303f08ca7a2d6867d815f188f5324461fb518875e4f02c074717`。

R5 既有默认路径性能差距仍未解决，详见 `git show 87098f3:docs/mla/status.md`：原调度/eager 固定矩阵对 CANN 为 0.723021/0.746345，新增默认路径为 1.104532，其收尾例外仅适用 R5。本轮不借用该例外，也不把历史结果计作本次性能通过。INT8 M21 eager 为 0.707703，graph 为 1.197126，图性能按既有授权暂缓；一般随机 INT8 的固定源/独立数学限制保持。本轮未重跑固定原生 CANN、Trap、完整迁移阶段或图性能，也未做 910C 实机验收。

唯一下一步：无。R6 单次提交后停止；后续数值、功能或性能工作另行确定范围。
