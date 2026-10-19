# MLA 当前状态

<!-- mla-loop-next: none -->

**R5 的测试精简与两处数值修复已完成，正确性及原性能矩阵复测通过；新增默认路径集合的既有性能差距按用户明确授权记录后收尾。** 基线 `7ea2bca`，分支 `feat/mla-cann-migration`。非量化 M17、INT8 M18–M21 与 R4 单文件整理已完成；本轮按用户“功能组合完整，同一组合减少重复输入”的要求整理，并按后续授权修复新用例检出的数值问题。2026-10-19 用户确认“记录既有性能差距，提交本轮改动”，例外仅限 R5，见 [design 第 7 节](design.md#7-验收口径)。新增集合仍未达标，不能替代 M17 原矩阵，或将其写成本次数值修复引入的整体退化。

## 当前交付

仍只有 [test_flash_attn_npu_v3_mla.py](../../tests/test_flash_attn_npu_v3_mla.py) 一个普通 MLA E2E 文件。用例 **1,814 → 1,578**，净减少 **236 项（13.01%）**：去掉 326 项重复展开，补充 90 项确定性数值回归。

- 主矩阵保留五布局 × 两种表示 × 双 dtype × 三 mask × 三 split × softcap 关/开，共 360 个完整组合。softcap 使用 saturation 输入，使忽略参数产生可检查的数值差异。
- 184 项具名边界保留空 B/Q/KV、packed/unpacked、窗口归一化、空行、长 KV、Q/KV 尾块、最大 split、舍入、饱和与抵消。旧 32 个前向场景中，7 个并入主矩阵，25 个进入边界表。
- GQA/页大小减少同类别取值；int32 长度/metadata 存储形式、无 host readback 和 grad 拒绝使用 BF16 代表；共享拒绝保留全部错误值、real/fake 及入口分派。stride、图/compile、动态长度/前缀、生命周期与 INT8 226 项保留。
- 修复跨 KV 块位修正将零 PV 变成 NaN、随后丢失非零贡献的问题；非零 softcap 使用实际送入 Cube 的 P 总和归一化，修复近常量 V 的偏差。复用源 RowSum、原缓冲及握手，LSE 统计不变。
- 新增块内抵消 60 项、softcap 归一化 30 项，均不依赖随机种子。原失败输入、独立 reference、断言和精度阈值保留；公共 conftest、构建配方、CI 和 CATLASS 不改。

数值差异见 [design 6.5](design.md#65-数值路径与资源布局)，组合保留规则见 [design 7.3](design.md#73-增量验收与证据继承)，产物与完成条件见 [plan](plan.md)。

## 最近一次验收

本轮证据根：`/tmp/flash-attention-npu-mla-46928a793876/mla-test-pruning/`。以下最终版结果使用同一源码、测试与扩展身份，见 `numerical-final-sha256.json`；不继承旧扩展的通过记录。

| 检查 | 最终结果与证据 |
| --- | --- |
| 完整 MLA + 普通 FA3 | **1,718 passed，0 failed/error/skipped，279.52 s**；其中 MLA 1,578、普通 140；`acceptance-numerical-final.log` / `.xml` / `acceptance-numerical-summary.json` |
| 完整集合与覆盖核对 | 收集 1,578 项；主矩阵 360 组合无缺失/重复，旧场景有映射，精简后的原 1,488 节点全部保留、另增 90 项；`coverage-audit-numerical-final.json` / `final-mla-nodes.txt` |
| 两处原失败与确定性回归 | 92 passed，19.49 s；`numerical-probe-final.log` / `.xml` |
| 原失败种子跨路径复核 | 双 dtype、五布局、三 split 及适用 metadata，共 54 条通过；`verify_original_paths.json` |
| 固定源与双数学精度 | 40 项目标均通过；37 项输出/LSE 与固定源逐元素一致，3 项修复源错误；全部 LSE 仍一致；`source-check-final.json` / `source-probes-final/` |
| 数值看护检查 | 忽略 softcap 的 60 次、忽略 mask 的 40 次错误公开调用均被原数值断言检出；`sensitivity_numerical_final.json` |
| 修复前后公开耗时 | 60 配置交替采样平均比 **1.012591**，40/60 变慢；未改动普通 FA3 对照为 1.016307–1.042033；`interleaved-timing/` |
| 固定原生 CANN 预检 | 50 配置双 dtype 的独立数值与 template 5/原生 FD 分派均核验；10 个自动 split 适用、40 个单片/显式 split 单列；`native-timing-final/*-dispatch.json` |
| 新增固定源公开诊断集合 | **该集合平均未达标**：50 配置实测，10 个适用配置平均比 **1.034434 > 1**，其中 5 个变慢；FP16 0.980363、BF16 1.088505；40 个非同义 FD 对照单列；`native-timing-final/summary.json` |
| 恢复原调度矩阵 | 20 个适用配置，修复前/CANN **0.727590**、修复后/CANN **0.723021**；当前 0/20 慢于 CANN；历史 M17 为 0.726740 |
| 恢复原 eager 矩阵 | 40 个适用配置，修复前/CANN **0.750347**、修复后/CANN **0.746345**；当前 0/40 慢于 CANN；历史 M17 为 0.796747 |
| 新增默认路径同场对照 | 同一 10 项，修复前/CANN **1.128327**、修复后/CANN **1.104532**，两版均 8/10 慢于 CANN；修复后/修复前平均 **0.998012**；既有差距仍未达标，不能写成新增回退 |
| 同调度组件性能 | 36 配置的 Kernel 平均比 **1.018209**、预分配 probe wall **1.008474**；3,600 条 Kernel、7,200 条 wall 样本；`component-timing-final/summary.json` |
| 质量与身份 | 仓库 pre-commit、`git diff --check`、变更 Markdown 链接及最终源码/测试/reference/扩展哈希核对通过；`quality-final.log` / `final-checks.json` |

普通 FA3 的 140 项是相关选跑子集（metadata 签名/兼容 129、数值 6、compile 3、图 2），不是全部普通 FA3 用例。最终集合的 JUnit 记录是实际节点清单的依据：CANN 警告与收集 stdout 混排时曾粘连一个节点名称，未影响收集或执行，核对使用结构化 JUnit。正常测试只依赖仓库已有 pytest 与公共 helper；固定源诊断在仓库外复用 R4 归档工具，不恢复旧测试框架。

历史性能口径复核：M17 调度 20 个适用配置和 eager 40 个均含 compressed/prefill。调度自动项为显式 pack/margin、window、Q=(3,1)、KV=(4097,1057)，eager 另有短 KV 单片；新增 10 个自动诊断项只有 compressed、默认 pack/margin、无 mask、Q=(7,3)、KV=(4097,513)，实际分派也不同。10 个原输入/helper 与 M17 字节一致，25 份固定 native 包/分派原始证据的哈希已核对（`historical-evidence-check.json`）。恢复原矩阵 60 项，加新增诊断 10 项，在相同设备 1/CPU 48–51 的两个常驻进程交替测修复前后；各进程也交替测固定 CANN。全部 70 项在两扩展上均实际通过独立数值及 native 预检，输入 tensor 哈希逐项一致；5 次预热、5×20 wall、原入口/R、分派适用性和计时边界不变。

完整原始样本、逐例/入口/R、退化数和身份在 `historical-timing/`，严格汇总见 `comparison.json`；精确命令为 `historical-timing-command.json`，脚本为工作根的 `historical_timing.py`。调度/eager 的修复后/修复前直接耗时比分别 1.020642/1.012373，同进程的未修改 CANN 对照分别 1.027295/1.018061，因此不把小幅差异全归因于修复，也不宣称确定加速。新增 10 项的前次 1.034434 和同场 1.104532 均保留，不混合不同集合或挑较好的一轮验收；同场修复前已为 1.128327，确认其差距早于本次修复。M17 历史测量为设备 7/CPU 0，历史扩展也不同，只用于解释原固定矩阵；本轮判断修复影响使用保留的 `7ea2bca` 扩展作同场基线。

复测准备保留 `historical-timing-preflight/` 的 10 项成功代表检查。首次完整尝试完成两项调度后，核对发现原 eager 驱动在计时前释放上一结果，遂主动中止、修正临时驱动对应边界并从头完成 70 项；中止数据及旧脚本保存在 `historical-timing-interrupted-boundary-check/` 和同名日志，仅作诊断，不混入最终汇总。生产、用例和数值预检不变。

修复前后顺序采样的平均比为 1.269700（57/60 变慢，`numerical-final-timing/`），随后因未改动路径也变慢而补充交替对照：两个常驻进程在导入 torch 前绑定相同 CPU 48–51，交替测同设备相同输入，完整保留 60 配置、5 次预热、5×20 wall 和 R=1/4。普通 FA3 对照也有 1.6%–4.2% 差异，不能将前后比值全归因于修复，也不能据此宣称无开销；softcap 10 配置的前后平均比为 1.015111，没有同义 CANN 对照。固定源的公开平均按 design 的分派适用规则另行判定，不用这些诊断比值替代。

公开与组件采样串行使用设备 1；固定 native 包 `/tmp/mla-m13b/native` 的配方、文件哈希及实际二进制注册/分派已校验。公开保存所有入口/R 的原始样本，按配置内及配置间等权汇总；复现参数见 `performance-commands.json`。组件复用同调度的已验收输入，Kernel CSV 与原始 `op_summary` 按设备、名称、起始时间和时长逐行匹配；profiler 的 ACL/NPU flow 关联与 FWK API DB 导出报错，未用于 Kernel 时长结论，不称完整关联导出成功。临时公开驱动曾在计时开始前因 tuple/JSON list 清单比较错误退出，未产生样本；修正后保留原预检脚本身份与失败日志 `native-timing-preflight-error.log`，再完成上述采样，生产及输入未变。

两次中止检查如实保留：`acceptance.log` 为 269 passed / 1 failed，`acceptance-final.log` 为零值修复版的 333 passed / 1 failed；二者不计最终通过。原零值失败 seed 为 880500170，softcap 原节点为 `window-softcap-single-paged_bsnd-compressed-bf16`，阈值不变。新回归加载原始扩展时，抵消组 20 failed / 40 passed、归一化组 15 failed / 15 passed，均为预期检错证据（`old-regression`、`old-normalization` 日志和 XML）。

加载 CANN 环境后直接运行普通入口：

```bash
ASCEND_RT_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 python3 -m pytest --collect-only -q tests/test_flash_attn_npu_v3_mla.py
ASCEND_RT_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 python3 -m pytest -q tests/test_flash_attn_npu_v3_mla.py
```

完整 1,718 项的精确参数保存在 `acceptance-command.json`，可在本机重放；`run.py` 仅恢复已验证 CANN 环境：

```bash
python3 - <<'PY_REPLAY'
import json
import subprocess
from pathlib import Path
work = Path("/tmp/flash-attention-npu-mla-46928a793876/mla-test-pruning")
subprocess.run(json.loads((work / "acceptance-command.json").read_text()), check=True)
PY_REPLAY
```

## 环境与历史边界

Python 3.11.15、torch 2.7.1+cpu、torch_npu 2.7.1、CANN 9.0.0-beta.2；最终验收使用物理设备 1（Ascend910B3，可见逻辑设备 0）。最初的设备 7 后被其他作业占用，旧扩展复现曾报初始化错误 507033，改用设备 1 后重试完成；未干预其他作业。最终性能前后也均使用设备 1。

实际加载 `flash_attn_npu_3/flash_attn_npu_3.cpython-311-aarch64-linux-gnu.so`，SHA256 `ee17fa9fda69303f08ca7a2d6867d815f188f5324461fb518875e4f02c074717`；CATLASS `769cd40a8716b28650b6bebb08db4834eea4462f`。生产通过原 `setup.py build_ext --inplace`、v3/910 配方重建，日志 `build-numerical-final.log`；固定源为 `ca18aee610729ca5270a44e2d84c751bc687fbf0`，源码只读。旧扩展保留为 `baseline-extension.so`，SHA256 `1e5ee5b0c7d5aafbd8e0613e917f1040e6bc4fe5f2a6de7a7244d88fa9c6a015`。

M21 历史正确性 1,677 项通过；39 个 INT8 配置 eager 平均比 0.707703，通过，graph 1.197126，失败并按用户授权暂缓。它是独立的 INT8 集合，不能与本轮非量化比值混比。INT8 固定源 scale 语义、动态 P 量化与独立数学限制仍保留；本轮解析输入的 E2E 不等价于一般随机量化精度验收。Trap、完整迁移阶段性能矩阵、graph 性能和 910C 本轮未重跑；本轮重测 M17 原平均的 60 个适用配置、新增的 512+64 公开修复前后/固定源与同调度组件集合。新增默认路径的 eager 性能差距依据本次 R5 单独授权接受，不沿用历史图性能例外。历史证据见工作根 `mla-m21/` 与 `git show d433904:docs/mla/status.md`。

唯一下一步：无。本轮按用户确认记录既有性能差距后提交并停止；无待开发的已授权步骤，默认路径进一步性能优化未纳入本轮，不将已接受的未达标结果改写为通过。
