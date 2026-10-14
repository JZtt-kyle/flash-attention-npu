# MLA 当前状态

当前分支 `feat/mla-cann-migration`，本轮起点 `b2c233007df57b60efbf1608cea13bc8803db42b`。**M09e 与 M09 五布局窗口阶段已闭环。** 本步接通 TND Q + ND PA KV window 只读 cache；下一步为 [M10a 独立 CPU split-KV 分区与 workspace 布局](plan.md#唯一下一步-m10a独立-cpu-split-kv-分区与-workspace-布局)，本轮不开始实现。

## 当前能力与本步边界

512+64、FP16/BF16 的 BSND、TND Q/KV、TND Q + BSND KV、BSND/TND Q + ND PA KV 五种布局均已接通无 mask、causal、有限/单侧 window 及 causal 叠加。TND PA window 使用独立 v16，精确字数 `411+5B`，接通 CPU/AICPU、设备消费者、生产分派及公开 real/fake/compile；复用现有 window 数值核、页 Copy 和四路 stride。

Q 分配前缀决定起点，内部 used 不压紧间隙；公开 cache/metadata 签名不增加 Q-used。动态前缀/长度、窗口身份和全部有效页在空值捷径前检查；支持显式记录复用、长度更新和流生命周期。有效全遮罩/零 KV 为 O=0、LSE=-inf。未开放的 split-KV、192/128、softcap、其他调度选项仍明确拒绝，`num_splits` 仍只接受 1；完整边界见 [design](design.md#22-后端选项与返回值)。

本模块新增 `unit/window_paged_tnd/` 和 `e2e/window_paged_tnd/`，日常清单新增 8 项能力回归。未修改 CANN 源仓、CATLASS 或 Block/Tile 数值算法。阶段收尾按五布局窗口契约核对 design/plan/status，未使用旧无 mask 测试充当新能力验收。

## 本次闭环证据

证据根目录 `/tmp/mla-m09e/`；构建/reference 缓存为 `/tmp/flash-attention-npu-mla-46928a793876/mla-validation-artifacts/`。`build/` 仅保留 `mla-codex-loop/` 的日志和恢复状态。以下全部为本轮实际执行；**继承 E2E 为 0，历史单元及性能继承为 0**。上一闭环的 111 项 E2E 原始证据仍在，但共享生产构建改变后已实际重跑累计 119 项。

| 检查 | 本次结果与耗时 | 证据（相对根目录） |
| --- | --- | --- |
| 模块/阶段验收工具 | 39/39，8.2 秒 | `tools-final/summary.json` |
| 本模块 CPU 固定源/边界 | 42/42，31.7 秒 | `acceptance/summary.json` |
| 本模块 ASan/UBSan | 42/42，114.7 秒 | 同上 |
| 双 dtype 代表与成本边界 | 8/8，329.3 秒；固定源与双数学通过，reference 首次计算 8 次 | `representative/summary.json` |
| 累计小输入 E2E | 119/119，33.9 秒；reference 命中 260、首次计算 6 | `acceptance/summary.json` |
| 五布局窗口约定组合 | 60/60，30.4 秒；reference 命中 120、首次计算 60 | 同上 |
| 普通 FA3 | 516/516，34.5 秒 | 同上 |
| TND PA window 公开专项 | 90/90，1100.1 秒；双 dtype 数值、real/fake/compile、长度/stride/空值、复用/更新/生命周期 | 同上 |
| TND PA window release Trap | 52/52，999.4 秒；双 dtype 各 26 种错误实际发射并在设备拒绝，含零 Q/used 的有效页检查 | 同上 |
| 五布局阶段性能 | 9 个测试节点通过；100 配置完整实测，最终平均值通过，详见下表 | `performance/summary.json`、`performance/stage-average.json` |
| 清单/命令核对 | dry-run 与 collect-only 成功，仅计清单检查 | `*-dry-run.json`、`device-collect/summary.json`、`performance-collect/summary.json` |
| 源码/原始证据与加载身份 | 14 组原始证据完整、源码未变，扩展哈希一致；不把核验当成重跑或继承 | `final-evidence-check.json` |
| 文档链接与仓库质量 | 339 个本地链接/锚点、`git diff --check` 和 staged 质量检查通过 | `doc-links-final.json`、`quality-final.log` |

代表成本输入 FP16/BF16 分别约 89.8/99.5 秒，host 峰值约 6.6 GiB、NPU 峰值约 257 MiB。公开专项的冷长 KV reference 是主要耗时；同一步相同输入后续复用内容缓存，待测实现仍实际执行。新增 TND PA 性能组的数值/计时 reference 均命中此前缓存，未重复计算。

本模块固定源对照中的 O/适配后 LSE 逐元素一致，双数学通过；本轮未出现需要按源精度限制豁免的结果。原 CANN 全遮罩行 LSE 的 +inf 先独立核验再按 FA3 归一为 -inf，完全空值等不可比源输入仍独立验数学与 FA3 契约，不能称为未适配源逐元素一致。性能中的原生 CANN 非同调度结果按既定双数学阈值验收。

### 阶段公开性能

第 6 卡空闲、无其他 MLA 验收进程后顺序采样，记录在 `performance-isolation.log`。固定输入选择在计时前保存，R=1/4、5 次 warmup、5 轮×20 次 wall 和同步/转换边界保持不变。聚合器核验五布局完整、原始证据哈希、实际分派、共享源码和运行身份，并从原始耗时重算。

| 布局 | 实测配置 | 适用 / 原生 FD 单列 | 适用平均耗时比 | 组耗时 |
| --- | --- | --- | --- | --- |
| BSND | 20 | 14 / 6 | 0.577630 | 338.9 秒 |
| TND Q/KV | 20 | 14 / 6 | 0.383578 | 831.2 秒 |
| TND Q + BSND KV | 20 | 14 / 6 | 0.420403 | 863.4 秒 |
| BSND PA | 20 | 10 / 10 | 0.385739 | 1415.1 秒 |
| TND PA | 20 | 10 / 10 | 0.376034 | 448.4 秒 |
| 合计 | 100 | 62 / 38 | **0.434843 ≤ 1** | 五组均通过 |

适用公开配置退化 0，入口/R 分段退化 0；全部单例、分段和 38 个原生 FD 配置保留于 `performance/stage-average.json` 及各组原始报告。FD 尚未迁移，不计入本阶段非 FD 平均值。

源组件采样单列：TND PA FP16 三例比值为 1.230271/1.165243/1.451335，BF16 为 1.417297/1.275723/1.752317，六例均较慢；BSND PA 复测 FP16 为 1.304985/1.173988/1.366554，BF16 为 1.390559/1.054784/1.671572。完整结果在各组 `component/{fp16,bf16}/profile.json`；它们不替代公开 E2E，按 design 以公开平均值验收。

### 复现命令

本轮在第 6 卡执行，先分别 dry-run/collect-only；以下为最终实际执行集合，不使用 `--max-cases` 或历史组件组填充：

```bash
python3 -m tests.mla --suite window-paged-tnd-tools --output /tmp/mla-m09e/tools-final
python3 -m tests.mla --suite window-paged-tnd-check --device 6 --output /tmp/mla-m09e/representative
python3 -m tests.mla --suite window-paged-tnd-cpu window-paged-tnd-sanitize e2e window-stage-correctness ordinary window-paged-tnd-public window-paged-tnd-trap --device 6 --output /tmp/mla-m09e/acceptance
python3 -m tests.mla --window-stage-performance --suite window-bsnd-profile window-tnd-profile window-mixed-profile window-paged-profile window-paged-tnd-profile --device 6 --output /tmp/mla-m09e/performance
python3 -m tests.mla.support.mla_causal_stage_performance --mask window --evidence /tmp/mla-m09e/performance/summary.json --output /tmp/mla-m09e/performance/stage-average.json
```

最终代表 attempt 为 `representative/attempt-1791969854274673910`，正确性为 `acceptance/attempt-1791970262080143600`，性能为 `performance/attempt-1791972690258257980`。恢复时先核对这些目录与 summary 的源/运行身份及原始结果；路径失效后不可继续继承。单元和性能每次实际执行，数学内容缓存独立复用。

仓库质量命令为 `ci/code_quality.sh --staged`，本机以 `PYTHONPATH=/tmp/mla-m09e/quality-deps` 使用隔离安装的 pre-commit；该变量不用于设备测试。格式检查未修改验收源码。

## 环境与恢复记录

Python 3.11.15、torch 2.7.1+cpu、torch_npu 2.7.1、CANN 9.0.0-beta.2；实机 Ascend 910B3 第 6 卡。CATLASS `769cd40a8716b28650b6bebb08db4834eea4462f`，固定 CANN 源 `ca18aee610729ca5270a44e2d84c751bc687fbf0`。实际加载仓库 `flash_attn_npu_3/flash_attn_npu_3.cpython-311-aarch64-linux-gnu.so`，新扩展 SHA-256 `23c1d9992ec9dae5eb9d1a9d2b7a04f586995313c8831c7307fe20888e7ebbe4`，见各 attempt 的 `runtime.json`。生产/测试源码固定后完成上述验收，长测期间仅整理文档。

接手时 HEAD/分支与起点一致，工作树干净，无遗留 MLA 验收进程；上次无头 attempt 因 `workspace routing discovery timed out` 退出，未开始实现。本轮早期失败原始日志保留且不计通过：沙箱内 LSAN ptrace/设备不可见，改用可访问环境后原配置实跑；probe 漏重命名 v16 符号导致构建失败，修正后重建；准备期间源码改动触发身份拒绝；TND host reference 误按三维切片，改为 `[1,T,H,D]` 的 allocation 切片并补缓存/间隙测试；一次工具命令误列不存在节点，修正后 39 项通过。受影响检查已重跑，无未解决失败。

未运行：显式 full-matrix、历史单元矩阵和未开放 split-KV 等能力，不属于本步门槛。无必要验收待补。当前闭环提交由 `git log -1 -- docs/mla/status.md` 定位，无需为自身提交号追加提交。

## 唯一下一步

[M10a：独立 CPU split-KV 分区与 workspace 布局](plan.md#唯一下一步-m10a独立-cpu-split-kv-分区与-workspace-布局)。本轮只完成 M09e 与 M09 阶段收尾，提交后立即停止。
