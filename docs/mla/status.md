# MLA 当前状态

**M09c2 必要验收已闭环。** 分支 `feat/mla-cann-migration`，接手基线 `23238f64aa8e69f9eef1ab6bc89f3ed4331e6b07`。本步完成非 PA 混合 window cache，一个提交后立即停止。唯一下一步是 [M09d：BSND PA window cache](plan.md#唯一下一步-m09dbsnd-pa-window-cache)，本次未启动。

## 当前能力与本步边界

新增 512+64、FP16/BF16、Hkv=1、`num_splits=1` 下 **非 PA TND Q + BSND KV window 只读 cache**，支持 Q 分配前缀/调用级 maxQ、KV 容量/实际长度、有限/单侧窗口、causal 叠加及四路合法 stride。公开 cache 接受 None/整数/tensor KV 长度，支持 metadata 生成和显式复用；内部 Q-used 每次生成且拒绝显式记录，公开签名不增加 Q-used。

v14 使用 `413+8B`，绑定混合布局、allocation/used 快照和归一化窗口，与 v9/v12/v13 等旧记录互斥；非默认窗口归一化成 causal/无 mask 后仍保留 v14。公开生成器在设备前缀校验后推导总 Q，无 host 读回。生成和消费沿用当前 stream 顺序、提交队列排空和 allocator 生命周期。契约见 [design 5.16](design.md#516-v14混合-window-记录)，实际影响和验收选择见 [本步计划](plan.md#最近完成-m09c2非-pa-混合-window-cache)。

复用 CPU window/causal 算法、既有地址/Copy、Block/Tile；新增混合 window AICPU、共享消费者默认关闭分支、TND launcher 特化和 Python/C++ real/fake 分派。PA window、split-KV 等未开放能力继续明确拒绝。CATLASS 和 CANN 源仓未改；本步不是 M09 阶段收尾。

## 最近一次闭环证据

证据根 `build/mla-m09c2/`。`acceptance.json` 核验 **29 组、1311 次本次 pytest 执行**，有效验收失败/跳过/未运行均为 0；核验 **6349 个原始产物**、901 个冻结源文件及当前构建身份，最终审计 **50.4 秒**。组运行合计 **4146.1 秒**，不含收集、构建/runtime 和被替代尝试；工具专项与 tools 的重叠按组执行计数。

| 必跑组 | 本次通过 | 秒数 |
| --- | ---: | ---: |
| `window-mixed-cpu` | 80 | 30.4 |
| `window-mixed-sanitize` | 80 | 108.6 |
| `window-bsnd-cpu` | 34 | 27.1 |
| `window-bsnd-sanitize` | 34 | 88.8 |
| `window-tnd-cpu` | 43 | 28.9 |
| `window-tnd-sanitize` | 43 | 107.2 |
| `causal-bsnd-cpu` | 100 | 55.2 |
| `causal-bsnd-sanitize` | 100 | 166.1 |
| `causal-tnd-cpu` | 41 | 25.1 |
| `causal-tnd-sanitize` | 41 | 89.9 |
| `causal-mixed-cpu` | 76 | 24.4 |
| `causal-mixed-sanitize` | 76 | 91.0 |
| `window-mixed-tools` | 40 | 7.4 |
| `tools` | 100 | 14.3 |
| `cpu` | 2 | 3.6 |
| `window-mixed-check` | 10 | 715.7 |
| `window-mixed-public` | 86 | 136.6 |
| `window-mixed-trap` | 34 | 470.0 |
| `regression-nonpaged` | 86 | 737.7 |
| `window-bsnd-check` | 10 | 115.3 |
| `window-tnd-check` | 10 | 140.4 |
| `causal-bsnd-check` | 56 | 108.3 |
| `causal-tnd-check` | 6 | 99.3 |
| `causal-mixed-check` | 6 | 98.1 |
| `causal-paged-check` | 6 | 132.3 |
| `causal-paged-tnd-check` | 6 | 129.4 |
| `ordinary` | 100 | 23.0 |
| `causal-mixed-boundary` | 4 | 19.9 |
| `window-mixed-profile` | 1 | 352.0 |

有效记录：CPU 前三组 `cpu/attempt-1791933576518292270/summary.json`，剩余九组 `cpu/attempt-1791933874139896630/summary.json`；工具 `tools/attempt-1791938069131712830/summary.json`；代表/成本 `check/attempt-1791934149393633490/summary.json`；正式集合 `acceptance/attempt-1791934933460712770/summary.json`；性能 `performance/attempt-1791938163618794890/summary.json`。完整日志为 `cpu-run.log`、`cpu-resume.log`、`tools-recovery-run.log`、`check-run.log`、`acceptance-run.log` 和 `performance-recovery-run.log`。以下命令用于复现；接手先核对依赖和证据，不无条件重跑有效组。

```bash
python3 -m tests.mla_validation --output build/mla-m09c2/cpu --suite window-mixed-cpu window-mixed-sanitize window-bsnd-cpu window-bsnd-sanitize window-tnd-cpu window-tnd-sanitize causal-bsnd-cpu causal-bsnd-sanitize causal-tnd-cpu causal-tnd-sanitize causal-mixed-cpu causal-mixed-sanitize --resume
python3 -m tests.mla_validation --output build/mla-m09c2/tools --suite window-mixed-tools tools cpu --resume
python3 -m tests.mla_validation --output build/mla-m09c2/check --device 6 --suite window-mixed-check --resume
python3 -m tests.mla_validation --output build/mla-m09c2/acceptance --device 6 --suite window-mixed-public window-mixed-trap regression-nonpaged window-bsnd-check window-tnd-check causal-bsnd-check causal-tnd-check causal-mixed-check causal-paged-check causal-paged-tnd-check ordinary causal-mixed-boundary --resume
TMPDIR=/dev/shm python3 -m tests.mla_validation --output build/mla-m09c2/performance --device 6 --suite window-mixed-profile
```

各批先 dry-run/collect；`cpu-selection-audit.json`、`tools-selection-audit.json`、`device-selection-audit.json` 和 `performance-recovery-selection.json` 保存候选、选中与未选中清单。新 check/public/trap 分别 10/86/34 且互斥，ordinary 从 804 个候选固定选择 100；其他各组均不超过 100，没有 full-matrix。移除旧混合 window 拒绝项，由新正向专项覆盖；剩余 PA window 拒绝边界仍实测。旧无 mask 组仅作共享回归。

## 数学精度与公开性能

`window-accuracy.json` 汇总 **200 份原始数值记录、76 个有有效 Q 的 dtype/输入配置、1260 组 O/LSE 比较**，全部逐元素等于固定 CANN 源并通过双数学；另 2 个全零 Q-used 配置没有有效数值行。固定源实际传 band sparse 4、三角压缩 mask 及归一化 pre/next，causal/无 mask 特例为 3/0。数学从原始 Q 前缀和 BSND KV 容量起点独立映射，公开/专用/eager/raw 和显式复用入口均实际执行。保存文件沿用共享驱动的 `window-tnd-source-*` / `tnd-source-*.json` 名称。

继承 M09b1 的三条 O 源精度限制，不能计为数学通过；原始输入和误差见 `build/mla-m09b1/source-precision-limits.json`。本次新窗口数值组未新增限制。性能原生预检另保存 **176 次 O/LSE 数学比较**，全部通过；实际原生分派和运算次序差异保存在对应配置记录。

`public-performance.json` 汇总预定 **12 输入×双 dtype，共 24 个配置**；6 个原生 FD 配置保留计时和数值结果并单列。其余 **18 个适用配置平均公开 E2E 比为 0.433848 ≤1**，退化配置 0、退化分段 0。FP16/BF16 适用配置均值分别为 **0.431454 / 0.436241**。先在配置内对 implicit/generate/reuse 三入口与 R=1/4 六段等权平均，再跨配置等权平均，不按耗时筛选。

保持 5 次预热、5 轮×20 次 wall 和既有 profiler/同步边界；原生混合 packing/输出恢复计入 E2E。单例、分段、wall 样本、profiler CSV、固定源包和分派证明在性能 attempt 的 `public-profile/`。`npu-before-profile.log`、`npu-after-profile.log`、前后进程记录及采样内快照确认第 6 卡隔离；预检后才启动绑定单 CPU 的计时子进程。独立连续组件性能按不变依赖继承，没有代替本步混合公开 E2E。

## 缓存、继承与环境

数学缓存为 `build/mla-validation-artifacts/golden`，公式、阈值、内容键与最多四样本并行策略均不改、不刷新。数值验收合计 **485 次命中、148 次冷算/写入**，无缓存错误或刷新，性能预检 88 次全部命中。工具故障注入另计 10 次命中、21 次 miss、23 次写入、2 次预期 read_error、3 次禁用和 3 次主动刷新；不与设备数值缓存混算，分类见 `reference-cache-audit.json`，逐组统计见 `acceptance.json`。新混合输入先单列检查成本边界，后续正确性/性能复用相同参考，待测实现仍实际执行。最长 **322.7 秒**，主机峰值约 **15.0 GiB**、NPU 峰值约 **543 MiB**；原始 `mixed-cost-*.json` 在代表组 probes 下。

`baseline-audit.json` 复核 M09c1 的 26 组、1122 次执行及构建/加载身份；`inheritance.json` 另复核 M09a `window-cpu/window-sanitize/causal-cpu/causal-sanitize` 与 M09b1 `window-check/window-attention/window-profile`，**7 组、366 项**单列继承。算法/驱动、固定 FA3/CANN、工具链、原始产物、编译依赖和实际加载库身份有效；新增 v14/混合 stride/公开入口由本步实测。

初次 CPU 前三组完成后，仅界面 docstring 和新增 CPU Python 测试的空行格式变化；`cpu-evidence-compatibility.json` 与 `format-compatibility.json` 核对 AST 语义相同，CPU 源/头/驱动、选择与工具链未变，保留这 194 次本次执行。其余 CPU 组在该变化后运行。性能环境恢复只改两个新增 profile/工具文件；`profile-recovery-compatibility.json` 确认 25 个正确性组的实际依赖均不包含它们，保留原结果，只重跑工具与性能。工具恢复通过后重新固定实现。`frozen-sources.json` 的 901 个源文件在最终审计时完全一致。

Python 3.11.15、torch 2.7.1+cpu、torch_npu 2.7.1、CANN 9.0.0-beta.2；第 6 卡 Ascend910B3。CATLASS `769cd40a8716b28650b6bebb08db4834eea4462f`，固定 CANN `ca18aee610729ca5270a44e2d84c751bc687fbf0`；固定 FA3 源和完整工具链身份见构建清单。接手工作树干净，无遗留 MLA 验收；其他进程和源仓保持原状。

生产构建 **6330 个依赖、11 个产物**，清单 `build-manifest.json`，日志 `build.log` / `production-build.log`；实际加载仓库 `flash_attn_npu_3/flash_attn_npu_3.cpython-311-aarch64-linux-gnu.so`，SHA256 `ba4595a4f2d49e0f8e16d834222b8f3dbd2db4b22b185acec78e85204af5f569`。统一入口核验构建与加载，已有旧扩展不作为当前源码产物。

## 已解决尝试与收尾

初次 CPU 的 `window-bsnd-sanitize` 虽 pytest 断言通过，但运行期间界面 docstring 改变，统一入口正确判为失效；仅续跑失效及未完成组。两批设备 collect 最初在设备不可见的沙箱内失败，改为主机可见环境后完整收集。性能首次在 FP16 数值预检通过后失败：CANN 报输出分区可用空间 0、低于 20 MiB，主动跳过采集，未生成 kernel CSV。失败 wall/日志保留，不作为性能通过证据。运行 NPU 的容器仅有 64 MiB tmpfs，不能放整组调试日志；新增每 dtype 计时临时输出和按哈希归档，显式 `TMPDIR=/dev/shm` 后重跑性能组。所有原始采集、报表和输入身份留在仓库忽略目录，`profile-archive-audit.json` 另核验每 dtype 78 个采集文件和仅路径变化的报表，临时副本在核验后释放；没有清理历史证据或改变计时规则。恢复详情 `performance-recovery.json`，CANN 原始失败日志 `performance-disk-failure.log`。旧工具结果因后续测试内容变化被替代；`superseded-attempts.json` 单列这四个失败/失效组和六个被替代成功组，原始记录保留且不计入闭环总数，没有遗留数值失败。

本步相关契约、三份文档与链接/锚点核对见 `doc-links.json`，质量记录为 `quality-staged.log`。提交前只暂存本步 28 个文件，执行 `git diff --check` 和 `ci/code_quality.sh --staged`，不绕过提交钩子；一次提交后立即停止，不启动 M09d。命令使用 `login=false`；实机、设备收集和 sanitizer 需要主机可见性，pre-commit 复用 `PYTHONPATH=/tmp/mla-m01-precommit`。
