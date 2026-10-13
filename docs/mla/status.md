# MLA 当前状态

**M09c1 必要验收已闭环。** 分支 `feat/mla-cann-migration`，接手基线 `4cd5d507c403fc866d2f1075e955b963a15ce7aa`。本步只完成一个独立可调用行为，提交一次后立即停止。唯一下一步是 [M09c2：非 PA 混合 window cache](plan.md#唯一下一步-m09c2非-pa-混合-window-cache)，本次未启动。

## 当前能力与本步边界

新增 512+64、FP16/BF16、Hkv=1、`num_splits=1` 下 **非 PA TND Q/KV window varlen**，包含双分配前缀、可选 `seqused_q/k`、调用级 maxQ/maxK、有限/单侧窗口、causal 叠加及四路合法 stride。v13 使用 `413+8B`，绑定 TND 布局、allocation/used 快照与归一化窗口；与 BSND v12 和旧版本互斥。非默认窗口归一化成 causal/无 mask 后仍保留 v13；公开 varlen 每次内部生成，不增加显式 metadata 参数。

复用 CPU window/causal 算法、既有地址/Copy、Block/Tile；新增 TND window AICPU、共享消费者的默认关闭分支、TND launcher 和 Python/C++ real/fake 分派，沿用当前 stream 与 allocator 生命周期。混合 cache、PA window、split-KV 等未开放能力继续明确拒绝。CATLASS、CANN 源仓未改。契约见 [design 5.15](design.md#515-v13tnd-window-记录)，实际影响和验收选择见 [本步计划](plan.md#最近完成-m09c1非-pa-tnd-qkv-window-varlen)。

## 最近一次闭环证据

证据根 `build/mla-m09c1/`。`acceptance.json` 核验 **26 组、1122 次本次 pytest 执行**，有效验收失败/跳过/未运行均为 0；核验 **5533 个原始产物**、893 个冻结源文件及当前构建身份，审计合计 **23.6 秒**。组运行合计 **3562.0 秒**，不含收集、构建/runtime 和被替代的开发检查；工具专项与 tools 的重叠按组执行计数。

| 必跑组 | 本次通过 | 秒数 |
| --- | ---: | ---: |
| `window-tnd-cpu` | 43 | 28.2 |
| `window-tnd-sanitize` | 43 | 104.2 |
| `window-bsnd-cpu` | 34 | 27.5 |
| `window-bsnd-sanitize` | 34 | 92.1 |
| `causal-bsnd-cpu` | 100 | 54.9 |
| `causal-bsnd-sanitize` | 100 | 170.1 |
| `causal-tnd-cpu` | 41 | 26.2 |
| `causal-tnd-sanitize` | 41 | 90.2 |
| `causal-mixed-cpu` | 76 | 24.7 |
| `causal-mixed-sanitize` | 76 | 91.5 |
| `window-tnd-tools` | 39 | 7.3 |
| `tools` | 100 | 14.1 |
| `cpu` | 2 | 3.4 |
| `window-tnd-check` | 10 | 636.1 |
| `window-tnd-public` | 74 | 127.8 |
| `window-tnd-trap` | 22 | 306.6 |
| `regression-nonpaged` | 86 | 721.3 |
| `window-bsnd-check` | 10 | 115.2 |
| `causal-bsnd-check` | 56 | 106.0 |
| `causal-tnd-check` | 6 | 97.6 |
| `causal-mixed-check` | 6 | 99.2 |
| `causal-paged-check` | 6 | 134.0 |
| `causal-paged-tnd-check` | 6 | 129.1 |
| `ordinary` | 100 | 23.0 |
| `causal-mixed-boundary` | 10 | 20.1 |
| `window-tnd-profile` | 1 | 311.3 |

新增窗口入口、协议和动态长度，以及固定非分页、已有 window、五布局 causal、普通 FA3 和未开放边界，均有本次证据。本步不是 M09 阶段收尾；混合和 PA window 不属于已开放能力。

有效记录：CPU `cpu/attempt-1791928312749350130/summary.json`；工具 `tools/attempt-1791929344144456450/summary.json`；代表/成本 `check/attempt-1791929370748931530/summary.json`；正式集合 `acceptance/attempt-1791930055066702740/summary.json`；性能 `performance/attempt-1791932162724863130/summary.json`。冻结实现为 `frozen-sources.json`，完整运行日志为 `cpu-final.log`、`tools-reference-run.log`、`check-reference-resume.log`、`acceptance-run.log`、`performance-run.log`。以下命令用于复现；接手先核对依赖和证据，不无条件重跑有效组。

```bash
python3 -m tests.mla_validation --output build/mla-m09c1/cpu --suite window-tnd-cpu window-tnd-sanitize window-bsnd-cpu window-bsnd-sanitize causal-bsnd-cpu causal-bsnd-sanitize causal-tnd-cpu causal-tnd-sanitize causal-mixed-cpu causal-mixed-sanitize --resume
python3 -m tests.mla_validation --output build/mla-m09c1/tools --suite window-tnd-tools tools cpu --resume
python3 -m tests.mla_validation --output build/mla-m09c1/check --device 6 --suite window-tnd-check --resume
python3 -m tests.mla_validation --output build/mla-m09c1/acceptance --device 6 --suite window-tnd-public window-tnd-trap regression-nonpaged window-bsnd-check causal-bsnd-check causal-tnd-check causal-mixed-check causal-paged-check causal-paged-tnd-check ordinary causal-mixed-boundary --resume
python3 -m tests.mla_validation --output build/mla-m09c1/performance --device 6 --suite window-tnd-profile
```

各批先 dry-run/collect；`selection-audit.json` 保存候选、选中、未选中数量和清单身份。新 check/public/trap 分别 10/74/22 且互斥，ordinary 从 804 个候选固定选择 100；其他各组均不超过 100，没有 full-matrix。旧 TND 测试中窗口拒绝项已移除，正向和剩余拒绝边界由新增专项覆盖；旧无 mask 组只作共享回归。

## 数学精度与公开性能

`window-accuracy.json` 核验 **162 条保存的原始数值记录、60 个含有效 Q 的 dtype/输入配置、618 组 O/LSE 比较**，全部逐元素等于固定 CANN 源并通过双数学；另两个配置全零 Q-used，没有有效数值行。固定源实际传 band sparse 4、三角压缩 mask 及归一化 pre/next，causal/无 mask 特例为 3/0。参考按原始分配起点、有效长度和调用级窗口独立构造，公开/专用/eager/raw 四入口均实际执行。

继承 M09b1 的三条 O 源精度限制，不能计为数学通过；原始输入和误差在 `build/mla-m09b1/source-precision-limits.json`。本次新窗口数值组未新增限制。性能原生预检另保存 **176 次 O/LSE 数学比较**，全部通过；实际原生分派及运算次序差异保存在对应配置记录。

`public-performance.json` 汇总预定 **12 输入×双 dtype，共 24 个配置**；实际原生分派全部为 MLA template 5，6 个原生 FD 配置保留计时和数值结果并单列。其余 **18 个适用配置平均公开 E2E 比为 0.444728 ≤1**，退化配置/分段均为 0。FP16/BF16 各 9 个适用配置，均值分别为 **0.438203 / 0.451254**。先在配置内对 R=1/4 等权平均，再跨配置等权平均，不按耗时筛选输入。

保持 5 次预热、5 轮×20 次 wall 和既有 profiler/同步边界；原生 TND packing、输出恢复计入 E2E。单例、分段、wall 样本、profiler CSV、固定源包和分派证明在性能 attempt 的 `public-profile/`。`npu-before-profile.log`、`npu-after-profile.log`、前后进程记录及采样内快照确认第 6 卡隔离；预检后才启动绑定单 CPU 的计时子进程。独立连续组件性能按不变依赖继承，没有代替本步 varlen E2E。

## 缓存、继承与环境

数学缓存为 `build/mla-validation-artifacts/golden`，公式、阈值、内容键均不改、不刷新。有效验收合计 **394 次命中、121 次冷算/写入**，无缓存错误；性能预检 88 次全部命中。新 TND 参考改为已有 BSND 的最多四样本并行，保持 intra-op 线程数，设备仍在调用者 stream 执行；执行策略改变后单列复测两个成本边界的双 dtype。最长 299.7 秒，主机峰值约 14.7 GiB、NPU 峰值约 334 MiB，原始 `tnd-cost-*.json` 在代表组 probes 下。中断前完成的 reference 已通过内容缓存复用，不重复冷算或计入有效测试执行数。

`baseline-audit.json` 复核 M09b2 的 23 组闭环身份；`inheritance.json` 另复核 M09a `window-cpu/window-sanitize/causal-cpu/causal-sanitize` 与 M09b1 `window-check/window-attention/window-profile`，**7 组、366 项**单列继承。算法/驱动、固定 FA3/CANN、工具链、原始产物、编译依赖和实际加载库身份有效；新增 v13/TND stride/入口由本步实测。

十个 CPU 组通过后只变更了设备 reference/profile 驱动；`cpu-evidence-compatibility.json` 确认 CPU 源/头/驱动、选中用例、命令和工具链未变，保守快照中仅有一个不执行的设备 profile 文件变化。该差异不使这些本次 CPU 证据失效。893 个当前冻结源文件在最终审计时完全一致。

Python 3.11.15、torch 2.7.1+cpu、torch_npu 2.7.1、CANN 9.0.0-beta.2；第 6 卡 Ascend910B3。CATLASS `769cd40a8716b28650b6bebb08db4834eea4462f`，固定 CANN `ca18aee610729ca5270a44e2d84c751bc687fbf0`；固定 FA3 源和完整工具链身份见构建清单。接手工作树干净，无遗留 MLA 验收；其他进程、CANN 源仓和 CATLASS 保持原状。

当前生产构建 **6330 个依赖、11 个产物**，清单 `build-final-manifest.json`；实际加载仓库 `flash_attn_npu_3/flash_attn_npu_3.cpython-311-aarch64-linux-gnu.so`，SHA256 `492e206ac5fdbc55a5eadc9aa6be01c4cacaf9ae25e419fd3704f9909ab8c106`。最终重建日志 `build-final.log`；统一入口构建复用约 1.1 秒、runtime 核验约 21–24 秒。旧扩展快照不作为当前源码产物身份。

## 已解决尝试与收尾要求

初次构建遇到编译期间源码变更，被身份检查正确判为失效，最终重新构建并实际加载通过。初次设备代表因测试驱动把 `window_size` 直接传给通用 C++ schema 而失败；已拆为 left/right，补充四入口路由工具检查后重跑受影响组。随后串行长 KV 参考耗时偏高，中断未完成组并调整参考并行；输入集合、数学门槛和缓存键不变，代表/成本完整重跑通过。

`superseded-attempts.json` 单列四个失败/失效组、一个中断组和十五个被替代的成功组，原始记录保留且均不加入当前闭环总数；构建失败日志另存。最终生产构建后实现未再改变，修正后的新专项、共享回归与性能均完成。

本步相关契约、三份文档和链接/锚点结果见 `doc-links.json`；质量记录为 `quality-staged.log`。提交前只暂存本步 27 个文件，执行 `git diff --check` 和 `ci/code_quality.sh --staged`，不绕过提交钩子；一次提交后立即停止，不启动 M09c2。使用 `login=false`；实机与 sanitizer 需主机可见性，pre-commit 复用 `PYTHONPATH=/tmp/mla-m01-precommit`。
