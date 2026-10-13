# MLA 当前状态

**M09b2 必要验收已闭环。** 分支 `feat/mla-cann-migration`，接手基线 `55db0466645d65db06902c37cf6ee3511a96add4`。本步只完成一个独立可调用行为、提交一次后停止。唯一下一步为 [M09c1：非 PA TND Q/KV window varlen](plan.md#唯一下一步-m09c1非-pa-tnd-qkv-window-varlen)，本次未启动。

## 当前能力与本步边界

新增 512+64、FP16/BF16、Hkv=1、`num_splits=1` 下 **非 PA BSND window** 的 func、只读 cache、metadata 生成/复用及内部 Q-used；支持有限/单侧窗口、causal 叠加和四路合法 stride。v12 使用 `413+8B` 字，尾部保存归一化 kind/left/right，调用容量与实际 used 分离；归一化成 causal/无 mask 的非默认窗口仍用 v12，旧版本互斥。内部 Q-used 仍拒绝显式 metadata，公开 cache/生成器不增加 Q-used 参数。

复用 M09a CPU window 分核与 M09b1 window 任务环、Block/Tile；新增 AICPU、used 消费窗口分支及 Python/C++ real/fake 分派，沿用当前 stream 与 allocator 生命周期。其他布局的 window、split-KV 等未开放能力继续明确拒绝。CPU window/causal 算法、地址/Copy、Block/Tile、CATLASS 和 CANN 未改。契约见 [design 5.14](design.md#514-v12bsnd-window-记录)，实际影响路径和验收选择见 [plan](plan.md#最近完成-m09b2bsnd-window-完整公开行为)。

## 最近一次闭环证据

证据根 `build/mla-m09b2/`。`acceptance.json` 核验 **23 组、990 次本次 pytest 执行**，有效验收失败/跳过/未运行均为 0；核验 **4707 个原始产物**、885 个冻结源文件及当前构建身份，耗时 33.0 秒。组运行合计 **3422.3 秒**，不含收集、构建/runtime 和被替代的开发检查；工具专项与 tools 的重叠按组计数。

| 必跑组 | 本次通过 | 秒数 |
| --- | ---: | ---: |
| `causal-bsnd-cpu` | 100 | 55.3 |
| `causal-bsnd-sanitize` | 100 | 166.9 |
| `causal-mixed-cpu` | 76 | 24.5 |
| `causal-mixed-sanitize` | 76 | 95.6 |
| `causal-tnd-cpu` | 41 | 24.0 |
| `causal-tnd-sanitize` | 41 | 91.7 |
| `window-bsnd-cpu` | 34 | 27.1 |
| `window-bsnd-sanitize` | 34 | 90.9 |
| `cpu` | 2 | 4.2 |
| `tools` | 100 | 14.2 |
| `window-bsnd-tools` | 23 | 7.2 |
| `window-bsnd-check` | 10 | 334.7 |
| `causal-bsnd-check` | 56 | 107.2 |
| `causal-mixed-boundary` | 10 | 20.0 |
| `causal-mixed-check` | 6 | 98.2 |
| `causal-paged-check` | 6 | 135.2 |
| `causal-paged-tnd-check` | 6 | 131.3 |
| `causal-tnd-check` | 6 | 99.2 |
| `ordinary` | 100 | 23.0 |
| `regression-nonpaged` | 86 | 729.3 |
| `window-bsnd-public` | 50 | 115.9 |
| `window-bsnd-trap` | 26 | 363.0 |
| `window-bsnd-profile` | 1 | 663.6 |

窗口 metadata、生产前向和公开 func/cache 的新行为，以及固定非分页、五布局 causal、普通 FA3 与未开放边界，均有本次证据；其余布局的 window 仍拒绝。本步不是 M09 阶段收尾，不把尚未开放的布局纳入已完成范围。

有效记录：CPU `cpu/attempt-1791921996635527540/summary.json`；工具 `tools/attempt-1791923566605849980/summary.json`；代表/成本 `check/attempt-1791923659576328130/summary.json`；正式集合 `acceptance/attempt-1791924039360930890/summary.json`；性能 `performance/attempt-1791926039539574900/summary.json`。完整日志为 `acceptance-resume-reference.log` / `performance-run.log`。冻结实现为 `frozen-sources.json`。下列命令供复现；接手时核对依赖与证据，不无条件重跑有效组。

```bash
python3 -m tests.mla_validation --output build/mla-m09b2/cpu --suite window-bsnd-cpu window-bsnd-sanitize causal-bsnd-cpu causal-bsnd-sanitize causal-tnd-cpu causal-tnd-sanitize causal-mixed-cpu causal-mixed-sanitize --resume
python3 -m tests.mla_validation --output build/mla-m09b2/tools --suite window-bsnd-tools tools cpu --resume
python3 -m tests.mla_validation --output build/mla-m09b2/check --device 6 --suite window-bsnd-check --resume
python3 -m tests.mla_validation --output build/mla-m09b2/acceptance --device 6 --suite window-bsnd-public window-bsnd-trap regression-nonpaged causal-bsnd-check causal-tnd-check causal-mixed-check causal-paged-check causal-paged-tnd-check ordinary causal-mixed-boundary --resume
python3 -m tests.mla_validation --output build/mla-m09b2/performance --device 6 --suite window-bsnd-profile
```

各批次先 dry-run/collect；`selection-audit.json` 保存候选、选中、未选中及清单路径。代表/成本 10 与公开 50 互斥，总计 46 个数值节点及 14 个 metadata/空值/边界/compile/生命周期节点。ordinary 候选由 816 减至 804：移除已经开放的两个 BSND 窗口拒绝参数（两 dtype×三个入口），窗口正向由本步专项覆盖；仍固定选 100。没有启用 full-matrix。

## 数学精度与公开性能

`window-accuracy.json` 核验 **98 条原始数值记录、44 个含有效 Q 的 dtype/输入配置、298 组 O/LSE 比较**，全部逐元素等于固定 CANN 源并通过双数学。另两个配置全部 Q-used=0，没有有效数值行。固定源真实传 band sparse 4、压缩三角 mask 和归一化 pre/next，causal/无 mask 特例为 3/0；没有用旧无 mask 验收替代窗口。公开 func/cache/internal/raw、metadata 复用及四路 stride 均实际执行。

继承 M09b1 的三条 O 源精度限制，不能记为数学通过；原误差与输入在 `build/mla-m09b1/source-precision-limits.json`。本次窗口数值组未新增源精度限制；原生性能预检的 24 个配置另有 116 次 O/LSE 数学比较，均通过，原生 FD 的运算次序差异按约定单列。

`public-performance.json` 汇总预定 **12 个输入×双 dtype，共 24 个配置**；真实原生分派全部为 MLA template 5，其中 6 个原生 FD 配置保留计时与数值结果并单列。其余 **18 个适用配置的平均公开 E2E 比为 0.623719 ≤1**，退化配置/分段均为 0。FP16/BF16 各 9 个适用配置，均值分别为 **0.624081 / 0.623358**。每个配置内先对适用的 func/cache、生成/复用/内部生成及 R=1/4 等权平均，再跨配置等权平均；不按耗时筛选输入。

保持 5 次预热、5 轮×20 次 wall 采样及原 profiler/同步边界；所有单例、分段、wall 样本、profiler CSV、固定源包和实际分派证明均在性能 attempt 的 `public-profile/` 下。`npu-before-profile.log` / `npu-after-profile.log` 及采样内快照确认第 6 卡仅有本次活动；预检结束后才启动绑定单 CPU 的计时子进程。M09b1 不变的数值组件性能单列继承，没有用其 probe wall 代替本步公开 E2E。

## 缓存、继承与环境

数学缓存继续使用 `build/mla-validation-artifacts/golden`，不改公式、阈值或缓存键，不刷新 reference。新公开参考最初串行；排障后改用已有组件的最多四样本并行、保持 intra-op 线程数，并将两个长 KV 配置移到代表/成本组，原总集合不变。成本四例全部通过；冷算最长 149.7 秒，主机峰值约 14.8 GiB、NPU 峰值分配约 303.7 MiB，原始 `public-cost-*.json` 在代表组 probes 下。修正后的代表缓存命中 21、冷算/写入 11，前一轮结果没有重复冷算。有效验收数值记录合计 **328 次命中、101 次冷算/写入**，无缓存错误或刷新；开发失败/被替代组的参考通过命中复用，未重复计入上述有效组执行数。工具的刷新/损坏注入另计。

`baseline-audit.json`、`inheritance.json` 核验 M09a CPU window/causal 普通与 sanitizer，以及 M09b1 独立 window check/attention/profile，合计 **7 组、366 项、2354 个原始产物**，单列继承。CPU 算法/驱动/工具链、固定 FA3/CANN、独立组件编译依赖与实际加载库均有效；新增 v12/used/stride 及公开入口由本步实测。

旧设备参考 helper 只增加 `<-1` 的归一化分支，调用已与固定 FA3 对照的独立 CPU oracle；`inheritance-helper.json` 确认移除该分支后旧模块 AST 完全相同，旧目录 46 个用例描述全部使用 ≥-1 的窗口。本步 CPU 验收后仅修改 runner 的设备成本过滤和公开用例分组；CPU 命令、源/头/驱动、输入和工具链不变，不因此重跑八个 CPU 组。

Python 3.11.15、torch 2.7.1+cpu、torch_npu 2.7.1、CANN 9.0.0-beta.2；第 6 卡 Ascend910B3。CATLASS `769cd40a8716b28650b6bebb08db4834eea4462f`、CANN 固定源 `ca18aee610729ca5270a44e2d84c751bc687fbf0`。接手工作树干净，无遗留 MLA 验收；其他进程、CANN 源仓及 CATLASS 保持原状。

当前生产构建 **6330 个依赖、11 个产物**；实际加载 `flash_attn_npu_3/flash_attn_npu_3.cpython-311-aarch64-linux-gnu.so`，SHA256 `dcb7e46f8d893ee8d50e9dd8d65c6ab607845aaa74e33fa5b0134c9dffe1c151`。重建耗时 571.8 秒（含父进程等待收集核对），正式复用构建约 1.1 秒、加载核验约 21 秒；各 attempt 保存完整构建/runtime/loaded 身份。最初环境快照中的旧扩展不能作为当前产物身份。

## 已解决尝试与收尾要求

初次 sanitizer 在沙箱退出阶段因 ptrace/LeakSanitizer 限制失败，主机续跑后通过；开发期间源码变化导致的结果失效均被最终组替代。第一次扩展虽链接成功，但导入发现 `WindowMetadataWords` 仅有 AICPU 定义；改为头文件 inline 并重建后完成实际加载和实机核验，原始失败日志保留。

窗口正式组首次在 `(-2,-2)` 的参考构造处失败：旧数学适配器只接受 ≥-1，未执行该用例的生产前向。保留该轮 18 个通过、1 个失败节点的原始记录，修正参考归一化及串行成本后重跑受影响组，已全部通过；旧记录不重复计入有效验收，reference 通过内容缓存复用。静态边界测试的参数覆盖合并也已修正，output-only 返回路径补入同组。生产实现自成功重建后未再改变。

本步相关契约、三份文档、链接/锚点和表格已经核对；链接与质量记录为 `doc-links.json`、`quality-staged.log`。只暂存本步 34 个文件，以 `git diff --check` 与 `ci/code_quality.sh --staged` 为提交门槛，不绕过提交钩子；一次提交后立即停止。使用 `login=false`；实机与 sanitizer 需主机可见性，pre-commit 复用 `PYTHONPATH=/tmp/mla-m01-precommit`。未完成必要验证不得提交为闭环。
