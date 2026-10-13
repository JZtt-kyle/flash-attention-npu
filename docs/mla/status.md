# MLA 当前状态

**M08e 必要验收已闭环，M08 阶段完成。** 分支 `feat/mla-cann-migration`，接手基线 `757208fb8f8f9f30bf7373c5ae14f50add560d80`。本步交付 TND PA causal，并实测五种已开放 causal 布局的阶段组合和性能；仅提交一次，提交后停止。唯一下一步为 [M09a：独立 CPU window 归一化与块范围](plan.md#唯一下一步-m09a独立-cpu-window-归一化与块范围)，本次未启动。

## 当前可调用能力

512+64、FP16/BF16、Hkv=1、`num_splits=1` 下，BSND、TND varlen、混合 cache、BSND PA 和 TND PA 均支持右下 causal。新增 TND Q + ND PA KV 的 v11 记录、公开只读 cache、metadata 生成/复用及内部 Q-used；全部合法 P 为 16..1024 的 16 倍数。v6/v10/v11 互斥；每次消费检查当次前缀、有效长度及全部有效页，包括空 Q/零 used 样本。全遮罩和 PA 零 KV 为 `O=0、LSE=-inf`。公开 cache/metadata 不新增 Q-used；内部 Q-used 拒绝显式复用。

契约见 [design 5.13](design.md#513-v11tnd-pa-causal-记录)，实现与共享影响见 [plan](plan.md#最近完成-m08etnd-pa-causal-与阶段收尾)。复用既有 causal 任务环、Block/Tile 和页 Copy；有限/单侧窗口、split-KV 等后续能力继续明确拒绝。

## 最近一次闭环证据

证据根 `build/mla-m08e/`。`acceptance.json` 核验 **37 组、1782 项本次通过**，有效验收中失败/跳过/未运行均为 0；核对 859 个当前源码身份和 12592 个原始产物，核验耗时 10.4 秒。各组实跑耗时合计 8697.2 秒，不含收集、构建和失败尝试。历史继承另列，不填充本次阶段计数。

| 必跑组（花括号表示组名展开） | 本次通过 | 组运行秒数合计 |
| --- | ---: | ---: |
| `causal-paged-tnd-{cpu,sanitize}` | 42+42 | 158.1 |
| `causal-paged-{cpu,sanitize}` | 36+36 | 153.4 |
| `paged-shared-{cpu,sanitize}` | 100+100 | 921.6 |
| `cpu`、`tools`、`causal-paged-tools`、`causal-stage-tools` | 2+100+57+22 | 41.2 |
| `causal-paged-tnd-{check,public,trap}` | 6+76+40 | 1123.7 |
| `causal-bsnd-{public,trap}` | 100+24 | 477.5 |
| `causal-tnd-{check,public,trap}` | 6+74+18 | 480.5 |
| `causal-mixed-{check,public,trap}` | 6+80+26 | 610.7 |
| `causal-paged-{check,public,trap}` | 6+70+26 | 819.2 |
| `causal-attention` | 100 | 147.5 |
| `regression-{dense,nonpaged,paged}` | 84+86+100 | 1085.0 |
| `ordinary`、`public`、`metadata` | 100+100+100 | 96.0 |
| 五布局 `causal-*-profile` 及 `causal-profile` | 15+2 | 2582.8 |

新 TND PA 覆盖非均匀前缀/空段、allocation 与 used 分离、NaN 间隙、全部 P、跨页 mask、Sq<Sk/=/>、独立 stride/offset/广播/重叠、零 KV/全遮罩、记录复用与长度更新、真实/fake/compile/流生命周期及 40 个隔离 Trap。共享 CPU/消费者和固定 `regression-*` 覆盖受影响的 PA 与非 PA 路径；普通公开入口及 metadata 补测 v11 分派、失败清零和前缀更新，未把无 mask 组当作 causal 验收。

新路径 public 的每个 dtype 各 91 个数值输入，O/LSE 分别 672 次固定源逐元素一致且双数学检查通过，另有 27 次空值契约检查；见 `tnd-public-accuracy.json`。新代表和 profile 精度检查也通过。阶段独立 causal 组件中，FP16 的 2 例 O 超出数学阈值，但与固定源逐元素一致，按既定优先级记为 **source_precision_limit**，不记数学通过：shape `[2,129,3,1025]`、长度 `[17,1025]`、max-change、amplitude 4/8；LSE 和 BF16 均通过。原始输入/输出/reference 与误差在 `stage-component-accuracy.json` 所指 `.pt` 和 `accuracy.jsonl`；未放宽阈值。

## 阶段性能

`profile-final/attempt-1791910268526663460` 完成五布局及独立组件采样；`stage-average.json` 从绑定的原始耗时重新计算并校验跨布局运行/共享源码身份。采样前固定 **100 个含 dtype 配置**，每布局 10 输入×2 dtype；**62 个适用配置平均比 0.448559 ≤1**，38 个原生 FD 配置单列。适用配置和分段退化均为 0。先在配置内等权平均入口/R，再跨配置等权平均；R=1/4、5 次 warmup、5 轮×20 次 wall 不变。

| 布局 | 适用 / FD 单列 | 公开平均耗时比 | profile 组秒数 |
| --- | ---: | ---: | ---: |
| BSND | 12 / 8 | 0.498550 | 495.0 |
| TND | 16 / 4 | 0.411584 | 464.1 |
| 混合 | 14 / 6 | 0.404621 | 469.7 |
| BSND PA | 12 / 8 | 0.466804 | 450.8 |
| TND PA | 8 / 12 | 0.497049 | 451.8 |

独立 causal 组件组 2 项、251.3 秒。全部组件采样共 94 个 dtype/输入记录，39 个 kernel 比值 >1，全部保留于 `component-performance.json` 及原始 profiler CSV。新 TND PA 组件 FP16 比值 0.909–1.193、BF16 1.046–1.361；其他生产组件最大比值 2.526，独立 causal 组件为 0.918–1.057。组件与公开 E2E 计时边界不同，不能互相替代；提交门槛为上述适用公开平均值。

性能串行执行，第 6 卡采样前健康、AICore=0 且无其他运行进程，快照在 `before-performance-retry-device.log`，各原生驱动保留前后设备快照。实际固定 CANN 分派、key、设备二进制及库哈希随组保存，必要布局转换留在公开计时内。TND PA 使用已核验的 `build/mla-m07e3c-tnd-paged-public/vendor/mla_ca18aee_transformer`；BSND PA 使用对应 m07e2c 包，非 PA 使用既有固定源包。

## 复现与证据继承

统一正确性命令为 `python3 -m tests.mla_validation --output build/mla-m08e/<批次> --device 6 --max-cases 100 --suite <上表组名>`，先 dry-run/collect。每组最多 100 节点，未启用 full-matrix；候选/选中/未选中清单均保存。20 个阶段正确性组与工具修正前完全相同，见 `stage-selection-unchanged.json`；性能 100 配置、PA 六类页大小及 vendor 路径修正后的清单一致性分别见 `stage-performance-selection.json`、`performance-vendor-selection-unchanged.json`。

有效 attempt 的 `summary.json`、命令、环境、XML、source/evidence 身份和各组耗时位于：

- CPU：`cpu-fix/attempt-1791899603502785420`；构建工具 `cpu` 2 项在 `cpu/attempt-1791899413942121560`。
- 工具：`stage-tools/attempt-1791901938009309570` 的 tools/PA tools；汇总工具由 `stage-identity-tools/attempt-1791909513931449030` 的 22 项替代。
- 重建与代表：`representative-fixed/attempt-1791899698858824880`，生产重建 584.9 秒，实际加载核验通过。
- 阶段正确性：`devices-final/attempt-1791904532912958350`，20 组 1222 项；新代表的 6 项只在上一批执行/计数一次。
- 性能：`profile-final/attempt-1791910268526663460`。完整公开及组件数据见本组 `public-profile/`、`component/` 或 `probes/`，各段均值和所有单例由 `stage-average.json` 保留。

```bash
python3 -m tests.mla_validation --output build/mla-m08e/profile-final --device 6 --max-cases 100 --causal-stage-performance --suite causal-paged-tnd-profile causal-paged-profile causal-bsnd-profile causal-tnd-profile causal-mixed-profile causal-profile
python3 -m tests.mla_causal_stage_performance --evidence build/mla-m08e/profile-final/attempt-1791910268526663460/summary.json --output build/mla-m08e/stage-average.json
```

接手时核验 M08d 的 851 个源码身份、20 组及 4683 个产物，见 `baseline-audit.json`。`component-inheritance.json` 另核验 13 个未变算法依赖和 M08b2 的 **2 条历史记录、326 节点、1577 个产物**（8.1 秒），仅继承未变算法与最长 reference/最大内存成本边界，不计入阶段实测。阶段正确性与性能均在本步执行。

数学缓存为 `build/mla-validation-artifacts/golden`。有效数值组统计为 2176 次命中、6 次新计算/写入；工具中的故障注入缓存统计单独保留。public 修正后重跑的 182 个 reference 全命中，76 项耗时 179.7 秒。构建、数学 reference、历史证据分别管理；`acceptance.json` 与列出的依赖审计说明性能工具/独立汇总器/报错断言及 vendor 路径调整为何不使已完成的无关数值组失效。原生实现和数学阈值冻结后未再改动。

## 已解决失败及环境差异

`nonpassing-attempts.json` 保留 **7 条未计通过的记录**：两次收集在工具修改中失效（33.3/31.7 秒）；新 AICPU 函数缺少 namespace 限定导致 CPU 编译失败（27.0 秒）；相应旧生产构建因源码变化失败（246.2 秒，仅保留外层及编译日志，原 attempt summary 不可用）；为修正阶段性能全局上限中断未完成 public（633.3 秒）；缩短前缀时合法拒绝的报错文本断言过窄（2248.3 秒、70 通过/1 失败，不计闭环通过）；TND profile 误选 BSND vendor（151.8 秒，计时前失败）。受影响组均已重新通过，其余只续跑未完成项。

Python 3.11.15、torch 2.7.1+cpu、torch_npu 2.7.1、CANN 9.0.0-beta.2；第 6 卡 Ascend910B3，20 AIC/40 AIV。CATLASS `769cd40a8716b28650b6bebb08db4834eea4462f`；固定 CANN `ca18aee610729ca5270a44e2d84c751bc687fbf0`。CATLASS 工作树和所用 FIA/common 源路径未改；CANN 其他目录的删除/未跟踪文件保留，见 `external-worktrees-final.json`。未终止无关外部上下文。设备及 LSan 检查使用主机可见性，shell 为 `login=false`。

实际加载 `flash_attn_npu_3/flash_attn_npu_3.cpython-311-aarch64-linux-gnu.so`，SHA256 `43d465633e8b858933b241b7cdefc31af25d26f73070ce529948e238010d555d`。原始产物在忽略目录，迁机须复制完整证据或重验；恢复先核对进程、身份和完整记录，只补未完成或失效部分。

本步相关契约、文档链接/锚点及表格已核对，`git diff --check` 和 `ci/code_quality.sh --staged` 通过。质量日志为 `quality-staged.log` / `quality-staged-final.log`；只暂存本步 38 个文件，提交一次后立即停止。
