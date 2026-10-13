# MLA 当前状态

**M09b1 必要验收已闭环。** 分支 `feat/mla-cann-migration`，接手基线 `e58150c4fdf19cbfbe88dd9e74773915a5bd33fb`。本步交付独立连续 BSND window 数值组件；只提交一次，提交后停止。唯一下一步为 [M09b2：BSND window 完整公开行为](plan.md#唯一下一步-m09b2bsnd-window-完整公开行为)，本次未启动。

## 当前能力与本步边界

生产能力仍为 512+64、FP16/BF16、Hkv=1、`num_splits=1` 下 BSND、TND varlen、混合 cache、BSND PA、TND PA 的无 mask/右下 causal；v7–v11、只读 cache、metadata 生成/复用和内部 Q-used 边界不变。公开有限/单侧窗口、split-KV 等后续能力继续明确拒绝。

新增 `MlaWindowBsnd` 独立组件与 `MlaWindowKernelParams`；复用 M09a CPU window 分核及现有 Kernel / Block / Tile，在任务环接入非零 S2 起点，在 Vector 接双侧 mask。首块、任务环槽位、softmax/aMLA、空行和排空同步均有设备证据。CPU window/causal 算法、地址/Copy、metadata 协议、公开入口、CATLASS 和 CANN 未改。接口与来源见 [design 6.2.7](design.md#627-独立-bsnd-window-数值组件)，实际影响路径和验收选择见 [plan](plan.md#最近完成-m09b1独立-bsnd-window-数值组件)。

## 最近一次闭环证据

证据根 `build/mla-m09b1/`。`acceptance.json` 核验 **17 组、702 次本次 pytest 执行**，有效验收失败/跳过/未运行均为 0；核验 **5565 个原始产物**、固定源码和当前生产构建身份，耗时 6.2 秒。组运行合计 **2688.0 秒**，不含收集、构建/runtime 和被替代的工具开发检查；生产构建 **471.9 秒**、正式加载核验 **20.0 秒**。工具专项与 tools 的重叠按组分别计数。

| 必跑组 | 本次通过 | 秒数 |
| --- | ---: | ---: |
| `window-check` | 16 | 507.0 |
| `window-attention` | 66 | 200.0 |
| `window-profile` | 2 | 134.1 |
| `regression-dense` | 84 | 314.6 |
| `regression-nonpaged` | 86 | 734.5 |
| `regression-paged` | 100 | 38.3 |
| `causal-check` | 36 | 119.5 |
| `causal-bsnd-check` | 56 | 107.3 |
| `causal-tnd-check` | 6 | 99.3 |
| `causal-mixed-check` | 6 | 99.2 |
| `causal-paged-check` | 6 | 133.4 |
| `causal-paged-tnd-check` | 6 | 133.3 |
| `ordinary` | 100 | 24.0 |
| `causal-mixed-boundary` | 10 | 21.1 |
| `window-device-tools` | 20 | 5.0 |
| `tools` | 100 | 13.4 |
| `cpu` | 2 | 3.9 |

窗口两组共 **82 个节点、80 个含 dtype 独立输入、84 条数值记录**，stream 节点再次执行代表输入。覆盖有限双侧/单侧无限/causal 叠加、无 mask/causal 归一化特例、调用级上界与 used、Sq<Sk/=/>、非零 S2 起点、M/N/头尾块、零 KV/全遮罩/空容量、scale、任务环复用与 stream。固定源实际传 2048×2048 bool mask、band sparse mode 4 和独立归一化的 pre/next；causal/无 mask 特例分别传 3/0，没有用旧无 mask 组冒充窗口验收。

`selection-audit.json` 保存候选、选中、未选中及节点清单。窗口 16+66 互斥且全选；regression-dense/nonpaged 为 84/86 全选，paged 120 中选 100，未选主要为空值组合变体及两个已有跨 dtype 对应的任务环变体。ordinary 816 中固定选 100。工具候选由 M09a 的 217 增至 228，差异为新增窗口注册及既有失败/身份/恢复检查参数化，仍选 100；新增窗口工具专项 20 全选。固定选择策略、旧设备命令与数学阈值不变，未启用 full-matrix。

有效原始记录：

- 窗口代表：`check/attempt-1791916484641757720/summary.json`。
- 工具与构建 CPU：`tools/attempt-1791916715269854040/summary.json`。
- 其余正式组及性能：`acceptance/attempt-1791917044266747720/summary.json`。
- 固定实现身份：`frozen-sources.json`；独立代表的源/库/实际加载核验：`check-audit.json`。代表没有调用生产前向，后续生产重建不使其独立 probe 证据失效。

```bash
python3 -m tests.mla_validation --output build/mla-m09b1/check --device 6 --suite window-check
python3 -m tests.mla_validation --output build/mla-m09b1/tools --suite window-device-tools tools cpu --resume
python3 -m tests.mla_validation --output build/mla-m09b1/acceptance --device 6 --suite window-attention regression-dense regression-nonpaged regression-paged causal-check causal-bsnd-check causal-tnd-check causal-mixed-check causal-paged-check causal-paged-tnd-check ordinary causal-mixed-boundary window-profile --resume
```

各批次先 dry-run/collect 并核对集合；性能组始终重新采样，其他组恢复需核对依赖与原始证据。上述命令用于复现，不需要下一步接手时无条件重跑全部组。

## 数学精度与组件性能

窗口 84 条记录的 O/非空 LSE 均与固定 CANN 源逐元素一致；77 个独立输入双数学通过，3 个输入的 O 单列 `source_precision_limit`，LSE 均数学通过。`window-accuracy.json` 保存完整误差及原始 `.pt` 路径，阈值未改变：

| dtype / 输入 | O 最大误差 | 原数学阈值 |
| --- | ---: | ---: |
| FP16，B=2/Sq=129/H=3/Sk=1025、窗口 (513,7)、used=[17,1025]，max-change 幅度 4 | 0.015625 | 0.00390625 |
| 同上，幅度 8 | 0.015625 | 0.005859375 |
| BF16，B=41/Sq=7/H=73/Sk=513、窗口 (2,1)、used 循环 [0,1,3,513] 后接 0 | 1.7578125 | 0.09375 |

三例源也同样超限，按经确认的源一致性优先规则验收，不能记为数学通过。causal-check 另当次复现原有两条 FP16 O 源精度限制，源仍逐元素一致；与新增窗口三条分别保存在 `source-precision-limits.json`。

`component-performance.json` 汇总预定 8 个输入 × 双 dtype 的 **16 个配置**；保留逐例/分 dtype、全部 wall 样本、profiler CSV 和原始 source/migration kernel 对应。平均 kernel 比 **0.995162**，其中 6 个配置退化；预分配 probe 调用 wall 平均比 **1.003150**，其中 10 个配置退化。FP16 kernel/wall 均值为 **1.002464/1.003525**，BF16 为 **0.987860/1.002774**。保持 5 次预热、5 轮×20 次 wall 与 5 轮×10 次 profiler、交替顺序及完成同步，不按耗时筛选。

这是同调度非 FD 的组件对照，包含 ctypes/发射/同步的 probe wall 略有退化；**不是公开或 metadata E2E，也不声称公开窗口已满足平均性能门槛**。公开窗口及其适用 R=1/4 E2E 由 M09b2 独立验收。采样前/后的设备和进程证据表明第 6 卡只有本次验收活动。

## 继承、缓存与环境

`baseline-audit.json` / `inheritance.json` 核验 M09a window/causal 普通及 sanitizer **4 组、282 项、430 个原始产物**，单列继承，不计入本次 702。CPU 源/头/驱动、固定 FA3/CANN 与工具链全部匹配；宽快照中的差异仅为设备 `.hpp`、设备 builder 和 runner 注册，它们不属于这四组的 CPU 编译路径。新增 window 设备文件不进入这些 CPU 驱动。地址/Copy 与原数学公式不变；已开放设备路径用上述当次回归验收，没有用旧设备结果填充本次计数。

数学缓存仍为 `build/mla-validation-artifacts/golden`。数值组 **575 次命中、258 次冷算/写入**，无数值缓存错误或刷新；stream 复用代表 reference，性能不重复计算数学结果。两个新窗口成本边界为 `(1,1,128,16384)` / `(10,3,86,8193)`；双 dtype 单例最长 134.9 秒，主机峰值约 14.3 GiB，NPU 峰值分配约 178.2 MiB。工具组的刷新/损坏注入另计，不当作数值缓存故障。

Python 3.11.15、torch 2.7.1+cpu、torch_npu 2.7.1、CANN 9.0.0-beta.2；第 6 卡 Ascend910B3。CATLASS `769cd40a8716b28650b6bebb08db4834eea4462f`、CANN 固定源 `ca18aee610729ca5270a44e2d84c751bc687fbf0`。接手分支/工作树/进程与 M09a 一致，无未完成 MLA 验收；外部进程、CANN 其他目录和 CATLASS 保持原状。

原生产构建的 6324 个依赖和 11 个产物在接手时匹配；共享头修改后强制重建，当前 **6325 个依赖、11 个产物全部匹配**。正式实际加载扩展为 `flash_attn_npu_3/flash_attn_npu_3.cpython-311-aarch64-linux-gnu.so`，SHA256 **`639175c3de536dd84ceb172b6d3eb36a2e1977eb0d2333456e55863452dcfee0`**；完整构建/运行栈/loaded 身份在正式 attempt。初始环境快照在 `environment.json`，不能将其中的旧扩展 SHA 当作当前产物。

## 已解决尝试与收尾

初次共享回归收集因沙箱无法访问 NPU 而失败，未执行数值；在主机只续收集失败及未收集组后通过。原始记录保留在 `acceptance/attempt-1791916527975538590/`，前后日志为 `acceptance-collect.log` / `acceptance-collect-host.log`。工具开发结果在格式/探针空容量边界修正后被最终工具组替代；正式数值实现自代表开始未变，无数值失败或中断，不重跑有效组。

shell 使用 `login=false`，实机与隔离检查使用主机可见性；pre-commit 使用已有 `PYTHONPATH=/tmp/mla-m01-precommit` 安装。按本步契约核对相关文档、链接/锚点和表格；差异及质量记录为 `doc-links.json`、`quality-development.log` / `quality-staged.log`。只暂存本步 18 个文件，`git diff --check` 与 `ci/code_quality.sh --staged` 均通过，不绕过提交钩子；一次提交后停止。
