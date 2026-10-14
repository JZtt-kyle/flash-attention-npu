# MLA 当前状态

**M09d 已闭环：BSND Q + ND PA KV window 只读 cache。** 分支 `feat/mla-cann-migration`，本轮起点 `2b8441efe150aa22faff71d0f71efbf36a839cfa`；本步提交可由 `git log -1 -- docs/mla/status.md` 定位。唯一下一步是 [M09e：TND PA window 与阶段收尾](plan.md#唯一下一步-m09etnd-pa-window-与阶段收尾)，本轮未启动它。

## 当前能力与本步边界

512+64、FP16/BF16、Hkv=1、`num_splits=1` 的 BSND PA cache 已接通有限/单侧窗口、causal 叠加、非零 S2 起点、跨页/页尾 mask、全部合法 P，以及公开 metadata 生成/复用。`version=kind=15` 精确字数 `411+5B`，绑定 Q 容量、P/C、allocation/used、当次 KV 长度和归一化窗口；归一化使用逻辑 KV 容量 C，不能用页对齐的有效 S2 替代。内部 Q-used 每次生成且拒绝显式记录，公开签名不增加 Q-used。生产生成与消费沿用调用者 stream；即使 Q-used 为零仍检查全部有效页。契约见 [design 5.17](design.md#517-v15bsnd-pa-window-记录)。

复用既有 window 分核/任务环、PA 地址/Copy 和 Block/Tile。CATLASS 与固定 CANN 源未改。TND PA window、split-KV 等未开放能力继续明确拒绝；本步不是 M09 阶段收尾。

## 最近一次闭环证据

证据根 `build/mla-m09d/`，最终汇总 `acceptance.json`：**本轮 25 组、1113 项有效通过，必跑集合无未解决失败、跳过或未运行；历史继承另计 9 组、837 项。** 每组最多 100 个 pytest 节点，无 full-matrix。新 check/public/trap 分别 8/70/36 且互斥；分页固定回归 120 选 100、非分页 86、ordinary 804 选 100、CPU 共享各 1198 选 100，补公开/schema/未开放边界 42。窗口新行为使用独立 window 组，旧无 mask 组仅作回归。

| 统一测试组 | 本轮通过 | 耗时（秒） |
| --- | ---: | ---: |
| `window-paged-cpu` | 38 | 52.4 |
| `window-paged-sanitize` | 38 | 139.2 |
| `causal-paged-cpu` | 36 | 32.5 |
| `causal-paged-sanitize` | 36 | 109.5 |
| `causal-paged-tnd-cpu` | 42 | 31.5 |
| `causal-paged-tnd-sanitize` | 42 | 108.5 |
| `paged-shared-cpu` | 100 | 53.4 |
| `paged-shared-sanitize` | 100 | 790.6 |
| `window-paged-tools` | 35 | 7.3 |
| `causal-paged-tools` | 57 | 5.0 |
| `tools` | 100 | 14.6 |
| `cpu` | 2 | 3.4 |
| `window-paged-check` | 8 | 325.6 |
| `window-paged-public` | 70 | 1208.5 |
| `window-paged-trap` | 36 | 862.5 |
| `regression-paged` | 100 | 38.2 |
| `regression-nonpaged` | 86 | 745.0 |
| `causal-paged-check` | 6 | 141.5 |
| `causal-paged-tnd-check` | 6 | 130.4 |
| `window-bsnd-check` | 10 | 116.4 |
| `window-tnd-check` | 10 | 136.4 |
| `window-mixed-check` | 10 | 159.5 |
| `ordinary` | 100 | 39.4 |
| `window-paged-boundary` | 42 | 24.8 |
| `window-paged-profile` | 3 | 490.9 |

`acceptance.json` 核验 6188 份原始产物、914 个最终源文件、6332 项构建依赖、11 个构建输出与实际加载身份；四份临时性能采集共 318 个文件逐一哈希归档，临时目录已释放。核验耗时 50.7 秒，分组执行耗时合计 5767.1 秒（含并行组，非墙钟总时长）。候选/选中/未选中清单见各组 `selection.json` 及根目录 `*-selection-audit.json`、`summary-selection-audit.json`。

最终性能为 `performance/attempt-1791949041171701690/`；CPU、工具、设备恢复的具体 attempt、身份与原始目录逐组记录在 `acceptance.json`，没有用未完成记录代替通过。复现统一使用以下组；先 `--dry-run` / `--collect-only` 核对集合，恢复前核对进程、原始证据和实际依赖，只续未完成或失效组：

```bash
python3 -m tests.mla_validation --output build/mla-m09d/<批次> --suite window-paged-cpu window-paged-sanitize causal-paged-cpu causal-paged-sanitize causal-paged-tnd-cpu causal-paged-tnd-sanitize paged-shared-cpu paged-shared-sanitize
python3 -m tests.mla_validation --output build/mla-m09d/<批次> --suite window-paged-tools causal-paged-tools tools cpu
python3 -m tests.mla_validation --output build/mla-m09d/<批次> --device 6 --suite window-paged-check window-paged-public window-paged-trap regression-paged regression-nonpaged causal-paged-check causal-paged-tnd-check window-bsnd-check window-tnd-check window-mixed-check ordinary window-paged-boundary
TMPDIR=/dev/shm python3 -m tests.mla_validation --output build/mla-m09d/<批次> --device 6 --suite window-paged-profile
```

## 数值与性能

`window-accuracy.json` 汇总代表、公开与性能预检：双 dtype 共 186 个不同配置、216 条报告、212 份原始 tensor 记录；2792 次 O/LSE 比较均与同调度固定源逐元素一致，其中 **2783 次数学通过，9 次 O 比较按源精度限制通过**，另有 54 次空输出数学检查。9 次均为同一个 BF16 配置（公开 8 路、性能预检 1 路）：P=112、Q=[3,3]、KV=[8193,1025]、H=128、C=8400、window=(129,2)。O 最大误差为 0.205078125，固定源同样超出原阈值；LSE 数学通过，shape/NaN/inf/空行契约检查不变。原阈值、误差及输入/源/迁移/双 reference 全部保留，不能把这 9 次写成数学通过；详见 `window-accuracy.json` 的 `source_limits` 和 `source-precision-limits.json`。

`public-performance.json`：固定 12 输入×双 dtype=24 配置，全部实测；原生 template=5，其中 **10 个非 FD 适用配置平均公开 E2E 比值 0.385079 ≤1**，FP16/BF16 分别 0.386625/0.383534，适用配置及分段退化均为 0。其余 14 个原生 FD 配置完整保留并单列，不纳入未迁移 FD 的门槛。implicit/generate/reuse 的 R=1/4 六段均保留；5 次预热、5 轮×20 次 wall、必要 PA 转换和完成同步边界不变。两种 dtype 的原生 O/LSE 共 48 次双数学检查通过。六段平均比值依次为 implicit 0.463449/0.373819、generate 0.570014/0.302588、reuse 0.356970/0.243634。

代表组件三输入×双 dtype 的 Kernel 比值分别为 FP16 **1.35712/1.10719/1.50184**、BF16 **1.42433/1.07735/1.62162**，六例均比同调度固定源慢；生产 Kernel 包含 v15/当次有效页校验。这些组件结果用于记录与定位，未混作公开 E2E，也未隐去退化。原始各例耗时和 profiler CSV 在最终性能组的 `component/`、`public-profile/`，硬门槛按 [design 第 7 节](design.md#7-验收口径) 的适用平均公开 E2E 判定。

## 缓存、继承与恢复

数学 reference 默认复用 `build/mla-validation-artifacts/golden`。本轮最终有效设备组统计 **470 hit、195 miss/write**；其中新 window 代表与公开 185 次冷算，ordinary 10 次，其余命中；最终性能 54 次全部命中。开发、代表和提交验收未刷新同输入 reference，待测实现和固定源仍实际执行。工具故意测试的关闭/刷新/损坏另列在 `acceptance.json`，不冒充设备缓存异常或精度结果。新成本边界 FP16/BF16 为 94.7/88.6 秒，主机最大 RSS 7061628 KiB、NPU 峰值 272626176 字节，原始 `cost.json` 随代表组保存。

接手 `baseline-audit.json` 核验 M09c2 的 29 组 1311 项和当时构建/加载，未将旧扩展当作本步生产证据。`inheritance.json` 继承依赖未变的 M09a CPU window/causal、M09b1 独立连续 window 数值/组件性能和 M07 独立 PA 地址/Copy，9 组 837 项、12791 份原始身份、编译依赖及工具链均匹配，最终复核 4.6 秒。M09b1 的三条 O 源精度限制继续单列于 `build/mla-m09b1/source-precision-limits.json`，不计为数学通过。

原无头路由中断、AICPU 测试包装器重复符号、窗口包装结构未传内层调度、旧 TND 拒绝消息断言，以及性能身份/汇总的 tuple/list 比较问题均已修复；失败与中断尝试保留但不计通过。`resume-compatibility.json`、`message-change-audit.json`、`profile-compatibility.json`、`summary-change-audit.json` 分别核对实际编译/调用依赖、AST、原始产物、构建与加载，保留未受影响结果；无完成记录的 sanitizer 重新运行。性能失败组按统一入口现行规则重新采样，reference 全部复用；最后三节点完整通过才记闭环。此前两轮失败性能仅作诊断，未写成通过。

## 环境与收尾

Python 3.11.15、torch 2.7.1+cpu、torch_npu 2.7.1、CANN 9.0.0-beta.2；第 6 卡 Ascend910B3。CATLASS `769cd40a8716b28650b6bebb08db4834eea4462f`，固定 CANN `ca18aee610729ca5270a44e2d84c751bc687fbf0`。实际加载本仓 `flash_attn_npu_3/flash_attn_npu_3.cpython-311-aarch64-linux-gnu.so`，SHA256 `1c15c067b2be43bcac922affc7a52be701165a0e7c2c815a0e9d208a9550c603`。生产构建、固定源库及各探针依赖见 `build-manifest.json` 和各组 manifest；采样前第 6 卡健康空闲，性能与其他本轮验收隔离。

本步相关契约、计划与状态已同步；349 个本地链接/锚点、`git diff --check` 和 `ci/code_quality.sh --staged` 均通过，质量工具未改动冻结源码，日志见 `doc-links.json`、`quality-final.log`。只暂存本步 39 个文件并保留提交钩子，提交后立即停止；唯一下一步为 M09e，阶段完整窗口组合尚未在本步运行。
