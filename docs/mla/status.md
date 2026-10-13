# MLA 当前状态

**M09a 必要验收已闭环。** 分支 `feat/mla-cann-migration`，接手基线 `0ded391a5d5b21a9f1872cf7c15085f000b0071c`。本步交付独立 CPU window 归一化、逐样本范围及完整 M 分核；只提交一次，提交后停止。唯一下一步为 [M09b1：独立 BSND window 数值组件](plan.md#唯一下一步-m09b1独立-bsnd-window-数值组件)，本次未启动。

## 当前可调用能力

生产能力仍为 512+64、FP16/BF16、Hkv=1、`num_splits=1` 下 BSND、TND varlen、混合 cache、BSND PA、TND PA 的无 mask/右下 causal；v7–v11、只读 cache、metadata 生成/复用与内部 Q-used 边界不变。公开有限/单侧窗口、split-KV 等后续能力继续明确拒绝，CPU 验收不代表 window 前向已开放。

新增 `mla_window.h/.cpp` 复用不变的 causal 几何校验/causal 特例、allocation/used 解析及固定源 SplitCore band 分支；不修改既有生产源码、构建列表、metadata、Kernel/Block/Tile、公开入口、CATLASS 或 CANN。范围与来源见 [design 6.2.6](design.md#626-独立-cpu-window-归一化与范围)，实施和验收选择见 [plan](plan.md#最近完成-m09a独立-cpu-window-归一化与块范围)。

固定 FA3 按调用级 Q/K 上界归一化，再按逐样本有效 Sq/Sk 右下对齐。保留单 query causal 优化、双侧/单侧无限、causal 覆盖右窗及空值；host 归一化后仍低于 `INT32_MIN` 的负值明确返回 `FieldOverflow`，防止 forward 的 int 窄化与 metadata 的 int64 行为不一致。大正 int64 窗口先归一为无限侧。完整 M 分核包含全遮罩和零 KV 行，失败不改变输出。

## 最近一次闭环证据

证据根为 `build/mla-m09a/`。`acceptance.json` 核验 **6 组、405 次本次 pytest 执行**，有效验收失败/跳过/未运行均为 0；核验 445 个原始产物、当前源码及固定源/工具链身份，耗时 1.7 秒。有效组运行合计 **221.4 秒**，不含收集、代表检查与失败/被替代尝试。普通与 sanitizer 是同一节点集合的两种执行模式，工具专项与工具回归的重叠按组分别计数。

| 必跑组 | 本次通过 | 秒数 |
| --- | ---: | ---: |
| `window-cpu` | 73 | 40.2 |
| `window-sanitize` | 73 | 82.9 |
| `causal-cpu` | 68 | 20.8 |
| `causal-sanitize` | 68 | 57.1 |
| `window-tools` | 23 | 6.4 |
| `tools` | 100 | 14.0 |

窗口组覆盖固定 FA3 forward/metadata 原文归一化（含实际 int 参数转换）、固定 CANN 范围/成本/全部 FD 候选字段、独立逐行 key 和任务唯一归属、有限双侧/单侧无限/causal 叠加、Sq<Sk/=/>、五布局 allocation/used 分离、空段/全遮罩/零 KV、头跨块/MN 尾块、1..26 核/C1V1/C1V2、随机重复调用及宽字段/累计 tile 溢出。每组四个独立 CPU 驱动的请求/响应、编译命令、源文件、退出码和 stderr 均保留；sanitizer 保留退出时 LeakSanitizer 检查。

复用契约的 causal 两组当次实测，不重跑无关设备矩阵。工具组检查 CPU 隔离、固定源缺失/损坏、依赖身份/运行中失效、成功/失败/中断/恢复、选择一致性及驱动协议。`selection-audit.json` 保存新旧清单差异：窗口 73/73、causal 68/68，未选均为 0；工具候选由 211 增至 217，仍选 100，新增窗口专项 23 全选。差异仅为新增窗口工具参数及原依赖失效用例参数化；固定选择策略与已有 CPU 命令不变。未启用 full-matrix。

有效原始记录：

- 四个 CPU 组：`acceptance/attempt-1791914918843312910/summary.json`。
- 窗口工具：`tools/attempt-1791914738266674020/summary.json` 的 `window-tools`。
- 通用工具：`tools/attempt-1791915243575239850/summary.json` 的 `tools`。
- 固定实现的开发代表为 4 项、20.8 秒；正式窗口 73 项已包含其覆盖，代表结果不再加入正式计数。

```bash
python3 -m tests.mla_validation --output build/mla-m09a/acceptance --suite window-cpu window-sanitize causal-cpu causal-sanitize --dry-run
python3 -m tests.mla_validation --output build/mla-m09a/acceptance --suite window-cpu window-sanitize causal-cpu causal-sanitize --collect-only
python3 -m tests.mla_validation --output build/mla-m09a/acceptance --suite window-cpu window-sanitize causal-cpu causal-sanitize --resume
python3 -m tests.mla_validation --output build/mla-m09a/tools --suite window-tools tools --resume
```

窗口固定 FA3 源从 `/tmp/mla-fa3-reference/d15f1531a460ba456f41b01a774f33ab2db8febf/flash_api.cpp` 读取，SHA256 `012547af36b3a780b2849b2ac26bda75d4aa26b0c83921449b81109128782fab`；可用 `MLA_FA3_SOURCE_ROOT` 指定相同内容的目录。每个窗口组保存完整源、许可证及提取 oracle，迁机可从这些证据复制固定源；不得依赖临时目录永久存在或回退到其他版本。

## 继承证据与环境

`baseline-audit.json` 复核 M08e 的 **37 组、1782 项、12592 个产物和 859 个源码身份**，耗时 9.0 秒。旧阶段快照的三个后续工具/断言差异与 M08e 已有依赖审计、当前 HEAD 一致，未误判为数值回归。本步单列继承其中 **27 个设备组、1245 项历史记录**，不计入本次 405；包括五布局 causal、相关公开/metadata/生命周期/Trap、`regression-dense/nonpaged/paged`、普通 FA3 与性能。其余历史记录仅核验，不填充本步计数。

`inheritance.json` 确认原生产构建的 **6324 个依赖和 11 个产物全部匹配**，新增 window 文件无生产调用者、未进入编译依赖，现有生产源码及构建列表无差异。实际加载扩展仍为 `flash_attn_npu_3/flash_attn_npu_3.cpython-311-aarch64-linux-gnu.so`，SHA256 `43d465633e8b858933b241b7cdefc31af25d26f73070ce529948e238010d555d`；加载库身份也与历史匹配，不因已有 .so 就假定新 window 已被编入。

历史 M08 阶段公开性能：100 个含 dtype 配置中，62 个适用配置平均比 0.448559，38 个原生 FD 配置单列；这是继承结果，本次未采样。历史独立 causal 组件的两例 FP16 O 数学阈值冲突仍按源逐元素一致记录 `source_precision_limit`，不称数学通过；原始数据在 `build/mla-m08e/stage-component-accuracy.json` 所指产物，阈值未改变。

Python 3.11.15、torch 2.7.1+cpu、torch_npu 2.7.1、CANN 9.0.0-beta.2，第 6 卡 Ascend910B3；CATLASS `769cd40a8716b28650b6bebb08db4834eea4462f`、CANN 固定源 `ca18aee610729ca5270a44e2d84c751bc687fbf0`。环境、主机进程及设备快照在 `environment.json`、`processes.log`、`npu-smi.log`。接手无未完成 MLA 验收进程；外部上下文未终止，CANN 其他目录的已有删除/未跟踪文件保留。shell 使用 `login=false`；设备身份核验与 LSan 使用主机可见性。

数学 reference 缓存仍为 `build/mla-validation-artifacts/golden`，统一入口默认 cache；本步没有设备数学 reference 或数值执行，无新增设备 reference 冷算。工具中的缓存故障注入属于工具验收，不作为窗口数值证据。

## 已解决尝试与收尾

`acceptance.json` 保留两条失败记录且不计通过：固定 FA3 oracle 包装的 `attention_chunk` 桩类型错误导致编译失败（13.9 秒）；收紧负值窄化边界后，一条正向范围矩阵仍保留 `INT64_MIN`（18.7 秒，失败前 13 项通过）。分别修正桩类型、把越界值列入拒绝检查并补 `INT32_MIN` 合法边界，失败组和未运行组均已通过。早期代表/工具结果在相关依赖修改后被替代；最终仅窗口用例修改使通用 tools 的保守身份失效，单独续跑 tools，未重复已有效的窗口工具组。

初次环境探测受沙箱设备可见性限制，已用主机核验；默认 Python 没有 pre-commit，质量检查使用已有 `PYTHONPATH=/tmp/mla-m01-precommit` 安装。相关契约、链接/锚点及表格按本步影响核对，不做无关整篇审计。差异检查、`git diff --check` 和 `ci/code_quality.sh --staged` 通过，日志为 `quality-staged.log` / `quality-staged-final.log`；只暂存本步 13 个文件，提交一次后立即停止。
