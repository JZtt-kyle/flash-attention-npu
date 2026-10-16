# MLA 当前状态

**M15c 非 PA TND Q/KV softcap 完整公开接入已完成。唯一下一步是 [M15d 非 PA 混合布局 softcap](plan.md#唯一下一步-m15d)，本轮提交后停止。** 本轮从 `feat/mla-cann-migration` 的 `0b4529d` 干净工作树开始。

## 当前能力与边界

512+64 和 192/128 的非 PA BSND、TND Q/KV 均支持正 softcap，包含三 mask、自动/1/显式 split、真实 accumulator、metadata、公开 real/fake/compile 与 raw C++。本轮 TND 压缩使用 v35/v36，prefill 使用 v37；记录只绑定是否启用，正 cap 可改值复用。压缩单片复用 partial/merge，公开 accumulator 仍为空。数值组件沿用 M15a，CANN 源仓和 CATLASS 未修改。

公开 varlen 每次内部生成 TND Q/KV 记录。官方 metadata 生成器没有 KV allocation 前缀参数，TND Q/KV 显式生成仍限内部算子。混合与 PA 的正 cap、非默认调度和 graph replay 仍未开放；910C 未实机验证。契约见 [design 2.2](design.md#22-后端选项与返回值) 和 [5.17](design.md#517-tnd-softcap-记录)。

## 最近一次闭环证据

证据根 `/tmp/mla-m15c/`，测试均通过 `python3 -m tests.mla`。收尾使用驱动既有 `suite_sources`、`suite_dependencies`、`reusable`、`matching_files` 和 `build_inputs`：21 组当前来源/依赖与原始证据有效，生产输入、依赖、产物及实际加载一致，native 包配方与内容有效，结果见 `closeout-validation.log`。

| summary / attempt | 本轮结果与耗时 |
| --- | --- |
| `final-cpu/summary.json` / `1792143841822740590` | 工具 29 passed / 29.4 s；CPU 80 passed / 87.4 s；ASan/UBSan 80 passed / 304.2 s |
| `preflight/summary.json` / `1792142576526855110` | 生产重建 568.0 s、实际加载校验通过；native TND 代表性能夹具 1 passed / 205.2 s，该夹具的当前来源/依赖仍一致 |
| `final/summary.json` / `1792144267422210530` | 代表 32 passed / 29.8 s；正 cap 性能夹具 4 passed / 124.6 s；公开专项 186 passed / 399.6 s，含 24 个隔离 Trap；固定源 32 passed / 473.0 s；累计 E2E 555 passed / 105.2 s；ordinary 508 passed / 29.7 s；正 cap 正式采样 4 passed / 118.7 s，覆盖 10 配置 |
| `perf-prefill/summary.json` / `1792145631097877700` | 五布局各 12 配置全部实测，分组耗时 253.8–277.0 s；既有汇总器通过 |
| `perf-compressed/summary.json` / `1792147005944188260` | 五组各 6 配置全部实测，共 1238.2 s；指定组汇总通过 |

必跑集合已 dry-run/collect 核对，工具/native 包和代表夹具先行，固定实现后执行最终集合。本轮没有继承测试通过证据：生产源码变化后实跑累计 E2E，历史单元不累计。数学 reference 使用内容缓存；最终 E2E 命中 740 次，ordinary 命中 31 次，正 cap 正式性能命中 20 次，待测实现均实际执行。

固定源逐元素一致且通过双数学，本轮没有源精度例外。长度原地更新/记录重生成、正 cap 改值复用、独立 stride、空值、真实 accumulator、fake/compile、raw C++ 和多 stream/生命周期由新增专项覆盖。累计 E2E 从 515 项变为 555 项：新增 42 项所属日常用例，移除 BSND softcap 专项中 2 项过期 TND 拒绝断言；其余旧 TND 拒绝用例也按新契约修正，ordinary 仍为 508 项。

早期 CPU 单片计划测试把实际任务数等同于静态容量上界，已修正独立预期并重跑 CPU 与 sanitizer。沙箱设备不可见及并行 collect 的 NPU 初始化阻塞已通过主机环境和顺序运行处理，仅终止本轮阻塞进程。失败/中断日志保留，未记为通过；当前必要验收无失败或未运行项。

## 公开性能

采样前固定 100 个目标配置（含 dtype）：正 cap TND 10 项、共用 prefill 五布局 60 项、压缩显式 TND/混合和自动 BSND/TND/混合 30 项。R=1/4、预热 5 次、wall 5×20、kernel 5×10 不变，未按耗时筛选或截断。

| 集合 | 实测 / 同义适用 / 单列 | 配置平均耗时比 | 退化配置 |
| --- | --- | --- | --- |
| prefill 五布局 | 60 / 30 / 30 | 0.812071 | 0 / 30 |
| 压缩显式 TND | 6 / 2 / 4 | 0.288605 | 0 / 2 |
| 压缩显式混合 | 6 / 2 / 4 | 0.430102 | 0 / 2 |
| 压缩自动 BSND | 6 / 6 / 0 | 1.031627 | 2 / 6 |
| 压缩自动 TND | 6 / 6 / 0 | 0.814527 | 2 / 6 |
| 压缩自动混合 | 6 / 6 / 0 | 0.657791 | 0 / 6 |
| **压缩五组整体** | **30 / 22 / 8** | **0.748231** | **4 / 22** |

`prefill-performance.json` 与 `compressed-performance.json` 从绑定证据重算。压缩采用 `selected-groups`，仅为本步固定子集，不是 M10 阶段收尾；组均值作诊断，按配置等权整体平均≤1 验收。prefill/压缩分别有 39/25 个退化入口分段，所有单例、分段和原生 FD 单列结果保留。prefill 的 60 个 kernel 对照均退化，公开平均通过不代表 kernel 全部提速。

正 cap 没有同义 CANN attention。10 个配置的 cap=15/cap=0 公开调用配置平均比为 1.065446，6 个高于 1，仅作开销参照，不计同义 CANN 门槛。kernel 统计为 79.45–461.44 μs，metadata 为 22.55–51.01 μs；全部入口/R、wall 原样本和 profiler 分段在 `final/attempt-1792144267422210530/softcap-tnd-profile/` 保留。

## 复现与收尾

```bash
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export MLA_NATIVE_ROOT=/tmp/mla-m13b/native
python3 -m tests.mla --suite softcap-tnd-tools unit-softcap_tnd softcap-tnd-sanitize --output /tmp/mla-m15c/final-cpu
python3 -m tests.mla --suite prefill-tnd-profile-check --output /tmp/mla-m15c/preflight
python3 -m tests.mla --suite softcap-tnd-check softcap-tnd-profile-check detail-softcap_tnd softcap-tnd-source e2e ordinary softcap-tnd-profile --output /tmp/mla-m15c/final
python3 -m tests.mla --prefill-stage-performance --suite prefill-bsnd-profile prefill-tnd-profile prefill-mixed-profile prefill-paged-bsnd-profile prefill-paged-tnd-profile --output /tmp/mla-m15c/perf-prefill
python3 -m tests.mla.support.mla_prefill_stage_performance --evidence /tmp/mla-m15c/perf-prefill/summary.json --output /tmp/mla-m15c/prefill-performance.json
python3 -m tests.mla --split-stage-performance --suite split-tnd-profile split-mixed-profile auto-split-bsnd-profile auto-split-tnd-profile auto-split-mixed-profile --output /tmp/mla-m15c/perf-compressed
python3 -m tests.mla.support.mla_split_stage_performance --evidence /tmp/mla-m15c/perf-compressed/summary.json --suite split-tnd-profile split-mixed-profile auto-split-bsnd-profile auto-split-tnd-profile auto-split-mixed-profile --output /tmp/mla-m15c/compressed-performance.json
```

提交收尾使用 `git diff --check` 与 `PYTHONPATH=/tmp/mla-m10c8a/quality ci/code_quality.sh --staged`，质量输出保留在 `quality-final.log`；本次影响的文档链接记录在 `handoff-links.log`。当前闭环提交可由 `git log -1 -- docs/mla/status.md` 定位。

## 环境与产物

Python 3.11.15、torch 2.7.1+cpu、torch_npu 2.7.1、CANN `/usr/local/Ascend/cann-9.0.0-beta.2`、CATLASS `769cd40a8716b28650b6bebb08db4834eea4462f`，卡 6 Ascend910B3。固定 CANN 源仓为 `ca18aee610729ca5270a44e2d84c751bc687fbf0`。生产扩展已重建，驱动实际加载本仓 `flash_attn_npu_3/flash_attn_npu_3.cpython-311-aarch64-linux-gnu.so`，SHA256 `5d58ef957eb50220508991c51b31cd0331ce4b1bcc6b263ba525b2deca7a1bd1`。

默认工作根 `/tmp/flash-attention-npu-mla-46928a793876/` 保存生产 `prepare.json` 与数学缓存；native 包使用 `/tmp/mla-m13b/native`，默认旧包配方已过期，不可误用。`build/` 仅保留无头循环日志/状态。`/tmp` 非持久存储，证据缺失或内容变化后不可继承。
