# MLA 当前状态

**M15d 非 PA 混合布局 softcap 完整公开接入已完成。唯一下一步是五轮开发后的[专门文档修正轮](plan.md#唯一下一步-文档修正轮)，本轮提交后停止。** 本轮从 `feat/mla-cann-migration` 的 `9a55a98` 干净工作树开始；文档轮后继续 M15e BSND Q + ND PA KV softcap。

## 当前能力与边界

本轮接通 512+64 和 192/128 非 PA TND Q + BSND KV cache 正 softcap，包含三 mask、自动/1/显式 split、真实 accumulator、metadata、公开 kvcache real/fake/compile 与 raw C++。压缩使用 v38/v39，prefill 使用 v40；记录只绑定是否启用，正 cap 可改值复用。压缩单片复用 partial/merge，公开 accumulator 仍为空。数值组件沿用 M15a，CANN 源仓和 CATLASS 未修改。

非 PA BSND、TND Q/KV 正 cap 保持已开放。公开混合 metadata 接收 Q allocation 前缀/max 与非累计 KV used；KV 起点为矩形槽 `b*Sk`，不受 used 影响。公开接口仍无 Q-used，内部压缩 Q-used 仍不允许显式 metadata 复用，内部 prefill 支持带 Q-used 的记录。PA 的正 cap、非默认调度和 graph replay 仍未开放；910C 未实机验证。契约见 [design 2.2](design.md#22-后端选项与返回值) 和 [5.18](design.md#518-混合-softcap-记录)。

## 最近一次闭环证据

证据根 `/tmp/mla-m15d/`，测试均通过 `python3 -m tests.mla`。收尾使用驱动既有 `suite_sources`、`suite_dependencies`、`reusable`、`matching_files` 和 `build_inputs`：21 组当前来源/依赖与原始证据有效，生产输入、依赖、产物及实际加载一致，native 包配方与内容有效，见 `closeout-validation.log`。无头循环记录为会话进程中断，最终测试进程实际已完成；没有把中断记为测试通过。

| summary / attempt | 已确认结果与耗时 |
| --- | --- |
| `cpu/summary.json` / `1792149568258100230` | 工具 29 passed / 29.1 s；CPU 86 passed / 93.5 s；ASan/UBSan 86 passed / 305.3 s |
| `preflight/summary.json` / `1792149771212267200` | 生产重建 580.0 s、实际加载校验通过；native 混合代表性能夹具 1 passed / 207.1 s；正 cap 性能夹具 4 passed / 158.7 s |
| `final/summary.json` / `1792150886789569710` | 代表 32 passed / 37.1 s；公开专项 202 passed / 442.5 s，含 24 个隔离 Trap；固定源 32 passed / 280.6 s；累计 E2E 595 passed / 105.9 s；ordinary 508 passed / 25.5 s；正 cap 正式采样 4 passed / 118.2 s，覆盖 10 配置 |
| `perf-prefill/summary.json` / `1792282393312721450` | 五布局各 12 配置全部实测，分组耗时 237.8–263.3 s；既有汇总器通过 |
| `perf-compressed/summary.json` / `1792283717420686910` | 五组各 6 配置全部实测，共 1126.7 s；指定组汇总通过 |

必跑集合已 dry-run/collect 核对，工具/native 包和代表夹具先行，固定生产实现后执行最终集合。代表长度更新用例补充 Q 前缀原地重分配后，重新执行最终代表与专项；不使用旧代表结果替代。本轮没有继承历史测试通过证据：生产源码变化后实跑累计 E2E，历史单元不累计。数学 reference 使用内容缓存；最终 E2E 命中 786 次，ordinary 命中 31 次，正 cap 正式性能命中 20 次，待测实现均实际执行。

正 cap 双数学和 cap=0 固定源逐元素对照通过，没有源精度例外。长度原地更新/记录重生成、正 cap 改值复用、独立 stride、空值、真实 accumulator、fake/compile、raw C++ 和多 stream/生命周期由新增专项覆盖。累计 E2E 从 555 项变为 595 项：新增 40 项所属日常用例，原有用例全部保留；过期 mixed 拒绝断言按新契约修正，ordinary 仍为 508 项。最终必要验收无失败或未运行项。

## 公开性能

采样前固定 100 个目标配置（含 dtype）：正 cap 混合 10 项、共用 prefill 五布局 60 项、压缩显式 TND/混合和自动 BSND/TND/混合 30 项。R=1/4、预热 5 次、wall 5×20、kernel 5×10 不变，未按耗时筛选或截断。

| 集合 | 实测 / 同义适用 / 单列 | 配置平均耗时比 | 退化配置 |
| --- | --- | --- | --- |
| prefill 五布局 | 60 / 30 / 30 | 0.801354 | 0 / 30 |
| 压缩显式 TND | 6 / 2 / 4 | 0.290502 | 0 / 2 |
| 压缩显式混合 | 6 / 2 / 4 | 0.427003 | 0 / 2 |
| 压缩自动 BSND | 6 / 6 / 0 | 1.037449 | 2 / 6 |
| 压缩自动 TND | 6 / 6 / 0 | 0.797413 | 2 / 6 |
| 压缩自动混合 | 6 / 6 / 0 | 0.679972 | 2 / 6 |
| **压缩五组整体** | **30 / 22 / 8** | **0.751092** | **6 / 22** |

`prefill-performance.json` 与 `compressed-performance.json` 由既有汇总器从绑定证据重算并通过。压缩采用 `selected-groups`，仅为本步固定子集，不是 M10 阶段收尾；组均值作诊断，按配置等权整体平均≤1 验收。prefill/压缩分别有 34/31 个退化入口分段，所有单例、分段和原生 FD 单列结果保留。prefill 的 60 个 kernel 对照均退化，公开平均通过不代表 kernel 全部提速。

正 cap 没有同义 CANN attention。10 个配置的 cap=15/cap=0 公开调用配置平均比为 1.101108，9 个高于 1，仅作开销参照，不计同义 CANN 门槛。kernel 统计为 68.19–445.35 μs，设备 metadata 为 23.11–50.20 μs，metadata 公开调用为 171.80–303.59 μs；全部入口/R、wall 原样本和 profiler 分段在 `final/attempt-1792150886789569710/softcap-mixed-profile/` 保留。

## 复现与收尾

```bash
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export MLA_NATIVE_ROOT=/tmp/mla-m13b/native
python3 -m tests.mla --suite softcap-mixed-tools unit-softcap_mixed softcap-mixed-sanitize --output /tmp/mla-m15d/cpu
python3 -m tests.mla --suite prefill-mixed-profile-check softcap-mixed-profile-check --output /tmp/mla-m15d/preflight
python3 -m tests.mla --suite softcap-mixed-check detail-softcap_mixed softcap-mixed-source e2e ordinary softcap-mixed-profile --output /tmp/mla-m15d/final
python3 -m tests.mla --prefill-stage-performance --suite prefill-bsnd-profile prefill-tnd-profile prefill-mixed-profile prefill-paged-bsnd-profile prefill-paged-tnd-profile --output /tmp/mla-m15d/perf-prefill
python3 -m tests.mla.support.mla_prefill_stage_performance --evidence /tmp/mla-m15d/perf-prefill/summary.json --output /tmp/mla-m15d/prefill-performance.json
python3 -m tests.mla --split-stage-performance --suite split-tnd-profile split-mixed-profile auto-split-bsnd-profile auto-split-tnd-profile auto-split-mixed-profile --output /tmp/mla-m15d/perf-compressed
python3 -m tests.mla.support.mla_split_stage_performance --evidence /tmp/mla-m15d/perf-compressed/summary.json --suite split-tnd-profile split-mixed-profile auto-split-bsnd-profile auto-split-tnd-profile auto-split-mixed-profile --output /tmp/mla-m15d/compressed-performance.json
```

提交收尾使用 `git diff --check` 与 `PYTHONPATH=/tmp/mla-m10c8a/quality ci/code_quality.sh --staged`，质量输出保留在 `quality-final.log`；本次影响的文档链接记录在 `handoff-links.log`。当前闭环提交可由 `git log -1 -- docs/mla/status.md` 定位。

## 环境与产物

Python 3.11.15、torch 2.7.1+cpu、torch_npu 2.7.1、CANN `/usr/local/Ascend/cann-9.0.0-beta.2`、CATLASS `769cd40a8716b28650b6bebb08db4834eea4462f`，卡 6 Ascend910B3。固定 CANN 源仓为 `ca18aee610729ca5270a44e2d84c751bc687fbf0`。生产扩展已重建，驱动实际加载本仓 `flash_attn_npu_3/flash_attn_npu_3.cpython-311-aarch64-linux-gnu.so`，SHA256 `e37136919b4800d28f0658f41f68ae3ebdb6645e40bab67e427a991a2cd59347`。

默认工作根 `/tmp/flash-attention-npu-mla-46928a793876/` 保存生产 `mla-validation-artifacts/prepare.json` 与数学缓存；native 包使用 `/tmp/mla-m13b/native`，默认旧包配方已过期，不可误用。`build/` 仅保留无头循环日志/状态。`/tmp` 非持久存储，证据缺失或内容变化后不可继承。
