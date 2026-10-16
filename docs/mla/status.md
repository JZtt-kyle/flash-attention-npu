# MLA 当前状态

**M14 prefill 组合验收已通过；唯一下一步是 [M15a softcap 独立数值闭环](plan.md#唯一下一步-m15a)。** 分支 `feat/mla-cann-migration`，接手提交 `07b462d`，接手工作区干净。本轮完成阶段工具、过期拒绝测试和文档收尾，提交一次后停止；未改生产 host/AICPU/Kernel、metadata、地址/发射、构建参数或 CATLASS。

## 当前能力与边界

512+64 压缩 MLA 与 192/128 独立 K/V prefill 均支持 FP16/BF16、五布局、三 mask、自动/1/显式 split、长度/空值、合法 stride 和真实 FP32 accumulator。M14 已完成 192/128 的阶段组合核对和公开性能汇总，覆盖映射见 [design 7.6](design.md#76-m14-prefill-阶段覆盖)。

公开 kvcache/metadata 没有 Q-used 参数；192/128 内部 TND Q 路径支持 Q-used 与显式 metadata，内部 BSND Q-used 仍只支持隐式记录。PA 检查全部 live 页，包括 Q allocation/used 为零的样本。softcap、非默认调度仍明确拒绝，完整 graph replay 留待 M16，910C 未实机验证。接口与拒绝边界以 [design 2.2](design.md#22-后端选项与返回值) 为准。

本轮删除 BSND 专项中两项“合法 TND Q + BSND cache 必须拒绝”的过期用例；PA 拒绝明确为非法 65-token 页大小，TND 的 mixed 拒绝明确为 cache batch 不匹配。累计日常清单仍为 477 项，各 prefill 布局保留双 dtype × 三 mask × 三 split 模式的 18 项日常用例；公开专项从 618 调整为 616 项，没有截断其余覆盖。

## 最近闭环证据

证据根 `/tmp/mla-m14/`。统一驱动保留各 attempt 的 `command.json`、`cases.json`、`selection.json`、源码/构建身份、实际加载和原始证据；下表路径相对此根。

| summary / attempt | 本轮有效结果与耗时 |
| --- | --- |
| `tools/summary.json` / `attempt-1792123296910101620` | `prefill-stage-tools` 177 passed / 100.0 s；固定选择、原样本重算、FD 分类、缺失/损坏/身份冲突及成功/失败传播 |
| `preflight/summary.json` / `attempt-1792122011399393820` | 五布局 native profile-check 各 1 passed；BSND/TND/混合/BSND PA/TND PA 分别 206.5/201.2/202.2/211.9/233.5 s |
| `acceptance/summary.json` / `attempt-1792124216738362770` | 日常 E2E 477 passed / 73.7 s；TND 103 / 32.0 s、混合 118 / 36.1 s、BSND PA 120 / 36.9 s、TND PA 132 / 39.1 s 均 passed |
| 同一 acceptance 的本轮继承 | ordinary 516 项，原实跑 `attempt-1792123167814029130` / 29.1 s；BSND 143 项，原实跑 `attempt-1792124101186119210` / 31.2 s。由同目录 `--resume` 校验身份和原始证据后继承 |
| `performance/summary.json` / `attempt-1792124480297511050` | 五布局 profile 各 1 passed，60 配置全部实测；各组耗时及阶段结果见下表 |

最终正确性/回归合计 **1,609 项**，其中最终 attempt 实跑 950 项、继承本轮已实跑 E2E 659 项；未继承 M13b 的通过结果、历史单元或性能。五布局固定源专项共 60 个测试、120 项 O/LSE 检查，全部 `math=passed, source=equal`，无源精度限制项；FA3 空行/长度/真实 partial 另由公开专项检查。PA 源片段使用独立 CPU gather 的逻辑 K/V，迁移侧实际访问 PA，原生 PA 分派另有固定包证据。

本轮有效正确性组的 reference 共 1,204 次 cache hit，预检 10 次、正式性能 60 次，均无 miss/write；相同数学结果复用，待测实现实际执行。没有改变算法或 reference 策略，不重复额外最长成本检查，专项内已有成本用例仍完整运行。

开发期工具 synthetic varlen 夹具缺行，以及 BSND 的旧 TND 拒绝和 PA real/fake 错误文案断言均出现过失败；已修复并按最终实现重跑受影响组。早期失败 attempt 和日志保留，不计通过。所有本步必跑组已通过，无遗留失败或未运行项。未新增设备协议或数值路径，本轮未累计重跑历史模块单元/Trap；完整图执行、softcap/非默认调度和 910C 实机仍未验收。

## 阶段公开性能

`stage-performance.json` 从绑定的完整 5×20 wall、5×10 kernel 原样本重算，校验固定输入、双 dtype、native 包/分派、共享源码、运行环境、单 CPU 亲和性及原始证据。每布局固定 12 配置，合计 60；30 个 automatic 配置参与平均，30 个 single/explicit 原生 FD 对照因分片策略不等价单列。完整单例/分段、metadata 生成、初始化/转换及原始样本保留在各组 `public-profile/`，不按耗时筛选。

**阶段平均 FA3/CANN = 0.790252，达到配置与入口/R 等权平均 ≤1 的门槛。** 适用配置退化 0/30，分段退化 34；kernel-only 为 60/60 退化（含单列项），不将公开 E2E 达标写成 kernel 提速。固定 CPU 0、卡 6，R=1/4、预热及计时协议不变；必要 native packing 在逐次计时内。

| 布局 | 平均 FA3/CANN | 适用 / 单列 | 分段退化 | profile 秒 |
| --- | --- | --- | --- | --- |
| BSND | 0.944973 | 6 / 6 | 14 | 259.2 |
| TND | 0.731534 | 6 / 6 | 0 | 242.3 |
| 混合 | 0.726421 | 6 / 6 | 3 | 253.1 |
| BSND PA | 0.772093 | 6 / 6 | 9 | 243.5 |
| TND PA | 0.776240 | 6 / 6 | 8 | 248.9 |

## 复现、环境与收尾

运行前按 [design 7.4](design.md#74-统一验收入口) dry-run/collect；原集合见 `collect-acceptance/`、修正后的 E2E/非分页专项见 `collect-corrected/`，工具与性能见 `collect-tools-final/`、`collect-performance/`。`--resume` 只继承同目录中身份及原始证据匹配的 E2E；性能与工具始终实跑。

```bash
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export MLA_NATIVE_ROOT=/tmp/mla-m13b/native
python3 -m tests.mla --suite prefill-stage-tools --output /tmp/mla-m14/tools
python3 -m tests.mla --suite prefill-bsnd-profile-check prefill-tnd-profile-check prefill-mixed-profile-check prefill-paged-bsnd-profile-check prefill-paged-tnd-profile-check --output /tmp/mla-m14/preflight
python3 -m tests.mla --suite e2e ordinary prefill-bsnd-public prefill-tnd-public prefill-mixed-public prefill-paged-bsnd-public prefill-paged-tnd-public --resume --output /tmp/mla-m14/acceptance
python3 -m tests.mla --suite prefill-bsnd-profile prefill-tnd-profile prefill-mixed-profile prefill-paged-bsnd-profile prefill-paged-tnd-profile --prefill-stage-performance --output /tmp/mla-m14/performance
python3 -m tests.mla.support.mla_prefill_stage_performance --evidence /tmp/mla-m14/performance/summary.json --output /tmp/mla-m14/stage-performance.json
```

接手 `takeover-evidence.json` 校验 M13b 15 组原始证据、生产产物与 native 包；本轮 runtime 与 M13b 的 Python/torch/torch_npu、芯片、扩展和加载库集合一致。环境为 Python 3.11.15、torch 2.7.1+cpu、torch_npu 2.7.1、CANN `/usr/local/Ascend/cann-9.0.0-beta.2`、CATLASS `769cd40a8716b28650b6bebb08db4834eea4462f`；卡 6 Ascend910B3。实际扩展为本仓 `flash_attn_npu_3/flash_attn_npu_3.cpython-311-aarch64-linux-gnu.so`，SHA256 `b3e440a12d210894df945ae07b5ba74fc2b7068a3bd3f31465acb059eebc3ab6`；生产构建经 manifest 校验复用，未重建。

生产 manifest/reference 根为 `/tmp/flash-attention-npu-mla-46928a793876/mla-validation-artifacts/`，native 包为 `/tmp/mla-m13b/native`。所有日志和产物在仓库外；`/tmp` 非持久存储，文件缺失后不能继承。

收尾 `evidence-final.json` 用统一驱动校验器复核 18 组原始证据、记录依赖、构建/实际加载库、native 包及阶段汇总身份，全部通过 / 6.1 s；`doc-checks.json` 的变更链接/锚点与表格检查通过。`git diff --check` 和 `env PYTHONPATH=/tmp/mla-m10c8a/quality ci/code_quality.sh --staged` 通过，质量日志为 `quality-closeout.log`。当前提交可用 `git log -1 -- docs/mla/status.md` 定位，历史开发证据从 Git 查询。
