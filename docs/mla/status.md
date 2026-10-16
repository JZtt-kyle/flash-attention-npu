# MLA 当前状态

**M15a softcap 独立数值组件已闭环；唯一下一步是 [M15b BSND softcap 完整公开接入](plan.md#唯一下一步-m15b)。** 本轮在 `feat/mla-cann-migration` 从 `f7105ef` 的干净工作树开始；当前闭环提交可用 `git log -1 -- docs/mla/status.md` 定位。

## 当前能力与边界

公开能力仍为 M14：512+64 与 192/128 的五布局、三 mask、自动/1/显式 split、真实 FP32 accumulator。公开 softcap、非默认调度仍明确拒绝，graph replay 留待 M16，910C 未实机验证。公开契约见 [design 2.2](design.md#22-后端选项与返回值)，M14 闭环见 `git show f7105ef:docs/mla/status.md`。

本步组件在完整 scaled score 后、mask 前计算 softcap，复用原 32 KiB scratch；补近零多项式、非零 cap 的最终取偶舍入，以及 scale=0 均匀权重的 aMLA 精确更新，见 [design 6.5.3](design.md#653-softcap-独立数值组件)。默认零 cap 保留固定源算术。沿用现有 Kernel / Block、partial/FD、probe 与 reference，没有修改 host/AICPU、序列化 metadata、公开 real/fake、页地址、发射策略、构建参数或 CATLASS。组件通过不代表公开 softcap 已接通；新公开 E2E 随 M15b 注册。

## 最近闭环证据

证据根 `/tmp/mla-m15a/`。最终统一驱动为 `final/summary.json`，attempt `1792132016265557880`；保存实际用例清单、命令、源码/选择身份、构建依赖、加载库和原始证据。收尾 `closeout-evidence.json` 调用既有 `reusable`、`matching_files`、`validate_causal_builds` 核验全部通过组、生产产物和四套组件 probe，均有效。

提交收尾的 `git diff --check`、`ci/code_quality.sh --staged` 通过，日志为 `quality-staged.log`；三份文档本步影响范围的 13 个链接及相关表格已核对。本轮构建和验收进程均已退出，只暂存本步 28 个文件。

| 本次实跑组 | 结果 | 驱动耗时 |
| --- | --- | --- |
| `softcap-tools` | 52 passed；独立标量 reference、ABI、缓存失效、归属及成功/失败/中断传播 | 22.4 s |
| `unit-softcap` | 94 passed；74 数值与 20 非法/源拒绝检查 | 62.8 s |
| `e2e` | 累计小输入 477 passed | 77.1 s |
| `ordinary` | 普通 FA3 516 passed | 34.8 s |
| `softcap-profile` | 8 passed；双表示/双 dtype 的 partial 与 FD 分段采样 | 110.5 s |

代表检查 `uniform-check/summary.json`（attempt `1792130480097197350`）46 passed / 329.3 s，包含受影响 probe 重建；补 FD 计时后的廉价夹具 `merge-fixtures/summary.json`（attempt `1792131885806493120`）8 passed / 42.7 s。首次完整闭环 `acceptance/attempt-1792130909212687610/` 已通过数值/E2E/普通回归；因最终 cast 涉及 FD，补分段计时后在最终源码身份下重跑完整集合，未减少或替换原用例。

58 个非零 cap 数值用例全部满足既有双数学阈值。其余 16 个 cap=0 数值用例实跑固定源并逐元素一致；其中两种表示的 FP16 `rounding-bypass` 有 3 项最终 O 比较属于源中点舍入精度限制，单列在各 `accuracy/*.json`。总体 1811 项数学比较通过、3 项源精度限制；**非零 cap 没有源豁免**。输入、双 reference、raw partial、真实 accumulator、最终 O/LSE、重复执行与 guard 均保留。

接手 `takeover-evidence.json` 与 `takeover-runtime.json` 已核对 M14 的 18 组原始证据和环境。本步共享数值头变化，继承 E2E 为零；未累计运行历史模块单元。最终 softcap reference 为 98 hit / 0 miss，E2E 为 552 hit / 94 miss，普通回归为 1 hit / 30 miss；数学缓存与测试执行分离。初始成本边界用例耗时：压缩 FP16/BF16 56.2/68.8 s，prefill 13.2/21.5 s，峰值 RSS 2,432,764 KiB；之后相同输入复用缓存。用例均已 dry-run/collect，不使用 `--max-cases`。

## 组件性能

固定 16 个输入配置（含表示/dtype），分别测预分配 partial attention 和单 M 块 canonical FD；每项预热 5 次，wall 5×20、kernel 5×10，CPU 固定核 0、卡 6 隔离。每个分段中 cap=0/15 各 8 项，所有单例、分轮 wall、kernel/原始 CSV 对应和分段结果均在最终 `softcap-profile/{compressed,prefill}/probes/{fp16,bf16}/{profile,merge-profile}/`。

| 分段 / cap | wall 平均比 | kernel 平均比 | wall / kernel 比值 >1 数量 |
| --- | --- | --- | --- |
| partial / 0 | 0.9963 | 1.0036 | 3/8、3/8 |
| partial / 15 | 1.0115 | 1.1353 | 5/8、6/8 |
| canonical FD / 0 | 1.0046 | 1.8671 | 4/8、8/8 |
| canonical FD / 15 | 1.0315 | 1.7783 | 8/8、8/8 |

比值为迁移/CANN。cap=15 的 partial 对照是无 cap CANN，仅作额外开销参照；FD 两侧消费相同 capped partial，仅比较归并与 cast。FD 原始行适配及适用的多片 accumulator 写回另存 5×20 wall，各配置中位数 145.705–198.645 μs。上述均为组件测量，保留 FD kernel 退化，不计为同义 softcap 或公开 E2E 平均；本步没有新公开入口或 native 包配方，固定源 probe 来自只读 CANN `ca18aee610729ca5270a44e2d84c751bc687fbf0`。

## 失败修复与环境

早期 `acceptance/attempt-1792128166687925330/` 在 `zero-scale-used` 发现 FP16 最终中点舍入失败（71 passed / 1 failed），诊断 raw 在 `diagnostic/attempt-1792128963951812570/`。改取偶舍入后，`rounding/attempt-1792129408811817000/` 的跨块中点用例暴露 ±4.768e-7 的 aMLA epsilon 扰动；`cof-check/attempt-1792130221419363250/` 的宽泛修正又引起 BF16 饱和失败。最终只对非零 cap 且 scale=0 的均匀权重去掉 epsilon，原失败与新增正负中点、单双片、旁路及饱和用例全部重跑通过。早期 Min API 编译错误亦已修复；这些失败 attempt 保留，不作通过证据。

环境为 Python 3.11.15、torch 2.7.1+cpu、torch_npu 2.7.1、CANN `/usr/local/Ascend/cann-9.0.0-beta.2`、CATLASS `769cd40a8716b28650b6bebb08db4834eea4462f`，卡 6 Ascend910B3。最终实现已生产重建（569.8 s）；最终驱动校验复用耗时 1.3 s、实际加载检查 22.9 s。加载本仓 `flash_attn_npu_3/flash_attn_npu_3.cpython-311-aarch64-linux-gnu.so`，SHA256 `2bd4136ca8f46b5a872517adebbb2be80770a19fdfa9913591ded0d0ae2462a0`，不是接手旧产物。

```bash
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
python3 -m tests.mla --suite softcap-check softcap-profile-check --output /tmp/mla-m15a/recheck
python3 -m tests.mla --suite softcap-tools unit-softcap e2e ordinary softcap-profile --output /tmp/mla-m15a/final
```

默认 `MLA_WORK_DIR=/tmp/flash-attention-npu-mla-46928a793876/` 保存生产 manifest/reference、`softcap-compressed-probes/` 和 `prefill-probes/`；`build/` 只保留无头循环状态。下一步先核对驱动 summary、进程、实际加载和依赖；`--resume` 仅可继承身份及原始证据均有效的 E2E，单元/性能仍实跑。`/tmp` 非持久存储，清理或内容变化即失效。910C、公开非零 softcap、非默认调度和 graph replay 本步未运行、未开放。
