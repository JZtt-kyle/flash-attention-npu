# MLA 当前状态

**M15f（TND Q + ND PA KV softcap）的完整公开接入与必要实机验收已完成。唯一下一开发步是 [M15g：调度选项](plan.md#唯一下一步-m15g-调度选项)，本轮不开始该步。** 本轮基线为 `feat/mla-cann-migration` 的 `e52b6dd`，接手工作树干净；本轮是专门文档轮后的第二个开发轮。

## 本轮交付与边界

- 两种表示（512+64、192/128）的 TND PA 正 softcap 完整接通 host/AICPU、launcher/Kernel、metadata、公开 kvcache/raw C++ 和 real/fake/compile；三 mask 与单/显式/自动 split 同步支持。四个公开函数的参数签名与基线一致。
- v44/v45/v46 沿用原分页/TND 字段及调度，只绑定 softcap 启用状态；不同正 cap 值可以复用。保留 allocation/used 原位置、独立 stride、只读 cache、全部 live 页检查及 Q-used 复用边界。实现复用既有数值组件，CANN/CATLASS 未修改，契约见 [design 5.20](design.md#520-tnd-pa-softcap-记录)。
- 非默认 `pack_gqa` / `sm_margin` 仍拒绝；完整 graph replay 未验收，生产平台仍只接受 Ascend910B。M15 的调度选项及后续里程碑见 plan。

## 最近一次验收证据

证据根 `/tmp/mla-m15f/`。接手通过统一驱动核对 M15e 的 21 个组及构建 manifest，记录在 `handoff-evidence.log`。共享实现改变后，本轮实际重跑累计 E2E，**继承 E2E 为 0**；不继承历史单元或性能。收尾用驱动已有的来源、依赖、实际用例、原始证据及构建产物校验复核下表 **20 个组全部有效**，记录在 `closing-evidence.log`；本步必跑组无失败、跳过或未运行。

| summary / attempt | 本次通过 |
| --- | --- |
| `cpu-final/summary.json` / `1792295248336556340` | 本模块 CPU 139 / 100.3 s，ASan/UBSan 139 / 329.7 s；该目录较早的工具结果不作为最终证据 |
| `tools-final/summary.json` / `1792296051973418320` | 工具 45 / 36.0 s |
| `device-accepted/summary.json` / `1792296052115799460` | 代表 48 / 37.2 s；PA prefill native 预检 1+1 / 203.1+203.3 s，正 cap 采样预检 4 / 132.7 s |
| 同一 `device-accepted` | 完整专项 222（含 32 个隔离 Trap）/ 541.0 s；cap=0 固定源 36 / 479.4 s |
| 同一 `device-accepted` | 累计小输入 E2E 711 / 146.6 s，普通 FA3 508 / 26.1 s；正 cap profile 4 / 121.9 s |
| `perf-prefill/summary.json` / `1792298056421185320` | PA BSND、TND 两组 / 247.5、243.9 s |
| `perf-compressed/summary.json` / `1792298581085804690` | PA BSND、TND 单 split / 342.8、347.5 s；显式 split / 232.0、237.1 s；自动 split / 233.0、235.3 s |

生产构建首次实际执行 544.1 s；最终构建 manifest 复核 1.2 s，实际加载核验 21.1 s。数学缓存累计命中 1465 次、未命中并写入 120 次，待测实现均实际执行。最终公开用例清单见 `collect-accepted/summary.json`；性能 dry-run/collect 记录见 `prefill-collect/`、`compressed-collect/`，未使用诊断截断或扩大穷举。

## 公开性能

采样前固定 **70 配置（含 dtype）**，R=1/4、预热、重复次数及计时边界保持不变。全部单例、分段和退化结果保留在原始报告中：

- 正 cap 10 配置全部完成，报告在 `device-accepted/attempt-1792296052115799460/softcap-paged-tnd-profile/*/profile.json`。配置内等权平均后，正 cap/零 cap 开销参照为 **1.066678**，7/10 高于 1；它没有同义 CANN softcap 基线，不是验收比值。
- PA prefill 24 配置全部采样，12 个适用配置平均公开比 **0.771041 ≤1**，配置级退化 0、分段退化 15；12 个原生 FD 配置单列。kernel 单列退化 24，不能据公开均值声称所有单例或 kernel 提速。汇总为 `prefill-performance.json`。
- 压缩 PA 六组共 36 配置全部采样，24 个适用配置平均公开比 **0.587658 ≤1**，配置级退化 4、分段退化 24；12 个原生 FD 配置单列。汇总为 `compressed-performance.json`。

## 复现命令

在设备可见环境运行；普通沙箱不映射 NPU。下列使用统一入口，`--dry-run`、`--collect-only` 已在正式执行前核对。恢复或重跑前先检查进程与 summary，避免重复占用卡 7。

```bash
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export MLA_NATIVE_ROOT=/tmp/mla-m13b/native
python3 -m tests.mla --suite unit-softcap_paged_tnd softcap-paged-tnd-sanitize --output /tmp/mla-m15f/cpu-final
python3 -m tests.mla --suite softcap-paged-tnd-tools --output /tmp/mla-m15f/tools-final
python3 -m tests.mla --device 7 --suite softcap-paged-tnd-check prefill-paged-bsnd-profile-check prefill-paged-tnd-profile-check softcap-paged-tnd-profile-check detail-softcap_paged_tnd softcap-paged-tnd-source e2e ordinary softcap-paged-tnd-profile --output /tmp/mla-m15f/device-accepted
python3 -m tests.mla --device 7 --prefill-stage-performance --suite prefill-paged-bsnd-profile prefill-paged-tnd-profile --output /tmp/mla-m15f/perf-prefill
python3 -m tests.mla --device 7 --split-stage-performance --suite window-paged-profile window-paged-tnd-profile split-paged-bsnd-profile split-paged-tnd-profile auto-split-paged-bsnd-profile auto-split-paged-tnd-profile --output /tmp/mla-m15f/perf-compressed
python3 -m tests.mla.support.mla_prefill_stage_performance --evidence /tmp/mla-m15f/perf-prefill/summary.json --suite prefill-paged-bsnd-profile prefill-paged-tnd-profile --output /tmp/mla-m15f/prefill-performance.json
python3 -m tests.mla.support.mla_split_stage_performance --evidence /tmp/mla-m15f/perf-compressed/summary.json --suite window-paged-profile window-paged-tnd-profile split-paged-bsnd-profile split-paged-tnd-profile auto-split-paged-bsnd-profile auto-split-paged-tnd-profile --output /tmp/mla-m15f/compressed-performance.json
```

## 环境、预检修正与收尾

Python 3.11.15、torch 2.7.1+cpu、torch_npu 2.7.1、CANN `/usr/local/Ascend/cann-9.0.0-beta.2`；卡 7 Ascend910B3。CATLASS 为 `769cd40a8716b28650b6bebb08db4834eea4462f`，固定 CANN 来源为 `ca18aee610729ca5270a44e2d84c751bc687fbf0`。实际扩展为本仓 `flash_attn_npu_3/flash_attn_npu_3.cpython-311-aarch64-linux-gnu.so`，SHA256 `e678dc4108fcadfd5302701de25fb8b3440bbea1785f477f25522834fbbdc05d`，由各 summary 的 `runtime_identity` 绑定。

默认工作根 `/tmp/flash-attention-npu-mla-46928a793876/` 保存构建 manifest、探针和数学缓存，native 包在 `/tmp/mla-m13b/native`；`build/` 只保留无头循环日志和状态。所有 `/tmp` 证据缺失或内容变化后不得继承。

早期预检修正了 profiler 损坏 ABI 夹具，以及 CPU prefill 夹具的 TND PA `totalK`/tiling 模式；普通沙箱的接口导入失败在设备可见环境重收通过。首次代表测试实际 48 passed，但测试夹具在执行期间调整，驱动按来源变化标为失败，已按最终来源重跑。原始失败保留在 `tools-preflight/`、`cpu-debug/` 和 `device-final/`，不作为通过证据。

文档链接、四个公开签名、差异及适用仓库质量检查均通过，日志为 `docs-final.log`、`quality-final.log`。质量入口为 `PYTHONPATH=/tmp/mla-m10c8a/quality ci/code_quality.sh --staged`；只暂存本轮 42 个文件，提交后立即停止。本轮闭环提交可用 `git log -1 -- docs/mla/status.md` 定位。
