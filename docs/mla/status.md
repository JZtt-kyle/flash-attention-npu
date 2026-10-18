# MLA 当前状态

**M15g1（独立 GQA 调度坐标与资源预算）的实现与必要验收已完成。唯一下一步是 [M15g2：调度选项完整公开接入](plan.md#唯一下一步-m15g2-调度选项完整公开接入)，本轮不开始该步。** 接手基线为 `feat/mla-cann-migration` / `d40ce0f`，工作树原本干净；本轮是专门文档轮后的第三个开发轮。

## 本轮交付与公开边界

- 新增 `mla_gqa_schedule.{h,cpp}`：有效 pack 归一、C1V2 剩余资源、原生端点/显式片段、原 allocation 的任务与 O/LSE/FP32 accumulator 行地址查询。复用 `ParseUsedLengths`、固定 `SplitCore`、既有窗口/split 几何；覆盖两种表示、五布局逻辑长度、三 mask 和单/显式/自动 split。
- 拆分依据是独立 query head 到原 KV head、Q allocation 和写回坐标的新共享协议风险，见 [plan](plan.md#m15g1-的拆分依据与已交付边界)。这是可调用 CPU 组件，未新增序列化版本，实际生产链接未包含该组件；没有开放新的设备/公开行为。契约见 [design 6.2.13](design.md#6213-独立-gqa-调度坐标与资源预算)。
- M15f 已开放的两种表示、五布局、三 mask、全部 split 与非负 softcap 保持原入口。非默认 `pack_gqa` / `sm_margin` 仍明确拒绝；完整 graph replay 未验收，生产平台仍只接受 910B。CANN 与 CATLASS 未修改。

## 最近一次验收证据

证据根 `/tmp/mla-m15g/`。`handoff-evidence.log` 用统一驱动已有校验核对了接手时 M15f 的 20 个组及构建产物。新增源码及构建身份使既有结果不作继承，本轮 **继承 E2E 为 0**，历史单元/性能不继承。收尾通过驱动来源、依赖、实际用例、原始证据、构建产物及实际加载库校验，下面 **6 个组全部有效**，记录为 `closing-evidence.log`。

| summary / attempt | 本次通过 |
| --- | --- |
| `preflight/summary.json` / `1792301314105261720` | 工具 18 / 10.3 s；固定源与大地址代表 14 / 25.3 s |
| `cpu-final/summary.json` / `1792301377819296300` | `unit-gqa_schedule` 75 / 27.7 s；同目录失败的 sanitizer 不作为通过证据 |
| `sanitize-final/summary.json` / `1792301505857153590` | ASan/UBSan 75 / 86.2 s，泄漏检查正常结束 |
| `device-final/summary.json` / `1792301441260913360` | 累计 E2E 711 / 139.7 s；普通 FA3 508 / 26.1 s |

生产扩展与探针按驱动保守身份重建，耗时 551.0 s；实际加载核验 20.9 s。数学缓存命中 **987** 次（E2E 956、普通 FA3 31），未发生未命中或重算，待测实现均实际执行。

`cpu-dry-run.log`、`cpu-collect/`、`device-dry-run.log`、`device-collect/` 保存命令及集合；`selection-check.log` 确认单元/sanitizer 清单相同、累计 E2E/普通 FA3 与 M15f 清单逐项一致。未使用诊断截断或扩大穷举。本组件未接通生产，尚不注册新的公开日常 E2E；M15g2 随入口增加自己的能力清单。

## 失败、未运行与适用范围

普通沙箱中的 sanitizer 完成断言后，LeakSanitizer 因 ptrace 限制在退出时报错，整个组按失败保留在 `cpu-final/`；未关闭检测，随后在非沙箱环境重跑同一完整集合通过。当前必跑组没有未解决失败、跳过或未运行项。

本步不新增设备数值算法或公开调用开销，因此未运行新的设备组件精度、公开调度选项专项或性能采样，不以本次 CPU 证据宣称这些能力完成；M15g2 必须随完整入口实际验收。此前 M15f 公开性能的原始摘要由 Git 中的上一版 status 查询。

## 复现命令

先核对进程和 summary，避免重复构建或占用卡 7。sanitizer 需允许 ptrace 的环境，设备组需 NPU 可见；普通沙箱不满足这两项条件。正式执行前已核对 dry-run/collect-only、工具及廉价夹具，`native-preflight.log` 确认既有 native 包配方与内容有效。

```bash
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export MLA_NATIVE_ROOT=/tmp/mla-m13b/native
python3 -m tests.mla --suite gqa-schedule-tools gqa-schedule-check --output /tmp/mla-m15g/preflight
python3 -m tests.mla --suite unit-gqa_schedule --output /tmp/mla-m15g/cpu-final
python3 -m tests.mla --suite gqa-schedule-sanitize --output /tmp/mla-m15g/sanitize-final
python3 -m tests.mla --device 7 --suite e2e ordinary --output /tmp/mla-m15g/device-final
```

## 环境与收尾

Python 3.11.15、torch 2.7.1+cpu、torch_npu 2.7.1、CANN `/usr/local/Ascend/cann-9.0.0-beta.2`；卡 7 Ascend910B3。CATLASS `769cd40a8716b28650b6bebb08db4834eea4462f`，固定 CANN `ca18aee610729ca5270a44e2d84c751bc687fbf0`，FA3 `d15f1531a460ba456f41b01a774f33ab2db8febf`。实际扩展为本仓 `flash_attn_npu_3/flash_attn_npu_3.cpython-311-aarch64-linux-gnu.so`，SHA256 `adc0482b1562ff2ab2447985d568833c8d8cedfd2aa83b3cc4a56b819dbfee80`，由 `device-final` 的 `runtime_identity` 绑定。

默认工作根 `/tmp/flash-attention-npu-mla-46928a793876/` 保存构建、探针与数学缓存，native 包为 `/tmp/mla-m13b/native`；`build/` 只有 `mla-codex-loop/`。这些外部证据缺失或改变后不得继承。

本次文档契约/链接、差异及适用质量检查记录在 `docs-final.log`、`quality-final.log`。质量入口为 `PYTHONPATH=/tmp/mla-m10c8a/quality ci/code_quality.sh --staged`；只暂存本轮 11 个文件，提交一次即停。当前闭环提交可用 `git log -1 -- docs/mla/status.md` 定位。
