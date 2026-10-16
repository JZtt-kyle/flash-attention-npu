# MLA 当前状态

**M13a BSND PA prefill 已完成。** 分支 `feat/mla-cann-migration`，接手基线 `6329ef8` 时工作树干净。唯一下一步是 [M13b TND PA prefill](plan.md#唯一下一步-m13b)，本轮提交后停止。

## 当前实现与边界

192/128 BSND Q + ND PA KV 已接通 host/AICPU/Kernel/v30/公开 real/fake/compile，支持 qv=None、独立 K/V、GQA、合法 P、完整 storage/stride、非单调/重复/共享页、三 mask、自动/1/显式 split 和真实 FP32 accumulator。复用既有页段 Copy、通用 prefill 数值与调度；契约见 [design 5.14](design.md#514-192128-bsnd-pa-prefill-记录)。

v30 在既有有效长度和原生 partition 记录上增加 P 身份；逻辑容量为页表宽度乘 P。每次消费检查当前有效页，即使 Q-used=0；页表和独立 K/V 按当前流保活。metadata 可跨克隆、连续 offset view、内容/scale、合法 stride 和有效页映射复用；长度、P 或调度几何变化须重生成。PA 复用 TND 的统一设备初始化和按独立行块分配向量核的归并，单行运算次序不变，全部空 KV 行 O/LSE 为 0/-inf。

192/128 TND PA、softcap、非默认调度及完整 graph replay 仍未开放，910C 未实机验证。既有 512+64 五布局和非 PA prefill 三布局保持支持。

## 本轮证据

证据根 `/tmp/mla-m13a/`。最终 CPU 组为 `cpu-optimized/summary.json`（`attempt-1792112660486344900`），设备组为 `optimized/summary.json`（`attempt-1792112805051335030`）。`collect-optimized/summary.json`、`optimized-dry-run.log` 核对集合；无截断或随机采样。

上一轮闭环 E2E 继承为 0；本轮仅由 `--resume` 继承同一优化源码/构建的 8 项代表 E2E，来自 `optimized/attempt-1792112313700097760`，原实跑 28.4 s，身份和原始证据均由驱动核验。其余本轮必要组实际执行。最终 `evidence-review.json` 复用统一驱动的 `reusable`、源码依赖、构建/loaded 和 native 校验，13 组原始证据、当前依赖及实际用例清单全部有效；无失败或未运行的必跑组。

| 必跑组 | 结果 | 秒 |
| --- | --- | --- |
| `prefill-paged-bsnd-tools` | 46 passed | 10.0 |
| `unit-prefill_paged_bsnd` | 130 passed | 23.6 |
| `prefill-paged-bsnd-sanitize` | 130 passed | 77.6 |
| `prefill-paged-bsnd-check` | 本轮 E2E 继承 8 | 0（原 28.4） |
| `prefill-paged-bsnd-profile-check` | 1 passed | 213.6 |
| `prefill-paged-bsnd-public` | 120 passed | 37.6 |
| `e2e` | 437 passed | 70.0 |
| `ordinary` | 516 passed | 26.1 |
| `prefill-paged-bsnd-profile` | 1 passed | 245.9 |
| `prefill-bsnd-profile` | 1 passed | 264.6 |
| `prefill-tnd-profile` | 1 passed | 245.4 |
| `prefill-mixed-profile` | 1 passed | 254.4 |
| `prefill-paged-bsnd-trap` | 40 passed | 553.7 |

固定源三 mask × 单/多 split 的 24 项 O/LSE 对照逐元素一致，双数学及真实 FP32 partial 通过。迁移侧实际读 PA，固定源数值片段接收独立 CPU 页表 gather 的逻辑 K/V；原生公开 PA 分派另由固定包验证，实测为 template 3，含 PA/BRC/FD keys。

新增日常清单 34 项，累计 437 项。最大成本配置 `[B,Sq,Hq,C,Hkv]=[2,257,8,4144,2]`、P=112、used=4097，双 dtype 各约 2.97/2.99 s，峰值约 2.83 GiB，见公开组 `cost-*.json`。最终公开/reference 命中 77 次、E2E 598 次、普通 31 次；profile-check 命中 2 次，四个完整性能组各命中 12 次。待测实现均实际执行，未重复计算相同 reference。

### 公开性能

每组固定 12 配置（含双 dtype），其中 6 个自动 split 配置参与平均，另 6 个原生 FD 单/显式 split 配置按既有边界单列。配置及入口/R 等权，门槛 ≤1；全部单例、分段、wall/kernel 样本和 native 分派保存在各组 `public-profile/`。

| 布局 | 平均耗时比 FA3/CANN | 配置退化 | 分段退化 | kernel-only 退化（含单列项） |
| --- | --- | --- | --- | --- |
| BSND PA | 0.778481 | 0/6 | 9 | 12/12 |
| 非 PA BSND | 0.948506 | 0/6 | 14 | 12/12 |
| TND | 0.744971 | 0/6 | 0 | 12/12 |
| 混合 | 0.728750 | 0/6 | 4 | 12/12 |

| PA 分段 | R=1 | R=4 |
| --- | --- | --- |
| 显式复用 | 1.009813 | 0.531630 |
| 生成后调用 | 0.807560 | 0.585231 |
| 隐式生成 | 1.113886 | 0.622766 |

kernel-only 仍全部慢于原生，原始结果完整保留；验收按约定的平均公开 E2E。采样固定 CPU 0、卡 6，预热、5×20 wall、5×10 kernel 与同步边界不变；native 必需的 Q/K/V/页表 packing 位于逐次计时内。

### 失败修正与恢复

首次完整 PA 平均 1.076908 未过门槛，证据为 `acceptance-final/attempt-1792110624392155150`；驱动当时停止，未把失败记作通过。优化仅为 PA 复用既有 TND 的设备统一初始化与按行并行归并，原输入与门槛未变。随后 profile-check 发现解析器仅允许 TND 初始化，修正为同时核对 PA，并新增双 dtype 的缺失/重复/顺序/dtype/非 PA 拒绝检查，保留完整初始化计时；最终性能均实际重跑。

开发期 native 配方缺少 BSND PA key、诊断数量夹具和旧能力拒绝断言均已修正。沙箱设备不可见的收集、源码变化导致失效的 sanitizer、等待集合收集时中断的代表检查均保留原始记录，未计通过。固定 CANN 源仓和 CATLASS 未修改。构建中间产物、数学缓存与验收原始数据在仓库外，`build/` 仅有无头循环日志。

```bash
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export MLA_NATIVE_ROOT=/tmp/mla-m13a/native
python3 -m tests.mla --suite prefill-paged-bsnd-tools unit-prefill_paged_bsnd prefill-paged-bsnd-sanitize --output /tmp/mla-m13a/cpu-optimized
python3 -m tests.mla --suite prefill-paged-bsnd-check prefill-paged-bsnd-profile-check prefill-paged-bsnd-public e2e ordinary prefill-paged-bsnd-profile prefill-bsnd-profile prefill-tnd-profile prefill-mixed-profile prefill-paged-bsnd-trap --resume --output /tmp/mla-m13a/optimized
```

Python 3.11.15、torch 2.7.1+cpu、torch_npu 2.7.1、CANN `/usr/local/Ascend/cann-9.0.0-beta.2`；卡 6 Ascend910B3、20 AIC。CATLASS `769cd40a8716b28650b6bebb08db4834eea4462f`，固定 CANN `ca18aee610729ca5270a44e2d84c751bc687fbf0`。实际加载本仓 `flash_attn_npu_3/flash_attn_npu_3.cpython-311-aarch64-linux-gnu.so`，SHA256 `2875a1bba07f77ece050eb359ce3c8f3196920a31c3dd16cda2497ef12d4b72c`；构建 manifest 在 `/tmp/flash-attention-npu-mla-46928a793876/mla-validation-artifacts/prepare.json`。质量检查 `env PYTHONPATH=/tmp/mla-m10c8a/quality ci/code_quality.sh --staged` 通过，日志 `quality-final.log`；`git diff --check` 及本次改动的 23 个文档链接检查通过，见 `doc-links.json`。本轮仅提交一次，随后停止；当前闭环提交可用 `git log -1 -- docs/mla/status.md` 定位。
