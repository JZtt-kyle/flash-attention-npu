# MLA 当前状态

**M12b 混合 prefill 已闭环，M12 非 PA 布局完成。** 分支 `feat/mla-cann-migration`，接手基线 `2185af9` 时工作树干净。本轮交付一个完整可调用行为；唯一下一步为 [M13a：BSND PA prefill](plan.md#唯一下一步-m13a)，本轮提交后停止。

## 当前实现与边界

192/128 TND Q + BSND cache 已接通 host/AICPU/Kernel/v29/公开 real/fake/compile，支持 qv=None、独立 K/V、GQA、Q allocation、cache used、完整 storage/stride、三 mask 及自动/1/显式 split，返回 O/LSE 和真实 FP32 accumulator。复用 M11/M12a 数值与调度；Q/O/归并沿用紧凑 allocation 坐标，K/V 各自按真实 BSND batch stride 定位。接口、版本与空值规则见 [design 5.13](design.md#513-192128-混合-prefill-记录)。

公开 cache 与 metadata 沿用 FA3 签名，内部算子另支持 Q-used。v29 复用 v28 的 span/partition 格式，以独立布局身份拒绝跨版本复用；AICPU 从 Q 前缀获得总 token 数，消费者绑定实际 Q/cache 几何与当前长度。克隆、连续 offset view、内容/scale/合法 stride 变化可复用。

PA prefill、softcap、非默认调度与完整 graph replay 仍未开放，910C 未实机验证。原有 512+64 五布局及 192/128 BSND/TND 保持支持；本轮共享源码改变，累计 E2E 已实际运行，继承旧 E2E 为 0。

## 最近闭环证据

证据根 `/tmp/mla-m12b/`。最终固定源码的 16 组必要检查全部实际通过，无失败、跳过或未运行，汇总为 `acceptance-final/summary.json`，尝试为 `attempt-1792105685818507750`。`collect-final/summary.json` 核对完整集合，`evidence-review.json` 使用统一驱动既有校验确认用例、原始哈希、当前依赖、生产构建、实际加载、固定源 probe 与 native 包有效。接手复核见 `handoff-review.json`；继承旧 E2E 为 0。BSND/TND 单元是本轮共享 producer/consumer 的直接影响检查。

| 必跑组 | 本次结果 | 秒 |
| --- | --- | --- |
| `prefill-mixed-tools` | 56 passed | 10.2 |
| `unit-prefill_mixed` | 110 passed | 22.1 |
| `prefill-mixed-sanitize` | 110 passed | 71.0 |
| `unit-prefill_tnd` | 113 passed | 22.1 |
| `prefill-tnd-sanitize` | 113 passed | 70.5 |
| `unit-prefill_bsnd` | 82 passed | 21.9 |
| `prefill-bsnd-sanitize` | 82 passed | 70.4 |
| `prefill-mixed-check` | 8 passed | 23.0 |
| `prefill-mixed-public` | 118 passed | 36.4 |
| `prefill-mixed-profile-check` | 1 passed | 206.1 |
| `e2e` | 403 passed | 68.3 |
| `ordinary` | 516 passed | 26.1 |
| `prefill-mixed-profile` | 1 passed | 256.3 |
| `prefill-tnd-profile` | 1 passed | 242.4 |
| `prefill-bsnd-profile` | 1 passed | 263.4 |
| `prefill-mixed-trap` | 48 passed | 652.1 |

固定源三 mask × 单/多 split 的 24 项 O/LSE 对照逐元素一致，双数学和真实 FP32 partial 通过。最终 check/public 的 reference 命中 14/160 次，无重算；待测实现实际执行。最大成本输入 `[B,maxQ,Hq,maxK,Hkv]=[2,257,8,4097,2]`，双 dtype 缓存命中后各约 0.70 s，进程峰值约 2 GiB，见公开组的 `cost-*.json`。新日常清单 33 项，累计集合 403 项。

### 公开性能

三组各固定 12 配置（含双 dtype），其中 6 个自动 split 配置参与平均，另外 6 个原生 FD 单/显式 split 配置按既有边界单列。配置及入口/R 等权，门槛 ≤1；全部单例、分段、wall/kernel 样本、native 分派和采样环境保存在各组 `public-profile/`。

| 布局 | 平均耗时比 FA3/CANN | 配置退化 | 分段退化 | kernel-only 退化（含单列项） |
| --- | --- | --- | --- | --- |
| 混合 | 0.722767 | 0/6 | 3 | 12/12 |
| TND 受影响路径 | 0.746656 | 0/6 | 0 | 12/12 |
| BSND 受影响路径 | 0.957891 | 0/6 | 14 | 12/12 |

| 分段均比 | 混合 R=1 / R=4 | BSND R=1 / R=4 |
| --- | --- | --- |
| 显式复用 | 0.913307 / 0.511874 | 0.988472 / 0.735452 |
| 生成后调用 | 0.759579 / 0.557634 | 0.962632 / 0.798898 |
| 隐式生成 | 0.994772 / 0.599435 | 1.253239 / 0.838283 |
| func | 不适用 | 1.279030 / 0.916650 |

TND varlen 的 R=1/4 为 0.762468/0.730844。kernel-only 全部慢于原生，原始样本完整保留；验收按约定的平均公开 E2E。采样固定 CPU 0、卡 6，预热与 5×20 wall、5×10 kernel 次数和同步边界不变；native 混合 KV gather 位于逐次计时内。

最终 profile-check 命中 reference 2 次，混合性能命中 7 次并首次计算 5 项，TND/BSND 性能各命中 12 次，累计 E2E/ordinary 命中 558/31 次。没有为自检或重跑重复计算相同 reference。

### 修正与复现

`cpu` 的窄化编译错误、`public` 的 metadata adapter 缺少内部默认参数已修正。`cpu-fixed` 在 sandbox 下 LeakSanitizer 因进程环境受限退出，改在主机环境实跑并通过，未把异常退出计为通过。新增物理 Sk=0 边界及修正接口/用例说明后，旧源码身份的结果不再用于最终闭环；`acceptance` 在性能采样期间主动停止，最终全部检查在 `acceptance-final` 重跑，缓存 reference 仍复用。一次 collect 期间说明变化被驱动拒绝，修正后已完整收集。

最终 dry-run/collect 见 `final-dry-run.log`、`collect-final/summary.json`；本轮未使用截断或随机采样。文档与相关契约、链接、表格已核对，结果见 `doc-links.json`；实现预检与完整 staged 质量检查见 `quality-code-final.log`、`quality-final.log`。所有产物及原始失败/中断记录均留在仓库外。

```bash
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export MLA_NATIVE_ROOT=/tmp/mla-m12a/native
python3 -m tests.mla --suite \
  prefill-mixed-tools unit-prefill_mixed prefill-mixed-sanitize \
  unit-prefill_tnd prefill-tnd-sanitize unit-prefill_bsnd prefill-bsnd-sanitize \
  prefill-mixed-check prefill-mixed-public prefill-mixed-profile-check e2e ordinary \
  prefill-mixed-profile prefill-tnd-profile prefill-bsnd-profile prefill-mixed-trap \
  --device 6 --output /tmp/mla-m12b/acceptance-final
```

## 环境与产物

卡 6 为 Ascend910B3、20 AIC；Python 3.11.15、torch 2.7.1+cpu、torch_npu 2.7.1，CANN `/usr/local/Ascend/cann-9.0.0-beta.2`。CATLASS `769cd40a8716b28650b6bebb08db4834eea4462f`，固定 CANN 源 `ca18aee610729ca5270a44e2d84c751bc687fbf0`，FA3 `d15f1531a460ba456f41b01a774f33ab2db8febf`；CANN 源只读，CATLASS 未修改。

实际扩展为仓库 `flash_attn_npu_3/flash_attn_npu_3.cpython-311-aarch64-linux-gnu.so`，SHA256 `799bd5d56298bd1865202f972e5c198754c250d064acbf1f4dfd4aa273fdc788`；最终 runtime 位于上述 attempt 的 `runtime.json`。独立准备 561.7 s 后，统一驱动因 visible-device 环境身份不同完整重建 545.6 s；最终准备校验复用 1.4 s。生产 manifest 在工作根 `/tmp/flash-attention-npu-mla-46928a793876/mla-validation-artifacts/prepare.json`；构建/reference 在该 artifacts 目录，数值 probe 在工作根 `prefill-probes/`，`build/` 只保留无头日志/状态。

native 包复用 `/tmp/mla-m12a/native/opp/vendors/mla_ca18aee_transformer`，已含通用 TND template 3/4 keys，混合预检查双 dtype 实测 template 3。这些 `/tmp` 原始证据被清理后不得继续继承。当前闭环提交由 `git log -1 -- docs/mla/status.md` 定位。
