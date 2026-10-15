# MLA 当前状态

**M12a 紧凑 TND prefill 已闭环。** 分支 `feat/mla-cann-migration`，接手 `55844d0` 时工作树干净。本轮交付一个完整可调用行为；唯一下一步为 [M12b：TND Q + BSND cache](plan.md#唯一下一步-m12b)，本轮提交后停止。

## 当前实现与边界

192/128 TND Q/KV 的 qv=None、独立 K/V、GQA、allocation/used、完整 storage/stride、三 mask 和自动/1/显式 split 已接通 host/AICPU/Kernel/v28/公开 real/fake/compile。数值核复用 M11；任务按 allocation 起点重定位，128 维归并写真实 TND token/KV head 与 FP32 accumulator。TND 合并输出初始化发射、按独立行块并行归并；prefill FD start 校验复用单次分片遍历，保持全部边界检查。接口及记录见 [design 5.12](design.md#512-192128-tnd-prefill-记录)。

公开 varlen 每次生成记录；内部显式 v28 支持 Q/K-used、克隆及连续 offset view，并核对当前前缀与长度。混合/PA prefill、softcap、非默认调度与完整 graph replay 仍未开放；910C 未实机验证。原有 512+64 五布局和 192/128 BSND 保持支持，受影响路径本轮实际回归。

## 最近闭环证据

证据根 `/tmp/mla-m12a/`。以下 13 组为本轮实际完成的必要检查，无失败、跳过或未运行；**继承旧 E2E 为 0**。接手复核见 `handoff-review.json`；收尾 `evidence-review.json` 调用统一驱动既有校验，确认用例数量、原始证据哈希、当前依赖、实际加载、生产构建、固定源 probe 和 native 包均有效。BSND 单元属于本轮共享 prefill 校验的直接影响检查，不累计无关历史单元。

| 必跑组 | 本次结果 | 秒 | 汇总 |
| --- | --- | --- | --- |
| `prefill-tnd-tools` | 56 passed | 9.9 | `cpu-init-fixed/summary.json` |
| `unit-prefill_tnd` | 113 passed | 22.7 | `cpu-init-fixed/summary.json` |
| `prefill-tnd-sanitize` | 113 passed | 73.9 | `cpu-init-fixed/summary.json` |
| `unit-prefill_bsnd` | 82 passed | 22.4 | `cpu-init-fixed/summary.json` |
| `prefill-bsnd-sanitize` | 82 passed | 69.7 | `cpu-init-fixed/summary.json` |
| `prefill-tnd-check` | 8 passed | 23.3 | `public-init/summary.json` |
| `prefill-tnd-public` | 103 passed | 31.8 | `public-init/summary.json` |
| `prefill-tnd-profile-check` | 1 passed | 205.3 | `public-init/summary.json` |
| `prefill-tnd-profile` | 1 passed | 241.4 | `acceptance-init/summary.json` |
| `prefill-bsnd-profile` | 1 passed | 267.0 | `remaining/summary.json` |
| `prefill-tnd-trap` | 48 passed | 660.0 | `remaining/summary.json` |
| `e2e` | 370 passed | 63.9 | `remaining/summary.json` |
| `ordinary` | 516 passed | 27.5 | `remaining/summary.json` |

固定源三 mask × 单/多 split 的 24 项 O/LSE 对照全部逐元素一致，双数学及真实 FP32 partial 检查通过。公开 check/public/profile-check 的 reference 缓存命中 14/118/2 次，TND/BSND 性能正确性各命中 12 次，累计 E2E/ordinary 命中 480/31 次；均无重算，相同输入的待测实现仍实际执行。最大成本输入 `[B,maxQ,Hq,maxK,Hkv]=[2,257,8,4097,2]`，本轮双 dtype 缓存命中后分别 0.69/0.72 s，进程峰值约 2 GiB，见公开专项的 `cost-*.json`。

### 公开性能

两组各固定 12 配置（含双 dtype），6 个自动 split 配置参与平均；其余 6 个原生 FD 单/显式 split 配置按既有边界单列。配置及入口/R 等权，门槛 ≤1；全部单例、wall/kernel 样本、native 分派和采样环境保存在各组 `public-profile/`。

| 布局 | 平均耗时比 FA3/CANN | 配置退化 | 分段退化 | kernel-only 退化（含单列项） |
| --- | --- | --- | --- | --- |
| TND | 0.743493 | 0/6 | 0 | 12/12 |
| BSND 受影响路径 | 0.966986 | 0/6 | 13 | 12/12 |

TND 分段均比：R=1 为 0.753628，R=4 为 0.733358。BSND 分段均比：显式复用 R=1/4 为 1.002142/0.741329，生成后调用 R=1/4 为 0.948305/0.805980，隐式生成 R=1/4 为 1.271569/0.851231，func R=1/4 为 1.301350/0.928179。kernel-only 仍慢于原生，保留记录；验收按约定的平均公开 E2E。采样固定 CPU 0、卡 6，预热与 5×20 wall、5×10 kernel 次数及同步边界不变。

### 修正与复现

本轮前三次 TND 性能均比 1.133630、1.081344、1.052853 失败，证据分别为 `acceptance`、`acceptance-parallel`、`acceptance-validator`；定位并修正串行行组归并、FD start 重复扫描及四次独立输出初始化，固定输入与阈值未改变。早期 varlen 分派、同名测试包收集及廉价编译准备错误均已修复；失败构建、中断 CPU 尝试与旧扩展结果未计入通过表。

TND 性能完成后同步了 ordinary 中两处旧的 TND 拒绝预期，主动中断刚启动的 BSND 性能并在 `remaining` 完整重跑。`contract-case-review.json` 确认普通回归集合不变（12 节点仅重命名）；已完成组和构建的驱动依赖清单均不含该测试文件。日常 E2E 新增 TND 30 项，总数 370；最终 dry-run/collect 见 `cpu-init-collect`、`final-device-collect` 与 `remaining-collect`。最终文档及仓库质量检查见 `doc-links.json`、`quality-final.log`。

```bash
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export MLA_NATIVE_ROOT=/tmp/mla-m12a/native
python3 -m tests.mla --suite prefill-tnd-tools unit-prefill_tnd prefill-tnd-sanitize unit-prefill_bsnd prefill-bsnd-sanitize --output /tmp/mla-m12a/cpu-init-fixed
python3 -m tests.mla --suite prefill-tnd-check prefill-tnd-public prefill-tnd-profile-check --device 6 --output /tmp/mla-m12a/public-init
python3 -m tests.mla --suite prefill-tnd-profile --device 6 --output /tmp/mla-m12a/acceptance-init
python3 -m tests.mla --suite ordinary prefill-bsnd-profile prefill-tnd-trap e2e --device 6 --output /tmp/mla-m12a/remaining
```

## 环境与产物

卡 6 为 Ascend910B3、20 AIC；Python 3.11.15、torch 2.7.1+cpu、torch_npu 2.7.1，CANN `/usr/local/Ascend/cann-9.0.0-beta.2`。CATLASS `769cd40a8716b28650b6bebb08db4834eea4462f`，固定 CANN 源 `ca18aee610729ca5270a44e2d84c751bc687fbf0`，FA3 `d15f1531a460ba456f41b01a774f33ab2db8febf`；CANN 源只读，CATLASS 未修改。

当前扩展从仓库 `flash_attn_npu_3/flash_attn_npu_3.cpython-311-aarch64-linux-gnu.so` 加载，SHA256 `3a3b61db5f1902440729ab3ce5bacb03eef4a0e18e79a4f4886db688b6e3e759`；runtime 见 `public-init/attempt-1792100625142168510/runtime.json`，其余两次设备验收加载同一产物。完整构建 577.6 s 通过，见 `prepare-init-result.json`；prepare manifest 位于工作根 `mla-validation-artifacts/prepare.json`。工作根 `/tmp/flash-attention-npu-mla-46928a793876`，构建/reference 在 `mla-validation-artifacts/`，数值 probe 在 `prefill-probes/`；`build/` 只保留无头日志/状态。

固定 native 包 `/tmp/mla-m12a/native/opp/vendors/mla_ca18aee_transformer` 已按通用 TND 103/104 keys 重建，TND 实测双 dtype template 3；旧包保留。这些 `/tmp` 原始证据被清理后不得继续继承。当前闭环提交由 `git log -1 -- docs/mla/status.md` 定位。
