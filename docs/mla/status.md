# MLA 当前状态

**M13b TND PA prefill 已完成。** 分支 `feat/mla-cann-migration`，接手基线 `9d47fcd`、工作树原本干净。唯一下一开发步是 [M14 prefill 组合收尾](plan.md#唯一下一步-m14)；本轮仅完成 M13b 的一次提交，提交后停止。

## 当前实现与边界

192/128 TND Q + ND PA KV 已接通 host/AICPU/Kernel/v31、公开 metadata/kvcache 及内部 real/fake/compile，支持 qv=None、独立 K/V、GQA、合法 P、Q allocation/used、完整 storage/stride、非单调/重复/共享页、三 mask、自动/1/显式 split 和真实 FP32 accumulator。复用既有通用数值、原生调度、PA Copy、统一初始化和按行归并；契约见 [design 5.15](design.md#515-192128-tnd-pa-prefill-记录)。

v31 复用 v29 allocation/used span，并以头 21 绑定 P。Q/O 按 allocation 起点定位，PA 仅移动当前 batch 的页表行，K/V 物理池地址不移动；每次消费检查全部 live 页，包括 Q allocation 或 Q-used 为零的样本。公开 Q-used 边界不变，内部 mixed 算子支持带 Q-used 的显式 metadata 复用。softcap、非默认调度及完整 graph replay 仍未开放，910C 未实机验证。

## 本轮证据

证据根 `/tmp/mla-m13b/`；CPU 组为 `cpu-verified/summary.json`（`attempt-1792116210662129410`），新增 native key 工具检查为 `native-tools/summary.json`（`attempt-1792116497426704640`）。正确性与 native 夹具为 `acceptance/summary.json`（`attempt-1792117094308361490`），其余公开专项、五布局性能和 Trap 为 `performance/summary.json`（`attempt-1792117588283847380`）。集合由 `collect-cpu`、`collect-device`、`collect-public-final`、`collect-tools-final`、`collect-performance` 记录，命令见 `*-dry-run.json`；无截断或随机采样。

接手已用统一驱动校验前轮 M13a 原始证据，见 `takeover-evidence.json`。本轮生产改变后不继承旧提交 E2E；仅 `--resume` 继承本轮同一冻结实现的 8 项代表 E2E，来自 `acceptance/attempt-1792116800721003950`，原实跑 27.1 s。其余必要组实际执行。最终 `evidence-review.json` 复用统一驱动的 `reusable`、源码依赖、构建/loaded 和 native 校验，15 组原始证据、当前依赖及实际用例清单均有效；无失败或未运行的必跑组。

| 必跑组 | 结果 | 秒 |
| --- | --- | --- |
| `prefill-paged-tnd-tools` | 47 passed | 9.9 |
| `unit-prefill_paged_tnd` | 126 passed | 23.3 |
| `prefill-paged-tnd-sanitize` | 126 passed | 73.3 |
| `prefill-paged-tnd-check` | 本轮 E2E 继承 8 | 0（原 27.1） |
| `prefill-paged-tnd-profile-check` | 1 passed | 209.1 |
| `prefill-paged-tnd-public` | 132 passed | 40.6 |
| `prefill-paged-bsnd-public` | 120 passed | 39.0 |
| `e2e` | 477 passed | 74.3 |
| `ordinary` | 516 passed | 33.2 |
| `prefill-paged-tnd-profile` | 1 passed | 255.7 |
| `prefill-paged-bsnd-profile` | 1 passed | 247.8 |
| `prefill-mixed-profile` | 1 passed | 245.4 |
| `prefill-tnd-profile` | 1 passed | 239.5 |
| `prefill-bsnd-profile` | 1 passed | 274.0 |
| `prefill-paged-tnd-trap` | 60 passed | 799.3 |

固定源三 mask × 单/多 split 的 24 项 O/LSE 对照逐元素一致，双数学和真实 FP32 partial 通过。迁移侧实际访问 PA，固定源数值片段接收独立 CPU gather 的逻辑 K/V；原生公开 PA 路径另由固定包验证，双 dtype 实测 template 3、TND query digit 3、ND PA KV digit 1，含 PA/FD keys。

新增日常清单 40 项，累计 477 项。最大成本配置 `[B,Sq,Hq,C,Hkv]=[2,257,8,4144,2]`、P=112、Q allocation `[257,129]`：首次边界用例 13.5/17.0 s、峰值约 2.45 GiB；最终公开组复用 reference 后为 2.98/2.93 s，进程峰值约 2.53 GiB，见对应 `cost-*.json`。最终 TND PA 公开/reference 命中 91 次、累计 E2E 646 次、普通 31 次、BSND PA 公开 77 次；四个旧布局性能各命中 12 次。新 TND PA profile-check 生成 2 个 reference，完整性能命中 6 次、生成 6 次；相同输入未重复计算 reference，待测实现仍实际执行。

### 公开性能

每组固定 12 配置（含双 dtype），6 个自动 split 配置参与平均，6 个原生 FD 单/显式 split 配置按既有规则单列。配置及入口/R 等权，门槛 ≤1；各组 `public-profile/` 保存全部单例、分段、wall/kernel 样本和 native 分派。

| 布局 | 平均耗时比 FA3/CANN | 配置退化 | 分段退化 | kernel-only 退化（含单列项） |
| --- | --- | --- | --- | --- |
| TND PA | 0.718409 | 0/6 | 4 | 12/12 |
| BSND PA | 0.775169 | 0/6 | 6 | 12/12 |
| 混合 | 0.715828 | 0/6 | 0 | 12/12 |
| TND | 0.734106 | 0/6 | 0 | 12/12 |
| BSND | 0.941425 | 0/6 | 14 | 12/12 |

| TND PA 分段 | R=1 | R=4 |
| --- | --- | --- |
| 显式复用 | 0.857731 | 0.537031 |
| 生成后调用 | 0.757352 | 0.586561 |
| 隐式生成 | 0.943694 | 0.628081 |

kernel-only 仍全部慢于原生，验收按约定的平均公开 E2E。采样固定 CPU 0、卡 6，预热、5×20 wall、5×10 kernel 与同步边界不变；native 所需 Q/K/V/页表 packing 位于逐次计时内。60 项双 dtype 隔离 Trap 均在设备事件同步阶段拒绝错误输入，实际扩展身份一致。

### 失败修正与恢复

开发期沙箱 sanitizer 的断言通过但 LeakSanitizer 退出失败；随后宿主 sanitizer 因运行期间驱动修改失效，冻结后 126 项完整重跑通过。首次生产编译成功，但准备期间验收工具修改使驱动判失效；后续通过构建 manifest 复核复用，没有重复编译。首次 native 配方颠倒 Q/KV key 字段，编译器拒绝发布；已按固定源 `TILING_KEY_IS` 注册修正为 `103/104000000010200003` 及 FD/dtype 变体，并增加廉价注册检查，完整包构建通过。

公开组先后因 raw C++ metadata 夹具遗漏必填默认参数、旧 TND PA 拒绝断言失败；已修复共享 helper 和契约断言，TND/BSND PA 公开组均完整重跑通过。补充集合的沙箱收集因 NPU 不可见失败，宿主重新收集通过。全部失败及失效记录保留，未计入通过。固定 CANN 源仓和 CATLASS 未修改；产物、数学缓存及证据均在仓库外，`build/` 仅保留无头循环日志。

```bash
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export MLA_NATIVE_ROOT=/tmp/mla-m13b/native
python3 -m tests.mla --suite prefill-paged-tnd-tools unit-prefill_paged_tnd prefill-paged-tnd-sanitize --output /tmp/mla-m13b/cpu-verified
python3 -m tests.mla --suite prefill-paged-tnd-check prefill-paged-tnd-public e2e ordinary prefill-paged-tnd-profile-check --resume --output /tmp/mla-m13b/acceptance
python3 -m tests.mla --suite prefill-paged-bsnd-public prefill-paged-tnd-profile prefill-paged-bsnd-profile prefill-mixed-profile prefill-tnd-profile prefill-bsnd-profile prefill-paged-tnd-trap --output /tmp/mla-m13b/performance
```

Python 3.11.15、torch 2.7.1+cpu、torch_npu 2.7.1、CANN `/usr/local/Ascend/cann-9.0.0-beta.2`；卡 6 Ascend910B3、20 AIC。CATLASS `769cd40a8716b28650b6bebb08db4834eea4462f`，固定 CANN `ca18aee610729ca5270a44e2d84c751bc687fbf0`。实际加载本仓 `flash_attn_npu_3/flash_attn_npu_3.cpython-311-aarch64-linux-gnu.so`，SHA256 `b3e440a12d210894df945ae07b5ba74fc2b7068a3bd3f31465acb059eebc3ab6`；构建 manifest 在 `/tmp/flash-attention-npu-mla-46928a793876/mla-validation-artifacts/prepare.json`，该目录也保存内容 reference 缓存。

质量检查 `env PYTHONPATH=/tmp/mla-m10c8a/quality ci/code_quality.sh --staged` 通过，日志 `quality-final.log`；`git diff --check` 和本次影响的 22 个文档链接核对通过，见 `doc-links.json`。最终状态文字更新后再次运行同一质量入口，日志 `quality-commit.log`。本轮仅提交一次并立即停止；当前闭环提交可用 `git log -1 -- docs/mla/status.md` 定位。
