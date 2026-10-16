# MLA 当前状态

**本轮为专门文档修正，开发仍完成到 M13b；唯一下一开发步是 [M14 prefill 组合收尾](plan.md#唯一下一步-m14)。** 分支 `feat/mla-cann-migration`，接手提交 `4033fe9`，工作区干净。本轮只修改 design/plan/status，提交后停止，不推进 M14。

## 当前能力与边界

512+64 压缩 MLA 和 192/128 prefill 的五布局（BSND、紧凑 TND、TND Q + BSND cache、BSND PA、TND PA）均已接通公开前向，支持 FP16/BF16、三 mask、自动/1/显式 split、合法 stride、长度与空值语义；多 split 返回真实 FP32 accumulator。192/128 使用独立 K/V、GQA 与 v27–v31 共用生产路径。入口/协议映射见 [design 2.1](design.md#21-入口与记录选择)。

公开 kvcache/metadata 没有 Q-used 参数；192/128 内部 TND Q 路径支持 Q-used 与显式 metadata，内部 BSND Q-used 仍只支持隐式记录。PA 每次检查全部 live 页，包括 Q allocation/used 为零的样本。softcap、非默认调度选项仍明确拒绝；完整 graph replay 尚未验收，910C 未实机验证。M13b 的各布局验收不等于 M14 阶段收尾已完成。

## 本轮文档核对

对照 Python/C++ real/fake 分派、v27–v31 生成器/消费者/launcher、setup 构建清单、日常清单及性能驱动，修正 Q-used 复用范围、prefill 共用构建说明及 FD 单列含义；合并重复契约与性能边界，压缩已完成步骤和开发流水。固定来源按 [design 第 1 节](design.md#1-范围与固定依据) 核对：读取 CANN 固定 Git 对象、FA3 本地固定副本（`flash_api.cpp` 哈希与 oracle 声明一致）及 CATLASS 固定提交，未改源仓或子模块。

离线核对结果在 `/tmp/mla-doc-review-20261016/`。`evidence-review.json` 使用统一驱动的 `reusable` / `matching_files` 与 native 包校验：下列 15 组原始证据及记录的源码文件、生产构建依赖/产物和 native 包内容均匹配，耗时 3.4 s。**这只是现有证据的文件校验，本轮未探测设备、加载扩展、重建或运行设备测试，也未运行 `--resume` 产生新的继承结果。** 单元与性能保留为上轮记录，不列为本轮通过；下一开发轮仍须核对环境、选择与实际加载身份后决定 E2E 是否可继承。

本地链接/锚点、表格和固定 FA3 链接的本地源码映射核对通过；12 条 list/dry-run 命令通过，未运行设备收集或测试，见 `doc-checks.json`。`git diff --check` 及 `env PYTHONPATH=/tmp/mla-m10c8a/quality ci/code_quality.sh --staged` 通过，日志 `quality.log`；最终文字更新后复查见 `quality-final.log`。质量检查仅包含本轮三份 Markdown，其他语言检查按文件类型跳过。

## 最近开发闭环：M13b

代码基线 `4033fe9`。证据根 `/tmp/mla-m13b/`，以下路径均相对此根；原始命令、用例清单和选择记录在各 attempt 的 `command.json`、`cases.json`、`selection.json`，完整历史说明可用 `git show 4033fe9:docs/mla/status.md` 查询。

| summary / attempt | 上轮结果与耗时 |
| --- | --- |
| `cpu-verified/summary.json` / `attempt-1792116210662129410` | `unit-prefill_paged_tnd` 126 passed / 23.3 s；`prefill-paged-tnd-sanitize` 126 passed / 73.3 s |
| `native-tools/summary.json` / `attempt-1792116497426704640` | `prefill-paged-tnd-tools` 47 passed / 9.9 s；取代 CPU attempt 中较早的 46 项工具结果 |
| `acceptance/summary.json` / `attempt-1792117094308361490` | TND PA 公开 132 passed / 40.6 s；日常 `e2e` 477 passed / 74.3 s；`ordinary` 516 passed / 33.2 s；native `prefill-paged-tnd-profile-check` 1 passed / 209.1 s |
| 同一 `acceptance/summary.json` 的 `prefill-paged-tnd-check` | M13b 同一冻结实现内继承 8 项 E2E；原实跑位于 `attempt-1792116800721003950`，27.1 s |
| `performance/summary.json` / `attempt-1792117588283847380` | BSND PA 公开 120 passed / 39.0 s；TND PA 隔离 Trap 60 passed / 799.3 s；五布局 profile 各 1 passed，耗时见下表 |

M13b 最终必跑组无失败或未运行项；未继承更早提交的 E2E。开发期 sanitizer 环境失败、依赖变化失效、native key 配方错误及公开夹具失败均已处理，按冻结实现重跑受影响组，旧失败不计通过；细节及日志保留在原证据目录和 Git。固定源三 mask × 单/多 split 的双 dtype 24 项 O/LSE 逐元素一致，双数学及真实 partial 检查通过。迁移侧实际访问 PA，固定源数值片段接受独立 CPU gather 的逻辑 K/V；原生 PA 另有固定包实际分派证据。

### 上轮公开性能

每布局固定 12 配置（含 dtype），6 个自动 split 配置参与平均；6 个 single/explicit 配置因原生 FD 策略不等价单列。各组 `public-profile/average.json` 及其原始样本保留全部配置/分段结果；五组均达到配置与入口/R 等权平均 ≤1 的门槛，但尚未形成 M14 阶段覆盖与汇总结论。

| 布局 | 平均 FA3/CANN | 配置退化 | 分段退化 | profile 秒 |
| --- | --- | --- | --- | --- |
| BSND | 0.941425 | 0/6 | 14 | 274.0 |
| TND | 0.734106 | 0/6 | 0 | 239.5 |
| 混合 | 0.715828 | 0/6 | 0 | 245.4 |
| BSND PA | 0.775169 | 0/6 | 6 | 247.8 |
| TND PA | 0.718409 | 0/6 | 4 | 255.7 |

每布局 kernel-only 均为 12/12 退化（含单列项），不把公开平均达标写成 kernel 提速。上轮采样固定 CPU 0、卡 6，预热及 5×20 wall、5×10 kernel 协议不变；必要 native packing 在逐次计时内。最长成本与 reference 缓存统计保留在原 `cost-*.json`、`golden.tsv`，本轮未重跑。

### 复现与环境记录

以下为 M13b 的组级复现入口，本轮仅核对命令；运行前按 [design 7.4](design.md#74-统一验收入口) dry-run/collect 并检查环境。`--resume` 只复用同目录中身份匹配的 E2E，性能与单元仍实跑。完整历史命令以 summary 及 attempt 为准。

```bash
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export MLA_NATIVE_ROOT=/tmp/mla-m13b/native
python3 -m tests.mla --suite unit-prefill_paged_tnd prefill-paged-tnd-sanitize --output /tmp/mla-m13b/cpu-verified
python3 -m tests.mla --suite prefill-paged-tnd-tools --output /tmp/mla-m13b/native-tools
python3 -m tests.mla --suite prefill-paged-tnd-check prefill-paged-tnd-public e2e ordinary prefill-paged-tnd-profile-check --resume --output /tmp/mla-m13b/acceptance
python3 -m tests.mla --suite prefill-paged-bsnd-public prefill-paged-tnd-profile prefill-paged-bsnd-profile prefill-mixed-profile prefill-tnd-profile prefill-bsnd-profile prefill-paged-tnd-trap --output /tmp/mla-m13b/performance
```

上轮实测环境：Python 3.11.15、torch 2.7.1+cpu、torch_npu 2.7.1、CANN `/usr/local/Ascend/cann-9.0.0-beta.2`；卡 6 Ascend910B3、20 AIC。本轮未重新探测。实际加载扩展记录为本仓 `flash_attn_npu_3/flash_attn_npu_3.cpython-311-aarch64-linux-gnu.so`，SHA256 `b3e440a12d210894df945ae07b5ba74fc2b7068a3bd3f31465acb059eebc3ab6`；生产 manifest 在 `/tmp/flash-attention-npu-mla-46928a793876/mla-validation-artifacts/prepare.json`，reference 缓存同属该工作目录。所有证据位于仓库外；`/tmp` 非持久存储，文件被清理后不能继续继承。

本轮文档提交可用 `git log -1 -- docs/mla/status.md` 定位；开发历史从 Git 查询，不在本页累计。
