# MLA 当前状态

**M15d 及五轮开发后的专门文档修正轮已完成；唯一下一步是 [M15e：BSND PA softcap](plan.md#唯一下一步-m15e-bsnd-pa-softcap)。** 本轮从 `feat/mla-cann-migration` 的 `26d7a70` 干净工作树开始，只修改三份 Markdown 与公开 Python 入口的 docstring；不计开发轮数，提交后停止。

## 当前能力与边界

| 表示与布局 | 已接通行为 |
| --- | --- |
| 512+64、192/128，全部五布局 | cap=0；无 mask、右下 causal、window；单/显式/原生自动 split，真实 FP32 accumulator，公开 real/fake/compile |
| 两种表示，非 PA BSND / BSND、TND / TND、TND / BSND | 正 softcap；三 mask 与全部 split 模式，v32–v40；正 cap 改值可复用同一特征记录 |

公开 varlen 支持 allocation 前缀与 Q/K-used，每次内部生成 metadata；kvcache 与公开生成器没有 Q-used 参数。内部压缩 Q-used 仅隐式生成；内部 prefill 的 TND Q 可与显式记录组合，BSND Q 不可。布局、记录和返回契约以 [design 第 2、5 节](design.md#2-fa3-入口与后端边界) 为准。

PA 正 softcap、非默认 `pack_gqa` / `sm_margin` 仍拒绝；完整 graph replay 未验收。生产 `DensePlatform` 当前明确只接受 **Ascend910B**，910C 尚未开放，仍需完成实现与资源兼容；实机验收目标继续为 910B。fake/compile 不证明图重放或 910C 可用。迁移排除项见 [design 第 4 节](design.md#4-功能边界)。

## 本轮文档检查

- 修正压缩/prefill softcap 共同协议范围、stream 版本范围、stride 组件支持的维度、cap=0 分派限定和 910C 边界；合并 softcap 重复契约，补性能与固定 FA3 源码路径，压缩已完成步骤历史。
- 对照固定 CANN Git 对象、FA3 本地快照及实际 Python/C++/AICPU 分派；公开 docstring 删除过期 PA prefill 拒绝说明和重复版本表。去除 docstring 后的 AST 与原提交一致，四个公开签名未变且位置参数/默认值与固定 FA3 一致；455 个本地链接/锚点、44 张表、10 条具体验收命令检查通过，见 `document-checks.log`。
- 使用驱动既有 `suite_sources`、`reusable`、`validate_result`、`matching_files` 和 native 包校验，复核下表 21 组原始结果、用例选择及构建文件，耗时 29.3 s；从已有样本重算两份性能汇总，没有重新采样。记录在 `/tmp/mla-docs-20261018/`。

`git diff --check` 与 `PYTHONPATH=/tmp/mla-m10c8a/quality ci/code_quality.sh --staged` 通过，质量工具为 Python 3.11.15 / pre-commit 4.6.2，输出见上述目录的 `quality-final.log`。固定来源外链在线可达性未验证（抓取返回 cache miss）；七条链接对应的本地快照文件存在，`flash_api.cpp` 已通过既有固定 SHA256 校验。

本轮没有运行 CPU/设备测试、`--resume`、设备探测或扩展构建。最近实机结果仍属于 M15d；本轮只复核原始证据。docstring 改变了 Python 文件哈希，旧组的来源身份不再逐字匹配，**不能声称可直接 `--resume` 继承**；下轮由驱动按实际依赖判定并实跑受影响集合。构建源码、产物和原始证据的文件内容与历史 manifest 相符，不代表本轮重新验证了运行环境或实际加载。

## 最近开发闭环：M15d

提交 `26d7a70`，证据根 `/tmp/mla-m15d/`；以下均为当时实际执行结果。本轮原始证据校验见 `/tmp/mla-docs-20261018/evidence-review.log`，旧收尾记录为 `closeout-validation.log`。

| summary / attempt | 原始结果与耗时 |
| --- | --- |
| `cpu/summary.json` / `1792149568258100230` | 工具 29 passed / 29.1 s；CPU 86 passed / 93.5 s；ASan/UBSan 86 passed / 305.3 s |
| `preflight/summary.json` / `1792149771212267200` | 生产重建 580.0 s、实际加载校验通过；native 混合代表性能夹具 1 passed / 207.1 s；正 cap 性能夹具 4 passed / 158.7 s |
| `final/summary.json` / `1792150886789569710` | 代表 32 passed / 37.1 s；公开专项 202 passed / 442.5 s，含 24 个隔离 Trap；cap=0 固定源 32 passed / 280.6 s；累计 E2E 595 passed / 105.9 s；ordinary 508 passed / 25.5 s；正 cap 正式采样 4 passed / 118.2 s |
| `perf-prefill/summary.json` / `1792282393312721450` | 五布局各 12 配置实测，分组耗时 237.8–263.3 s |
| `perf-compressed/summary.json` / `1792283717420686910` | 指定五组各 6 配置实测，共 1126.7 s |

preflight 中的旧 `softcap-mixed-check` 在 Q 前缀原地重分配用例更新后失效，已由 final 的代表与专项重跑替代，不列入上述 21 组。最终必要集合无失败或未运行项，正 cap 双数学与 cap=0 固定源逐元素对照通过，无源精度例外。M15d 未继承历史测试通过证据，数学内容缓存复用时待测实现仍实际执行。595 个日常节点与当时的完整选择及当前注册范围一致。

## 最近公开性能

M15d 采样前固定合计 100 配置（含 dtype）；R=1/4、预热 5 次、wall 5×20、kernel 5×10。下表由既有汇总器从绑定的原始样本重算，输出 `/tmp/mla-docs-20261018/{prefill,compressed}-performance.json`，与原汇总一致。

| 集合 | 实测 / 同义适用 / 单列 | 配置平均耗时比 | 退化配置 |
| --- | --- | --- | --- |
| prefill 五布局 | 60 / 30 / 30 | 0.801354 | 0 / 30 |
| 压缩指定五组 | 30 / 22 / 8 | 0.751092 | 6 / 22 |
| 正 cap 混合 | 10 / 0 / 10 | cap=15 / cap=0 为 1.101108，仅作开销参照 | 9 / 10 |

压缩汇总为 `selected-groups`，不是 M10 全阶段收尾；按适用配置等权整体平均≤1 验收，组均值仅诊断。prefill/压缩分别保留 34/31 个退化入口分段；prefill 的 60 个 kernel 对照均退化，公开平均通过不表示 kernel 或每个单例提速。原生 FD 单列依据为分片策略，不是自动 FD 未迁移。正 cap 无同义 CANN attention，不计入该平均门槛；全部 wall/profiler 样本仍在原 attempt 目录。

## 复现与环境

设备复现前按 AGENTS.md 核对运行环境、native 包和实际加载，并先 dry-run/collect-only。本轮未执行下列设备命令；完整 M15d 命令（含 preflight、性能采样与指定组汇总）可用 `git show 26d7a70:docs/mla/status.md` 查询。

```bash
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export MLA_NATIVE_ROOT=/tmp/mla-m13b/native
python3 -m tests.mla --suite softcap-mixed-tools unit-softcap_mixed softcap-mixed-sanitize --output /tmp/mla-m15d-recheck/cpu
python3 -m tests.mla --suite softcap-mixed-check detail-softcap_mixed softcap-mixed-source e2e ordinary softcap-mixed-profile --output /tmp/mla-m15d-recheck/final
```

M15d 记录的环境为 Python 3.11.15、torch 2.7.1+cpu、torch_npu 2.7.1、CANN `/usr/local/Ascend/cann-9.0.0-beta.2`、卡 6 Ascend910B3；CATLASS 固定为 `769cd40a8716b28650b6bebb08db4834eea4462f`，CANN 源为 `ca18aee610729ca5270a44e2d84c751bc687fbf0`。记录的实际扩展为本仓 `flash_attn_npu_3/flash_attn_npu_3.cpython-311-aarch64-linux-gnu.so`，SHA256 `e37136919b4800d28f0658f41f68ae3ebdb6645e40bab67e427a991a2cd59347`；本轮仅校验其文件内容。

默认工作根 `/tmp/flash-attention-npu-mla-46928a793876/` 保存 `mla-validation-artifacts/prepare.json` 与数学缓存；当前有效 native 包为 `/tmp/mla-m13b/native`，不可误用默认目录中的旧配方包。固定 FA3 快照路径及校验见 [design 7.4](design.md#74-统一验收入口)。上述 `/tmp` 路径均非持久存储，内容变化或缺失后不得继承。旧阶段记录从 Git 查询，当前交接提交用 `git log -1 -- docs/mla/status.md` 定位。
