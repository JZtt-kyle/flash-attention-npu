# MLA 当前状态

本轮专门修正文档，功能基线仍为 `737d5ae`（M10c8b），分支 `feat/mla-cann-migration`；未推进功能或作 M10 阶段收尾。**唯一功能下一步为 [M10c9a：TND PA 显式 split 计划与安全协议](plan.md#唯一下一步-m10c9atnd-pa-显式-split-计划与安全协议)。** 本轮完成文档检查后至多提交一次并停止。

## 当前能力与边界

512+64 共享 KV、Hkv=1、FP16/BF16 支持下表路径；均含无 mask、右下 causal/window、实际长度、合法四路 stride、空值与 real/fake/compile。公开接口、metadata 复用和内部 Q-used 的区别见 [design 第 2 节](design.md#2-fa3-入口与后端边界)。

| Q / KV 布局 | 单 split | 显式 2..256 / 原生自动 0 |
| --- | --- | --- |
| 非 PA BSND / BSND | 已接通 | v17 / v18 |
| 非 PA TND / TND | 已接通 | v19 / v20 |
| 非 PA TND / BSND | 已接通 | v21 / v22 |
| BSND / ND PA | 已接通 | v23 / v24 |
| TND / ND PA | 已接通 | 均明确拒绝，须显式传 `num_splits=1` |

显式 split 返回真实 FP32 accumulator；自动路径以 26 为静态返回容量，原生每 M 的片数由实际 AIC/FD 端点决定。最新 v24 使用独立 init/partial/merge、FD 持久槽和 Direct 临时槽，每阶段先校验当前长度/有效页；完整 allocation 输出与 accumulator 初始化为 0/-inf。生产不调用 CANN attention，CANN/CATLASS 未由本轮修改。

192/128 prefill、softcap、非默认 pack_gqa/sm_margin 尚未接通；追加 KV、内置 rotary、量化等未开放选项明确拒绝。完整 graph replay 尚未验收，fake/compile 和多流证据不能替代。实机记录来自 910B；没有 910C 实机通过声明。

## 最近功能闭环证据：M10c8b

以下是 `737d5ae` 的历史实测结果，**不是本轮重新执行或继承的测试**。证据根 `/tmp/mla-m10c8b/`，每行引用该子目录 `summary.json` 中的对应组，具体 attempt 在汇总内；节点收集对应关系在 `selection-completion-final.json`。所有最终选用组均 0 failed、0 skipped，未使用 `--max-cases` 截断。

| 检查 | 结果 | 耗时 | 结果子目录 |
| --- | --- | --- | --- |
| protocol CPU | 420 passed | 44.9 s | `cpu-final` |
| 同集合 ASan/UBSan | 420 passed | 147.6 s | `sanitize-final` |
| protocol 工具 | 14 passed | 8.9 s | `sanitize-final` |
| 模块工具/进程隔离 | 26 passed | 7.2 s | `tools-exec` |
| 固定源 PA/原生 FD 与双数学 | 12 passed | 228.2 s | `acceptance-cores` |
| 公开专项 | 80 passed | 102.8 s | `acceptance-cores` |
| release Trap | 40 passed | 530.8 s | `trap-exec-regression` |
| 日常 E2E | 229 passed | 49.2 s | `trap-exec-regression` |
| ordinary FA3 | 516 passed | 28.5 s | `trap-exec-regression` |
| 原生自动公开性能 | 1 汇总用例，12 配置 | 285.2 s | `profile-native` |

源数值组逐元素 O/LSE、真实 accumulator 和双数学通过，无源精度例外。公开性能为预先固定的 6 输入×双 dtype，隐式/生成/复用各 R=1/4，5 次预热、5×20 wall、10 次 profiler；12 配置均适用，含 8 个原生 FD 配置。配置内/配置间等权平均比 **0.852969≤1**，4/12 配置及 25/72 分段退化。完整逐例、分段及原始计时在性能组 `public-profile/average.json` 和双 dtype 归档中。

`acceptance-cores` 的早期 Trap 仍为 failed，随后的 E2E/ordinary 为 not_run；最终结果取后续 `trap-exec-regression`，不能把前一 attempt 整体写成通过。其他已修正的构建身份、溢出用例、物理 AIC/静态容量混淆和旧原生包配方问题，见 `git show 737d5ae:docs/mla/status.md` 及原失败日志；本轮未发现新的测试失败。

### 历史环境与复现

- 运行记录：Python 3.11.15、torch 2.7.1+cpu、torch_npu 2.7.1，CANN `/usr/local/Ascend/cann-9.0.0-beta.2`，CATLASS `769cd40a8716b28650b6bebb08db4834eea4462f`；卡 6 为 Ascend910B3、实际 20 AIC。当前设备状态本轮未探测。
- 历史实际加载为本仓 `flash_attn_npu_3/flash_attn_npu_3.cpython-311-aarch64-linux-gnu.so`，SHA-256 `dd1e0745d29ea659d83909fb7148c29c9b64dba40071a03990caba9d669c0f78`；各 attempt 的 `runtime.json`/组内 `loaded.json` 保存加载身份。本轮没有加载或重建扩展，后续开发仍须按依赖核验实际产物。
- 性能使用 `MLA_NATIVE_ROOT=/tmp/mla-m10c3b/native` 的固定 CANN `ca18aee610729ca5270a44e2d84c751bc687fbf0` 包；配方/内容核验记录为 `native-audit.json`。当时默认 native 目录的包配方已过期，不能仅因目录存在就复用。
- 构建与数学缓存为 `/tmp/flash-attention-npu-mla-46928a793876/mla-validation-artifacts/`；`/tmp` 非持久存储。下次继承须重新核对身份和原始文件，证据缺失即失效；只有 E2E 正确性可继承，历史单元/性能只供查询。

按影响选择复现；先对所选命令加 `--dry-run`、再 `--collect-only`，日常只运行 E2E，设备号按当时可用情况选择：

```bash
python3 -m tests.mla --unit auto_split_paged_bsnd_protocol --output /tmp/mla-m10c8b/repro-cpu
python3 -m tests.mla --suite auto-split-paged-bsnd-protocol-sanitize auto-split-paged-bsnd-protocol-tools auto-split-paged-bsnd-tools --output /tmp/mla-m10c8b/repro-tools
python3 -m tests.mla --unit auto_split_paged_bsnd --device 6 --output /tmp/mla-m10c8b/repro-source
python3 -m tests.mla --suite auto-split-paged-bsnd-public auto-split-paged-bsnd-trap e2e ordinary --device 6 --output /tmp/mla-m10c8b/repro-public
MLA_NATIVE_ROOT=/tmp/mla-m10c3b/native python3 -m tests.mla --suite auto-split-paged-bsnd-profile --device 6 --output /tmp/mla-m10c8b/repro-profile
```

## 本轮文档核对与既有工作

本轮证据在 `/tmp/mla-doc-maintenance-20261015/`。`evidence-audit.json` 实际核对上述 10 个最终组引用的 1,217 个原始文件哈希和 JUnit 数量，无缺失/不匹配。源码快照差异与 M10c8b 原审计一致：早期 CPU 组的后续设备 reference/工具注册变化、源/公开组的独立 Trap 工具变化均保留说明；最终 E2E/ordinary/性能源码快照与接手代码一致。此核对不代替构建/运行环境的恢复验收，也未把历史单元或性能记为继承。

固定 CANN Git 对象、FA3 本地固定源码及其 oracle 哈希、CATLASS gitlink、Python/C++ 分派、v1–v24 编码与 AICPU 构建清单已核对。design 合并记录选择/返回值说明，补齐 v24 的空值/流/性能边界；plan 压缩已完成日志、补齐 M10c8b 归档，并将 M10c9a 命令标为拟新增。11 组文档命令 dry-run 通过（`commands.json`）；本地链接/锚点与表格检查通过（`doc-links-final.json`），固定 FA3 远端链接只核对对应本地固定源码，未作在线可达性声明。

适用的 trailing-whitespace、end-of-file-fixer、check-merge-conflict 均通过（`quality-applicable.log`），`git diff --check` 通过。首次文件尾检查修正 plan 末尾多余空行后重跑通过；完整 pre-commit 在无关 actionlint 环境初始化时因网络权限限制未完成（`quality-docs.log`），不记为通过。本轮设备/工具单元测试及 pytest collection 均未运行，继承 E2E 为 0。

接手前已有 `AGENTS.md`、`scripts/mla_codex_loop.py`、`tests/mla/support/mla_validation.py`、`tests/mla/unit/tools/test_mla_codex_loop.py` 以及 plan/status 的独立循环修复备注；本轮保留且不暂存这些改动。最新用户约定“每五轮开发插入文档轮”继续有效。旧工具修复的 287 passed/91.0 s 与 8 个证据文件已只读核对，不算本轮执行。下方如附有该修复原始备注，其中“当前/本次”和循环计数均指当时状态，不作为现时 HEAD、设备或恢复状态的依据。
