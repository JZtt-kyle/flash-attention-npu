# MLA 当前状态

M10c9a 的 TND PA 显式 split 紧凑计划与 v25 安全协议已完成独立闭环。生产能力基线仍为 `737d5ae`（M10c8b），分支 `feat/mla-cann-migration`。唯一下一步为 [M10c9b：TND PA 显式 split 生产接入](plan.md#唯一下一步-m10c9btnd-pa-显式-split-生产接入)；本轮不开始该步。

## 当前能力与边界

512+64 共享 KV、Hkv=1、FP16/BF16 的已开放生产路径如下，均含无 mask、右下 causal/window、实际长度、合法四路 stride、空值与 real/fake/compile。接口和内部 Q-used 的区别见 [design 第 2 节](design.md#2-fa3-入口与后端边界)。

| Q / KV 布局 | 单 split | 显式 2..256 / 原生自动 0 |
| --- | --- | --- |
| 非 PA BSND / BSND | 已接通 | v17 / v18 |
| 非 PA TND / TND | 已接通 | v19 / v20 |
| 非 PA TND / BSND | 已接通 | v21 / v22 |
| BSND / ND PA | 已接通 | v23 / v24 |
| TND / ND PA | 已接通 | 均明确拒绝，须显式 `num_splits=1`；v25 仅独立 CPU 计划/共享协议 |

v25 为 `414+7B` 字，固定 C1V2；预算、Task/Merge/Row 独立处理 Q allocation/used、紧凑 totalQ、三种 mask、有效页和 FP32 accumulator 地址/溢出。复用已有 PA/mixed/split 组件，未改生产 AICPU/launcher/分派/构建清单，也未修改 CANN/CATLASS。新协议没有数值设备能力或公开性能通过声明；详见 [design 6.2.29](design.md#6229-tnd-pa-显式-split-计划与-v25-安全协议)。

192/128 prefill、softcap、非默认 pack_gqa/sm_margin 及 TND PA 自动 split 尚未接通。追加 KV、内置 rotary、量化等未开放选项明确拒绝；完整 graph replay 未验收。实机为 910B，没有 910C 实机通过声明。

## 本步验收与证据

证据根 `/tmp/mla-m10c9a/`，构建及数学缓存继续使用 `/tmp/flash-attention-npu-mla-46928a793876/mla-validation-artifacts/`，`build/` 仅有 `mla-codex-loop/`。这些 `/tmp` 路径非持久存储，后续继承须重验原始文件与依赖。

最终普通 CPU 集合为 415 passed、0 failed、0 skipped（47.2 s，`acceptance/summary.json` 的 `unit-split_paged_tnd_protocol`）。同集合 ASan/UBSan 415 passed（143.3 s）、协议工具 14 passed（9.2 s），均 0 failed/0 skipped，见 `sanitize-final/summary.json`。累计日常 E2E 229 passed、0 failed、0 skipped（48.6 s，`e2e/summary.json`），数学缓存命中 374 次，无重算。重建 533.4 s、实际加载核验 21.3 s 均通过。原代表/成本边界 9 passed（45.4 s，`preflight-2/`）；新增 C1V1 安全边界已包含于最终普通全集。最终集合、普通/sanitizer 节点集合一致性分别见 `selection-final/` 和 `selection-audit.json`，未使用 `--max-cases`。

本次修正和环境失败保留原证据：

- `preflight/` 首轮暴露 mixed shape 聚合初始化的字段顺序错误，已修正并重跑代表/完整组。
- `cpu/` 超界 maxQ 测试把超出 int32 的前缀送入测试驱动，导致流解析退出；改为合法零前缀独立测试超界 maxQ，`cpu-2/` 413 passed，随后新增 C1V1 拒绝边界后普通组重新 415 passed。
- `acceptance/` 的 sanitizer 415 项断言通过，但退出时 LeakSanitizer 因沙箱进程检查限制报错，该组仍为 failed；未关闭泄漏检查，改在宿主环境完整重跑。其后工具组 not_run，另在 `sanitize-final/` 执行。
- `e2e-collect/` 因沙箱无法访问 NPU 而收集失败；宿主环境 `e2e-collect-device/` 已完整收集 229 项（20.5 s）。

既有 258 组命令未改变，本步新增 4 个组（包含自动发现的模块单元组），见 `command-audit.json`。代码适用的六个 hooks 已通过，见 `quality-code-final.log/json`；文档三项 hooks 通过（`quality-docs.log`）；484 个本地链接/锚点及 32 张表格检查通过（`doc-links.json`），仓库 `ci/code_quality.sh --staged` 通过（`quality-staged.log`；六项适用 hooks 通过，YAML/shell/workflow 无适用文件）；固定 actionlint 环境在宿主完成初始化。`git diff --check` 与暂存差异检查通过，原始工作保留/暂存范围见 `staging-audit.json`。前期命令误将互斥 `--unit`/`--suite` 合用、选到旧版 clang-format 以及默认 Python 未带 pre-commit 的尝试均未计通过；已改用有效命令、clang-format 21.1.8/ruff 0.16.5 和既有 `/tmp/mla-m01-precommit` 工具环境。

复现先加 `--dry-run`，再 `--collect-only`；sanitizer 和 E2E 使用允许进程检查/设备访问的环境：

```bash
python3 -m tests.mla --unit split_paged_tnd_protocol --output /tmp/mla-m10c9a/repro-cpu
python3 -m tests.mla --suite split-paged-tnd-protocol-sanitize split-paged-tnd-protocol-tools --output /tmp/mla-m10c9a/repro-sanitize
python3 -m tests.mla --suite e2e --device 6 --output /tmp/mla-m10c9a/repro-e2e
```

## 环境、继承及已有工作

接手实际检查：Python 3.11.15、torch 2.7.1+cpu、torch_npu 2.7.1，CANN `/usr/local/Ascend/cann-9.0.0-beta.2`，CATLASS `769cd40a8716b28650b6bebb08db4834eea4462f`；卡 6 为 Ascend910B3。宿主进程检查仅见当前无头循环及无关任务，没有遗留 MLA 构建/pytest；不干扰其他卡。

`runtime-start.json` 保存实际加载的本仓扩展及全部运行库身份，与 M10c8b 的 E2E 一致，扩展 SHA-256 为 `dd1e0745d29ea659d83909fb7148c29c9b64dba40071a03990caba9d669c0f78`。`evidence-audit.json` 核对上一轮 E2E 的 13 个原始文件及 6421 个构建依赖，均有效且未改变。最终本次 CPU/sanitizer/工具原始证据与源码复核见 `acceptance-audit.json`，E2E 见 `e2e-audit.json`。本步新增协议/测试注册进入 runner 的保守闭包，因此按统一入口重建/实跑，继承 E2E 为 0。重建后实际加载扩展 SHA-256 为 `3ffca791e2fe82be17d98a3cfb7a55c5a0ea23877c410ebdceb39f0edf563738`，完整身份在本步 E2E attempt 的 `runtime.json`。历史模块单元、普通 FA3 专项和性能均未运行或继承；普通生产代码未改，不累跑历史单元。M10c8b 的历史性能和源数值记录可从 `git show 737d5ae:docs/mla/status.md` 查询。

接手前已有 `AGENTS.md`、`scripts/mla_codex_loop.py`、`tests/mla/support/mla_validation.py` 的独立注册修复、`tests/mla/unit/tools/test_mla_codex_loop.py` 及 plan/status 尾部循环修复备注，均保留且排除本步暂存。验收在保留这些工作树修复的环境执行；本步只提交新增协议、所属测试/注册及三份文档。原始内容和补丁保存在 `/tmp/mla-m10c9a/preexisting/`、`preexisting.patch`，下方原始备注中的计数/状态仍指当时，不覆盖当前状态。
