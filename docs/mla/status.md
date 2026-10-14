# MLA 当前状态

当前分支 `feat/mla-cann-migration`，本步起点 `cba3b77`。**M10c3a TND 显式 split 任务与持久布局已闭环。** 本轮只完成该独立 CPU 算法并提交一次，提交后停止；当前闭环提交可用 `git log -1 -- docs/mla/status.md` 定位。

## 本步产物与边界

接手发现现有显式计划的容量和输出地址绑定 BSND 的 B×Sq，按原 plan 允许的新地址算法拆分规则，将 M10c3 分为本步 CPU 计划与 M10c3b 整体生产接入。

新增 `mla_split_tnd_plan.h/.cpp`，用紧凑 totalQ 与分段尾部 padding 构造静态持久上界，复用 allocation/used 解析、固定源 mask 分片/workspace 和 TND accumulator。实际任务使用有效 Q 的 uint64 M 前缀，Task 返回双 allocation 起点与 used，Row 使用原分配 token，保留 used 间隙；覆盖跨波所有者、空片、字节溢出和失败原子性。契约见 [design 6.2.16](design.md#6216-tnd-显式-split-任务与持久布局)。

新增独占 `unit/split_tnd_plan/`、CPU driver/build、工具检查与统一驱动分组。本步未修改公共入口、metadata、设备数值核或 setup；TND split 仍明确拒绝。未将旧无 mask/BSND 回归算作新 TND split 能力验收，也未注册伪公开 E2E。CANN 源仓和 CATLASS 未改。

## 环境与证据身份

接手时分支/提交与交接一致、工作树干净。宿主机无遗留 MLA 构建/测试，第 6 张 910B3 空闲；本轮无头进程与其他任务保留。前步 E2E 的 13 个原始文件、926 个源码、生产 6,362 个依赖/12 个产物及 230 个加载库，共 7,225 个唯一文件逐项核对无差异，见 `/tmp/mla-m10c3a/takeover.json`。

Python 3.11.15、torch 2.7.1+cpu、torch_npu 2.7.1、CANN 9.0.0-beta.2，CATLASS `769cd40a8716b28650b6bebb08db4834eea4462f`。新 CPU 源/工具改变保守构建和证据身份，生产重建 497.9 秒，运行时核对 20.1 秒。实际加载仓库 `flash_attn_npu_3/flash_attn_npu_3.cpython-311-aarch64-linux-gnu.so`，SHA-256 `03a812f51233ec908efa3cb625c418e68994a05a2ac59983887e3a8f0554d6f6`，与 E2E loaded 报告一致。实际链接 52 个对象，不含新 TND CPU 计划，见 `production-boundary.json`。

累计 E2E 本次实跑、继承 0；未累计历史单元/性能。证据根 `/tmp/mla-m10c3a/`，构建/reference 继续使用仓库外 `/tmp/flash-attention-npu-mla-46928a793876/mla-validation-artifacts/`。build 仅保留 mla-codex-loop 无头状态，原始证据清理后不可继承。

## 最终验收

各集合均先 dry-run/collect-only；正式实际节点与收集清单逐项一致。普通模块与 sanitizer 同为 57 项，累计 E2E 仍为上步相同的 149 个小输入节点，未填充或截断，见 `selection-check.json`。代表检查完成后固定实现，再运行完整集合；期间仅更新文档。

| 集合 | 本次结果 | 证据目录（相对证据根） |
| --- | --- | --- |
| `split-tnd-plan-tools` | 14/14，8.7 秒 | `check/attempt-1792002289633518230/` |
| `split-tnd-plan-check` | 3/3，26.7 秒 | 同上 |
| `unit-split_tnd_plan` | 57/57，27.0 秒 | `acceptance/attempt-1792002497215833010/` |
| `split-tnd-plan-sanitize` | 57/57，78.0 秒 | 同上 |
| 累计 `e2e` | 149/149，28.7 秒，继承 0 | 同上 |

本模块每组新编译并实际执行固定 CANN 块范围/workspace 与 FA3 分片/shape 片段；独立枚举三种 mask、非均匀双分配/used、零段/全空、跨波/尾片/C1V1/C1V2，以及原分配输出和 accumulator 地址。小分配分区穷举验证静态上界，特别覆盖多个短段的尾块不能合并；边界查询验证 int32 总量、2^32 个组/2^40 个任务和 int64 字节界，不分配大 tensor。

普通与 sanitizer 的 plan/native/contracts 原始请求及响应逐字节一致，分别执行 4,837/43/1,112 次查询，退出码均为 0、stderr 为空；sanitizer 编译命令均含 ASan/UBSan。所有最终集合无失败或跳过。CPU 组不计算 attention reference；累计 E2E 的 303 次 reference 查询全部命中，无 miss/refresh/重复计算，设备实现始终实际执行。

`final-evidence-check.json` 核对五组 152 个原始文件、当前源码、12 个 CPU 可执行文件、生产 6,362 个依赖/12 个产物和 230 个实际加载库，共 7,372 个唯一文件，错误 0。当前相关契约、链接与质量检查见 `doc-links-final.json`、`quality-final.log`，提交前差异见 `final.patch`。

复现使用新输出目录，先附加 `--dry-run` 和 `--collect-only` 核对，再实跑；涉及 E2E 的收集/执行需可见 910B 设备：

```bash
python3 -m tests.mla --suite split-tnd-plan-tools split-tnd-plan-check --output /tmp/mla-m10c3a/repro-check
python3 -m tests.mla --unit split_tnd_plan --output /tmp/mla-m10c3a/repro-unit
python3 -m tests.mla --suite split-tnd-plan-sanitize e2e --device 6 --output /tmp/mla-m10c3a/repro-sanitize-e2e
```

正式执行使用等价的 `--suite unit-split_tnd_plan split-tnd-plan-sanitize e2e --device 6` 合并驱动。

## 环境失败与未运行

E2E 初次在沙箱内 collect-only 时因设备不可见而失败，退出 4；改到宿主机后完整收集成功，不改测试断言。CLI 自检曾将互斥 `--unit`/`--suite` 合用而被拒绝，改为上述等价 suite。全局 Python 未安装 pre-commit，一个旧缓存又缺 actionlint 环境；恢复使用既有 `/tmp/mla-m01-precommit` 与 `/tmp/mla-m01-precommit-cache` 后质量检查通过，未跳过 hook。这些环境/命令失败均不计通过；正式测试无待处理失败。

历史单元/性能、额外普通 FA3 大组、TND split 设备/公开行为、full-matrix 和 910C 未运行，不计通过。公共调用和设备数值未改变，本步不要求新增数值/性能矩阵，也不是 M10 阶段收尾。

## 唯一下一步

[M10c3b TND Q/KV 显式 split 生产接入](plan.md#唯一下一步-m10c3btnd-qkv-显式-split-生产接入)。先核对本步进程、完整证据和实际依赖，再将已闭环 CPU 地址计划、partial/FD 整体接入设备 metadata、varlen 入口与真实 accumulator，补齐 TND 专属数值、公开行为、Trap/生命周期、日常 E2E 和适用公开性能。本轮提交后立即停止，不开始该步。
