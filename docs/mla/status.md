# MLA 当前状态

当前分支 `feat/mla-cann-migration`，本步起点 `fb2086c`。**M10c2b1 自动 split metadata 安全协议已闭环。** M10c2a 的原生自动任务计划已闭环；本步按新共享协议的独立验收边界从 M10c2b 拆出，公开自动 split 仍拒绝。当前闭环提交用 `git log -1 -- docs/mla/status.md` 定位。

## 本步产物与边界

新增 `mla_auto_split_bsnd_metadata.h/.cpp` 与独立 `mla_auto_split_bsnd_consumer.h`：v18 连续 int32 `[758+10B]`，保存 v5 allocation/used、源基本块/资源及原生变长 AIC/FD 端点。消费者不调用生产构造器、CPU 计划或 SplitCore，独立验证可见 tile 覆盖、FD 片数/首片/紧凑槽、8 行 Vector 归约、64 位 M 前缀和 workspace/accumulator 容量，再提供任务、归约和分配行地址查询。仅 C1V2，非空 blockDim=AIC；失败不发布消费者输出，非法缓冲/重叠在写入前拒绝。

本步是独立 CPU 共享协议。生产 setup、metadata AICPU、Kernel/Block/Tile 数值、发射、公开 real/fake/compile、CANN 源仓和 CATLASS 均未修改；不能据本步 CPU 成功宣称设备 v18 或公开自动功能已接通。静态 AIC accumulator 容量仍不是已开放的自动返回契约。协议见 [design 6.2.14](design.md#6214-bsnd-自动-split-的-v18-安全协议)。

新增独占 `tests/mla/unit/auto_split_bsnd_metadata/`、固定源构建/CPU 驱动及工具用例。累计日常 E2E 134 项与上步逐项相同，见 `/tmp/mla-m10c2b/selection.json`；未注册伪自动 E2E，未将旧无 mask 用例算作本步新能力验收。

## 接手、环境与证据

接手分支与 `fb2086c` 交接一致、工作树干净，无遗留 MLA 构建/测试。第 6 张 910B3 空闲；其他仓库进程未触碰。上步 E2E 的 13 个原始文件、909 个源码及生产 6,349 个依赖/12 个产物完整，见 `/tmp/mla-m10c2b/takeover.json`。本步新共享头/驱动及统一入口改变保守依赖身份，累计 E2E 本次实跑，继承 0；不累计历史单元/性能。

Python 3.11.15、torch 2.7.1+cpu、torch_npu 2.7.1、CANN 9.0.0-beta.2；CATLASS `769cd40a8716b28650b6bebb08db4834eea4462f`。统一构建后实际加载仓库 `flash_attn_npu_3/flash_attn_npu_3.cpython-311-aarch64-linux-gnu.so`，SHA-256 `925b847bf22943ecebaf39ce240b59f4d2a2fead2f203045478c76763f2f894a`；runtime 与 E2E loaded 记录一致。构建耗时 494.8 秒，运行时核对 21.4 秒。接手旧库仅用于基线核对，未代替当前构建。

证据根 `/tmp/mla-m10c2b/`；构建/reference 沿用 `/tmp/flash-attention-npu-mla-46928a793876/mla-validation-artifacts/`，`build/` 仅有 `mla-codex-loop/`。原始证据清理后不可继续继承。

## 最近闭环验收

全部组先 dry-run/collect-only；普通与 sanitizer 使用同一 81 项完整集合。三种 mask、FD/非 FD、单核/最大核、双 dtype、变长/空值、独立几何/地址及记录损坏均有独立覆盖目的，无固定数量填充或截断。CPU 不计算 attention reference；本次 E2E 的 281 次数学 reference 查询全部命中内容缓存，无 miss、read/write error 或 refresh，待测设备实际执行。

| 最终集合 | 结果 | 证据目录 |
| --- | --- | --- |
| `auto-split-metadata-check` | 8/8 通过，26.3 秒；三种 mask 双 dtype 及两个容量边界 | `check-budget/attempt-1791995873487657400/` |
| `--unit auto_split_bsnd_metadata` | 81/81 通过，26.0 秒 | `cpu-budget/attempt-1791995944405006430/` |
| `auto-split-metadata-tools` | 13/13 通过，9.0 秒 | `sanitize-budget/attempt-1791995946350923770/` |
| `auto-split-metadata-sanitize` | 81/81 通过，85.9 秒；ASan/UBSan/LeakSanitizer 及进程退出检查 | `sanitize-budget/attempt-1791995946350923770/` |
| 累计 `e2e` | 134/134 本次 910B 实跑，27.7 秒，继承 0 | `e2e-budget/attempt-1791995948384615470/` |

最终 `final-evidence-check.json` 复核全部五组原始证据和当前源码、生产 6,349 个依赖/12 个产物、230 个实际加载库，共核对 7,353 个唯一文件，错误 0。普通 CPU 和 sanitizer 各 44 个原始文件完整，固定 Git blob、编译命令、二进制身份、驱动输入/输出与退出结果均保留。

最大逻辑容量为 5,033,164,800 个 M 块，只查询代表边界、不创建逐 M 描述。独立 accumulator 溢出用例只分配约 2.4 MB 记录，从 v5 构造绕过自动计划的合法形状/零 KV 记录，确认消费者拒绝超过 INT64_MAX 的静态容量；不分配巨量 attention tensor。

复现先加 `--dry-run`、`--collect-only`，再执行；输出使用新目录，保留当前证据。sanitizer 需可运行 LeakSanitizer 的宿主环境，E2E 需可访问 910B：

```bash
python3 -m tests.mla --suite auto-split-metadata-tools --output /tmp/mla-m10c2b/tools-repro
python3 -m tests.mla --unit auto_split_bsnd_metadata --output /tmp/mla-m10c2b/cpu-repro
python3 -m tests.mla --suite auto-split-metadata-sanitize --output /tmp/mla-m10c2b/sanitize-repro
python3 -m tests.mla --suite e2e --device 6 --output /tmp/mla-m10c2b/e2e-repro
```

## 失败处理与未运行

代表检查修复两处过严的源端点判断：window 的零成本前缀允许端点位于首个可见 tile 前，全零有效 KV 的原生端点允许位于 batch 末尾。完整 CPU 又发现 C1V1 构造/消费接受范围不一致，现于构造阶段明确拒绝。旧 80 项 CPU/sanitizer 曾通过，但复核后补了独立 accumulator 溢出校验和用例；旧结果不算最终闭环，完整 81 项重新运行并通过。旧基线 `e2e-final` 构建在 184.4 秒主动中断，驱动退出 130，未运行设备用例、不计通过；最终构建另在 `e2e-budget` 重启。早期 dry-run 的注册语法错误已修复，固定版本质量工具已通过冻结检查。

AICPU/v18 设备执行、自动数值/Trap/生命周期、公开自动 real/fake/compile、平均公开性能、普通 FA3 独立组、历史单元/性能、full-matrix 和 910C 未运行，不列为通过或继承。仅本步 CPU 协议的必要检查和既有能力 E2E 属于当前验收。最终必要集合均通过，无未解决失败。三份文档按新协议及提交边界核对；改动涉及的链接、仓库 pre-commit 和 `git diff --check` 通过，证据见 `doc-links-final.json`、`quality-final.log`。

## 唯一下一步

[M10c2b2 BSND 自动 split 生产接入](plan.md#唯一下一步-m10c2b2bsnd-自动-split-生产接入)。先核对本步进程、完整证据和实际依赖，明确自动返回形状与静态容量的适配，再整体接通 AICPU/设备、partial/8 行 merge、真实 accumulator、公开 real/fake/compile 及该能力 E2E/性能。本轮只提交 M10c2b1 一次，提交后停止。
