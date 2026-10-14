# MLA 当前状态

当前分支 `feat/mla-cann-migration`，本步起点 `14ad721`。**M10c4b1 TND 自动 split 的 v20 安全协议已完成。** 原 M10c4b 按共享协议风险拆为本步与 M10c4b2 生产接入；一次提交后立即停止，唯一下一步为 M10c4b2。

## 本步产物与边界

新增 v20 CPU 构造器和 CPU/AICore 共用消费者头，复用 M10c4a 原生计划、v5 双 allocation/used 和 v18 的原生 AIC/FD/Vector 校验。v20 独立检查 TND 记录身份、紧凑总量、空容量、accumulator 上界和原 allocation token 地址。保留 v18 wrapper 与 BSND Row 语义，调整的共享头已用重建生产后的累计 E2E 验收。

本步尚未接通 v20 AICPU、init/partial/merge 或公开 TND 自动入口；Direct 临时槽和设备数值/性能由下一步整体闭环。BSND 自动/显式与 TND 显式能力保持，TND 自动、混合和 PA split 仍拒绝。没有新公开行为，不注册伪自动 TND E2E；CPU 协议和旧无 mask 回归不构成新生产能力。CANN 源仓和 CATLASS 未修改。协议见 [design 6.2.19](design.md#6219-tnd-自动-split-的-v20-安全协议)，拆分依据见 [plan](plan.md#已完成-m10c4b1tnd-自动-split-的-v20-安全协议)。

## 环境与证据身份

接手工作树干净，宿主机无遗留 MLA 构建/测试，6 号 910B3 空闲。前步 E2E 的 13 个原始文件、1,104 个源码、生产 6,372 个依赖/12 个产物和 230 个加载库，共 7,400 个唯一文件逐项核验无差异，见 `/tmp/mla-m10c4b1/takeover.json`。

Python 3.11.15、torch 2.7.1+cpu、torch_npu 2.7.1、CANN 9.0.0-beta.2，CATLASS `769cd40a8716b28650b6bebb08db4834eea4462f`。生产重建 503.0 秒；首次准备因测试修正后的整树身份变化失效，最终重新核验构建缓存 1.2 秒、运行时检查 19.9 秒。实际加载仓库 `flash_attn_npu_3/flash_attn_npu_3.cpython-311-aarch64-linux-gnu.so`，SHA-256 `459f487db89b9a3bb06369d49fc9b6f16adb2245426deefc9c338107fa62079a`；接手旧扩展不计本次通过。

证据根 `/tmp/mla-m10c4b1/`；构建/reference 继续使用仓库外默认 `/tmp/flash-attention-npu-mla-46928a793876/mla-validation-artifacts/`。原生包保持 `MLA_NATIVE_ROOT=/tmp/mla-m10c3b/native`，本步不运行性能。build 仅保留 mla-codex-loop。共享头和构建身份已变，本轮累计 E2E 实跑、继承 0；不累计历史模块单元或性能。CPU 协议无需 attention reference，E2E 使用内容缓存并实际执行算子。

## 最终检查

各组均先 dry-run/collect-only。普通与 sanitizer 同为 125 节点；累计 E2E 与前步 161 节点逐项一致，无新增、删减或截断，见 `selection-check.json`。最终执行节点与收集清单一致，最终集合期间仅更新文档。

| 集合 | 本次结果 | 证据目录（相对证据根） |
| --- | --- | --- |
| `auto-split-tnd-metadata-tools` | 13/13，7.9 秒 | `cpu-fixed/attempt-1792009537817051280/` |
| `unit-auto_split_tnd_metadata` | 125/125，38.8 秒 | 同上 |
| `auto-split-tnd-metadata-sanitize` | 125/125，128.9 秒 | 同上 |
| 累计 `e2e` | 161/161，29.1 秒 | `e2e-fixed/attempt-1792010011573814140/` |

最终失败/跳过均为 0。固定 CANN AIC/FD、workspace 和 FA3 TND accumulator 片段重新编译执行；独立检查三种 mask、双 allocation/used 间隙、Direct/Partial/Empty、原生 Vector 8 行归约、隐式长度、零段/全空、所有尾字段/交叉版本/描述符损坏、精确预算和失败原子性。最大逻辑容量覆盖超过 uint32 的 M 前缀、int32 总 Q 及精确 int64 accumulator 字节界，不分配大 tensor。

普通与 sanitizer 的 v20/v18/native/contracts 四个驱动输入和输出逐字节一致，见 `cpu-evidence.json`。开发工具 13 项及代表/容量 8 项也曾通过，最终模块集合包含这些代表节点；以最终集合为准。累计 E2E 数学缓存 314 次命中，无 miss 或重算。

完整证据共 7,378 个唯一文件逐项核验无缺失或变化，见 `evidence-final.json`：包括 122 个原始文件、各组源码/CPU 产物、probe 源/依赖/产物、生产 750 个源码/6,372 个依赖/12 个产物及 230 个加载库。依赖变化或原始证据被清理后不可继续继承。

复现用新输出目录，各命令先 dry-run/collect-only；sanitizer 和 NPU 集合在宿主机执行：

```bash
export MLA_NATIVE_ROOT=/tmp/mla-m10c3b/native
python3 -m tests.mla --suite auto-split-tnd-metadata-tools --output /tmp/mla-m10c4b1/repro-tools
python3 -m tests.mla --unit auto_split_tnd_metadata --output /tmp/mla-m10c4b1/repro-cpu
python3 -m tests.mla --suite auto-split-tnd-metadata-sanitize --output /tmp/mla-m10c4b1/repro-sanitize
python3 -m tests.mla --suite e2e --device 6 --output /tmp/mla-m10c4b1/repro-e2e
```

## 失败处理与未运行

首轮普通集合的零 allocation 段对照误用调用级 maxQ 计算 scratch；固定源与消费者按实际最大 allocation 段长计算。修正本模块对照输入后普通/sanitizer 全通过，生产实现未因该失败改变；失败证据保留在 `cpu/`。修正发生在首轮 E2E 构建期间，驱动按保守整树身份拒绝该次准备，见 `e2e/summary.json` 的 `Source changed during preparation`；其 E2E 未运行、不计通过。编译输入/产物完整核验后，固定源码上的 `e2e-fixed/` 重新完成构建缓存、实际加载及全部 E2E，未改写或绕过身份。

登录 shell 初始化挂起，已结束本轮挂起命令并使用非登录 shell。首个 runtime 探测缺少输出目录，创建后重做成功；CPU 命令曾混用互斥 `--unit`/`--suite`，修正后收集成功，旧失败日志保留。sanitizer 直接在宿主机执行，保留 LeakSanitizer 检查。

pre-commit 使用 `/tmp/mla-m01-precommit`（`PYTHONPATH` 同路径）及 `/tmp/mla-m01-precommit-cache`。本步代码、最终文档质量/相关链接及 `git diff --check` 均通过，见 `quality-final-code.log`、`quality-final-docs.log` 和 `doc-links-final.json`。暂存模式的 pre-commit 因沙箱只读 `.git` 无法执行 `git write-tree`，宿主机按原配置重跑通过，见 `quality-staged-host.log`。历史模块单元、普通 FA3、设备新协议数值/Trap、性能矩阵、full-matrix、910C 和 M10 阶段收尾未运行；本步不开放生产 TND 自动能力。

## 唯一下一步

[M10c4b2：TND 原生自动 split 生产接入](plan.md#唯一下一步-m10c4b2tnd-原生自动-split-生产接入)。复用本步 v20，整体完成 AICPU/init/partial/merge、Direct 临时槽、静态真实 accumulator、公开 real/fake/compile E2E、Trap/生命周期和含原生 FD 的平均性能；当前提交不开始该步。
