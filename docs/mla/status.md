# MLA 当前状态

当前分支 `feat/mla-cann-migration`，本步起点 `e62e7c1`。**M10c4a TND 原生自动 split 任务与持久布局已完成。** 提交后立即停止；唯一下一步为 M10c4b TND 原生自动 split 生产接入。

## 本步产物与边界

新增独立 `mla_auto_split_tnd_plan.h/.cpp`，保留原生 AIC/FD 端点和 8 行 Vector 归约所有权；静态预算使用紧凑 totalQ，完整解析并拥有双 allocation/used，明确 Direct/Partial/Empty 任务、FD 持久槽及原 allocation token 的 O/LSE/accumulator 地址。共享 CPU 文件、数值 Kernel、AICPU、生产发射/公共入口、CANN 源仓和 CATLASS 未修改。

BSND 自动/显式与 TND 显式生产能力保持；TND 自动、混合和 PA split 仍拒绝。本步没有新公开行为，不注册伪自动 TND E2E。相关契约见 [design 6.2.18](design.md#6218-tnd-原生自动-split-任务与持久布局)，路径和完成条件见 [plan](plan.md#已完成-m10c4atnd-原生自动-split-任务与持久布局)。

## 环境与证据身份

接手工作树干净，宿主机无遗留 MLA 构建/测试，6 号 910B3 空闲。前步 E2E 的 13 个原始文件、952 个源码、生产 6,372 个依赖/12 个产物和 230 个加载库，共 7,251 个唯一文件核对无差异，见 `/tmp/mla-m10c4a/takeover.json`。

Python 3.11.15、torch 2.7.1+cpu、torch_npu 2.7.1、CANN 9.0.0-beta.2，CATLASS `769cd40a8716b28650b6bebb08db4834eea4462f`。新增源码改变保守构建身份，生产重新构建 507.0 秒、运行时检查 19.5 秒；实际加载仓库 `flash_attn_npu_3/flash_attn_npu_3.cpython-311-aarch64-linux-gnu.so`，SHA-256 `0d9373ad314ae9056c947acc2b99a696e97b88260d8abcf77ca937b85b184670`。不能将接手旧扩展或旧 E2E 当作本次通过。

证据根 `/tmp/mla-m10c4a/`；构建/reference 沿用仓库外默认 `/tmp/flash-attention-npu-mla-46928a793876/mla-validation-artifacts/`。本步 CPU 测试无需 attention reference；累计 E2E 使用内容缓存，待测生产实现实际执行。原生包保持 `MLA_NATIVE_ROOT=/tmp/mla-m10c3b/native`，本步不运行原生性能。build 仅保留 mla-codex-loop。

## 最终检查

全部先 dry-run/collect-only。普通与 sanitizer 节点逐项一致；累计 E2E 与前步 161 项清单一致，无新增、删减或截断，见 `selection-check.json`。实现固定后仅更新文档，最终执行节点与收集清单相同。

| 集合 | 本次结果 | 证据目录（相对证据根） |
| --- | --- | --- |
| `auto-split-tnd-plan-tools` | 13/13，8.3 秒 | `cpu/attempt-1792008047749112530/` |
| `unit-auto_split_tnd_plan` | 52/52，25.3 秒 | 同上 |
| `auto-split-tnd-plan-sanitize` | 52/52，85.9 秒 | 同上 |
| 累计 `e2e` | 161/161，29.4 秒 | `e2e/attempt-1792008114349595530/` |

最终集合失败/跳过均为 0，继承 E2E 为 0，不累计历史单元或性能。固定 CANN 原生 AIC/FD、workspace 和 FA3 TND accumulator 片段均重新编译执行；独立检查三种 mask、allocation/used 间隙、Direct/FD/Empty 混合、8 行归约边界、零段/全空、输入生命周期及失败原子性。最大逻辑查询覆盖超过 uint32 的组前缀、int32 总量和精确 int64 字节边界，不分配大 tensor。

普通与 sanitizer 的 plan/native/contracts/default-plan 四个驱动输入、输出逐字节一致，见 `cpu-evidence.json`。开发代表检查曾通过 4 项；正式模块集合包含其全部节点及补充的 KV 总量边界，以最终 52 项证据为准。累计 E2E 数学缓存 314 次命中，无 miss 或重算。

完整证据共 7,362 个唯一文件逐项核验，无缺失或变化，见 `evidence-final.json`：包含 116 个原始文件、各组源码、CPU 产物、probe 依赖/库、生产 745 个源码/6,372 个依赖/12 个产物及 230 个加载库。源码或原始证据被修改/清理后不可继续继承。

复现用新输出目录，各命令先 dry-run/collect-only；sanitizer 与 NPU 集合在宿主机执行：

```bash
export MLA_NATIVE_ROOT=/tmp/mla-m10c3b/native
python3 -m tests.mla --suite auto-split-tnd-plan-tools --output /tmp/mla-m10c4a/repro-tools
python3 -m tests.mla --unit auto_split_tnd_plan --output /tmp/mla-m10c4a/repro-cpu
python3 -m tests.mla --suite auto-split-tnd-plan-sanitize --output /tmp/mla-m10c4a/repro-sanitize
python3 -m tests.mla --suite e2e --device 6 --output /tmp/mla-m10c4a/repro-e2e
```

## 失败处理与未运行

登录 shell 初始化挂起，已结束本次挂起命令并改用非登录 shell。开发驱动曾错误引用不存在的 `mla_used_schedule.cpp`（实现位于既有 `mla_tnd_schedule.cpp`），修正后代表与最终模块组通过。开发质量检查发现的 Python 局部名字冲突已修正。

沙箱 E2E collect-only 因 NPU 不可见而失败；宿主机相同集合收集并运行成功。sanitizer 直接在宿主机执行，避免已有 LeakSanitizer/ptrace 限制，未关闭检查。失败目录 `check/` 与 `final-collect/` 保留，不计通过。

pre-commit 使用 `/tmp/mla-m01-precommit`（`PYTHONPATH` 同路径）和 `/tmp/mla-m01-precommit-cache`；最终质量检查、相关文档链接及 `git diff --check` 均通过，见 `quality-final.log` 与 `doc-links-final.json`。历史模块单元、普通 FA3、性能矩阵、full-matrix、910C 和 M10 阶段收尾未运行；本步未改变生产算法或公共调用行为。

## 唯一下一步

[M10c4b：TND 原生自动 split 生产接入](plan.md#唯一下一步-m10c4btnd-原生自动-split-生产接入)。复用本步原生任务与地址协议，整体接通 metadata、init/partial/merge、真实 accumulator、公开 E2E 与适用平均性能；当前提交不开始该步。
