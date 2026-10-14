# MLA 当前状态

当前分支 `feat/mla-cann-migration`，本步起点 `d62d44f`。**M10c5a 混合显式 split 计划与 v21 安全协议已完成。** 当前闭环提交用 `git log -1 -- docs/mla/status.md` 定位；唯一下一步为 [M10c5b 混合显式 split 生产接入](plan.md#唯一下一步-m10c5b混合布局显式-split-生产接入)，本次提交后立即停止。

## 当前能力与本步边界

本步新增非 PA TND Q + BSND KV 的显式 S=2..256 CPU 计划及 v21 构造/独立消费协议。紧凑 Q 的持久预算和原 allocation 行地址与矩形 KV 的 64 位 `b*Sk` 分开处理，覆盖无 mask、causal、window、allocation/used、空容量及 accumulator 溢出。v19 的双 TND KV 总量受 int32 前缀约束，不能直接代替混合协议；据此按 plan 的风险拆分规则先闭环新协议，再整体接入生产。

v21 尚未编入 AICPU，也未发射设备 partial/merge。公共能力维持前步：非 PA BSND 与 TND Q/KV 支持自动和显式 split；混合/PA split 仍明确拒绝。当前 CPU 证据不代表混合 split 的设备数值、公开入口或性能通过。未修改既有 v17/v19/v20、生产分派、构建/发射参数、CANN 源仓或 CATLASS；契约见 [design 6.2.21](design.md#6221-混合显式-split-计划与-v21-安全协议)。

## 环境与证据身份

接手工作树干净，HEAD 与交接一致；宿主机无前轮遗留 MLA 测试/构建，6 号 910B3 空闲。前步累计 E2E 的 13 个原始文件、975 个源码依赖、6,383 个生产依赖、12 个产物及 230 个加载库，共 7,271 个唯一文件核验无差异，见 `/tmp/mla-m10c5a/takeover.json`。新增原生文件及驱动改变保守构建/验收身份，本步 E2E 实跑、继承 0，不累计历史单元/性能。

Python 3.11.15、torch 2.7.1+cpu、torch_npu 2.7.1、CANN `/usr/local/Ascend/cann-9.0.0-beta.2`；CATLASS `769cd40a8716b28650b6bebb08db4834eea4462f`。接手实际加载仓库扩展 `flash_attn_npu_3/flash_attn_npu_3.cpython-311-aarch64-linux-gnu.so`，SHA-256 `fecd251a8638cd2b8ff9f40e15c6b1a941294a9ae4200017b617606cd72a3fe2`。本步以 505.9 秒完成重建，21.2 秒核验实际加载同一路径，最终 SHA-256 **`b9e618f86053327e2ce9eb191931276e49f798b682b9a5ee828e475e43374424`**；构建 manifest、依赖和加载库均已核验。

证据根 **`/tmp/mla-m10c5a/`**；构建/reference 使用仓库外默认 `/tmp/flash-attention-npu-mla-46928a793876/mla-validation-artifacts/`。`build/` 仅保留 `mla-codex-loop`。最终四组共 122 个原始文件，加上源码快照、固定源 blob、CPU/生产产物、编译依赖和加载库，共 **7,397 个唯一文件**核验无差异，见 `evidence-final.json`。各组实际节点与 collect-only 完全一致；累计 E2E 的 173 项与前步清单逐项相同。CPU 单独审计另保留在 `evidence-cpu.json`。

## 本次验收

先 dry-run/collect-only，完成工具和代表/最大成本检查后固定实现。最终验收期间只修改相关文档；本模块普通与 ASan/UBSan 运行相同 195 节点集合，无数量截断或按耗时筛选。

| 组 | 结果 | 耗时 | 证据目录（相对证据根） |
| --- | --- | --- | --- |
| `unit-split_mixed_protocol` | 195/195 通过 | 37.4 秒 | `cpu-final/attempt-1792014477364294090/` |
| `split-mixed-protocol-sanitize` | 195/195 通过 | 120.3 秒 | `sanitize-final/attempt-1792014659073072840/` |
| `split-mixed-protocol-tools` | 14/14 通过 | 8.9 秒 | 同上 |
| 累计小输入 `e2e` | 173/173 通过 | 30.2 秒 | `e2e-final/attempt-1792014578086860050/` |

本模块新编译固定 CANN 范围/workspace 与 FA3 显式分片/accumulator 片段，独立核对三种 mask 的逐行可见集合、所有权、持久槽与原 Q/64 位 KV 地址。具名用例覆盖跨波、尾行、used 间隙、零段、全空、192/193/255/256 分片、宽任务号、精确字节边界、输入生命周期和失败原子性；v21 覆盖双 dtype 标志、长度/资源/窗口/片数/前缀损坏、非法缓冲与旧版本拒绝。普通和宿主机 sanitizer 的 plan、metadata、CANN、FA3 四个执行器均正常退出、stderr 为空。

代表/成本检查 9/9、35.3 秒，见 `dev/attempt-1792014348252183330/`。最大实际 CPU 输入为 B=262146、Sk=16384、仅 2 个 Q token；v21 记录 10,487,504 字节，最后 KV 起点超过 uint32。最终普通组该协议用例耗时 0.753 秒，计划用例 0.212 秒；其余大逻辑容量只做静态预算/边界查询，不分配 attention tensor。本模块不执行浮点 attention，因此不新增数学 reference；累计 E2E 数学缓存 hit 326、miss 0，待测实现实际执行，没有重算相同 reference。

未运行历史模块单元、普通 FA3 专项、公开性能、完整矩阵或 910C 实机；本步未改变生产/公开调用行为，它们不计本步通过或继承。M10c5b 必须新增自身设备固定源/双数学、accumulator、真实/fake/compile、Trap、生命周期、日常 E2E 与适用平均公开性能，不能用既有无 mask 证据替代。

## 已处理问题与复现

首次 sanitizer 在沙箱中完成 195 项断言后，LeakSanitizer 退出时因 ptrace 受限报 fatal error；统一驱动正确判该组失败，并停止后续工具组。失败证据保留在 `cpu-final/attempt-1792014477364294090/split-mixed-protocol-sanitize/`，不计通过。未关闭任何 sanitizer 检查，改宿主机重跑同一集合后通过，工具组随后实跑通过。最初登录 shell 初始化延迟，后续均用非登录 shell；三个初始 shell 已正常退出，未终止其他进程。

以下以仓库为 cwd，使用宿主机非登录 shell；新结果使用独立目录，先追加 `--dry-run`，再 `--collect-only` 核对集合。`--unit` 与 `--suite` 是互斥选择器；合并运行时可用 `--suite unit-split_mixed_protocol split-mixed-protocol-sanitize split-mixed-protocol-tools`。

```bash
python3 -m tests.mla --suite split-mixed-protocol-tools --output /tmp/mla-m10c5a-repro/tools
python3 -m tests.mla --suite split-mixed-protocol-check --output /tmp/mla-m10c5a-repro/check
python3 -m tests.mla --unit split_mixed_protocol --output /tmp/mla-m10c5a-repro/unit
python3 -m tests.mla --suite split-mixed-protocol-sanitize --output /tmp/mla-m10c5a-repro/sanitize
python3 -m tests.mla --suite e2e --device 6 --output /tmp/mla-m10c5a-repro/e2e
```

本步相关 design/plan/status、协议表格和链接已核对，未作阶段全面审计。完整差异、`git diff --check` 及仓库适用 pre-commit 质量记录分别由提交收尾核验，日志见证据根的 `quality-final.log`，链接记录见 `doc-links-final.json`。质量工具使用仓库要求的 pre-commit 4.6.2 与既有缓存环境；只暂存本步文件，不绕过提交钩子。收尾进程和 NPU 状态记录在 `processes-final.json`、`npu-final.log`；生产构建日志保留在上述仓库外 artifact 目录。
