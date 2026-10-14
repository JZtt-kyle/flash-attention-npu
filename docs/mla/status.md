# MLA 当前状态

当前分支 `feat/mla-cann-migration`，本步起点 `94eedfe`。**M10c6a 混合原生自动 split 计划与 v22 安全协议已完成验收。** 唯一下一步为 [M10c6b 混合布局原生自动 split 生产接入](plan.md#唯一下一步-m10c6b混合布局原生自动-split-生产接入)；本次只提交 M10c6a，提交后立即停止。当前闭环提交可用 `git log -1 -- docs/mla/status.md` 定位。

## 本步产物与开放边界

新增非 PA TND Q + BSND KV 的原生自动计划及 v22 CPU 构造/独立消费协议。复用紧凑 Q 的拥有存储计划、原生 AIC 变长分区、FD 持久槽与 8 行 Vector 归约；共享构造显式选择 KV 布局，BSND 总量和 `b*Sk` 起点始终保持 64 位。静态预算使用 totalQ，Q/O/LSE/FP32 accumulator 保留原 allocation 地址；Direct/Partial/Empty、三种 mask、空容量和独立字节上界均由本模块验收。

v22 是精确 int32 `[758+10B]`、version=kind=22。独立消费者复用共享原生安全检查并显式选择 Q=TND/KV=BSND，重算前缀、槽和 accumulator 上界，核对当前长度及窗口；v18/v20 wrapper 保持原参数。尚未编入 v22 AICPU、发射混合自动设备 Kernel 或开放公开混合自动请求，混合自动与 PA split 仍明确拒绝。前步混合显式 split 及 BSND/TND 自动能力保持开放；CPU 证据不代表新设备数值/性能通过。契约见 [design 6.2.23](design.md#6223-混合原生自动-split-计划与-v22-安全协议)。未修改 CATLASS 或 CANN 源仓。

## 环境与证据身份

接手工作树干净，HEAD 与交接一致；宿主机无前步遗留 MLA 验收任务，6 号 910B3 空闲，其他卡任务未改动。Python 3.11.15、torch 2.7.1+cpu、torch_npu 2.7.1、CANN `/usr/local/Ascend/cann-9.0.0-beta.2`；CATLASS `769cd40a8716b28650b6bebb08db4834eea4462f`。接手实际加载仓库 `flash_attn_npu_3/flash_attn_npu_3.cpython-311-aarch64-linux-gnu.so`，SHA-256 `854c9a98b28e742ba099194897690a91b4b03cc98c1d81433432e5f2db25b5aa`，与前步一致。

证据根 `/tmp/mla-m10c6a/`。构建和数学缓存沿用仓库外 `/tmp/flash-attention-npu-mla-46928a793876/mla-validation-artifacts/`，`build/` 仅有 `mla-codex-loop`。接手前步 E2E 的 13 个原始文件、1,000 个源码依赖、生产输入/6,392 个依赖/12 个产物及 230 个加载库，共 7,299 个唯一文件核验无差异，见 `takeover.json`。本步共享代码和验收身份已改变，累计 E2E 全部实际执行，继承 0；不累计其他模块历史单元。

## 本次通过

正式源码固定后，以下四组通过且无失败/跳过；普通与 sanitizer 为同一 182 节点集合，各自与 collect-only 清单逐项一致。CPU 最终原始结果、源码、固定源/工具链及二进制核验通过，1,078 个唯一文件无差异，见 `evidence-cpu.json`。

| 组 | 通过 | 耗时 | 证据目录（相对证据根） |
| --- | --- | --- | --- |
| `auto-split-mixed-protocol-tools` | 14/14 | 9.1 秒 | `cpu-frozen/attempt-1792019742966917160/` |
| `unit-auto_split_mixed_protocol` | 182/182 | 42.3 秒 | 同上 |
| `auto-split-mixed-protocol-sanitize` | 182/182 | 132.9 秒 | 同上 |
| `e2e` | 185/185 | 31.6 秒 | `e2e-frozen/attempt-1792019741661207010/` |

本模块重新编译固定 CANN AIC/FD/Vector 与 workspace、FA3 accumulator 片段，独立枚举 mask 可见行、任务/槽归属、原 Q 行及矩形 KV 地址。最大实际 CPU 输入 B=262146、Sk=16384、totalQ=2；v22 记录 10,488,872 字节，最后 KV 起点超过 uint32。还覆盖宽 M/任务号、精确 accumulator 字节边界（含空 KV）、非法/陈旧描述符、损坏端点、交叉版本及失败原子性。普通/sanitizer 的 plan、metadata、CANN、FA3 驱动均正常退出且 stderr 为空。

新源码完整构建 533.7 秒、实际加载检查 20.1 秒均通过，见 `e2e-frozen/attempt-1792019741661207010/`。实际加载同一仓库扩展路径，最终 SHA-256 `9a4871c31d1eadb6c0d915798165dd3cd264843a9167f15fdaa95457065c43d2`；生产输入、6,392 个依赖和 12 个产物匹配当前源码。最终 `evidence-final.json` 核验 7,430 个唯一文件、原始结果、CPU 固定源/工具链、probe、构建和加载库，差异为 0。

185 项累计 E2E 与前步逐项相同，162 个既有显式组命令未改，见 `selection-audit.json`。本 CPU 步没有新公开行为，不注册伪自动 E2E。E2E 数学缓存 **337 次命中、0 次新计算**，待测实现实际执行；没有用旧无 mask 证据替代新协议验收。

## 复现与开发记录

以下命令均先 dry-run/collect-only；运行使用宿主机非登录 shell，新运行使用新的 output 目录：

```bash
python3 -m tests.mla --suite auto-split-mixed-protocol-tools auto-split-mixed-protocol-check --output /tmp/mla-m10c6a/check-dev
python3 -m tests.mla --suite auto-split-mixed-protocol-tools unit-auto_split_mixed_protocol auto-split-mixed-protocol-sanitize --output /tmp/mla-m10c6a/cpu-frozen
python3 -m tests.mla --suite e2e --device 6 --output /tmp/mla-m10c6a/e2e-frozen
```

开发代表/成本检查 15/15、39.8 秒通过；首次完整普通组 181/181、40.3 秒通过，复核后补充空 KV accumulator 精确边界，最终普通/sanitizer 182 项全跑。开发和旧源码结果不计最终通过或继承；本 CPU 模块不执行浮点 attention，无新增数学 reference，E2E 继续复用内容缓存并实际执行待测算子。

开发时新增 suite 曾插入错误的同后缀集合，dry-run 报语法错误，已修正后通过工具检查；sandbox 内 E2E 收集因 NPU 不可见失败，改宿主机收集通过。质量工具初次缺少 PYTHONPATH、随后所选缓存缺 actionlint 环境，均未绕过检查；使用现有 `/tmp/mla-m01-precommit` 与完整 `/tmp/mla-m01-precommit-cache` 后通过，见 `quality-fixed.log`。质量失败原始日志保留在 `quality-dev.log`、`quality-source.log`，收集失败保留在 `collect-dev/`；均不计验收通过。初始登录 shell 延迟，已全部正常退出。

相关 design/plan/status、v22 表格和链接已核对，最终记录见 `doc-links-final.json`；差异及适用仓库质量检查见 `quality-final.log`，使用既有完整 pre-commit 环境，未绕过钩子。收尾进程及设备状态见 `processes-final.json`、`npu-final.log`，本轮任务全部退出且 6 号卡空闲。未运行混合自动设备数值/公开性能/Trap、PA split、普通 FA3 专项、其他历史模块单元、完整矩阵或 910C 实机；这些不计本次通过或继承。
