# MLA 当前状态

当前分支 `feat/mla-cann-migration`，本步起点 `e7f9f7b`。**M10c3b TND Q/KV 显式 split 生产接入已完成。** 本步实现、必要验收与状态更新一并提交，提交后立即停止；唯一下一步为 M10c4a TND 原生自动 split 任务与持久布局。

## 本步产物与边界

新增 v19 CPU/AICPU 生成器、独立设备消费者及双 dtype partial/merge launcher，接入 M10c3a 的 TND 任务与紧凑持久预算。共享 task ring 的 TND allocation/used/stride 特化用于 partial；FD Block 以原 allocation token 写 O/LSE 和真实 `[S,H,totalQ,512]` / `[S,H,totalQ]` FP32 accumulator，保留 64 位地址及超 DMA stride 时逐头写入。

公开 varlen、通用 Python/C++ raw 和私有前向接通显式 S=2..256、三种 mask；公开 varlen 签名及二元返回不变，内部每次生成 v19。私有 metadata 可跨层/scale 复用并验证当次双前缀/used。TND 自动、混合和 PA split 继续拒绝。新增模块单元、公开 E2E、日常注册和固定公开性能驱动；CANN 源仓与 CATLASS 未改。相关契约见 [design 6.2.17](design.md#6217-tnd-显式-split-生产适配)，路径与完成条件见 [plan](plan.md#已完成-m10c3btnd-qkv-显式-split-生产接入)。

## 环境与证据身份

接手分支/提交与交接一致，工作树干净，宿主机无遗留 MLA 构建/测试，第 6 张 910B3 空闲。前步 E2E 的 13 个原始文件、931 个源码、生产 6,362 个依赖/12 个产物及 230 个加载库，共 7,230 个唯一文件核对无差异，见 `/tmp/mla-m10c3b/takeover.json`。

Python 3.11.15、torch 2.7.1+cpu、torch_npu 2.7.1、CANN 9.0.0-beta.2，CATLASS `769cd40a8716b28650b6bebb08db4834eea4462f`。生产构建 507.1 秒，最终构建缓存复核 1.1 秒、运行时检查 21.3 秒；实际加载仓库 `flash_attn_npu_3/flash_attn_npu_3.cpython-311-aarch64-linux-gnu.so`，SHA-256 `30c80f83cd76ee796c187e6d7922deedc47d8d61b54d4a9ae14efeae25cf152d`。

构建/reference 沿用仓库外默认 `/tmp/flash-attention-npu-mla-46928a793876/mla-validation-artifacts/`；本步证据根 `/tmp/mla-m10c3b/`。原生对照需设置 **`MLA_NATIVE_ROOT=/tmp/mla-m10c3b/native`**：新包保留原有 24 key，并为双 dtype 各加入固定源共用的空 KV key，源归档哈希不变，见 `native-key-check.json`。旧默认包保留原始内容，但不匹配新 recipe，不能继续用于验收。build 仅保留 mla-codex-loop。

共享数值模板、生产构建/发射及公共入口改变，本步累计 E2E 实跑、继承 0；不累计历史单元或性能。证据或原始文件被清理后不可继承。

## 最终检查

均已先 dry-run/collect-only。性能输入与 68/30/161/516 的最终节点清单未删减或截断，见 `selection-check.json`、`final-selection-check.json`；日常 E2E 保留原有 149 项，新增本模块 12 项。工具修复后实现再次固定，最终组期间只更新文档。

| 集合 | 最终结果 | 证据目录（相对证据根） |
| --- | --- | --- |
| `split-tnd-tools` | 21/21，7.2 秒 | `tools-symbols/attempt-1792006319601029840/` |
| `split-tnd-cpu` | 91/91，35.0 秒 | `cpu-symbols/attempt-1792006678139545510/` |
| `split-tnd-sanitize` | 91/91，113.2 秒 | 同上 |
| `split-tnd` | 68/68，170.3 秒 | `acceptance-symbols/attempt-1792006367482388540/` |
| `split-tnd-profile` | 1/1，267.2 秒，12 个配置完整实测 | 同上 |
| `split-tnd-trap` | 30/30，409.3 秒 | 同上 |
| 累计 `e2e` | 161/161，29.0 秒 | 同上 |
| `ordinary` | 516/516，25.9 秒 | 同上 |

固定源 10 份设备报告的 40 项 O/LSE 比较均逐元素一致且双数学通过；真实 accumulator 也经源部分结果与数学检查。193..256 按既定扩展口径验收，不声称原生 FD 支持该域。普通/sanitizer 三个驱动的输入、输出逐字节一致。代表集合包含最大 accumulator 138,682,368 字节，最终最大用例 10.3 秒；初次成本检查 8 份 reference 写入缓存，后续同输入未重复计算。最终专项 56 次、累计 E2E 314 次、ordinary 31 次缓存命中；性能 6 次命中、BF16 首次 6 次计算并写入。所有最终集合失败/跳过均为 0。

公开性能平均比值 **0.273009 ≤ 1**，R=1/4 分段平均 0.300431 / 0.245586，适用配置/分段退化均为 0。12 个配置全部保留；其中 2 个非 FD 配置纳入既定平均，8 个原生 FD、2 个空张量分派单列。双 dtype 共 4,800 个 wall 样本、1,200 组 profiler 样本，600 组完整 producer/partial/merge 明细；CPU 亲和性固定为 `[0]`，采样前后设备状态、原生分派、包/加载身份与全部单例保存在 `split-tnd-profile/public-profile/`，汇总 `average.json`，结构复核 `profile-check.json`。

完整证据共 10,321 个唯一文件逐项核验，无缺失或身份变化，见 `evidence-final.json`：包含 1,015 个原始文件、各组源码闭包、CPU/probe 产物与依赖、生产 741 个源码/6,372 个依赖/12 个产物、230 个加载库和原生包 2,027 个文件。实际执行集合与预先收集一致。

复现用新输出目录；先 dry-run/collect-only，再执行：

```bash
export MLA_NATIVE_ROOT=/tmp/mla-m10c3b/native
python3 -m tests.mla --suite split-tnd-tools --output /tmp/mla-m10c3b/repro-tools
python3 -m tests.mla --suite split-tnd-cpu split-tnd-sanitize --output /tmp/mla-m10c3b/repro-cpu
python3 -m tests.mla --suite split-tnd-profile split-tnd split-tnd-trap e2e ordinary --device 6 --output /tmp/mla-m10c3b/repro-device
```

原生包如需重建，用 `MLA_NATIVE_ROOT=<新目录> python3 -m tests.mla.support.native_package`，按新包实际路径更新环境；源仓仅做固定提交归档，构建发生在包目录。

## 失败处理与未运行

登录 shell 初始化挂起后改用非登录 shell。旧扩展缺算子造成首次 collect-only 失败，新构建后收集成功。CPU 开发用例曾错误地拒绝 FA3 允许归一化的负窗口、让溢出用例的文本前缀同时超 int32，均修正后完整重跑。沙箱 sanitizer 退出时报 LeakSanitizer/ptrace 限制，后续在宿主机按最终集合重跑；期间修改源码的开发结果未继承。

首轮性能预检因固定源包缺空 KV key 失败；补齐后第二轮 FP16 数值和采样完成，但汇总工具不识别 C++ 修饰符号而失败。分别保留在 `acceptance/`、`acceptance-final/`；增加包身份失败传播及两种符号格式的边界检查后，最终性能完整通过。固定性能输入、数学阈值、R、预热/重复与计时边界均未改变。工具修复使保守源码身份变化，最终必要组全部重跑，未把修复前通过项继承为最终结果。

pre-commit 使用 `/tmp/mla-m01-precommit`（`PYTHONPATH` 同路径）和 `/tmp/mla-m01-precommit-cache`；最终质量检查、相关文档链接和 `git diff --check` 通过。历史模块单元、full-matrix、910C 和 M10 阶段收尾未运行。

## 唯一下一步

[M10c4a：TND 原生自动 split 任务与持久布局](plan.md#唯一下一步-m10c4atnd-原生自动-split-任务与持久布局)。先独立闭环原生 AIC/FD 任务、紧凑总量和原 allocation 地址，再在后续步骤接通生产。当前提交不开始该步。
