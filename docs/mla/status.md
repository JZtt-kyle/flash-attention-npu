# MLA 当前状态

当前分支 `feat/mla-cann-migration`，本步起点 `a02f194`。**M10c6b 混合布局原生自动 split 生产接入已完成验收。** 唯一下一步为 [M10c7a BSND PA 显式 split 计划与安全协议](plan.md#唯一下一步-m10c7absnd-pa-显式-split-计划与安全协议)；本次只提交 M10c6b，提交后立即停止。当前闭环提交可用 `git log -1 -- docs/mla/status.md` 定位。

## 本步产物与开放边界

非 PA TND Q + BSND KV 的 `num_splits=0` 接通 v22 安全 prefix/AICPU、共享 init/partial/merge、双 dtype 实例及 Python/C++ real/fake/compile。保留固定 CANN 每 M 的变长分片、FD 持久槽和 8 行 Vector 归约；Q/O/LSE/accumulator 使用原 allocation 地址，KV 使用独立 batch/token stride。真实 FP32 accumulator 为 `[26,H,totalQ,512]` / `[26,H,totalQ]`，静态 26 是资源上界，未用片为 0/-inf。

覆盖无 mask、右下 causal/window、Q allocation/used、实际 KV 长度、四路 stride、空容量、metadata 复用/失败恢复和流生命周期。公开 cache/metadata 保持 FA3 签名；内部 Q-used 每次生成，本步补齐共享混合显式/自动入口对 Q-used 与显式 metadata 同用的 host/fake 拒绝。PA split 仍明确拒绝，要求 `num_splits=1`；其他未迁移选项不变。契约见 [design 6.2.24](design.md#6224-混合原生自动-split-生产适配)。未修改 CATLASS 或 CANN 源仓。

## 环境与证据身份

接手工作树干净，HEAD 与交接一致；宿主机无前步遗留 MLA 验收任务，6 号 Ascend910B3 空闲，其他卡任务未改动。Python 3.11.15、torch 2.7.1+cpu、torch_npu 2.7.1、CANN `/usr/local/Ascend/cann-9.0.0-beta.2`；CATLASS `769cd40a8716b28650b6bebb08db4834eea4462f`。实际加载仓库 `flash_attn_npu_3/flash_attn_npu_3.cpython-311-aarch64-linux-gnu.so`，接手 SHA-256 `9a4871c31d1eadb6c0d915798165dd3cd264843a9167f15fdaa95457065c43d2`，本步重建后为 `c070e52e30d08c60357acc22c8c1df4bd60bc0b965af137fc1d5271175a0787b`。

证据根 `/tmp/mla-m10c6b/`。构建与数学缓存沿用仓库外 `/tmp/flash-attention-npu-mla-46928a793876/mla-validation-artifacts/`；`build/` 仅有 `mla-codex-loop`。接手核验前步 E2E 原始文件、源码、构建输入/依赖/产物及加载库，共 7,309 个唯一文件无差异，见 `takeover.json`。本步生产、共享发射和入口变化，累计 E2E 全部实际执行，继承 0；不累计其他模块历史单元。

## 本次通过

以下为源码冻结后的最终运行，均无失败或跳过，节点逐项匹配各自的 collect-only 清单。协议普通与 sanitizer 为同一 238 节点集合；工具和协议原始结果、源码、固定源/工具链及二进制已核验，1,097 个唯一文件无差异，见 `evidence-cpu.json`。最终 `evidence-final.json` 核验 10,485 个唯一文件、252 个固定 CANN Git blob、所有组原始结果、probe、生产构建/加载库、原生包及性能归档，差异为 0。

| 组 | 通过 | 耗时 | 证据目录（相对证据根） |
| --- | --- | --- | --- |
| `auto-split-mixed-tools` | 43/43 | 14.1 秒 | `cpu-frozen/attempt-1792021476929122910/` |
| `unit-auto_split_mixed_protocol` | 238/238 | 42.8 秒 | 同上 |
| `auto-split-mixed-protocol-sanitize` | 238/238 | 129.2 秒 | 同上 |
| `auto-split-mixed-check` | 8/8 | 291.7 秒 | `check-fixed/attempt-1792021480143988870/` |
| `unit-auto_split_mixed` | 10/10 | 142.6 秒 | `device-frozen/attempt-1792022386522407580/` |
| `auto-split-mixed-public` | 89/89 | 67.1 秒 | 同上 |
| `e2e` | 201/201 | 32.0 秒 | 同上 |
| `ordinary` | 516/516 | 25.7 秒 | 同上 |
| `auto-split-mixed-trap` | 38/38 | 515.7 秒 | 同上 |
| `auto-split-mixed-profile` | 1/1，14 个固定配置 | 297.1 秒 | `profile-native/attempt-1792023348412427310/` |

完整构建 524.5 秒、实际加载检查 22.1 秒通过，见 `check-fixed/`；正式设备组验证构建内容后复用，1.2 秒，实际加载 19.9 秒。781 个生产输入、6,401 个依赖和 12 个产物匹配当前源码。所属数值单元重新编译固定源 partial/FD 与原生 AIC/Vector 对照：10 个双 dtype 输入的 O/LSE 均逐元素一致且双数学通过，未出现源精度限制；真实 FP32 accumulator 同时满足源和数学检查。覆盖 Direct/FD/Empty、跨 M、长 KV 和 Direct 临时槽复用。

最大实际成本输入为 KV 容量 16384、H=257、Q 段 `(1,2)`，accumulator 41,134,392 字节；FP16/BF16 首次含 reference 分别 118.5/138.1 秒，正式公开组复用缓存。代表组新计算 8 次，源单元命中 6/新计算 4，公开组命中 22/新计算 44，累计 E2E 命中 348/新计算 0，ordinary 命中 31/新计算 0，性能命中 2/新计算 12；所有待测调用实际执行，同输入未重复计算 reference。

日常清单新增本模块的三 mask、空容量、fake/compile 和 Q-used/metadata 拒绝边界，共 16 项，原 185 项全部保留；累计 201 项，未按数量填充或截断。227 个既有显式 suite 命令未改变，见 `selection-audit.json`；混合显式专项移除已经开放的自动模式拒绝断言。本步新增自动专项与源对照，不用旧无 mask 测试充当能力验收。

38 个 release Trap 各在独立进程先完成合法调用，再于设备事件同步阶段捕获预期错误，包含空容量、交叉版本、原生 AIC/FD/Vector 端点、陈旧前缀/长度/窗口及非法 used。扩展身份均与本步构建一致；全部进程正常退出，没有把 host 描述符拒绝冒充设备 Trap。

## 公开性能

固定七种输入、双 dtype 共 14 配置全部实测；每配置等权平均隐式/生成/复用 metadata 各 R=1/4 的六段，再对适用配置等权平均。保持 5 次预热、5 轮×20 次 wall / 10 次 profiler、完成同步与 CPU affinity `[0]`，计入原生 packing/gather、分配和输出恢复。原生包 `/tmp/mla-m10c3b/native` 的当前配方、输入声明及 2,027 个文件核验通过；双方数学、固定源分派与 AICPU→init→partial→merge 阶段均通过。

| 原生分派类别 | 配置数 | 平均耗时比 FA3/CANN | 退化配置 / 分段 | 本步门槛 |
| --- | --- | --- | --- | --- |
| 模板 5，FD 与非 FD 合计 | 12 | **0.630502** | 0/12、4/72 | ≤1，通过 |
| 其中原生自动 FD | 8 | 0.683859 | 0/8、4/48 | 纳入上述平均 |
| 其中非 FD | 4 | 0.523788 | 0/4、0/24 | 纳入上述平均 |
| 空 KV 模板 0 | 2 | 0.522699 | 0/2、0/12 | 独立模板，单列 |

适用六段均值为 implicit R1/R4 `0.710437/0.594472`、generate `0.807328/0.557876`、reuse `0.614435/0.498463`。没有按耗时筛选输入；全部配置及分段原始结果保存在性能组的 `public-profile/`，门槛汇总 `average.json`，分类与退化计数见证据根 `performance-groups.json`。调试分派与正式计时分开；采样前无其他验收任务，6 号卡空闲，见 `npu-before-profile-native.log`。

## 复现与开发记录

以下组均已先 dry-run/collect-only；运行使用宿主机非登录 shell。新运行使用新的 output 目录：

```bash
python3 -m tests.mla --suite auto-split-mixed-tools unit-auto_split_mixed_protocol auto-split-mixed-protocol-sanitize --output /tmp/mla-m10c6b/cpu-frozen
python3 -m tests.mla --suite auto-split-mixed-check --device 6 --output /tmp/mla-m10c6b/check-fixed
python3 -m tests.mla --suite unit-auto_split_mixed auto-split-mixed-public e2e ordinary auto-split-mixed-trap --device 6 --output /tmp/mla-m10c6b/device-frozen
MLA_NATIVE_ROOT=/tmp/mla-m10c3b/native python3 -m tests.mla --suite auto-split-mixed-profile --device 6 --output /tmp/mla-m10c6b/profile-native
```

开发工具/代表协议检查通过后，复核发现共用混合入口缺少 Q-used/metadata 拒绝，主动中断旧构建并补齐真实/fake 与日常边界，再重新冻结和运行受影响 CPU/设备组。旧构建仅记录为中断；开发源码、首次格式化前的结果不计最终通过。一次 dry-run 插入位置错误导致语法错误，已修正；一次收集与格式化并行被驱动判定源码变化，已在冻结后重新收集。原始日志保留，不改写为通过。首次性能启动 `profile-frozen/` 因默认原生包配方过期在采样前失败；随后核验 `/tmp/mla-m10c3b/native` 的当前配方、输入声明和全部 2,027 个文件均有效，切换到该包实际重测，见 `native-package.json`。没有修改配方、放宽检查或继承旧性能结果。

收尾进程及设备状态见 `processes-final.json`、`npu-final.log`：本轮验收/性能进程全部退出，6 号卡空闲。相关文档、链接/表格及适用仓库质量检查记录在 `doc-links-final.json`、`quality-code-final.log` 和 `quality-final.log`；使用既有完整 pre-commit 环境，未绕过钩子。未运行 PA split、其他历史模块单元、完整矩阵、完整图重放或 910C 实机，这些不计通过或继承。本次提交后不继续下一步。
