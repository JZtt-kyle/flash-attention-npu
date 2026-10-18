# MLA 当前状态

<!-- mla-loop-next: M19 -->

**M18 已完成；唯一下一步为 M19：BSND Q + ND PA INT8 完整公开前向。** 本轮在 `feat/mla-cann-migration`、接手 HEAD `c061bdb` 上交付独立 INT8 数值与 ND 适配组件。公开 INT8 的 real/fake/compile 仍明确拒绝，不能称为 M19 已接通。M18–M21 已获授权；本步单次提交后停止。

## 本步产物与边界

新增 INT8 内容 / BF16 RoPE 与输出 / FP32 scale 的 Kernel、Block、Tile，迁移固定源 QK、动态 P 量化、PV、普通/快速 FD，并直接读取 ND 页。复用已有页地址校验与 ND→NZ 片上拷贝，增加单字节存储校验和 INT8 C0=32 回退；旧双字节入口保持。CANN checkout 与 CATLASS 未改动，生产路径不转调 CANN。

固定源实际边界为 Hq=1/2/4/8/16/32/64/128，BSND Sq 与 TND 每段 Q 最大 32；33 拒绝。已验证无 mask、多 Q 右下 causal、页尾、实际空 KV、TND 空 Q 段及两种 FD。源 NZ 转换只用于对照测试；具体来源、长度初始化、空值同步及 LSE 适配见 [design 3.4](design.md#34-int8-压缩-mla-扩展)。原始 tiling 是测试夹具和私有数值描述，不是生产 metadata；TND 组件 LSE 仍为源 `[Tq,Hq]`，公开布局适配属于 M20。

## 最近闭环证据

证据根 `/tmp/mla-m18/`。下列均为本轮实际执行，无 E2E 继承、无最终失败或跳过；单元与性能未继承历史通过。统一驱动的实际节点、selection、依赖/构建 manifest 和原始证据均保留。收尾通过驱动已有 `validate_result`、`reusable` 及 `matching_files` 核对，摘要为 `closure-evidence.json`。

| 统一入口组 | 本次结果 | 耗时 | summary 路径（相对证据根） |
| --- | --- | --- | --- |
| `int8-tools` | 45 通过；CPU 固定源地址差分、存储/溢出、ASan/UBSan、夹具/构建/驱动失败与失效检查 | 28.6 s | `tools-final/summary.json` |
| `int8-source-check` | 36 通过；22 组源几何、13 项拒绝及 RoPE 缩放辨别 | 29.6 s | `source-final/summary.json` |
| `--unit int8` | 131 通过；56 数值/stride/量化组件、3 批 CPU 边界、36 源、32 工具、4 独立 Trap | 141.3 s | `unit-empty/summary.json` |
| `int8-profile-check` | 24 个固定性能配置预检通过 | 22.5 s | `profile-preflight/summary.json` |
| `int8-component-profile` | 24 配置采样及原始 trace 协议完整；性能差距见下 | 71.6 s | `performance/summary.json` |
| `e2e` | 879 通过；含 6 项 INT8 公开拒绝边界 | 219.4 s | `regression/summary.json` |
| `ordinary` / `graph-ordinary` | 504 / 2 通过 | 34.9 / 21.9 s | `regression/summary.json` |

本步生产重建耗时 588.0 s，运行环境核对 20.8 s，随后才运行回归。dry-run/collect 记录在 `tools-final-collect`、`unit-empty-collect`、`final-int8-collect` 和 `regression-collect` 等目录。数学缓存复用独立于通过证据：完整 INT8 组 50 次命中、28 次新计算；E2E 1083 次命中、148 次新计算，待测实现均实际执行。CPU sanitizer 合批保留所有边界及泄漏检查；模块组约 79 s 用于四个隔离 Trap 子进程，不进入日常 E2E。

56 组组件 O 均与固定源 probe 及真实 native 默认端点逐元素一致；非空 LSE 严格源对照、空行 0/-inf 和真实 FD partial 检查通过。独立 FP32/FP64 数学的 O 为 **21 通过、35 源限制**，LSE 为 **10 通过、46 源限制**，未把超限算作数学通过。源将内容与 RoPE 点积合并后整体乘 Q/K scale：`rope-scale-order.json` 的受控结果明确区分该语义与独立反量化公式。输入、双 reference、源/目标 O/LSE、误差及原始 stride 在最终模块组的 `.pt/.json` 中；源快速 FD 缺少 LSE，比较使用相同真实 partial 上的固定源普通归并。

24 配置的目标/源平均 Kernel 比为 **1.1450877174**（23 项退化），预分配 probe wall 比为 **1.0181407505**（21 项退化）；direct 20 配置为 1.1591949453 / 1.0170355163，原生 FD 4 配置为 1.0745515778 / 1.0236669210。端点是 O-only、同原生分区的源 NZ 与目标 ND 单 Kernel；目标 ND 转换和必要校验在计时内，源 NZ 准备耗时单列。全部配置、样本、分段、设备/CPU 隔离信息在 `performance/attempt-1792358984629355250/int8-component-profile/profile/`；私有 tmpfs 的 93 个文件已归档并核对哈希。这不是公开 E2E，不宣称已达平均不劣化；M19/M20 仍必须分别验收公开 eager/graph 平均 ≤1。

定位期间曾有失败，均已处理：早期源 probe 长 TND FD 的未绑定长度访问见 `component-first` 等日志；空 KV 同时初始化源 O/LSE 的 stack UB 竞态见 `unit-final`，目标增加完成同步，源 O/LSE 分别对照并保留其空值缺陷记录；TND Q=33 已按实际源拒绝纳入边界测试。native 重建首次下载依赖受沙箱网络限制，`native-host-build-network-failure.log` 保留，授权环境重试 `native-build-formatted-retry.log` 成功。这些诊断失败不冒充通过，最终结果以上表为准。

## 环境与构建身份

Python 3.11.15、torch 2.7.1+cpu、torch_npu 2.7.1；CANN `/usr/local/Ascend/cann-9.0.0-beta.2`；实测 910B3 物理设备 7，20 AIC/40 AIV。冻结包装器 `/tmp/mla-m16-schedule/run.py` 恢复正确库路径；910C 未实机验收。CATLASS `769cd40a8716b28650b6bebb08db4834eea4462f`，CANN 固定源 `ca18aee610729ca5270a44e2d84c751bc687fbf0`；源 checkout HEAD 不同，native/probe 只读固定 archive。

- 实际加载扩展：`flash_attn_npu_3/flash_attn_npu_3.cpython-311-aarch64-linux-gnu.so`，SHA256 `2adfe6e4b8308dccf1bd5794b4175dd9c1a3a41111ce55fe8b2bd1e40017999e`。回归 attempt 为 `1792359105965091850`，其中 `runtime.json`、`build-manifest.json` 和各组 `loaded.json` 绑定实际产物。
- 生产构建/数学缓存：`/tmp/flash-attention-npu-mla-46928a793876/mla-validation-artifacts`；独立 INT8 DSO 在同级 `int8-probes/`。目标 DSO 的已验收 SHA256 为 `b9bdde8e5e441b9df34e66666eaf109c119b55e035b351115cd035b1b44773f4`，源为 `1060d5bfd9ac1c4a684edc80bfede04c8ba2220b5913f5318ba34f3269909e9a`；编译器/头文件见其 `build-manifest.json`。
- INT8 native：`/tmp/mla-m18/native`，12 个 key、INT8/BF16 混合签名和完整包 hash 已核对；旧非量化 `/tmp/mla-m13b/native` 保留。有效原始调度夹具为 `/tmp/mla-m18/source-final/attempt-1792357915791404830/int8-source-check/output.log`，绑定输入生产者签名。

验收后仅修正三个新头文件开头的来源注释，其中快速 FD 保留源实际的 2026 版权年份；`notice-only-correction.json` 保存前后声明和 hash，确认 `#pragma once` 之后的全部代码逐字节不变。按 AGENTS 的纯注释规则未重跑 NPU，也未改写旧 manifest：上述执行证据基于注释修正前的相同代码正文。下一步统一驱动若因保守文件身份要求重建或拒绝继承，应照常执行，不能手工伪造缓存命中。`ci/code_quality.sh --staged`、`git diff --check` 及本步文档链接核对均通过，质量日志为 `/tmp/mla-m18/quality-final.log`。

M17 的非量化图性能例外仅属于原闭环 `ebdf3eb`，证据可从该提交的 status 定位；不能用它豁免 INT8。

## 复现与唯一下一步

```bash
MLA_INT8_NATIVE_ROOT=/tmp/mla-m18/native \
  python3 /tmp/mla-m16-schedule/run.py python3 -m tests.mla \
  --suite int8-source-check --device 7 --output /tmp/mla-m18/source-recheck
# 源组通过后，把 MLA_INT8_SOURCE_LOG 指向该成功 attempt 的 int8-source-check/output.log。
# 同环境执行 --unit int8，再显式执行 --suite int8-profile-check 和 int8-component-profile。
python3 /tmp/mla-m16-schedule/run.py python3 -m tests.mla \
  --suite e2e ordinary graph-ordinary --device 7 --output /tmp/mla-m18/regression-recheck
```

继续 [M19](plan.md#m19bsnd-公开-int8)：同一提交接通 BSND 的 host/AICPU/Kernel、metadata、公开 real/fake/compile、真实图及必要性能，稳定 causal/FD 一并接入；按实际开放范围调整 M18 的公开拒绝清单。不得继承历史单元或本轮组件性能代替新入口验收。当前闭环提交可用 `git log -1 -- docs/mla/status.md` 定位；无头入口仍为 `/home/zhangqijia/repo/mla_codex_loop.py`，不改写 pending 或成功次数。
