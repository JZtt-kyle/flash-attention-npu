# MLA 当前状态

当前分支 `feat/mla-cann-migration`，本步起点 `90d7dac`。**M10c4b2 TND 原生自动 split 生产接入已完成。** 当前闭环提交用 `git log -1 -- docs/mla/status.md` 定位；唯一下一步为 [M10c5 混合布局显式 split](plan.md#唯一下一步-m10c5混合布局显式-split)，本次提交后立即停止。

## 当前能力与边界

非 PA TND Q/KV 已开放 `num_splits=0`：v20 AICPU 与独立设备消费、共享 init/partial/merge 的 TND 特化、原生 AIC/FD 端点和 8 行 Vector 归约、紧凑 host 预算、原 allocation token 地址以及静态 26 片真实 FP32 accumulator。覆盖无 mask、causal、window、双 allocation/used、四路 stride、real/fake/compile、metadata 复用和流生命周期；全局 totalKV=0 与仅 used KV=0 保持各自空值语义。契约见 [design 6.2.20](design.md#6220-tnd-自动-split-生产适配)。

混合和 PA split 仍明确拒绝；本步未修改 CPU 计划/协议算法、CANN 源仓或 CATLASS，不代表 M10 阶段结束。新增本模块单元、公开/Trap/性能和 12 个日常 E2E 节点；前步 161 节点全部保留，累计 173 项，无数量截断或按耗时筛选。

## 环境、构建与证据身份

接手工作树干净，最近提交与 M10c4b1 一致，宿主机无前轮遗留 MLA 测试/构建，6 号 910B3 空闲。前步 E2E 的 13 个原始文件、1,112 个源码、生产 6,372 个依赖/12 个产物和 230 个加载库，共 7,408 个唯一文件核验无差异，见 `/tmp/mla-m10c4b2/takeover.json`。本步源码和生产身份改变，累计 E2E 实跑、继承 0；不累计历史单元或性能。

Python 3.11.15、torch 2.7.1+cpu、torch_npu 2.7.1、CANN `/usr/local/Ascend/cann-9.0.0-beta.2`；CATLASS `769cd40a8716b28650b6bebb08db4834eea4462f`。接手旧扩展 SHA-256 `459f487db89b9a3bb06369d49fc9b6f16adb2245426deefc9c338107fa62079a` 不计本步证据；已用 514.2 秒重建，22.3 秒核验实际加载。最终各设备组均加载仓库 `flash_attn_npu_3/flash_attn_npu_3.cpython-311-aarch64-linux-gnu.so`，SHA-256 **`fecd251a8638cd2b8ff9f40e15c6b1a941294a9ae4200017b617606cd72a3fe2`**。

证据根 **`/tmp/mla-m10c4b2/`**；构建/reference 使用仓库外默认 `/tmp/flash-attention-npu-mla-46928a793876/mla-validation-artifacts/`，原生包 `MLA_NATIVE_ROOT=/tmp/mla-m10c3b/native`。`build/` 仅保留 `mla-codex-loop`。原生分派、实际加载、profile/wall 样本及源部分缓冲均保留在各组目录。最终原始证据 971 个、源码快照 1,128 个，加上生产/探针/原生依赖和加载库，共 **8,434 个唯一文件**核验无差异，见 `evidence-final.json`；各组实际节点与 collect-only 完全一致。

## 本次验收

所有下列组本次实际执行，失败/跳过均为 0。先 dry-run/collect-only，工具与代表/最大成本检查后固定实现，再执行最终集合；最终验证期间只整理文档，未修改相关源码。

| 组 | 通过 | 耗时 | 最终证据目录（相对证据根） |
| --- | --- | --- | --- |
| `auto-split-tnd-tools` | 29/29 | 9.0 秒 | `tools-final/attempt-1792011844623937670/` |
| `unit-auto_split_tnd` | 10/10 | 143.8 秒 | `unit-final/attempt-1792011892876204440/` |
| `auto-split-tnd-public` | 59/59 | 50.7 秒 | `device-final/attempt-1792012066091031850/` |
| `auto-split-tnd-trap` | 38/38 | 509.3 秒 | 同上；独立错误子进程 |
| 累计小输入 `e2e` | 173/173 | 30.4 秒 | 同上 |
| 普通 FA3 `ordinary` | 516/516 | 25.8 秒 | 同上 |
| `auto-split-tnd-profile` | 1/1 | 275.4 秒 | `performance-final/attempt-1792013049999206540/` |

固定源单元新编译原生 AIC/FD 分区并实际执行未修改的 CANN partial/FD 核：双 dtype 的 O/LSE 均逐元素一致，双数学通过；真实 accumulator O 逐元素一致，accumulator LSE 按 `rtol=atol=1e-6` 对照源统计量转换，并独立检查双数学。具名用例覆盖三种 mask、Direct/FD/Empty 混合、非零起点、跨 M/尾行、长 KV、Direct 临时槽复用和原 allocation 间隙，无新增源精度限制。

公开专项包含双 dtype 代表、长度/stride/零容量、max 非零的空 tensor、静态 accumulator 溢出、所有实际入口、metadata 更新/复用、fake/compile 和 stream/allocator 生命周期。Trap 用例在非空场景先确认存在真实原生 FD，覆盖 v20 身份、预算、端点/Vector/片数损坏及非法长度，保留 release Trap 原始日志。工具组验证组选择、失败/身份失效、旧两阶段兼容、新三阶段识别、原生 FD 纳入平均及无效证据拒绝。

开发代表/成本组 `auto-split-tnd-check` 为 8/8、119.6 秒，见 `check/attempt-1792011128351602040/`。最大成本输入 Q=(1,2)、KV=(16384,8193)、H=128，FP16/BF16 冷缓存用例分别 35.9/47.1 秒，返回 accumulator 共 20,487,168 字节；这 8 项也包含在最终公开 59 项中。其后只修正测试错误信息匹配和 Trap 输入，开发快照不作为最终 E2E 继承证据；相同数学输入的内容缓存继续有效。

最终数学缓存：本模块单元 hit 6/miss 4，公开 hit 14/miss 33，累计 E2E hit 326/miss 0，普通 FA3 hit 31/miss 0，性能 hit 2/miss 12；各 miss 均成功写入。同一步重复输入不重算 reference，待测实现实际执行。未运行历史模块单元/sanitizer、穷举矩阵、910C 实机或 M10 阶段全组合；它们不计本步通过或继承。

## 公开性能与差距

空闲 6 号 910B3 隔离采样，固定 7 个输入 × FP16/BF16，共 14 配置全部实测。保持 R=1/4、5 次预热、5 轮各 20 wall/10 profiler 样本及完成同步，原生必要 packing/restoration 计入公开 wall。12 个原生 MLA template 5 配置适用，其中 8 个为原生 FD，全部纳入等权平均；2 个全空 used KV 的原生 template 0 Empty 配置单列，未按测得耗时删选。

平均 FA3/CANN 耗时比 **0.794557 ≤ 1**；R=1 为 0.841721，R=4 为 0.747393。适用配置退化 2/12、分段退化 6/24；退化配置为 Q=(1,3)、KV=(16384,17)、H=2，FP16/BF16 配置均值分别 1.341546/1.297686。两个 Empty 配置的 R=1/4 比值分别为 FP16 0.591034/0.526600、BF16 0.567492/0.519467。所有单例和分段保留在 `auto-split-tnd-profile/public-profile/average.json` 及双 dtype `profile.json`。

三阶段 Kernel 总耗时另列，适用配置 FA3/CANN 比为 3.216–9.555；公开 wall 通过不代表设备核已追平。原始 AICPU/init/partial/merge 与原生 task、实际分派和样本均已核验，未用 Kernel 时间代替公开 E2E 门槛。

## 复现、质量与已处理问题

使用宿主机非登录 shell；下列命令均以仓库为 cwd，先追加 `--dry-run`、再 `--collect-only` 核对命令/集合。新结果放独立输出目录，不覆盖上述验收证据：

```bash
export MLA_NATIVE_ROOT=/tmp/mla-m10c3b/native
python3 -m tests.mla --suite auto-split-tnd-tools --output /tmp/mla-m10c4b2-repro/tools
python3 -m tests.mla --suite auto-split-tnd-check --device 6 --output /tmp/mla-m10c4b2-repro/check
python3 -m tests.mla --unit auto_split_tnd --device 6 --output /tmp/mla-m10c4b2-repro/unit
python3 -m tests.mla --suite auto-split-tnd-public auto-split-tnd-trap e2e ordinary --device 6 --output /tmp/mla-m10c4b2-repro/device
python3 -m tests.mla --suite auto-split-tnd-profile --device 6 --output /tmp/mla-m10c4b2-repro/performance
```

初次工具注册的语法错误在质量检查中发现，修正后工具及最终检查通过。沙箱 NPU 收集因无法枚举设备而报 device_id=None，改宿主机收集成功；失败目录 `acceptance-collect/` 保留，不计通过。登录 shell 初始化挂起，改非登录 shell；终止四个挂起 PID 的请求曾被自动审批拒绝，之后只读复核确认这些 PID 已退出，未影响其他任务。首次证据审计误按 manifest 所在目录寻找继承的 oracle 头，改按记录的实际 partial/缓存库路径核验后通过；原审计保留为 `evidence-draft.json`，没有重算或改写测试通过证据。

本步相关 design/plan/status、链接/表格和完整差异已核对，`git diff --check` 及仓库适用 pre-commit 检查通过；只暂存本步文件，不绕过已有提交钩子。质量记录见证据根的 `quality-final.log`，最终链接记录见 `doc-links-final.json`。收尾只读进程检查仅发现既有无头循环，无遗留 MLA 验收/构建子进程，见 `processes-final.json` 与 `npu-final.log`。
