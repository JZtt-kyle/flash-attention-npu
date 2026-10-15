# MLA 当前状态

M10c8b **BSND Q + ND PA KV 原生自动 split 生产接入已完成**：公开 `num_splits=0` 使用 v24、固定 CANN 原生变长分区及 init/partial/merge；支持 FP16/BF16、无 mask/causal/window、原 allocation/Q-used、页与四路 stride、metadata 复用及真实 accumulator。TND PA split 仍明确拒绝。本步基线 `0dacc51d7061fcaba5fa69c79c04593f6d225ff1`，分支 `feat/mla-cann-migration`；当前闭环提交可用 `git log -1 -- docs/mla/status.md` 定位。

唯一功能下一步为 [M10c9a：TND PA 显式 split 计划与安全协议](plan.md#唯一下一步-m10c9atnd-pa-显式-split-计划与安全协议)。本步不开始该任务，也不作阶段收尾；无头循环按已确认规则另行安排到期文档修正轮。

## 实现与边界

- v24 AICPU 和双 dtype 独立三阶段 launcher 接通 host、公开 cache/metadata、通用 raw/custom op 与 fake/compile；复用 PA partial/Copy、原生 AIC/FD 查询和 FD Block，不调用原生 CANN attention。
- 生产预算在 FD 持久槽后增加独立 Direct 临时槽，公开 accumulator 固定为 FP32 `[26,B,H,Qcap,512]` / `[26,B,H,Qcap]`，实际分区按物理 AIC 数。各阶段在写入/同步前检查完整记录及当前有效页；原 Q 分配行保持不变，空容量/未使用行初始化 `O=0、LSE=-inf`，accumulator 为 `0/-inf`。
- 所属单元、公开/Trap/性能专项和日常 E2E 已注册。累计小输入清单由 215 项增至 229 项；新增 14 项覆盖本模块三 mask、双 dtype、空值及 fake/compile，未按固定数量填充或截断。CANN/CATLASS 未修改。

## 环境与证据身份

- Python 3.11.15、torch 2.7.1+cpu、torch_npu 2.7.1；CANN `/usr/local/Ascend/cann-9.0.0-beta.2`，CATLASS `769cd40a8716b28650b6bebb08db4834eea4462f`。实机为 6 号 Ascend910B3，实际 20 AIC；返回容量 26 不代表物理核数。
- 实际加载本仓 `flash_attn_npu_3/flash_attn_npu_3.cpython-311-aarch64-linux-gnu.so`，SHA-256 `dd1e0745d29ea659d83909fb7148c29c9b64dba40071a03990caba9d669c0f78`。生产扩展已重建；后续按完整输入/依赖/产物哈希复用。环境及加载记录在各 attempt 的 `runtime.json`/`loaded.json`。
- 性能显式设置 `MLA_NATIVE_ROOT=/tmp/mla-m10c3b/native`：固定 CANN `ca18aee610729ca5270a44e2d84c751bc687fbf0`，当前配方、输入声明及 2,027 个包文件有效（`native-audit.json`）。默认 native 目录是旧配方包，不可作为本步证据。
- 本步结果根 `/tmp/mla-m10c8b/`；构建/数学内容缓存沿用 `/tmp/flash-attention-npu-mla-46928a793876/mla-validation-artifacts/`。`build/` 仅有 `mla-codex-loop/` 日志与状态；原始证据缺失后不可继续继承。
- 接手复核上步 7,104 个原始证据/依赖文件（`takeover.json`），恢复时 HEAD 未变且卡 6 空闲。本步生产源码/构建改变，**继承 E2E 为 0**；下列均为本步实际执行，不累计历史单元或继承性能。独立循环修复的已有改动保留，不纳入本提交。

## 本步验收

全部测试通过 `python3 -m tests.mla`，选择均已 dry-run/collect-only；没有 `--max-cases` 截断。表中耗时包含统一驱动的该组执行与结果检查。

| 检查 | 结果 | 耗时 | 结果子目录 / attempt |
| --- | --- | --- | --- |
| 所属 protocol CPU | 420 passed | 44.9 s | cpu-final/attempt-1792035120695968580 |
| 同集合 ASan/UBSan | 420 passed | 147.6 s | sanitize-final/attempt-1792035124516212300 |
| protocol 工具 | 14 passed | 8.9 s | sanitize-final/attempt-1792035124516212300 |
| 新模块工具/进程隔离 | 26 passed | 7.2 s | tools-exec/attempt-1792039313913419540 |
| 固定源 PA/原生 FD 与双数学 | 12 passed | 228.2 s | acceptance-cores/attempt-1792038650808043040 |
| 公开专项 | 80 passed | 102.8 s | acceptance-cores/attempt-1792038650808043040 |
| release Trap | 40 passed | 530.8 s | trap-exec-regression/attempt-1792039328781886560 |
| 累计小输入 E2E | 229 passed | 49.2 s | trap-exec-regression/attempt-1792039328781886560 |
| ordinary FA3 | 516 passed | 28.5 s | trap-exec-regression/attempt-1792039328781886560 |
| 原生自动公开性能 | 1 汇总用例、12 配置通过 | 285.2 s | profile-native/attempt-1792040165109145570 |

各最终组均 0 failed、0 skipped。前置代表/最大 accumulator/最长 KV 成本检查 10 项通过，随后被完整公开组覆盖。源数值 12 项的 O/LSE 与固定源逐元素一致，真实 accumulator、双数学及 canary 均通过，无源精度例外。Trap 每例用全新解释器/设备 context，40 项均在预期事件阶段失败，PID/退出/超时和加载身份检查通过。

数学缓存：固定源组 10 hit / 2 个新输入，公开组 21 hit / 46 个新输入，累计 E2E 374 hit、ordinary 31 hit，后两组 0 重算；性能诊断只为 12 个新配置各计算一次 reference，采样复用已验证输入/精度证明。待测实现始终实际执行。

### 公开性能

预先固定 6 个输入 × 双 dtype，隐式生成、显式生成加前向、显式复用各 R=1/4；5 次预热、5×20 wall、10 次 profiler，CPU affinity `[0]`，卡 6 隔离采样。双方必要分配及原生 ND PA 页表/cache 转换计入公开调用；AICPU/init/partial/merge、原生分派、逐例和分段原始数据均保存。

全部 12 个配置适用，包含 8 个原生 FD 配置，无排除项。配置内/配置间等权平均比 **0.852969 ≤ 1**；4/12 配置、25/72 分段退化，按约定记录且不单独阻塞提交。分段平均依次为 implicit R1/R4 `0.926921/0.830605`、generate R1/R4 `1.054680/0.793031`、reuse R1/R4 `0.797773/0.714805`。完整结果在性能组的 `public-profile/average.json`、双 dtype `profile.json`、dispatch、wall 和 profiler CSV；未按耗时选择输入。

## 恢复、失败与收尾

本轮恢复未发现已完成提交；原失败记录保留。CPU 首轮溢出用例输入仍可表示，修正为真正越界后重跑 420 项；首轮构建期间测试源变化使外层验收失败，随后完整核验构建复用。源数值首轮把静态返回容量 26 当作物理 AIC 数，改为从已校验 metadata 读取实际 20 核后完整重跑。fork 优化被 seed/包导入的 NPU 初始化断言拦截，改为全新解释器并重跑工具与全部 40 个 Trap；没有复用失败 context。性能首轮仅因默认包配方过期而在采样前失败，核验有效包并显式选择后完整重跑；失败/未执行组未转记为通过。

`preclose-evidence.json` 与 `final-evidence-audit.json` 核验原始证据、构建/依赖/实际加载及 native 包/性能归档共 10,357 文件，无差异或哈希冲突；双 dtype 各 90 个性能归档文件完整。通过数值组之后仅独立 Trap 工具改变，未改变其生产实现、数学函数或源 oracle；CPU 后续快照中的设备参考/无关工具归属变化不影响 CPU 集合。`selection-completion-final.json` 核对实际节点与原收集清单，`selection-audit.json` 保存新增/旧清单差异依据。相关契约同步至 design，plan 仅推进唯一功能下一步；适用 `git diff --check` / `ci/code_quality.sh --staged` 的日志为 `quality-final.log`，481 个本地链接及 151 个锚点检查见 `doc-links-final.json`，最终进程/设备记录为 `processes-final.log`、`npu-final.log`。

复现（按需选择本模块单元；日常只运行累计 E2E）：

```bash
python3 -m tests.mla --unit auto_split_paged_bsnd_protocol --output /tmp/mla-m10c8b/repro-cpu
python3 -m tests.mla --suite auto-split-paged-bsnd-protocol-sanitize auto-split-paged-bsnd-protocol-tools auto-split-paged-bsnd-tools --output /tmp/mla-m10c8b/repro-tools
python3 -m tests.mla --unit auto_split_paged_bsnd --device 6 --output /tmp/mla-m10c8b/repro-source
python3 -m tests.mla --suite auto-split-paged-bsnd-public auto-split-paged-bsnd-trap e2e ordinary --device 6 --output /tmp/mla-m10c8b/repro-public
MLA_NATIVE_ROOT=/tmp/mla-m10c3b/native python3 -m tests.mla --suite auto-split-paged-bsnd-profile --device 6 --output /tmp/mla-m10c8b/repro-profile
```

先对所选命令加 `--dry-run`、`--collect-only`；不得以截断代替完整集合。上述原始证据和依赖完整有效时，后续仅按规则继承 E2E 正确性。
