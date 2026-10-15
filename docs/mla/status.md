# MLA 当前状态

当前分支 `feat/mla-cann-migration`。**M10c7b BSND PA 显式 split 生产接入已闭环。** 本轮基线为 `9e8e8498836b071194df9e4fe3c1b77866bd5e42`；闭环提交用 `git log -1 -- docs/mla/status.md` 定位。唯一下一步为 [M10c8a：BSND PA 原生自动计划与安全协议](plan.md#唯一下一步-m10c8absnd-pa-原生自动计划与安全协议)，本轮不开始该步。

## 已开放能力与边界

BSND Q + ND PA KV 的 512+64、FP16/BF16、显式 `num_splits=2..256` 已接通 v23 AICPU、静态预算、PA partial/merge、双 dtype 生产分派、Python/C++ raw、公开 cache/metadata 及 real/fake/compile。支持无 mask、右下 causal/window、Q allocation/used、实际 KV 长度和合法 Q/qv/cache/页表 stride，返回真实 FP32 `[S,B,H,Sq,512]` / `[S,B,H,Sq]` accumulator；跨波保持独立持久槽，静态空容量经校验后写入 0/-inf。每阶段写入或同步前校验当前记录、长度和有效页。详见 [design 6.2.26](design.md#6226-bsnd-pa-显式-split-生产适配)。

公开签名不变；PA 自动 `num_splits=0` 和 TND PA split 继续明确拒绝。内部 Q-used 不与显式 metadata 同用。保留既有单 split、三种非 PA 布局显式/自动和普通 FA3 行为。未扩展到 192/128、其他调度选项或完整图重放；CANN/CATLASS 未修改。

## 环境与证据身份

- Python 3.11.15、torch 2.7.1+cpu、torch_npu 2.7.1、CANN `/usr/local/Ascend/cann-9.0.0-beta.2`，6 号 Ascend910B3。CATLASS `769cd40a8716b28650b6bebb08db4834eea4462f`。
- 实际加载 `/home/zhangqijia/repo/flash-attention-npu/flash_attn_npu_3/flash_attn_npu_3.cpython-311-aarch64-linux-gnu.so`，SHA-256 `4c10918eb6ac3062a88e63e508449a6a40b082357cc3ae00aa92b84f9edfe7cc`。本轮源码重建及加载检查通过，后续构建缓存按内容校验。
- 结果根 `/tmp/mla-m10c7b/`；构建和数学缓存 `/tmp/flash-attention-npu-mla-46928a793876/mla-validation-artifacts/`。`build/` 只保留 `mla-codex-loop/` 日志、锁与恢复状态。证据位于 `/tmp`，缺失后不可继承。
- 原生性能明确使用 `MLA_NATIVE_ROOT=/tmp/mla-m10c3b/native`。固定 CANN `ca18aee610729ca5270a44e2d84c751bc687fbf0`，当前构建配方和 2,027 个包文件均有效；默认 native 目录的旧配方包未使用。

## 本轮实际验收

所有测试经 `python3 -m tests.mla`，先 dry-run/collect-only；下表各组均 0 failed、0 skipped。单元均为本轮实际执行，不继承历史单元。共享源码/扩展变化，本轮累计 E2E 实跑，历史 E2E 继承 0。

| 组 | 结果 | 耗时 | 证据目录（相对结果根） |
| --- | --- | --- | --- |
| split-paged-bsnd-tools | 23 passed | 9.5 s | tools-profile-fix/attempt-1792031379337280900 |
| unit-split_paged_bsnd_protocol | 337 passed | 42.0 s | cpu-final/attempt-1792027508688759100 |
| split-paged-bsnd-protocol-sanitize | 337 passed | 138.4 s | 同上 |
| 新扩展构建 / 加载 | passed | 522.3 / 22.2 s | check-fixed/attempt-1792027472727387580 |
| split-paged-bsnd-check | 8 passed | 67.1 s | 同上 |
| split-paged-bsnd-public | 82 passed | 121.2 s | device-public-fixed/attempt-1792028656832153500 |
| unit-split_paged_bsnd | 12 passed | 204.7 s | resumed-final/attempt-1792029759585598780 |
| split-paged-bsnd-trap | 36 passed | 525.5 s | 同上 |
| e2e | 215 passed | 41.8 s | 同上 |
| ordinary | 516 passed | 26.4 s | 同上 |
| split-paged-bsnd-profile | 1 汇总用例通过，12 个固定输入配置 | 258.5 s | profile-fixed/attempt-1792031420372455170 |

- 代表检查先覆盖双 dtype、mask/跨页、16K KV 和最大 accumulator 273,162,240 字节；随后固定实现运行完整集合。普通和 ASan/UBSan 是同一组 337 项，分别与自身收集顺序一致。CPU 协议之后仅有无关公开 Python 包装变化，其 CPU 源码、编译依赖和工具链未变；公开修复由最终设备组覆盖。
- 新公开专项覆盖三 mask、P=16/112/512/528/1024、跨页/尾页、S=2..256、Q-used/原 allocation、空值、独立 stride/广播、长度便利输入、metadata 重用、real/fake/compile、多流/生命周期及失败恢复。release Trap 的 18 类故障 × 双 dtype 均在设备 event 同步处失败，无 host 读回替代。
- 固定源 PA+FD partial 与原始 FD 归并实际执行。源 FD 支持的 10 个配置 O/LSE 逐元素一致且双数学通过；S=256 的两个配置明确记录源 FD 的 192 片 UB 上界，源 partial 仍实际运行，accum O 与源逐元素一致，accum LSE 与源 sum/max 的独立 log 适配按 rtol/atol=1e-6 核对；迁移 O/LSE/accumulator 双数学通过。原始输入、页表、部分结果和比较报告均保存。
- 本模块向 `regression_cases.py` 新增 14 项日常 E2E，累计从 201 到 215；不填充或截断。旧无 mask 或非 PA 用例不作为新增 PA split/mask 验收。旧组件单元组、完整穷举矩阵、910C 实机和完整图重放本轮未运行，不计通过。
- 数学 reference 按内容复用：最终公开组 52 hit / 15 新计算，恢复后的固定源组 9 hit / 3 新计算；累计 E2E 361 hit、ordinary 31 hit、最终性能 12 hit，后三组均 0 重算。待测实现每次实际执行。

### 公开性能

固定 6 个输入 × 双 dtype，隐式生成、显式生成加前向、显式复用各 R=1/4；预热 5、5×20 wall / 10 profiler，CPU affinity `[0]`，6 号卡隔离采样。原生 ND PA 的页表/cache 转换和分配包含在公开调用计时内，AICPU/partial/merge 和原生单例/分段原始结果全部保留。

原生实际分派均为 template 5；输入 3/5 为非 FD，共 4 个适用配置，平均耗时比 **0.465807 ≤ 1**，配置退化 **0/4**、分段退化 **0/24**。六段均值依次为 implicit R1/R4 `0.535268/0.466940`、generate R1/R4 `0.678060/0.372833`、reuse R1/R4 `0.434206/0.307536`。全零实际 KV、非零容量的输入 5 仍由实测分派归为 template 5，未按空模板排除。

输入 0/1/2/4 的原生自动 FD 尚未迁移，8 个配置按 design 单列：诊断平均比 `1.150669`，配置退化 4/8、分段退化 24/48，不纳入本步适用门槛。未按耗时筛选输入。完整结果为 `profile-fixed/attempt-1792031420372455170/split-paged-bsnd-profile/public-profile/average.json` 及双 dtype `profile.json`、dispatch、wall 和 profiler CSV。

## 恢复与收尾核验

恢复时 HEAD 仍为本轮基线，工作树为本步改动；上次无头进程异常退出，固定源组只有 9 个局部产物，未计通过，已完整重跑 12 项。开发中的缺头文件、FakeTensor 准备读取 tensor 索引和整数长度预检查漏传 split 已修复并由最终公开组覆盖。首轮性能完成采样后因 tuple/JSON list 汇总身份不一致失败，修复本模块适配和身份回归测试后完整重跑；失败/中断运行均未转记通过，原日志保留。

`final-evidence-audit.json` 核验源码、构建/依赖/产物、实际加载及原始证据共 8,512 个文件，无差异或哈希冲突。公开、源对照、Trap、累计 E2E、ordinary 和最终工具/性能的当前所属源码依赖均匹配；性能适配修复不在已通过正确性组的依赖中。`final-native-audit.json` 复核原生包、252 个固定 Git 源文件和双 dtype 各 90 个性能归档文件。`selection-completion-final.json` 核对完整实际节点与收集清单；选择新增及旧清单差异依据在 `selection-audit.json`。本次契约变化按影响核对 design 的入口/记录、返回值、PA 生产/生命周期与验收表，plan 仅推进唯一下一步。`git diff --check` 及本步全部 30 个文件的仓库质量检查通过（`quality-final.log`）；相关文档 473 个本地链接、150 个锚点有效（`doc-links-final.json`）。提交钩子正常执行。 验收进程全部退出，6 号卡已释放，见 `processes-final.json` 与 `npu-final.log`。

复现命令（各组先加 `--dry-run`，再 `--collect-only`；输出使用新目录）：

```bash
python3 -m tests.mla --suite split-paged-bsnd-tools --output /tmp/mla-m10c7b/repro-tools
python3 -m tests.mla --unit split_paged_bsnd_protocol --output /tmp/mla-m10c7b/repro-protocol
python3 -m tests.mla --suite split-paged-bsnd-protocol-sanitize --output /tmp/mla-m10c7b/repro-sanitize
python3 -m tests.mla --suite split-paged-bsnd-check --device 6 --output /tmp/mla-m10c7b/repro-check
python3 -m tests.mla --unit split_paged_bsnd --device 6 --output /tmp/mla-m10c7b/repro-source
python3 -m tests.mla --suite split-paged-bsnd-public split-paged-bsnd-trap e2e ordinary --device 6 --output /tmp/mla-m10c7b/repro-device
MLA_NATIVE_ROOT=/tmp/mla-m10c3b/native python3 -m tests.mla --suite split-paged-bsnd-profile --device 6 --output /tmp/mla-m10c7b/repro-profile
```
