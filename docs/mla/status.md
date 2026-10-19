# MLA 当前状态

<!-- mla-loop-next: M21 -->

**M20 已闭环，唯一下一步为 M21。** TND Q + ND PA INT8 完整调用链已接通，模块单元 47 项、完整正确性验收 1,667 项全部通过。公开性能 eager 平均比 **0.6788566514，通过**；graph **1.0893393116，实测未达 ≤1**。用户于 2026-10-19 明确“图模式性能暂不做要求”，当前 INT8 M19–M21 按此授权暂缓图性能门槛；图正确性及其他必要验收保持不变。范围与授权见 [design 3.4](design.md#34-int8-压缩-mla-扩展)，下一步见 [plan M21](plan.md#m21int8-收尾)。

分支 `feat/mla-cann-migration`，本步基线为 `1ed2b328a7f9b9b716556201dd31e729afe176b0`。本次恢复收尾保持已验收的实现和测试不变，仅更新验收授权与交接文档，复核同一步原始证据。未修改 CATLASS 或只读 CANN 源。闭环提交可用 `git log -1 -- docs/mla/status.md` 定位。

## 当前实现

TND 直接读取原 int32 `cu_seqlens_q[B+1]` allocation 前缀，Q-scale 为 `[Tq,Hq]`，O/LSE 为 `[Tq,Hq,512]` / `[Hq,Tq]`。v49 的 `int32[463+2B]` 绑定总 Q、完整前缀和 KV 长度，消费者逐字核对。沿用固定源 balanced 调度、普通/快速 FD 与 INT8/BF16 数值组件，真实 partial 为 `[26,Hq,Tq,512]` / `[26,Hq,Tq]`；不转调 CANN。BSND v48 不变，未开放组合仍明确拒绝。

本步覆盖公开 real/fake/compile、无 mask/右下 causal、自动/单片、非均匀/零长度、allocation 与 stride、scale 更新、metadata 复用、图重放、stream/线程生命周期和 Trap。修复源分核未访问的尾空段 O/LSE 初始化，保留数值任务终点和流水线 flush；TND O-only 关闭可选 LSE。TND 页预检跨完整向量 grid 分工，零 Q 仍查全部已用页，数值 Cube 每次形成读地址前继续独立检查。

图采样发现新 stream 每次重建固定 causal mask。TND 现按设备持有已在 eager 完成初始化的 mask，以 event 非阻塞查询确认完成后跨 stream 复用；未完成或首次冷捕获走原 stream 局部路径，不发布未执行的数据。独立进程的双冷图与线程退出后重放已通过。BSND 原缓存路径不变。四配置预检中的重复 mask 生成任务由 400 个降为 0，必要视图拷贝仍计入。

## 本步闭环证据

证据根 `/tmp/mla-m20/`，工作根 `/tmp/flash-attention-npu-mla-46928a793876`。当前证据为 `mask-*`，实际扩展 SHA256 为 `8866a0da1a4ce4f983512b6191a3047100f80766f850ae6a4b4984de227c114e`，加载路径 `flash_attn_npu_3/flash_attn_npu_3.cpython-311-aarch64-linux-gnu.so`。驱动核对完整构建输入、依赖、产物和实际加载；manifest 由 `mla_validation_build prepare` 生成。

下表为同一 M20 冻结实现的实际执行记录；本次恢复复核其有效性，没有重跑数值测试、单元或性能采样，也不将原结果冒充本次新增执行或跨步单元/性能继承。历史 E2E 继承为 **0**，同一步完整正确性 E2E 复核 **1,667** 项。

| 检查 | 当前结果 | 证据（相对证据根） |
| --- | --- | --- |
| 模块单元 | 47 通过，11.5 s；固定 native 调度、边界/溢出、独立总 Q 上限、ASan/UBSan、性能工具成功/失败夹具 | `mask-tools/summary.json` |
| dry-run / collect | 代表 28、TND 专项 151、累计 E2E 901 全部收集；无临时截断 | `mask-dry.json`、`mask-collect/summary.json` |
| 完整构建与质量 | 构建、适用 hooks、文档检查和 `git diff --check` 通过 | `mask-build.log`、`mask-quality.log`、`handoff-doc-quality.log` |
| 固定源对照 | 15 通过，25.4 s；O/非空 LSE 逐元素一致；独立数学 O/LSE 各 15 项源精度限制，非数学全通过 | `mask-check/attempt-1792380312581633390/int8-tnd-source/` |
| 代表检查 | 28 通过，79.7 s；含 O-only、尾空段、冷图/跨线程、后部第 65 页与零 Q 页越界 | `mask-check/summary.json` |
| 性能预检 | 4 配置通过，132.6 s；400 条 graph Kernel 样本，122 个归档文件校验 | `mask-profile-check/summary.json`、`mask-profile-check.log` |
| 正式性能 | 完整 15 配置，404.1 s；eager 通过；graph 失败、按授权暂缓，15/15 配置及 78/90 分段退化 | `mask-performance/summary.json`、`mask-performance.log` |
| 完整 TND 专项 | 151 通过，507.8 s，reference 命中 62 次；包含此前失败的尾空段、全部图和 Trap | `mask-acceptance/summary.json` |
| 完整 BSND 专项 | 109 通过，259.9 s，reference 命中 64 次 | `mask-acceptance/summary.json` |
| 累计小输入 E2E | 901 通过，217.8 s，reference 命中 1,240 次 | `mask-acceptance/summary.json` |
| 普通 FA3 / 普通图 | 504 / 2 通过，28.1 / 21.2 s；reference 命中 31 / 2 次 | `mask-acceptance/summary.json`、`mask-acceptance.log` |
| 本次恢复核对 | 实际加载、构建输入/依赖/产物、用例清单、E2E 原始证据及源/数学分类均匹配；原始 wall 样本重算与保存报告一致 | `closure-authorized-runtime.json`、`closure-authorized-evidence.json` |
| 本次质量检查 | 适用 pre-commit、文档链接/标记及 `git diff --check` 通过 | `closure-authorized-quality.log`、`closure-authorized-doc-quality.log` |

正式性能保存 27,000 条 wall 样本、eager/graph 各 1,500 条 Kernel 样本及全部单例/分段，221 个文件逐文件归档校验。结果目录为 `mask-performance/attempt-1792380678698910500/int8-tnd-profile/public-profile/`。三入口 R=1/4、5 次预热、5×20 wall 和 5×10 Kernel、固定 15 配置及工具默认门槛均未变。graph 原始失败记录保留；用户暂缓要求不等于性能达标。

完整专项中的源对照记录为 TND 20、BSND 28，全部逐元素一致。独立数学 O：TND 2 通过 / 18 源限制，BSND 4 通过 / 24 源限制；LSE：TND 20 源限制，BSND 1 通过 / 27 源限制。源精度优先级按 design 7 执行，不能记为双数学全部通过。

数学 reference 沿用内容缓存，本步 source/代表分别命中 30/15 次，待测实现仍实际执行。`closure-authorized-evidence.json` 复用驱动的结果、原始证据、源码与依赖、构建输入/依赖/产物校验，全部匹配原 `closure-evidence.json`；实测扩展和加载库也完全一致。未另做手写全树审计。

## 循环报错与环境

最近无头轮 `20261019-010500-development-bw2yz0ji` 的 Codex 子进程退出码为 **0**、turn 已完成；当时 graph 未达门槛且尚未获本步豁免，因此没有提交，循环脚本按约定记录 `blocked` 并退出 **1**。本次按新的用户授权完成同一 M20 收尾，用现有 `--retire-pending` 归档被人工接手替代的旧会话。`retired.json` 保留原因、原状态和回执，pending 已清空；无头计数保持原值，没有伪造成功回执或启动下一轮。

首轮完整 TND 在 Q `(0,1,1)`、KV `(129,0,0)`、自动 split 时遗漏末尾 LSE 初始化：131 通过、1 失败、剩余未运行，425.4 s，见 `acceptance/attempt-1792373846973427120/detail-int8_tnd/`。当前完整 151 项专项已确认修复，旧失败记录保留。首次正式性能 `tail-performance/` 的 eager 0.7506133088 / graph 1.3740589794 是较早实现的结果，不能替代当前验收。

Python 3.11.15、torch 2.7.1+cpu、torch_npu 2.7.1、CANN 9.0.0-beta.2；实测 910B3 物理设备 7（20 AIC/40 AIV），910C 未实测。CATLASS `769cd40a8716b28650b6bebb08db4834eea4462f`，固定 CANN 源 `ca18aee610729ca5270a44e2d84c751bc687fbf0`。INT8 native `/tmp/mla-m18/native` 的配方、12 keys 与内容由现有验证器核对；非量化 native `/tmp/mla-m13b/native`。

设备检查和公开用例收集在宿主运行，沙箱不能访问 NPU；ASan/UBSan 在宿主运行以保留 LeakSanitizer 检查。固定 CANN 环境包装器为 `/tmp/mla-m16-schedule/run.py`。本次默认 shell 直接运行 `npu-smi` 遇到 `drvSetDeviceInfo` 库符号冲突，使用该包装器后正常，设备 7 无残留任务，见 `closure-authorized-npu.log`。profiler 使用现有私有 tmpfs 包装器解决根文件系统可用空间限制，结束后逐文件校验归档。构建、reference 与原始结果放在仓库外，`build/` 只保留 `mla-codex-loop/` 日志与恢复状态。

## 唯一下一步

**M21：INT8 组合覆盖与用户示例收尾。** 按 plan 核对两布局及拒绝边界、补示例与 E2E、完成阶段 eager 性能和文档；图模式性能暂不要求，图正确性仍必验。本次提交后停止，尚未执行 M21。

当前正确性闭环复现命令如下；新步骤只复用身份及原始证据均有效的 E2E，源码/构建变化后按影响范围重跑：

```bash
export MLA_WORK_DIR=/tmp/flash-attention-npu-mla-46928a793876
export MLA_INT8_NATIVE_ROOT=/tmp/mla-m18/native
export MLA_INT8_SOURCE_LOG=/tmp/mla-m18/source-final/attempt-1792357915791404830/int8-source-check/output.log
python3 /tmp/mla-m16-schedule/run.py python3 -m tests.mla \
  --suite detail-int8_tnd detail-int8_bsnd e2e ordinary graph-ordinary \
  --device 7 --resume --output /tmp/mla-m20/mask-acceptance
```

性能先 `int8-tnd-profile-check`，再 `int8-tnd-profile`；输出必须使用尚不存在的新目录，命令前加 `unshare -Ur -m python3 /tmp/mla-m16-schedule/run_private_storage.py <新输出目录>`，后接环境包装器与 `python3 -m tests.mla ... --output <同一新目录>`。保持设备隔离、原采样协议和完整集合。工具仍会因 graph 未达阈值返回失败，交付按 design 3.4 区分 eager 门槛和已授权暂缓的 graph 结果。
