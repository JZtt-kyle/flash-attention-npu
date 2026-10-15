# MLA 当前状态

**开发流程已按完整可调用行为收敛，功能基线仍为 M10c9b。** 分支 `feat/mla-cann-migration`；本轮提交由 `git log -1 -- docs/mla/status.md` 定位，上次实机闭环为 `d42a7ff`。唯一下一步为 [M10c10：TND PA 原生自动 split 完整接入](plan.md#唯一下一步-m10c10tnd-pa-原生自动-split-完整接入)，原 a/b 合并为一次提交。M10 尚未阶段收尾，自动路径仍拒绝。

## 本轮流程收敛

- 根 AGENTS 和无头开发提示词统一为：稳定组件按布局交付完整可调用行为，只有新增数值算法或共享协议的独立未覆盖风险才拆组件步。prefill 先做共享数值，再按布局带齐 mask/split，M14 只补组合缺口和收尾。范围、精度/安全/性能门槛和每五轮文档修正约定保持不变。
- 接手/收尾复用统一验收驱动已有的身份、依赖、构建和证据校验；缺口才补检查。长测前先核对工具、廉价夹具及适用 native 包，避免重复准备和整库审计。本轮只改工作流文字，未改数值、构建/发射或验收驱动实现。
- Python `/usr/local/python3.11.15/bin/python3` 3.11.15。`--suite tools` 经 dry-run/collect-only 后实际 **287 passed、0 failed、0 skipped，88.4 s**，包含临时 Git/假 CLI 的开发/恢复、失败传播、单提交、归档和文档周期。269 个验收组命令前后一致；外部入口 dry-run 不写状态，新建/恢复均使用当前提示词。本轮未探测设备、重建扩展、运行或继承 NPU E2E/性能；下方实机数据仅保留上次记录。
- 证据根 `/tmp/mla-workflow-convergence-20261015/`：`validation/summary.json`、`collect/`、`workflow-audit.json`、`loop-preview.txt` 和 `resume-command.json`。15 个本地链接、7 个锚点、5 个表格及适用五项质量 hooks 通过，记录为 `doc-links.json`、`quality-final.log`；暂存和已有改动保留核对为 `preservation-final.json`。这些临时证据后续使用前须复核。
- 原待办 `20261015-072727-development-t62a_7dy` 已中断且仅完成接手检查，已用现有 `--retire-pending` 归档至原目录的 `retired.json`；原日志和失败状态保留，未计成功。计数保持 development=80/reviewed=78，pending 清空；下次运行 `/home/zhangqijia/repo/mla_codex_loop.py` 从新 HEAD 和合并后的 M10c10 开始，本轮不启动开发循环。
- 接手已有六处独立循环修复和 plan/status 尾部备注，原内容备份于 `preexisting/` 与 `preexisting.patch`，均保留并排除本轮提交。

复现工具检查：`python3 -m tests.mla --suite tools --output /tmp/mla-workflow-convergence-20261015/repro`，先 dry-run/collect-only；只预览下一轮用 `python3 /home/zhangqijia/repo/mla_codex_loop.py --dry-run --max-dev-runs 1`。

## 功能基线：M10c9b（上次实机闭环）

512+64、Hkv=1、FP16/BF16 的 TND Q + ND PA KV 显式 S=2..256 接入 v25 AICPU、host 紧凑预算、init/partial/merge、公开 cache/metadata 和 real/fake/compile。三种 mask、实际 KV 和内部 Q-used 均已实测；O/LSE 与真实 FP32 `[S,H,totalQ,512]` / `[S,H,totalQ]` accumulator 保留原 allocation token，初始化覆盖 used 间隙、空段和空片。每阶段写入/同步前完整检查当前前缀、KV-used、metadata 与有效页。

公开 metadata 在 AICPU 验证描述符后取得 totalQ，空总量使用 1 核证书；前向内部生成核对已知 allocation 总量，host 不读取动态长度。复用固定数值 Block/Tile，未修改 CANN 或 CATLASS。契约见 [design 6.2.30](design.md#6230-tnd-pa-显式-split-生产适配)。TND PA 自动 0、192/128、softcap 和非默认调度选项仍拒绝；完整 graph replay 与 910C 实机未验收。

### 上次环境与证据身份

M10c9b 记录为 Python 3.11.15、torch 2.7.1+cpu、torch_npu 2.7.1，CANN `/usr/local/Ascend/cann-9.0.0-beta.2`，CATLASS `769cd40a8716b28650b6bebb08db4834eea4462f`，宿主卡 6 为 Ascend910B3。实机检查在宿主执行；本轮未重新核验设备和扩展身份，恢复开发时须按依赖变化复核。

- 证据根 `/tmp/mla-m10c9b/`；构建与数学缓存为 `/tmp/flash-attention-npu-mla-46928a793876/mla-validation-artifacts/`。`build/` 仅含 `mla-codex-loop/`。这些临时路径不是持久存储，后续继承前须核对文件和依赖。
- `bootstrap-2/build-result.json`：生产扩展与固定源 PA 探针重建通过（573.7 s）。实际加载 `flash_attn_npu_3/flash_attn_npu_3.cpython-311-aarch64-linux-gnu.so`，SHA-256 `2bbbdaeac0b18209728a1a50da139a138482a6ed470f726d36e2a87adc76e1a6`；runtime 记录位于各实机 attempt。
- 性能必须显式设置 `MLA_NATIVE_ROOT=/tmp/mla-m10c3b/native`：固定源 `ca18aee610729ca5270a44e2d84c751bc687fbf0`，当前配方、输入声明及 2,027 个包文件有效（`native-audit.json`）。默认 native 目录为旧配方，不能直接使用。CANN 工作树 HEAD 为 `2060a3d`；工具仍从固定版本 Git blobs 提取源，不以工作树替代。

### 上次实机验收

以下为 `d42a7ff` 的 M10c9b 当次记录，最终必要组无失败/跳过，未按数量截断。`device-collect/` 保存清单，`final-evidence-audit.json` 保存当次核验。该步继承 E2E 为 **0**；本轮仅保留证据入口，未重新声明这些结果有效。

| 组 | 结果 | 用时 | 证据目录 |
| --- | --- | --- | --- |
| 本模块 CPU 普通 | 421 passed | 46.7 s | `cpu-final/` |
| 同集合 ASan/UBSan | 421 passed | 140.9 s | `cpu-final/` |
| 所属工具 | 23 passed | 7.3 s | `tools-final/` |
| 代表与成本边界 | 8 passed；最终公开组再次覆盖相同节点 | 64.4 s | `preflight/`、`acceptance-3/` |
| 新公开专项 | 94 passed | 105.6 s | `acceptance-3/` |
| 固定源与双数学单元 | 12 passed | 231.5 s | `acceptance-3/` |
| 累计日常 E2E | 243 passed | 65.1 s | `acceptance-3/` |
| 普通 FA3 回归 | 516 passed | 52.7 s | `acceptance-3/` |
| release Trap | 42 passed | 600.3 s | `acceptance-3/` |
| 公开性能 | 1 汇总节点、12 配置通过 | 339.5 s | `performance-native/` |

代表组覆盖最长 KV=16384 reference 和最大 accumulator 203821056 字节。支持 FD 的 10 个源配置 O/LSE 均逐元素一致且双数学通过；两个 S=256 配置按固定源 FD 的 S≤192 表示域限制单列源 partial 与双数学，无新增源精度豁免。真实 accumulator 同时对照源 partial。新增日常 14 项覆盖三 mask、双 dtype、空值及 fake/compile，累计清单由 229 增至 243。

数学缓存：最终公开组 67 hit / 8 个新输入，固定源组 10 hit / 2 个新输入，累计 E2E 387 hit、ordinary 31 hit，后两组无重算；性能诊断为 12 个新配置各计算一次 reference，采样复用精度证明。代表/重跑之间继续复用内容缓存，待测实现始终实际执行。

### 上次公开性能

预先固定 6 种形状×双 dtype，隐式生成、显式生成加前向、显式复用各 R=1/4；预热 5 次、5×20 wall、10 次 profiler，同进程双方 CPU affinity `[0]`，卡 6 隔离采样。必要分配和原生 TND/页表/cache 转换计入 E2E，保留 AICPU/init/partial/merge、原生分派、全部单例和分段原始数据。

按显式 split 的既有适用口径，4 个非 FD 配置的等权平均比 **0.513392 ≤1**，0/4 配置、0/24 分段退化。implicit R1/R4 为 `0.593072/0.482895`，generate 为 `0.730007/0.418426`，reuse 为 `0.497460/0.358495`。8 个原生自动 FD 配置按 design 单列，诊断平均比 `1.162300`，5/8 配置、29/48 分段退化；全部 12 配置的诊断平均比 `0.945998`。未根据计时选择输入。汇总为性能组 `public-profile/average.json`；双 dtype 归档分别 88/90 个文件，经哈希核验完整。

### 基线复核与复现

M10c9b 的失败修正、完整环境核验和历史质量记录见 `git show d42a7ff:docs/mla/status.md`；下列命令仅用于需要重验该功能基线时，先 dry-run/collect-only。后续开发优先用统一驱动复核实际依赖和原始证据，不重复历史手工审计。

```bash
python3 -m tests.mla --suite unit-split_paged_tnd_protocol split-paged-tnd-protocol-sanitize split-paged-tnd-tools --output /tmp/mla-m10c9b/repro-cpu
python3 -m tests.mla --suite split-paged-tnd-check --device 6 --output /tmp/mla-m10c9b/repro-check
python3 -m tests.mla --suite split-paged-tnd-public unit-split_paged_tnd e2e ordinary split-paged-tnd-trap --device 6 --output /tmp/mla-m10c9b/repro
MLA_NATIVE_ROOT=/tmp/mla-m10c3b/native python3 -m tests.mla --suite split-paged-tnd-profile --device 6 --output /tmp/mla-m10c9b/repro-profile
```
