# MLA 当前状态

**M08c1 已闭环：非 PA TND Q/KV causal varlen；唯一下一步是 M08c2：非 PA TND Q + BSND KV causal cache。** 本轮基线为 `10248d45a65c21d814153ed4bd3fcf0786399cb3`，分支 `feat/mla-cann-migration`。本步提交后立即停止；闭环提交用 `git log -1 -- docs/mla/status.md` 定位。

## 当前可调用能力

- 新增非 PA TND Q/KV `flash_attn_varlen_func(causal=True, num_splits=1)`，支持 FP16/BF16、512+64、Hkv=1、完整分配前缀、可选 `seqused_q/k` 和原合法四路 stride。公开、专用与通用前向汇入同一生产路径，真实/fake/compile 均已验收。
- v8 使用 `version=kind=8`、`410+8B` 个 int32，每次内部生成；分配前缀决定物理起点，used 决定有效 Sq/Sk 和右下对齐。保留既有空值及未使用 Q 槽位契约，不新增显式 TND metadata 复用。
- 原 BSND causal 与无 mask 的非分页/PA 能力继续保留。混合、PA causal 和未开放选项仍明确拒绝；本步没有开放 causal cache/metadata 的 TND query。

协议、来源与生命周期见 [design 5.10](design.md#510-v8tnd-causal-记录)，提交边界和下一步见 [plan](plan.md#唯一下一步-m08c2非-pa-混合-causal-cache)。Block/Tile、CATLASS 和 CANN 源代码未修改。

## 本步闭环证据

证据根为 `build/mla-m08c1/`。生产和设备用例冻结于 `boundary-frozen-sources.json`；随后仅修正失败的性能驱动，最终快照为 `profile-frozen-sources.json`。`final-evidence-audit.json` 核对下表 CPU/设备/性能组的完整原始产物哈希和源码差异，恢复依据见 `boundary-cpu-inheritance.json`、`profile-inheritance.json`。源码闭包保守包含的独立性能文件变化，不改变已完成数值组的执行依赖。

| 检查 | 本步通过数 | 耗时（秒） | 证据目录（相对证据根） |
| --- | ---: | ---: | --- |
| v8 CPU | 41 | 25.8 | `protocol-cpu/attempt-1791880775803920940` |
| v8 ASan/UBSan（含退出 LSan） | 41 | 94.9 | `protocol-cpu/attempt-1791880920396135230` |
| v7/v5 CPU / ASan+UBSan | 100 / 100 | 55.9 / 170.2 | 同上 |
| 构建 CPU 检查 | 2 | 4.1 | 同上 |
| 最终工具检查 | 100 | 15.7 | `final-tools/attempt-1791884906691086440` |
| BSND causal 代表 / 限量公开 | 56 / 100 | 109.9 / 176.2 | `protocol-final/attempt-1791882155655276880` |
| mixed/PA 真实/fake 拒绝补例 | 8 | 24.5 | 同上，`causal-tnd-boundary` |
| TND causal 代表 / 公开行为 | 6 / 74 | 102.5 / 127.7 | 同上，`causal-tnd-check/public` |
| TND release Trap | 18 | 264.7 | 同上，`causal-tnd-trap` |
| 固定非分页共享 / 普通 FA3 回归 | 86 / 100 | 784.3 / 25.0 | 同上，`regression-nonpaged/ordinary` |
| 公开性能 / 双 dtype 代表组件 | 1 / 2 | 548.9（全组） | `profile-final/attempt-1791884264941979770` |

以上全部通过，无跳过；v8 CPU 在后续 SAN 批次中按身份继承，不重复计数。各组最多 100 个节点；tools 191 选 100、普通 FA3 816 选 100、BSND 公开 254 选 100，其余上表组保留实际集合。选择记录在 `protocol-collection/`、`bsnd-boundary-collection/`、`boundary-collection/`、`profile-collection/`、`final-tools-collection/`，均保存候选/选中/未选中。未选中组合不记为通过，M08 全部已开放组合在 M08e 冻结后实测。

统一命令为 `python3 -m tests.mla_validation --output build/mla-m08c1/<批次> --suite <上表组> --max-cases 100 --device 6`，先核对 `--dry-run` 和 `--collect-only`。各批 `summary.json`、`command.json` 保存精确命令和环境。公开性能单独用 `--suite causal-tnd-profile`，不恢复历史采样；本轮源码未变的已完成组不再重跑。

## 精度与性能

`protocol-precision.json` 保存新 TND 代表和公开组的源/双数学汇总。公开组双 dtype 各 301 次 O、301 次 LSE 比较均通过固定源逐元素一致与双数学检查；stride、分配间隙、有效长度、全遮罩和空值另有明确用例，未用旧无 mask 组代替新增行为。数学缓存为 `build/mla-validation-artifacts/golden`：最终设备/性能组共命中 590 次，性能代表新增 6 个不同内容的 reference；待测实现和固定源仍实际执行。

公开性能在第 6 卡隔离采样，固定 28 个含 dtype 配置、R=1/4、5 次预热、5 轮×20 次同步 wall；原生必要 gather/scatter/布局转换计入 E2E。实际分派确认 18 个非 FD 配置适用平均门槛，10 个原生 FD 配置单列并保留全部采样。

- **平均耗时比 0.433036 ≤ 1，通过；**R=1 / R=4 分段均值为 0.466394 / 0.399678，适用配置退化 0、分段退化 0。
- 原生预检共 204 次 O/LSE 比较：185 次逐元素一致、19 次原生自动 FD 数值顺序不同，全部双数学通过。原生空行符号先独立检查再适配；未降低阈值。
- 同调度非 FD 组件各 3 例，FP16 Kernel 比值为 1.111443 / 1.204281 / 0.362804，BF16 为 1.507604 / 1.156476 / 0.374766。组件精度逐元素通过；单例耗时完整报告，不替代公开 E2E。

原始 `wall.json`、`kernel-samples.json`、profiler CSV、实际加载库和 native tiling key 在最终性能组的 `public-profile/`；平均结果为 `public-profile/average.json`。组件结果在该组 `probes/<dtype>/tnd-component-profile/summary.json`。

## 继承、失败与恢复

**继承历史证据：**M08b2 的 causal attention 324 项与组件性能 2 项，共 326 项，仅覆盖未变算法/Block/Tile/任务环及最长 reference、最大内存边界。`baseline-audit.json` 核对接手时 830 个冻结文件；`component-inheritance.json` 核对 15 个算法依赖；`historical-artifact-audit.json` 逐一核验历史完整产物。新 TND 入口、地址/参数、生命周期与 E2E 均为本步实测，未用继承替代。

**原失败保留，不计通过：**

- 首版运行期选择 v7/v8 协议时，BSND Q-used 代表在 pytest 捕获配置下两次 AICore Trap/timeout，耗时 92.0 / 87.0 秒，见 `regression/attempt-1791879264140568440` 和 `regression-recheck/attempt-1791880080061112290`。修正为 launcher 模板固定协议、恢复 BSND 固定 v7 后，正式代表 56、限量公开 100 和补充边界 8 项均通过；不将未复现的隔离脚本当成故障已解释的证据。日志/core 保留在证据根，诊断在 `debug/`。
- 首次性能组误用只含 BSND 实例的源包，缺少 TND key `105000000030000002`，43.2 秒失败，未形成性能结果。更换已有固定 TND 包并核验 `tnd-vendor-audit.json` 后，仅重跑性能组，548.9 秒通过。数学内容缓存继续复用。
- 初次 SAN 的断言虽通过，退出 LSan 因沙箱 ptrace 限制失败；保留 LSan 在主机重跑成功。性能 collect 的一次沙箱设备不可见失败，也在主机重新核对成功。早期测试断言错误、编译/测试期间源码变化导致的失效与中断仍保留在 `cpu-dev/`、`cpu-final/`、构建日志和诊断目录，不记为通过；正式构建依赖及实际加载已重新核验。

本步必要数值、性能及质量检查均已完成，没有后台验证进程需要续接；只提交本步，不启动下一步。

## 环境与构建

Python 3.11.15、torch 2.7.1+cpu、torch_npu 2.7.1、CANN 9.0.0-beta.2；实测第 6 卡 910B3，20 AIC/40 AIV。CATLASS `769cd40a8716b28650b6bebb08db4834eea4462f`；只读 CANN `ca18aee610729ca5270a44e2d84c751bc687fbf0`。CANN 工作树有本任务之外的已有变动，相关 FIA/common 源无差异，见 `cann-worktree-audit.json`；固定源 oracle 从指定 Git blob 提取。

实际加载本仓 `flash_attn_npu_3/flash_attn_npu_3.cpython-311-aarch64-linux-gnu.so`，SHA256 `45e6adfa5195b9845f23f314c3463d9375b75d7ca18e7948a72c8814fcc33224`。重建后共享清单 `build/mla-validation-artifacts/prepare.json` 已复核；正式设备批次构建缓存检查 1.1 秒、runtime 22.3 秒，最终性能批次对应 1.1 / 21.3 秒。原生 TND 包位于 `build/mla-m06c3a-varlen/vendor/mla_ca18aee_transformer`，双 dtype 二进制与历史构建记录及本次实际分派一致；旧目录只复用已核验产物，不依赖临时执行脚本。

设备执行需要主机设备可见性；不要 source CANN set_env 或覆盖原 PYTHONPATH。质量入口为 `ci/code_quality.sh --staged`；本机 pre-commit 包在 `/tmp/mla-m01-precommit`，使用时附加到原 PYTHONPATH 前。统一质量入口全部适用检查通过，YAML/shell/workflow 因本步无对应文件由入口自动跳过；日志为 `quality-staged.log`，最终复核及耗时见 `quality-final.json` / `quality-final.log`。`git diff --check` 与三个文档的 289 个链接核对通过，见 `document-link-check.json`；质量检查未改变冻结源码。

## 唯一下一步

[M08c2：非 PA TND Q + BSND KV causal cache](plan.md#唯一下一步-m08c2非-pa-混合-causal-cache)。本轮只提交已验收的 M08c1，不开始 M08c2。
