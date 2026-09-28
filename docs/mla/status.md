# MLA 当前状态

## 当前能力与下一步

- 分支：`feat/mla-cann-migration`；接手 HEAD 为 `677c14e`，工作区干净。M04a 源码映射和实现拆分已完成，本步仅修改三个 MLA 文档。
- 生产 MLA 前向与 MLA metadata 仍明确报未实现；普通 FA3 保持既有计算与 metadata 路径。独立 reference、固定源码 CANN 基线、分派溯源和 profiler 工具可用。
- [design 6.1..6.5](design.md#61-首个-dense-实现的源对应关系) 已固定调度、Kernel / Block / Tile 对应函数、数值路径、资源布局、事件依赖和组件复用条件。
- 首个前向范围为连续 BSND、512+64 共享 KV、B>0、`1<=Sq*Hq<=256`、KV `1..512`、Hkv=1、无 mask、显式 `num_splits=1`；FP16/BF16 分步接通。此范围尚未实现；后续范围和验收顺序见 [plan](plan.md)。
- 唯一下一步：M04b1，抽取首步范围的无 gert 调度模块，CPU 对照固定源的基本块、分核端点、核数、无 FD 和 workspace，并验证字段/资源边界。本步提交后停止。

## 本步验证

证据目录：`build/mla-m04a-source-map/`。

| 项目 | 结果 |
| --- | --- |
| 接手及上步复核 | PASS：生产源码/构建来源哈希与上步一致；历史 XML 的 CPU 88、MLA 契约 233、普通 FA3 11 项全部通过且无 skip；复核原命令，未重跑测试 |
| 固定来源 | PASS：53 个 CANN 文件由固定 commit 导出并与工作区逐字节核对；7 个 CATLASS 组件与固定子模块一致；来源仓库状态保持原样 |
| 源码映射 | PASS：调度使用 `split_core.cpp`；核对 288+288 的 QK 排列、aMLA nUpdate、504 KiB L1、UB/GM 分区、五项跨核依赖及 C1V1/C1V2 分支 |
| 实现拆分 | PASS：M04b1 调度 → M04b2 FP16 → M04b3 BF16 → M05 metadata；更大 S1G、多块/长 KV、空容量在 M04c 子步骤独立验收 |
| 文档及质量 | PASS：98 个源码锚点、17 个本地链接/文档锚点、唯一下一步和 metadata E2E 口径一致；差异检查及 3 项适用质量检查通过；仅暂存三个 MLA 文档 |
| 构建、CPU/NPU 回归及性能 | 未运行：本步仅文档，无生产源码或构建变化；没有新的精度或性能验收结论 |

源码核对确定了组件复用条件，不表示移植后的设备等价性已验证。metadata 性能门槛仍为逐例 `FA3 metadata 路径 E2E / 原 CANN E2E <= 1.0`；M04b 的无 metadata 数据仅作诊断。

## 最近一次实机证据与环境差异

最近实机闭环为 M03，提交 `677c14e`，证据目录 `build/mla-interface-guard-validation/`。本轮只核对文件和既有证据，未导入 torch/扩展、未访问设备；下次开发或实机验证前须重新检查环境、实际加载扩展和可用芯片。

| 项目 | M03 记录，非本轮重测 |
| --- | --- |
| Python / 包 | `/usr/local/python3.11.15/bin/python3` 3.11.15；torch `2.7.1+cpu`，torch_npu `2.7.1` |
| CANN / 芯片 | `/usr/local/Ascend/cann-9.0.0-beta.2`；Ascend910B3 |
| 固定 FIA / 基线包 | `ca18aee610729ca5270a44e2d84c751bc687fbf0`；`build/mla-m02b-validation/opp-dense/` |
| CATLASS | `769cd40a8716b28650b6bebb08db4834eea4462f`；本轮核实子模块干净 |
| 加载扩展 | 本仓 `flash_attn_npu_3/flash_attn_npu_3.cpython-311-aarch64-linux-gnu.so` |
| 扩展 SHA256 | `955ecbe4c2b199b2e1131d9c86c38ce23b15ed8704c55cf2e35de37986b84877`；本轮仅确认磁盘文件哈希相同 |
| 设备条件 | 物理 2 号共享 Warning 卡，告警 `81078603`，前后 ECC 八项计数均为 0；21 个监控快照 |

上述设备记录只说明 M03 的功能验证条件，不作为本轮设备可用或性能验收结论。CANN 源仓只读，其已有删除和未跟踪文件均保留；本次引用固定 Git blob。CANN 精度及 profiler 基线分别见 `build/mla-m02b-validation/`、`build/mla-m02b2-validation/`。

## 证据与复现

- 本步 `source-manifest.json`、`catlass-manifest.json`、`source-repositories.json`：固定源哈希、只读快照和来源工作区状态。
- 本步 `source-symbols.json`、`previous-step-review.json`、`docs-check.json`、`quality-final.log`、`checksums.sha256`：函数/关键语句定位、上步命令与历史 XML 复核、文档及提交质量。
- M03 的 `build-command.json`、`build-provenance.json`、`cpu-command.json`、`npu-command.json`、`cpu.xml`、`npu.xml` 保存强制构建、88 项 CPU、233 项契约及 11 项普通 FA3 的精确命令与结果；`npu-device-monitor.json` 保存设备条件。M03 的 `verify_acceptance.py --staged` 依赖当时暂存区，不直接用作本步检查。

本步文档闭环命令（提交前暂存状态）：

```bash
python3 build/mla-m04a-source-map/check_docs.py --staged
git diff --check
git diff --cached --check
PYTHONPATH=/tmp/mla-m01-precommit PRE_COMMIT_HOME=/tmp/mla-m01-precommit-cache bash ci/code_quality.sh --staged
```

当前文档闭环提交用 `git log -1 -- docs/mla/status.md` 定位。进入 M04b1 后，先按开发约定复核环境与上步证据，再实现调度，不同时开始 kernel 或 metadata 接入。
