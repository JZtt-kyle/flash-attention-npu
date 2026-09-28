# MLA 当前交接

## 状态与唯一下一步

- 当前步骤：M00a **metadata E2E 验收口径澄清，纯文档**。用户明确“不劣化指的是走 metadata 路径端到端 E2E 不劣化”；本次更新 design、plan 和 status，未修改生产代码、测试或构建产物。
- 当前分支：`feat/mla-cann-migration`。接手 HEAD 为 `7b84028`，工作区干净；生产代码仍与 `23d7304` 一致，支持连续 BSND、512+64、FP16/BF16、KV `1..1280`、显式 `num_splits=1`。更长 KV 及 MLA metadata 继续明确拒绝。
- 验收口径以 [design 第 7 节](design.md#7-验收口径) 为准：逐例比较 **FA3 metadata 路径 E2E / 原 CANN E2E <= 1.0**，kernel 时间用于诊断。现有无 metadata 的 kernel/同步调用数据不能证明此项验收已通过。
- 唯一下一步：**M05 提前到长 KV 优化之前**，在当前已支持范围接通 MLA metadata、AICPU 调度和运行时 scale 分离，完成有/无 metadata 精度、复用、普通 FA3 回归及同轮 CANN E2E 对照。按真实调用段记录 metadata 生成/更新与复用成本，再据 E2E 瓶颈选择优化。
- M04b2b2b2b2b2b 的 `KV >1280`（含 8193）接通留待 M05 之后；split-KV 仍在 M10。本轮不开始实现，文档提交成功后停止。

## 本次闭环证据

| 检查 | 结果 |
| --- | --- |
| 接手与上步复核 | PASS：分支、干净工作区、最近提交和 `7b84028` 差异一致；上步最终 JSON/XML 记录为恢复基线的 36 项极值回归及 11 项普通 FA3，无 kernel 候选合入 |
| 代码依据 | PASS：`mla_short_kernel.hpp` 标明从 CATLASS MLA 改编；`mla_online_mmad.hpp`、`mla_online_epilogue.hpp` 实现设备流水适配；Python/C++ 仍明确拒绝 MLA metadata |
| 文档一致性 | PASS：链接、支持范围、E2E 计时边界及 M05 唯一下一步一致；历史性能证据未重标为 metadata E2E 通过 |
| 提交质量 | PASS：仅三个 MLA 文档，`git diff --check`、暂存差异检查及 `ci/code_quality.sh --staged` 的适用检查通过 |
| 构建与 NPU | 未运行：本轮仅文档，无源码或构建变化，不将上轮实测记为本轮重跑 |

本轮文档检查与质量日志：`build/mla-m00a-metadata-e2e/`。复现命令：

```bash
python3 build/mla-m00a-metadata-e2e/check_docs.py
git diff --check
git diff --cached --check
PYTHONPATH=/tmp/mla-m01-precommit PRE_COMMIT_HOME=/tmp/mla-m01-precommit-cache bash ci/code_quality.sh --staged
```

## 最近一次实机基线与证据入口

上次实机闭环为 `7b84028` 的 M04b2b2b2b2b2a。恢复后的 14 个构建对象与 `23d7304` 逐字节一致；最终 36 项跨半区极值回归及 11 项普通 FA3 通过。两种 softmax 候选均撤回：候选 A 的有效旧新 kernel 对照为 0 改善、2 微退化、62 区间跨 1，候选 B 的 12 例区间均包含 1。既有撤回决定保持原样，不能将其记为 kernel 优化已上线。

证据根目录：`/home/zhangqijia/repo/flash-attention-npu/build/mla-m04b2b2b6-validation/`。

- `restored-baseline/acceptance-integrity.json`、`baseline-maxima-final.xml`、`baseline-regression.xml`：最终合入内容的验收；`verify_restored.py --staged` 为当时提交前检查，依赖当时 HEAD/暂存区，不能直接作为后续提交的通用检查脚本。
- `restored-baseline/build-provenance-restored.json`、`link-order-proof.json`、`environment-restored.json`、测试命令及 monitor 文件：源码、重建与设备溯源。
- `full-tile-candidate/`、`direct-max/`、`versions-summary.json`、`performance-acceptance.json`：未采用候选的源码及实测。A 首轮因新增外部进程而无效的性能采集仍保留，不能混作有效性能数据。
- `checksums.sha256` 保存上步证据完整性。此前 700 项 MLA 及 56 例 CANN 性能证据见 `23d7304` 的 status；完整上步命令及条件见 `git show 7b84028:docs/mla/status.md`。

## 环境与设备条件

本轮只检查文档，未加载扩展或访问 NPU；没有新的设备健康、占用或运行环境验收结论。CATLASS 当前为 `769cd40a8716b28650b6bebb08db4834eea4462f`，子模块干净。

以下为上次实机记录，下一次开发或验证前须重新核实：

| 项目 | 上次实测记录 |
| --- | --- |
| Python / 包 | `/usr/local/python3.11.15/bin/python` 3.11.15；torch `2.7.1+cpu`，torch_npu `2.7.1` |
| 芯片 / CANN | Ascend910B3；`/usr/local/Ascend/cann-9.0.0-beta.2` |
| 固定 FIA | `ca18aee610729ca5270a44e2d84c751bc687fbf0`；固定源码包 `build/mla-m02b-validation/opp-dense/` |
| 实际加载扩展 | 本仓 `flash_attn_npu_3/flash_attn_npu_3.cpython-311-aarch64-linux-gnu.so` |
| 恢复扩展 SHA256 | `e267a7fb331430dd1e1aeca5becd6ee3c6b5cd07f7c4c0756d76c4579306006b` |
| 设备条件 | 用户仅授权上步使用物理 2 号共享 Warning 卡，告警 `81078603`，前后 ECC 八项计数均为 0；该授权不扩展至下一步 |

下次验收仍须检查实际加载扩展与源码/构建一致性，重新确定设备。不得把历史共享 Warning 条件记为健康、独占设备验收。
