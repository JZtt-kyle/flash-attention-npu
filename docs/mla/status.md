# MLA 当前状态

**M10 阶段收尾已完成。** 分支 `feat/mla-cann-migration`，接手基线 `a6efbd2`。五种 512+64 布局的自动/单/显式 split 与三 mask 已完成组合、公开行为和阶段性能验收。唯一下一步是 [M11a：192/128 共享数值组件](plan.md#唯一下一步-m11a)，本轮提交后停止。

## 本轮改动

统一驱动新增 `--split-stage-performance`：固定十五组共 92 个配置（含双 dtype），保留候选/选中/未选中清单；原生分派核验后、计时前拒绝适用集合为空；阶段汇总从绑定的原始耗时重新计算配置等权平均。增加三个只含公开测试的组别，避免阶段收尾累计历史组件单元。生产 host/AICPU/Kernel、metadata、公开 real/fake、构建参数和 CATLASS 均未改。

计时仍为 R=1/4、5 次预热、5×20 wall，转换与完成同步边界不变。性能选择不影响正确性清单；组合覆盖与拒绝边界见 [design 7.5](design.md#75-m10-阶段覆盖)。

## 本次闭环证据

证据根 `/tmp/mla-m10-close/`。最终工具为 `tools-final/summary.json`，正确性为 `acceptance-final/summary.json`，性能为 `performance-final/summary.json` 及 `performance-stage.json`。全部先 dry-run/collect-only，固定源码后实际执行，无 `--resume`、`--max-cases` 或完整矩阵扩展；继承通过结果为 0。

| 检查 | 结果 | 用时 |
| --- | --- | --- |
| 受影响工具 | 195 passed | 81.9 s |
| 累计日常 MLA E2E | 314 passed | 61.2 s |
| 普通 FA3 回归 | 516 passed | 35.1 s |
| BSND 显式 / 自动公开专项 | 53 / 53 passed | 42.4 / 34.9 s |
| TND 显式 / 自动公开专项 | 58 / 59 passed | 65.0 / 49.1 s |
| mixed 显式 / 自动公开专项 | 85 / 89 passed | 71.4 / 61.0 s |
| BSND PA 显式 / 自动公开专项 | 80 / 80 passed | 114.3 / 100.5 s |
| TND PA 显式 / 自动公开专项 | 92 / 100 passed | 106.4 / 102.9 s |
| 阶段公开性能 | 15 组、92 配置全部实测，阶段平均通过 | 3632.2 s |

正确性各组均 0 failed/skipped；公开专项合计 749 次通过，包含与日常及组间重叠，不作独立用例总数。正确性组共用时 844.1 s；MLA reference 1004 次命中，普通 FA3 随机输入 1 次命中、30 次生成缓存。性能 reference 168 次全部命中，待测实现仍实际执行。30 份性能诊断精度报告的 216 个数学状态均为 passed，未使用源精度限制豁免；源分派和策略差异保留在原始报告中。

阶段实测 92 配置，**64 个适用配置的算术平均为 0.5997372294 ≤1**；28 个原生 FD 配置来自单/显式 split 的策略差异，单列不计平均。五个自动组全部纳入原生 FD。保留 8 个退化配置、50 个退化入口/R 分段，逐例和分段均值在 `performance-stage.json`，所有原始样本在各组 `public-profile/`。以下组均值仅作诊断，阶段按配置等权，不按组均值平均：

| 布局 | 单 split | 显式 split | 自动 split |
| --- | --- | --- | --- |
| BSND | 0.5559 | 0.6801 | 1.0103 |
| TND | 0.3489 | 0.2862 | 0.7656 |
| mixed | 0.4184 | 0.4241 | 0.6515 |
| BSND PA | 0.4084 | 0.3852 | 0.8007 |
| TND PA | 0.3976 | 0.4108 | 0.7649 |

BSND 自动组均值 1.0103 及其退化配置完整保留，不影响按约定全阶段平均验收。完整固定输入见 `performance-final-selection.json` 和各组 `split-stage-selection-*.json`；最终策略为 2。`final-check.json` 复用统一驱动函数核对 28 组原始证据、源码、外部依赖和收集/实测用例清单，以及实际加载库、构建源/依赖/产物和 native 包，全部有效。`accuracy-summary.json` 索引本次诊断精度证据。

本轮未运行历史数值组件、协议/sanitizer/Trap 或组件性能：对应实现未变，阶段回归走公开 E2E 与专项。192/128 prefill、softcap、非默认调度仍拒绝；完整 graph replay 和 910C 实机未验收。

## 失败修正与证据边界

首次 90 配置在 `performance/summary.json`：五个单 split 组完成后，BSND 显式三个输入全部触发原生 FD，排除后适用集合为空，汇总失败；其余组未运行。根据原生分派补入既有下标 4（used=512，非 FD template=5），保留原有全部输入，形成 92 配置；没有按耗时筛选。五个显式组双 dtype 的分派预核对在 `native-selection-check.json`，不继承其中的历史性能。新增计时前适用集合检查后，工具、正确性及完整性能全部按最终身份重跑。

开发工具旧 mock 参数失败保留于 `tools/`，修正后的最终工具组通过。沙箱无法查询 ascend_hal 的收集失败保留于 `acceptance-collect/`、`performance-collect/`，宿主最终收集与执行通过；这些失败不记为数值通过。接手证据核对在 `handoff-check.json`；修正前的 `tools-retry/`、`acceptance/` 和 `performance/` 不替代最终证据。`profile-format-check.json` 只检查历史报告格式。

## 环境与复现

Python 3.11.15、torch 2.7.1+cpu、torch_npu 2.7.1；CANN `/usr/local/Ascend/cann-9.0.0-beta.2`，CATLASS `769cd40a8716b28650b6bebb08db4834eea4462f`。卡 6 为 Ascend910B3；实际加载本仓 `flash_attn_npu_3/flash_attn_npu_3.cpython-311-aarch64-linux-gnu.so`，SHA256 `56641e271f3949ebb3e0f66d1f9757afae81a0f4c3f5071be672f01a3f3d7b93`。正确性准备复用构建 1.3 s、runtime 核验 20.8 s；性能准备 1.2 / 21.4 s。

构建/reference 缓存为 `/tmp/flash-attention-npu-mla-46928a793876/mla-validation-artifacts/`；native 包为 `/tmp/mla-m10c3b/native/opp/vendors/mla_ca18aee_transformer`，当前配方/内容有效。性能启动前卡 6 健康、无其他计算进程，快照在 `performance-final-device-before.log`，驱动持有卡锁。`build/` 仅保留无头日志/状态，未改循环计数；临时证据清理后不可继承。

```bash
python3 -m tests.mla --suite split-stage-tools --output /tmp/mla-m10-close/repro-tools
MLA_NATIVE_ROOT=/tmp/mla-m10c3b/native python3 -m tests.mla \
  --suite e2e ordinary split-bsnd-public auto-split-bsnd-public \
  split-tnd-public auto-split-tnd-public split-mixed-public auto-split-mixed-public \
  split-paged-bsnd-public auto-split-paged-bsnd-public \
  split-paged-tnd-public auto-split-paged-tnd-public \
  --device 6 --output /tmp/mla-m10-close/repro-correctness
MLA_NATIVE_ROOT=/tmp/mla-m10c3b/native python3 -m tests.mla \
  --split-stage-performance --suite split-bsnd-profile \
  window-paged-profile window-paged-tnd-profile window-bsnd-profile \
  window-tnd-profile window-mixed-profile split-tnd-profile split-mixed-profile \
  split-paged-bsnd-profile split-paged-tnd-profile \
  auto-split-bsnd-profile auto-split-tnd-profile auto-split-mixed-profile \
  auto-split-paged-bsnd-profile auto-split-paged-tnd-profile \
  --device 6 --output /tmp/mla-m10-close/repro-performance
python3 -m tests.mla.support.mla_split_stage_performance \
  --evidence /tmp/mla-m10-close/repro-performance/summary.json \
  --output /tmp/mla-m10-close/repro-stage.json
```

复现先同参数 dry-run/collect-only；实机和需要查询芯片的收集在宿主执行。日常仍为 `python3 -m tests.mla --device 6`。收尾的 `ci/code_quality.sh --staged`、`git diff --check` 均通过，质量记录为 `quality-final.log`；三份文档的契约、表格及 521 个本地链接/锚点已核对。
