# MLA 当前状态

**M10c10 TND PA 原生自动 split 完整接入，必要验收通过。** 分支 `feat/mla-cann-migration`，接手基线 `49a5704`；当前闭环提交可用 `git log -1 -- docs/mla/status.md` 定位。唯一下一步为 [M10 阶段收尾](plan.md#唯一下一步-m10阶段收尾)，本次提交后停止，不开始下一步。

## 本步行为

512+64、Hkv=1、FP16/BF16 的 TND Q + ND PA KV 支持 `num_splits=0`，包含 none/causal/window、host/AICPU、v26、init/partial/merge、公开 cache/metadata 和 real/fake/compile。五布局现均接受 0..256。v26 为 `756+7B` 字，复用紧凑 Q 原生分区、PA partial 和 TND merge；返回真实 FP32 `[26,H,T,512]` / `[26,H,T]` accumulator，保留原 allocation 地址与未用槽 0/-inf。协议与表示域差异见 [design 6.2.31](design.md#6231-tnd-pa-原生自动-split-完整接入)。

修正上轮搬迁遗留的 12 个文件中的组件导入名，复用共享 PA 固定源 fixture；受影响九个历史模块仅做收集检查，272 个节点成功收集，不计为数值通过。历史 TND 自动拒绝断言改为负 split 拒绝，新自动能力由本步三 mask 正向用例验收。CANN/CATLASS 未修改，共享数值 Block/Tile 算法未更换。

## 当次闭环证据

证据根 `/tmp/mla-m10c10/`，所有集合先 dry-run/collect-only，未使用 `--max-cases` 或 `--resume`，继承通过结果为 0。最终集合启动后只修改文档。源码、实际用例、工具链、原始证据及构建输入/依赖/产物经统一驱动既有函数复核有效，结果见 `final-evidence-check.json`。

| 检查 | 结果 | 用时 |
| --- | --- | --- |
| 协议工具 | 14 passed | 9.7 s |
| 生产工具，含真实 AICPU 编译 | 38 passed | 14.9 s |
| 本模块完整协议 | 452 passed | 47.4 s |
| 同集合 ASan/UBSan/LSan | 452 passed | 155.0 s |
| 固定源 partial/merge 与双数学 | 12 passed | 210.6 s |
| 公开专项 | 100 passed | 123.4 s |
| release Trap | 48 passed | 657.7 s |
| 累计日常 MLA E2E | 314 passed | 60.2 s |
| 普通 FA3 回归 | 516 passed | 26.9 s |
| 固定公开性能 | 12 配置，平均比 0.832628 ≤1 | 285.7 s |

各组均为 0 failed/skipped；公开专项与日常 E2E 有重叠，不相加作为独立用例总数。固定源 24 项 O/LSE 对照全部逐元素一致，双数学通过；真实 FP32 accumulator 同时核对源和双数学，没有新增源精度限制。48 项 Trap 各用新进程/context，全部在设备事件同步阶段拒绝，无超时。

公开性能全部 12 个 dtype/输入配置适用，含八个原生 FD 配置，无排除；三入口及 R=1/4 等权平均。4/12 配置、24/72 分段退化，单例不阻塞平均门槛；保留全部单例、分段、原生分派及 AICPU/init/partial/merge 明细。六段平均比为：隐式 R1/R4 **0.9159/0.8090**，生成 R1/R4 **1.0269/0.7506**，复用 R1/R4 **0.8065/0.6869**。

- `cpu-acceptance/summary.json` 保存协议工具、完整协议与 sanitizer；生产工具最终证据在 `tools-final/summary.json`。完整设备证据为 `acceptance/summary.json`，对应 `attempt-1792066747635981340/`；性能在该 attempt 的 `auto-split-paged-tnd-profile/public-profile/average.json` 及各 dtype 原始报告。
- 长测前 10 项代表/成本检查通过（`device-check/summary.json`，121.8 s），最终公开组重跑覆盖。最大代表 accumulator 52.3 MiB；最慢代表含 reference 33.9 s。代表后补入非空 allocation、Q-used 全零的正向双 dtype 边界，验证无任务初始化；最终公开集合为新模块 94 节点及六个相关历史边界，性能配置未变。
- 当前源码构建 558.4 s，runtime 核验 21.0 s；最终准备核验并复用产物 1.3 s，runtime 20.9 s。数学缓存按内容复用：源数值组 10 hit/2 miss，公开组 27 hit/52 miss，日常 E2E 400 hit/0 miss，普通回归 31 hit/0 miss；实际实现全部执行。
- 开发预检修正了负例夹具的 int32 输入溢出及一个使用 BSND 总量的旧溢出配置；失败保留于 `cpu/`、`cpu-retry/`。沙箱 sanitizer 的 452 项断言通过，但退出时 LSan 被 ptrace 限制阻断，整组记失败（`cpu-final/`）；在宿主保持原配置重跑通过（`cpu-host/`、最终 `cpu-acceptance/`），未关闭检查或缩减集合。

## 环境与复现

Python 3.11.15、torch 2.7.1+cpu、torch_npu 2.7.1；CANN `/usr/local/Ascend/cann-9.0.0-beta.2`，CATLASS `769cd40a8716b28650b6bebb08db4834eea4462f`。宿主卡 6 为 Ascend910B3。实际加载本仓 `flash_attn_npu_3/flash_attn_npu_3.cpython-311-aarch64-linux-gnu.so`，SHA256 `56641e271f3949ebb3e0f66d1f9757afae81a0f4c3f5071be672f01a3f3d7b93`。

产物与数学缓存为 `/tmp/flash-attention-npu-mla-46928a793876/mla-validation-artifacts/`，构建身份由 `prepare.json` 核验；native 包为 `/tmp/mla-m10c3b/native/opp/vendors/mla_ca18aee_transformer`，配方、内容及实际分派均已核验。`build/` 只保留 `mla-codex-loop/`；未改无头循环状态。临时原始证据清理后不可继承，持久保存应设置持久 `MLA_WORK_DIR`。

```bash
python3 -m tests.mla --suite auto-split-paged-tnd-protocol-tools \
  auto-split-paged-tnd-tools unit-auto_split_paged_tnd_protocol \
  auto-split-paged-tnd-protocol-sanitize --output /tmp/mla-m10c10/repro-cpu
MLA_NATIVE_ROOT=/tmp/mla-m10c3b/native python3 -m tests.mla \
  --suite unit-auto_split_paged_tnd auto-split-paged-tnd-public \
  auto-split-paged-tnd-trap e2e ordinary auto-split-paged-tnd-profile \
  --device 6 --output /tmp/mla-m10c10/repro-device
```

复现先同参数 dry-run/collect-only，sanitizer 需可正常执行 LSan 的宿主环境。日常仅用 `python3 -m tests.mla --device 6`，不默认重跑本步单元/专项。提交前差异与 `git diff --check`、仓库适用六项质量 hooks、55 处受影响链接/锚点检查通过（`quality-final.log`、`doc-links.json`）。M10 阶段收尾未运行，192/128 prefill、softcap、非默认调度仍拒绝，完整 graph replay 和 910C 实机尚未验收。
