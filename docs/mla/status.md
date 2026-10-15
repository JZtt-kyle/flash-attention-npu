# MLA 当前状态

**M11b 完整 BSND prefill 已闭环。** 分支 `feat/mla-cann-migration`，接手 `58a5abb` 时工作树干净；本轮只交付 M11b。当前闭环提交以 `git log -1 -- docs/mla/status.md` 定位。唯一下一步是 [M12a 紧凑 TND prefill](plan.md#唯一下一步-m12a)，本轮提交后停止。

## 当前能力与边界

192/128 BSND 的 qv=None、独立 K/V、GQA、完整 storage/stride、有效长度与空输入已接通 host/AICPU/Kernel/metadata/公开 real/fake/compile。无 mask、右下 causal、window 及自动/1/显式 split 同步开放；多片返回真实 FP32 accumulator。v27 复用固定 CANN 通用 M/N、SplitCore 和既有端点消费者，数值与归并按真实 BSND 头布局寻址，见 [design 5.11](design.md#511-192128-bsnd-prefill-记录) 和 [6.5.2](design.md#652-bsnd-prefill-生产接入)。

显式 metadata 可克隆或使用连续 offset view，绑定当前形状、mask、split、平台、dtype 和长度；None 长度按完整容量校验。内部 Q-used 每次生成记录，不接收显式 metadata。TND/混合/PA prefill、softcap、非默认调度和完整 graph replay 尚未开放；910C 未实机验证。本步未新增这些范围的完成声明。

## 最近一次闭环证据

证据根 `/tmp/mla-m11b/`。下表均为本次实际执行，**0 failed、0 skipped；继承 E2E 为 0**。接手时用统一驱动复核的 M11a 六组证据有效，但本步共享入口、归并和构建变化后，累计 E2E 与普通回归全部重跑。未累计历史单元或性能。

| 检查 | 本次结果 | 耗时 | summary |
| --- | --- | --- | --- |
| prefill-bsnd-tools | 32 passed，工具/清单/构建配方/分类与失败传播 | 16.0 s | `cpu-routing/summary.json` |
| unit-prefill_bsnd | 82 passed，固定源切块/分核、边界、损坏与失败原子性 | 20.0 s | 同上 |
| prefill-bsnd-sanitize | 82 passed，宿主 ASan/UBSan | 62.3 s | 同上 |
| prefill-bsnd-check | 8 passed，双 dtype 代表与最大成本输入 | 22.3 s | `public-routing/summary.json` |
| prefill-bsnd-public | 145 passed，公开专项及 12 个固定源数值对照 | 227.7 s | 同上 |
| prefill-bsnd-profile-check | 1 passed，双 dtype native 分派及完整计时夹具 | 204.2 s | 同上 |
| prefill-bsnd-profile | 1 passed，固定 12 配置，适用平均 0.940973≤1 | 267.6 s | `acceptance/summary.json` |
| prefill-bsnd-trap | 38 passed，双 dtype 隔离 release Trap | 535.9 s | 同上 |
| e2e | 340 passed，含新增 prefill_bsnd 小输入清单 | 70.1 s | 同上 |
| ordinary | 516 passed，普通 FA3 回归 | 34.3 s | 同上 |

FP16/BF16 的三 mask × 单/多 split 固定源对照 O/LSE 均逐元素一致，双数学均通过；真实 FP32 partial accumulator 同源检查通过，无新增源精度例外。原始输入、输出和 probe 构建/加载记录在 `public-routing/attempt-1792090679242140430/prefill-bsnd-public/source/`。最大成本输入为 `(B,Sq,Hq,Sk,Hkv)=(1,257,8,4097,2)`、S=8、window、非连续视图：首次 FP16/BF16 约 10.2/12.6 s，进程峰值 RSS 约 1.99 GiB，见 `public-final/attempt-1792089852486032230/prefill-bsnd-check/cost-*.json`；后续相同 reference 命中缓存，最终 check 8/8 命中。

性能目录 `acceptance/attempt-1792091238758619130/prefill-bsnd-profile/public-profile/` 保存 `average.json`、双 dtype `profile.json`、wall/Kernel 原始样本、分派与设备状态。12 配置均实测为固定包 template 3 / native FD；自动 split 的 6 配置参与平均，单/显式 split 的 6 个原生 FD 对照单列。配置均值退化 0，分段退化 12；implicit R=1 与 func R=1 的分段均比为 1.214636、1.240903，其余入口/R 均值保存在汇总。Kernel 12/12 较原生慢，所有耗时保留；公开验收按 design 的配置内入口/R 等权、再配置/dtype 等权口径通过。预热、重复次数、阈值及同步边界未变。

复现均使用 `python3 -m tests.mla`，各集合已先 dry-run、再 collect-only：

```bash
python3 -m tests.mla --suite prefill-bsnd-tools unit-prefill_bsnd prefill-bsnd-sanitize --output /tmp/mla-m11b/cpu-routing
MLA_NATIVE_ROOT=/tmp/mla-m10c3b/native python3 -m tests.mla --suite prefill-bsnd-check prefill-bsnd-public prefill-bsnd-profile-check --device 6 --output /tmp/mla-m11b/public-routing
MLA_NATIVE_ROOT=/tmp/mla-m10c3b/native python3 -m tests.mla --suite prefill-bsnd-profile prefill-bsnd-trap e2e ordinary --device 6 --output /tmp/mla-m11b/acceptance
```

统一驱动在完成时核对实际清单、源码/构建依赖和加载身份；收尾使用其 `validate_result`、`reusable`、`matching_files` 复核原始证据，结果 `evidence-review.json`，不是额外整树审计。新公开源 probe 位于 `source/<dtype>`，另用同一文件校验函数核对两份 manifest 的引用依赖、库与加载记录。数学缓存与测试通过证据分开管理；这些 `/tmp` 证据被清理后不得继承。

开发中修正了设备地址空间/宏定义、单 split 空 accumulator 默认 dtype、None 长度复用，以及 func 对 qv=None 的分派。早期沙箱 sanitizer 的 ptrace 限制、源码变化后被驱动拒绝或主动停止的构建、`public-final` 的 func 路由失败均未计为通过；修正后受影响组已重跑。当前实现无遗留失败。

收尾核对本次 26 个文档链接/锚点、`git diff --check` 与暂存差异均通过；仓库 `ci/code_quality.sh --staged` 通过，日志为 `quality-final.log`。只提交本步 45 个文件。

## 环境与产物

卡 6 为 Ascend910B3、20 AIC；Python 3.11.15、torch 2.7.1+cpu、torch_npu 2.7.1，CANN `/usr/local/Ascend/cann-9.0.0-beta.2`。CATLASS `769cd40a8716b28650b6bebb08db4834eea4462f`，固定 CANN 源 `ca18aee610729ca5270a44e2d84c751bc687fbf0`，FA3 `d15f1531a460ba456f41b01a774f33ab2db8febf`。CANN 源仓只读，CATLASS 未修改。

实际加载仓库 `flash_attn_npu_3/flash_attn_npu_3.cpython-311-aarch64-linux-gnu.so`，SHA256 `3ce3ab7ab45ebed4a3499c9ddf43ddbeef1253bb436c018465332116c69d28d8`；完整重建 544.6 s，后续 prepare 校验命中。最终环境/库记录为 `acceptance/attempt-1792091238758619130/runtime.json`，生产 manifest 为 `/tmp/flash-attention-npu-mla-46928a793876/mla-validation-artifacts/prepare.json`。

`MLA_WORK_DIR` 使用默认 `/tmp/flash-attention-npu-mla-46928a793876`，构建/reference 位于其 `mla-validation-artifacts/`，组件位于 `prefill-probes/`；`build/` 仅保留无头日志与状态。固定 native 包为 `/tmp/mla-m10c3b/native/opp/vendors/mla_ca18aee_transformer`，103/104 配方完整，实际分派证据如上。FA3 固定副本位于 `/tmp/mla-fa3-reference/d15f1531a460ba456f41b01a774f33ab2db8febf/`。
