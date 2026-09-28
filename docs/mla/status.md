# MLA 当前交接

## 状态与下一步

- 当前步骤：M04b2b2b2b1 已完成，将在线输出分子的缩放提前到跨核 PV 等待前，允许独立向量计算与 PV/partial 加载重叠；支持范围仍为 KV `1..1024`。
- 当前分支：`feat/mla-cann-migration`；接手基点 `cf4593d`。方案已确认，本轮一个提交后停止。
- 本步产物：[在线向量 Block](../../csrc/ascend910/flash_attn_npu_3/mla/mla_online_epilogue.hpp)、[Kernel 调度](../../csrc/ascend910/flash_attn_npu_3/mla/mla_short_kernel.hpp)、[缓冲复用检查](../../tests/test_flash_attn_npu_v3_mla_online.py)。
- 唯一下一步：M04b2b2b2b2，继续 KV `>1024` 的长 KV 调度优化；验收包含 8193 单/多 query、FP16/BF16 O/LSE、重复执行及同轮 CANN 性能，保持显式 `num_splits=1`。

M04b2b2b2b 在实现前拆为本步独立向量指令提前发出和后续长 KV 调度。**本步未完成 8193 验收，没有接通 >1024。** 910B 的 `WaitEventImpl` 仍使用标量 `wait_flag_dev`；这次只将不依赖当前 partial 的 numerator 缩放移到等待之前，不能据此推断长 KV 已达标。

## 当前入口与实现

- `flash_attn_func(q, k, v, qv=...)` 支持 FP16/BF16、Dq=64/Dv=512/Hkv=1、连续 BSND、正 batch/query 长度、KV `1..1024`、无 mask、显式 `num_splits=1`。O 为 `[B,Sq,Hq,512]`，自然对数 FP32 LSE 为 `[B,Hq,Sq]`。
- KV `1..128` 保留单 tile 路径；`129..1024` 使用 128 列 tile 的在线 softmax。最大值、分母和输出分子保留 FP32；默认 scale 为 `1/sqrt(576)`，显式有限 scale 支持零/负值。
- `Softmax(step)` 写 `alpha[step % 2]` 后，先发出 `RescaleNumerator(step - 1)`，再等待 PV、加载并累加 partial。缩放读取另一 alpha 槽；tile 0 和零行 lane 不缩放。原向量数学运算顺序、独立 score/partial UB 释放事件、最终归一化及 task 排空保持。
- Q 常驻 L0A、PV 四段一次写回、KV 的 L1 双缓冲、workspace、Cube 调度和 host 调度保持。每个 task 遍历完整 KV，没有 split/合并；KV >512 且空闲核足够时按 8 个 head 分工，否则 16 个。仅实测 910B C1V2。
- 公共入口、metadata、CATLASS、CANN 源仓及生产调用路径未改；保留来源和版权声明。MLA prefill、varlen/kvcache、KV >1024、空 KV/零 query、外层非连续 stride、mask/window、自动/多 split、softcap、pack_gqa/sm_margin、metadata、追加 KV、out、实际长度/页表/rotary/descale 和梯度请求仍明确拒绝。MLA 图执行尚未验收。

## 最近闭环证据

| 检查 | 结果 |
| --- | --- |
| 接手核对 | PASS：工作区干净；上轮 273 项 checksum、27 项源码/扩展/对象/依赖哈希匹配，CANN/CATLASS 状态一致 |
| 构建溯源 | PASS：仅 v3/910，退出码 0；四个 MLA 单元均跟踪 online epilogue header；仅两个 online 对象变化，short、flash_api 和 9 个普通 FA3/AICPU 对象哈希保持 |
| 910B MLA | PASS：完整 `606 passed`，无 skip；契约 233、short 141、online 232 |
| 针对性覆盖 | 扩大既有 8 个复用节点至 129/255/257/384/513/1024 × 1/17/32/33 heads × FP16/BF16；最大值变化、首尾块、跨 task 满行/单行/空 lane 切换、重复执行及输入保持通过 |
| 固定源码 CANN | PASS：77 个 shape/scale/输入模式 × FP16/BF16，共 154 组；原双 reference/comparator 阈值未改；154 次 FIA launch 匹配配套 vendor binary |
| 普通 FA3 | PASS：`11 passed`，无 skip；dense/varlen/kvcache/FlashDecode、metadata、编译图及 dense/varlen backward |
| CANN 性能 | PASS（已授权共享卡）：既定 32 例同轮交替，kernel 比值 `0.571..0.959`、同步调用比值 `0.834..0.887`，逐例低于 1；kernel 配对 bootstrap 95% 区间上界最高约 `0.9661` |
| 上版对照 | 42 例 O/LSE 逐位一致；14 例 kernel 区间完全低于 1，中位比值 `0.9814..0.9957`，约 0.4%..1.9% 收益；其余 28 例区间跨 1，没有证实提速或退化；无明确退化例 |
| 既有 CPU | 本步未修改 reference/profile/provenance，继承既有记录，不记为本轮重跑 |
| 质量与完整性 | 提交前执行 `ci/code_quality.sh --staged`、文档检查、`git diff --check` 和 `final_integrity.py`；结果及哈希保存在本轮证据目录 |

本轮最初尝试在 softmax 前等待并预取上一块 partial：针对性 10 项精度通过，但 6 例旧新版 pilot kernel 中位比值为 `1.013..1.063`，减少了 Cube/Vector 重叠，已撤销该顺序。最终提前缩放版本的完整验收独立进行；两个候选的源码、扩展、构建与 pilot 原始数据均保留，决策及缓冲生命周期见 `schedule-review.json`。一次旧新版汇总因 producer 尚未写出 JSON 而失败，待采集成功结束后重跑通过，记录在 `summary-early-attempt.json`。

CANN 同轮性能沿用每实现预热 10 次，同步调用 7 轮 × 50 对 AB/BA 交替，profiler 独立 3 轮 × 10 对，1920 次计时调用各对应一个 kernel，无 Slice 转换。相同输入/dtype/scale，CANN 原生自动策略、MLA 单 split，无 metadata；逐例门槛 `<=1.0` 保持。旧新版使用相同 C++ 参数，既定 32 例加 10 例跨 task 复用，共 2520 次计时调用，单列并排除 1 个 FIA setup kernel。正式旧新版结果只在上述 14 例证实小幅收益，不把 pilot 或局部结果推广到其余场景。

## 设备条件与环境

- 使用此前已授权的共享物理 3 号 Ascend910B3，进程内 npu:0；两轮正式性能采集前健康、AICore 0%，唯一后台 PID `2939247`，136 MB；没有终止其他进程。
- CANN 对比采集窗口 13 个、旧新版对照 33 个设备快照（含采集结束后的首个边界样本）持续健康，仅出现本轮 driver PID 和已授权后台；所有原始快照中该后台均为 136 MB。两份正式 profiler 日志无 ERROR。
- **本轮属于共享设备测量，不是独占卡计时。** 两实现同轮交替，但不能排除已知后台进程的瞬时影响。后续验收须重新检查设备并记录实际条件。

| 项目 | 本轮结果（2026-09-28） |
| --- | --- |
| Python / 包 | `/usr/local/python3.11.15/bin/python` 3.11.15；torch `2.7.1+cpu`，torch_npu `2.7.1` |
| CANN / 固定 FIA 源 | `/usr/local/Ascend/cann-9.0.0-beta.2`；`ca18aee610729ca5270a44e2d84c751bc687fbf0` |
| CATLASS | `769cd40a8716b28650b6bebb08db4834eea4462f`，干净且未修改 |
| 实际扩展 | 本仓 `flash_attn_npu_3/flash_attn_npu_3.cpython-311-aarch64-linux-gnu.so`；SHA256 `55226b848b28ce7072d643df5508d74aab46ec392bc205988cb049fdd4859d43` |
| 芯片 | Ascend910B3，物理 3 号；20 Cube/40 Vector，用户授权共享卡测量 |

## 证据与复现

持久证据根目录：`/home/zhangqijia/repo/flash-attention-npu/build/mla-m04b2b2b3-validation/`。

- `takeover-audit.json`、`environment-before.json`、`environment-after.json`、`before/`：接手核对、实际加载扩展及已验收上版。
- `build-rescale-command.json` / `.log` / `-exit.json`、`build-provenance-rescale.json`：最终构建命令及源码/对象/依赖/扩展哈希。
- `mla-final-*`、`regression-final-*`、`run_final_checks.py`、`run_regression.py`：完整验收命令、退出码、日志/JUnit 及精确节点。
- `cann-accuracy.py` / `.json`、`dispatch-accuracy.log`、`verify_dispatch.py`、`dispatch-proof.json`：154 组精度及 vendor binary 来源。
- `compare.py`、`compare-cann-final.json`、`performance-acceptance.json`、`profile-cann-final/`：CANN 同轮计时及逐例验收。
- `compare-versions-final.py` / `.json`、`versions-summary.json`、`profile-versions-final/`：旧新版逐位一致、逐例计时和被排除的 setup kernel。
- `*-device-monitor.json`、`*-device-proof.json`、`verify_monitor.py`：原始快照、driver PID 映射、采集窗口和共享条件。
- `schedule-review.json`、`partial-prefetch-candidate/`、`compare-*-pilot*`、`build-prefetch-*`、`mla-prefetch-targeted-*`：同步源码核查、被撤销候选及两种调度的探索结果。
- `quality-staged.log`、`document-check-final.log`、`final-integrity.json`、`checksums.sha256`：质量、文档和完整性。

实机使用 `login=false`、沙箱外。运行器固定物理 3 号、线程数及隔离 cache/plog，保留 CANN PYTHONPATH；仅 CANN/比较/分派标签加载固定 vendor。复跑前检查设备，使用新标签/输出路径，不覆盖既有证据；设备/共享条件变化时同步调整运行器、监控和 proof。

```bash
FLASH_ATTN_BUILD_VERSION=v3 FLASH_ATTN_BUILD_NPU=910 \
  FLASH_ATTN_SKIP_SUBMODULE_INIT=1 python setup.py build_ext --inplace

python build/mla-m04b2b2b3-validation/run_command.py mla-repro \
  timeout 600 python -m pytest -q tests/test_flash_attn_npu_v3_mla_online.py \
  tests/test_flash_attn_npu_v3_mla_short.py tests/test_flash_attn_npu_v3_mla_contract.py

python build/mla-m04b2b2b3-validation/monitored_run.py compare-repro \
  timeout 300 python build/mla-m04b2b2b3-validation/compare.py \
  --output build/mla-m04b2b2b3-validation/compare-repro.json \
  --profile-dir build/mla-m04b2b2b3-validation/profile-repro --iterations 50 --rounds 7
```

质量工具沿用 `/tmp/mla-m01-precommit`（PYTHONPATH）和 `/tmp/mla-m01-precommit-cache`（PRE_COMMIT_HOME），执行 `ci/code_quality.sh --staged`，不绕过钩子。固定源码包仍为 `build/mla-m02b-validation/opp-dense/`；上一步闭环证据在 `build/mla-m04b2b2b2-validation/`。未修改 reference/comparator、CANN 基线执行器/包、CATLASS 或 CANN 源仓，源仓既有其他状态保留。当前闭环提交可用 `git log -1 -- docs/mla/status.md` 定位。
