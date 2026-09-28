# MLA 当前交接

## 状态与下一步

- 当前步骤：M04b2b2b2b2a 已完成，KV ≥385 的在线路径将 Q 的 L1 分区压紧至 16 行，增加第三个 KV 槽并在 PV 前预取下一块；支持范围仍为 KV `1..1024`。
- 当前分支：`feat/mla-cann-migration`；接手基点 `c624a4d`。方案已确认，本轮一个提交后停止。
- 本步产物：[在线 Cube Block](../../csrc/ascend910/flash_attn_npu_3/mla/mla_online_mmad.hpp)、[Kernel 调度](../../csrc/ascend910/flash_attn_npu_3/mla/mla_short_kernel.hpp)、[缓冲复用检查](../../tests/test_flash_attn_npu_v3_mla_online.py)。
- 唯一下一步：M04b2b2b2b2b，继续 KV `>1024` 的长 KV 调度优化；验收包含 8193 单/多 query、FP16/BF16 O/LSE、重复执行及同轮 CANN 性能，保持显式 `num_splits=1`。

M04b2b2b2b2 在实现前拆为本步 KV 预取和后续长 KV 调度。**本步未完成 8193 验收，没有接通 >1024。** 历史 8193 探测存在明显退化，本轮 1..1024 的收益不能外推为长 KV 达标。

## 当前入口与实现

- `flash_attn_func(q, k, v, qv=...)` 支持 FP16/BF16、Dq=64/Dv=512/Hkv=1、连续 BSND、正 batch/query 长度、KV `1..1024`、无 mask、显式 `num_splits=1`。O 为 `[B,Sq,Hq,512]`，自然对数 FP32 LSE 为 `[B,Hq,Sq]`。
- KV `1..128` 保留单 tile 路径；`129..1024` 使用 128 列在线 softmax。最大值、分母和输出分子保留 FP32；默认 scale 为 `1/sqrt(576)`，显式有限 scale 支持零/负值。
- KV `129..384` 保留双缓冲及原 L1 分区；`385..1024` 使用三槽预取。三槽路径的 Q L1 分区为 18 KiB，三个 KV 槽各 144 KiB，P 为 4 KiB，总计 454 KiB，并以静态断言检查 512 KiB 上限。Q 仍常驻 L0A；PV 使用另一半 L0A，四段输出在同一 L0C 缓冲中一次写回。
- 三槽路径先加载 Q，再预取 KV(0)；每步 QK(i) 后发出 KV(i+1) 加载，再执行 PV(i-1)。三槽独立于双槽 L0C 和 GM workspace；内容在 PV 最后一次 L1 读取后释放，RoPE 在 QK 最后一次读取后释放。三槽 KV 使用事件 0..5，Q 加载使用独立事件 6，P 使用事件 7。双槽路径保留 KV 事件 0..3，Q 在首次 KV 加载前使用并消费事件 0。
- numerator 提前缩放、FP32 数学运算顺序、score/partial UB 释放和 task 排空保持。每个 task 遍历完整 KV，没有 split/合并；KV >512 且空闲核足够时按 8 个 head 分工，否则 16 个。仅实测 910B C1V2。
- 公共入口、metadata、短路径事件顺序、CATLASS 和 CANN 源仓未改；保留来源和版权声明。MLA prefill、varlen/kvcache、KV >1024、空 KV/零 query、外层非连续 stride、mask/window、自动/多 split、softcap、pack_gqa/sm_margin、metadata、追加 KV、out、实际长度/页表/rotary/descale 和梯度请求仍明确拒绝；图执行尚未验收。

## 最近闭环证据

| 检查 | 结果 |
| --- | --- |
| 接手核对 | PASS：工作区干净；上轮 472 项 checksum、27 项源码/扩展/对象/依赖哈希匹配，CANN/CATLASS 状态一致；本轮复核原版 `606 passed` |
| 构建溯源 | PASS：仅 v3/910，退出码 0；四个 MLA 单元跟踪 online MMAD header；最终仅两个 online 对象变化，short、flash_api 和 9 个普通 FA3/AICPU 对象哈希保持 |
| 910B MLA | PASS：完整 `620 passed`，无 skip；契约 233、short 141、online 246 |
| 针对性覆盖 | 内容/RoPE 首尾及每个 Cube K 分段边界；新增 385/641/769/897 尾块，覆盖三槽 KV 与双槽 workspace 的不同轮换组合、满行/单行/空 lane 跨 task 复用、重复执行和输入保持 |
| 固定源码 CANN | PASS：100 个 shape/scale/输入模式 × FP16/BF16，共 200 组；原双 reference/comparator 阈值未改；200 次 FIA launch 匹配配套 vendor binary |
| 普通 FA3 | PASS：`11 passed`，无 skip；dense/varlen/kvcache/FlashDecode、metadata、编译图及 dense/varlen backward |
| CANN 性能 | PASS（已授权共享卡）：既定 32 例同轮交替，kernel 比值 `0.548..0.824`、同步调用比值 `0.825..0.850`，逐例低于 1；kernel 配对 bootstrap 95% 区间上界最高约 `0.8329` |
| 上版对照 | 50 例 O/LSE 逐位一致；25 例 kernel 区间完全低于 1，中位比值 `0.8402..0.9905`；其余 25 例区间跨 1，没有证实提速或退化；无明确 kernel 退化例 |
| 既有 CPU | 本步未修改 reference/profile/provenance，继承既有记录，不记为本轮重跑 |
| 质量与完整性 | PASS：`ci/code_quality.sh --staged`、文档检查、`git diff --check` 和 `final_integrity.py`；结果及哈希保存在本轮证据目录 |

第一版先加载 KV(0) 再加载 Q：完整 620 项 MLA、200 组 CANN 和 11 项普通 FA3 通过，32 例 CANN 性能也通过，但旧新版 D01 BF16 默认/显式 scale 的 kernel 中位数分别退化约 0.7%/1.1%。仅恢复 Q 先加载的候选仍未消除 D01 小幅退化，证据在 `build/mla-m04b2b2b4q-validation/`。最终版本保留 KV 129..384 的双缓冲及原 L1 分区，仅在 ≥385 时启用三槽预取；先加载 Q，使 Q 的 L1→L0A 与首次 KV 加载可以重叠。旧新版计时使用相同函数包装器绑定入口。最终源码与扩展独立完成上述全部验收，原始结果保留，不以补测覆盖第一版结果。

选择性分派曾因模板参数与方法同名而构建失败，已修复。随后误启动的 `compare-selective-pilot` 已取消，明确记为无效，不计入验收；相关日志和 `invalid-pilot.json` 保留。重跑脚本使用构建成功后才启动测试的顺序。

第一版的 8 例 pilot 还曾在采集前被设备监控拦截（精度测试 driver PID `2836026` 正在退出清理），等待退出后使用新标签重跑；未采集被拦截窗口的性能。第一版与其 pilot 证据保留在 `build/mla-m04b2b2b4-validation/`。

CANN 同轮性能沿用每实现预热 10 次，同步调用 7 轮 × 50 对 AB/BA 交替，profiler 独立 3 轮 × 10 对，1920 次计时调用各对应一个 kernel，无 Slice 转换。相同输入/dtype/scale，CANN 原生自动策略、MLA 单 split，无 metadata；逐例门槛 `<=1.0` 保持。旧新版使用相同 C++ 参数，既定 32 例加 18 例跨 task 复用，共 3000 次计时调用，单列并排除 1 个 FIA setup kernel。

## 设备条件与环境

- 使用此前已授权的共享物理 3 号 Ascend910B3，进程内 npu:0；正式采集前健康、AICore 0%，唯一后台 PID `2939247`，136 MB；没有终止其他进程。
- CANN 对比与旧新版对照采集窗口分别有 13、41 个设备快照（含结束后的首个边界样本），持续健康，仅出现本轮 driver PID 和已授权后台；原始快照中该后台均为 136 MB。正式 profiler 日志无 ERROR。
- **本轮属于共享设备测量，不是独占卡计时。** 两实现同轮交替，但不能排除已知后台进程的瞬时影响；后续须重新检查设备并记录实际条件。

| 项目 | 本轮结果（2026-09-28） |
| --- | --- |
| Python / 包 | `/usr/local/python3.11.15/bin/python` 3.11.15；torch `2.7.1+cpu`，torch_npu `2.7.1` |
| CANN / 固定 FIA 源 | `/usr/local/Ascend/cann-9.0.0-beta.2`；`ca18aee610729ca5270a44e2d84c751bc687fbf0` |
| CATLASS | `769cd40a8716b28650b6bebb08db4834eea4462f`，干净且未修改 |
| 实际扩展 | 本仓 `flash_attn_npu_3/flash_attn_npu_3.cpython-311-aarch64-linux-gnu.so`；SHA256 `47682e0180958826ccb08e2b4142a03ff7b9f83029f93287eec27fcab1ffb7bb` |
| 芯片 | Ascend910B3，物理 3 号；20 Cube/40 Vector，用户授权共享卡测量 |

## 证据与复现

持久证据根目录：`/home/zhangqijia/repo/flash-attention-npu/build/mla-m04b2b2b4f-validation/`。

- `takeover-audit.json`、`environment-before.json` / `-after.json`、`before/`、`mla-takeover-*`：接手核对、实际加载扩展及原版复核。
- `build-final-command.json` / `.log` / `-exit.json`、`build-provenance-final.json`：最终构建命令及源码/对象/依赖/扩展哈希。
- `mla-final-*`、`regression-final-*`、`run_validation.py`、`run_final_checks.py`、`run_regression.py`：完整验收命令、退出码、日志/JUnit 及精确节点。
- `cann-accuracy.py` / `.json`、`dispatch-accuracy.log`、`verify_dispatch.py`、`dispatch-proof.json`：200 组精度及 vendor binary 来源。
- `compare.py`、`compare-cann-final.json`、`performance-acceptance.json`、`profile-cann-final/`：CANN 同轮计时及逐例验收。
- `compare-versions-final.py` / `.json`、`versions-summary.json`、`profile-versions-final/`：旧新版逐位一致、逐例计时和被排除的 setup kernel。
- `*-device-monitor.json`、`*-device-proof.json`、`verify_monitor.py`：原始快照、driver PID 映射、采集窗口和共享条件。
- `schedule-review.json`、`compare-selective-pilot*`、`build-selective-pilot*`、`build-selective-retry*`：L1 分区、事件与生命周期审查、构建失败及重跑、最终调度的针对性探测；初版候选及完整证据见上文目录。
- `quality-staged.log`、`document-check-final.log`、`final-integrity.json`、`checksums.sha256`：质量、文档和完整性。

实机使用 `login=false`、沙箱外。运行器固定物理 3 号、线程数及隔离 cache/plog，保留 CANN PYTHONPATH；仅 CANN/比较/分派标签加载固定 vendor。复跑前检查设备，使用新标签/输出路径，不覆盖既有证据；设备/共享条件变化时同步调整运行器、监控和 proof。

```bash
FLASH_ATTN_BUILD_VERSION=v3 FLASH_ATTN_BUILD_NPU=910 \
  FLASH_ATTN_SKIP_SUBMODULE_INIT=1 python setup.py build_ext --inplace

python build/mla-m04b2b2b4f-validation/run_command.py mla-repro \
  timeout 600 python -m pytest -q tests/test_flash_attn_npu_v3_mla_online.py \
  tests/test_flash_attn_npu_v3_mla_short.py tests/test_flash_attn_npu_v3_mla_contract.py

python build/mla-m04b2b2b4f-validation/monitored_run.py compare-repro \
  timeout 300 python build/mla-m04b2b2b4f-validation/compare.py \
  --output build/mla-m04b2b2b4f-validation/compare-repro.json \
  --profile-dir build/mla-m04b2b2b4f-validation/profile-repro --iterations 50 --rounds 7
```

质量工具沿用 `/tmp/mla-m01-precommit`（PYTHONPATH）和 `/tmp/mla-m01-precommit-cache`（PRE_COMMIT_HOME），执行 `ci/code_quality.sh --staged`，不绕过钩子。固定源码包仍为 `build/mla-m02b-validation/opp-dense/`；上一步证据在 `build/mla-m04b2b2b3-validation/`，全部原始 checksum 保持。未修改 reference/comparator、CANN 基线执行器/包、CATLASS 或 CANN 源仓；源仓既有其他状态保留。当前闭环提交可用 `git log -1 -- docs/mla/status.md` 定位。
