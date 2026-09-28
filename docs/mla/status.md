# MLA 当前交接

## 状态与下一步

- 当前步骤：M04b2b2b2a 已完成，在线 score/partial UB 使用独立释放事件，消除对另一缓冲区向量计算的额外等待；支持范围仍为 KV `1..1024`。
- 当前分支：`feat/mla-cann-migration`；接手基点 `63bbb8d`。方案已确认，本轮一个提交后停止。
- 本步产物：[在线向量 Block](../../csrc/ascend910/flash_attn_npu_3/mla/mla_online_epilogue.hpp)、[缓冲复用与 lane 切换检查](../../tests/test_flash_attn_npu_v3_mla_online.py)。
- 唯一下一步：M04b2b2b2b，继续 KV `>1024` 的长 KV 调度；先处理跨核等待/指令发出顺序的限制，验收包含 8193 单/多 query、FP16/BF16 O/LSE、重复执行及同轮 CANN 性能，保持显式 `num_splits=1`。

M04b2b2b2 在实现前拆为本步 UB 事件解耦和后续长 KV 调度。**本步没有证实提速**：旧新版 40 例 kernel 置信区间全部跨 1。910B 的 `WaitEventImpl` 仍使用标量 `wait_flag_dev`；仅移动 UB 释放点不保证跨核等待下的加载提前执行，见 `sync-review.json`。此前仅放宽限界的 8193 探测存在性能退化；本步未完成 8193 验收，没有接通 >1024。

## 当前入口与实现

- `flash_attn_func(q, k, v, qv=...)` 支持 FP16/BF16、Dq=64/Dv=512/Hkv=1、连续 BSND、正 batch/query 长度、KV `1..1024`、无 mask、显式 `num_splits=1`。O 为 `[B,Sq,Hq,512]`，自然对数 FP32 LSE 为 `[B,Hq,Sq]`。
- KV `1..128` 保留单 tile 路径；`129..1024` 使用 128 列 tile 的在线 softmax。最大值、分母和输出分子保留 FP32；默认 scale 为 `1/sqrt(576)`，显式有限 scale 支持零/负值。
- score 的 `V_MTE2` 事件 0 在最后一次归约读取之后置位；partial 的独立事件 1 在复制/累加到 numerator 后置位。加载各自只等待对应缓冲区被释放；构造时各置位一次，析构各等待一次。零行 lane 不消费或重复置位，跨 task 切换仍保留一个可用事件。
- 数学指令顺序、Q 常驻 L0A、PV 四段一次写回、KV 的 L1 双缓冲、workspace、host 调度和跨 task 排空保持。每个 task 遍历完整 KV，没有 split/合并；KV >512 且空闲核足够时按 8 个 head 分工，否则 16 个。仅实测 910B C1V2。
- 未修改公共入口、metadata、CATLASS、CANN 源仓或生产 CANN 调用路径；保留来源和版权声明。MLA prefill、varlen/kvcache、KV >1024、空 KV/零 query、外层非连续 stride、mask/window、自动/多 split、softcap、pack_gqa/sm_margin、metadata、追加 KV、out、实际长度/页表/rotary/descale 和梯度请求仍明确拒绝。MLA 图执行尚未验收。

## 最近闭环证据

| 检查 | 结果 |
| --- | --- |
| 接手核对 | PASS：工作区干净；上轮 296 项 checksum、27 项源码/扩展/对象/依赖哈希匹配，CANN/CATLASS 状态一致 |
| 构建溯源 | PASS：仅 v3/910，退出码 0；四个 MLA 单元均跟踪 online epilogue header；仅两个 online 对象变化，short、flash_api 和 9 个普通 FA3/AICPU 对象哈希保持 |
| 910B MLA | PASS：完整 `604 passed`，补充 `2 passed`，共 606 个不同节点，无 skip；契约 233、short 141、online 232 |
| 针对性检查 | PASS：先执行 8 项（包含 6 项新增复用检查及既有连续入队），完整检查后补充 33-head 的 FP16/BF16 两项；新增 8 项覆盖 129/257/1024、1/17/32/33 heads、最大值变化、重复执行和输入保持 |
| 固定源码 CANN | PASS：65 个 shape/scale/输入模式 × FP16/BF16，共 130 组；均通过原双 reference/comparator，阈值未改；130 次 FIA launch 匹配配套 vendor binary |
| 普通 FA3 | PASS：`11 passed`，无 skip；dense/varlen/kvcache/FlashDecode、metadata、编译图及 dense/varlen backward |
| CANN 性能 | PASS（用户授权共享卡）：既定 32 例同轮交替，kernel 比值 `0.590..0.958`、同步调用比值 `0.820..0.881`，逐例低于 1；kernel 配对 bootstrap 95% 区间上界最高约 `0.9693` |
| 上版对照 | 40 例 O/LSE 逐位一致；32 例既定场景加 8 例跨 task 复用场景，全部 kernel 区间跨 1，没有证实提速或退化 |
| 既有 CPU | 本步未修改 reference/profile/provenance，继承既有记录，不记为本轮重跑 |
| 质量与完整性 | 提交前执行 `ci/code_quality.sh --staged`、文档检查、`git diff --check` 和 `final_integrity.py`；结果及哈希保存在本轮证据目录 |

完整检查收集 604 项后，复核调度发现 17 heads 的两个 head tile 在 20 核步进时不能让同一核切换满行/单行，因此再补充 33 heads（3 个 head tile）两项。3 与 20 互质，38 个 query token 使每个核都覆盖满行和单行/空 lane 切换。补充只增加参数及说明，没有改变原 604 项行为；JUnit 节点并集核对为 606。

CANN 同轮性能沿用每实现预热 10 次，同步调用 7 轮 × 50 对 AB/BA 交替，profiler 独立 3 轮 × 10 对，1920 次计时调用各对应一个 kernel，无 Slice 转换。相同输入/dtype/scale，CANN 原生自动策略、MLA 单 split，无 metadata；逐例门槛 `<=1.0` 保持。旧新版使用相同 C++ 参数，2400 次计时调用，单列并排除 1 个 FIA setup kernel。两份报告各自保留原始数据及配对区间，不把本步结果推断为 8193 达标。

## 设备条件与环境

- 使用此前用户授权的相对空闲共享物理 3 号 Ascend910B3，进程内 npu:0；每次性能采集前健康、AICore 0%，唯一后台 PID `2939247`，136 MB。没有终止其他进程。
- CANN 对比采集窗口 14 个、旧新版对照 28 个设备快照（含采集结束后的首个边界样本）持续健康，仅出现本轮 driver PID 和已获授权后台；所有原始快照的后台占用均为 136 MB。两份 profiler 日志无 ERROR。
- **本轮属于共享设备测量，不是独占卡计时。** 两实现同轮交替，但不能排除已知后台进程造成的瞬时影响。后续验收仍需重新检查设备并记录实际条件。
- 首次质量检查因 Ruff 自动格式化测试文件退出 1；接受格式调整后重跑通过，最终 staged 检查另存。

| 项目 | 本轮结果（2026-09-28） |
| --- | --- |
| Python / 包 | `/usr/local/python3.11.15/bin/python` 3.11.15；torch `2.7.1+cpu`，torch_npu `2.7.1` |
| CANN / 固定 FIA 源 | `/usr/local/Ascend/cann-9.0.0-beta.2`；`ca18aee610729ca5270a44e2d84c751bc687fbf0` |
| CATLASS | `769cd40a8716b28650b6bebb08db4834eea4462f`，干净且未修改 |
| 实际扩展 | 本仓 `flash_attn_npu_3/flash_attn_npu_3.cpython-311-aarch64-linux-gnu.so`；SHA256 `1a477155b742a95f1dce1a62f5da85ca8deccd025ff6e2eb64ddc5bb1fb1935d` |
| 芯片 | Ascend910B3，物理 3 号；本机平台配置为 20 Cube/40 Vector；用户授权共享卡测量 |

## 证据与复现

持久证据根目录：`/home/zhangqijia/repo/flash-attention-npu/build/mla-m04b2b2b2-validation/`。

- `takeover-audit.json`、`environment-before.json`、`environment-after.json`、`before/`：接手核对、实际加载扩展和已验收上版。
- `build-buffer-release-command.json` / `.log` / `-exit.json`、`build-provenance-buffer-release.json`：构建命令及源码/对象/依赖/扩展哈希。
- `mla-targeted-*`、`mla-final-*`、`mla-transitions-*`、`regression-final-*`：命令、退出码、日志/JUnit；`run_final_checks.py` 和 `run_regression.py` 保存本轮执行流程/精确节点。
- `cann-accuracy.py` / `.json`、`dispatch-accuracy.log`、`verify_dispatch.py` / `dispatch-proof.json`：130 组精度及 vendor binary 来源。
- `compare.py`、`compare-cann-final.json`、`performance-acceptance.json`、`profile-cann-final/`：CANN 同轮计时及逐例验收。
- `compare-versions-setup.py`、`compare-versions-final.json`、`versions-summary.json`、`profile-versions-final/`：旧新版逐位一致、逐例计时和被排除的 setup kernel。
- `*-device-monitor.json`、`*-device-proof.json`、`verify_monitor.py`：原始快照、driver PID 映射、采集窗口和共享条件；`platform-910b3.ini`、`sync-review.json` 保存 lane 切换和同步实现的核查依据。
- `quality-staged.log`、`document-check-final.log`、`final-integrity.json`、`checksums.sha256`：质量、文档和完整性。

实机使用 `login=false`、沙箱外。运行器固定物理 3 号、线程数及隔离 cache/plog，保留 CANN PYTHONPATH；仅 CANN/比较/分派标签加载固定 vendor。复跑前检查设备，使用新标签/输出路径，不覆盖既有证据；设备/共享条件变化时同步调整运行器、监控和 proof。

```bash
FLASH_ATTN_BUILD_VERSION=v3 FLASH_ATTN_BUILD_NPU=910 \
  FLASH_ATTN_SKIP_SUBMODULE_INIT=1 python setup.py build_ext --inplace

python build/mla-m04b2b2b2-validation/run_command.py mla-repro \
  timeout 600 python -m pytest -q tests/test_flash_attn_npu_v3_mla_online.py \
  tests/test_flash_attn_npu_v3_mla_short.py tests/test_flash_attn_npu_v3_mla_contract.py

python build/mla-m04b2b2b2-validation/monitored_run.py compare-repro \
  timeout 300 python build/mla-m04b2b2b2-validation/compare.py \
  --output build/mla-m04b2b2b2-validation/compare-repro.json \
  --profile-dir build/mla-m04b2b2b2-validation/profile-repro --iterations 50 --rounds 7
```

质量工具沿用 `/tmp/mla-m01-precommit`（PYTHONPATH）和 `/tmp/mla-m01-precommit-cache`（PRE_COMMIT_HOME），执行 `ci/code_quality.sh --staged`，不绕过钩子。固定源码包仍为 `build/mla-m02b-validation/opp-dense/`；上一步闭环证据在 `build/mla-m04b2b2b-validation/`。未修改 reference/comparator、CANN 基线执行器/包、CATLASS 或 CANN 源仓，源仓既有其他状态保留。当前闭环提交可用 `git log -1 -- docs/mla/status.md` 定位。
