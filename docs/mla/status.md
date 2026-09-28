# MLA 当前交接

## 状态与下一步

- 当前步骤：M04b2b2b1 已完成，在线 PV 的四段结果在同一 L0C 缓冲区中形成后，一次写回完整 512 维 partial；连续 BSND 支持范围仍为 KV `1..1024`。
- 当前分支：`feat/mla-cann-migration`；接手基点 `6562a9a`。方案已确认，本轮一个提交后停止。
- 本步产物：[在线 PV Block](../../csrc/ascend910/flash_attn_npu_3/mla/mla_online_mmad.hpp)、[输出列边界及复用检查](../../tests/test_flash_attn_npu_v3_mla_online.py)。
- 唯一下一步：M04b2b2b2，继续 KV `>1024` 的流水与长 KV 调度优化；包含 8193 单/多 query、FP16/BF16 O/LSE、重复执行及同轮 CANN 性能，保持显式 `num_splits=1`。

M04b2b2b 在实现前拆为本步 PV 写回合并和后续长 KV 调度。此前仅放宽限界的 1025/8193 探测精度通过，但 1025 单 query kernel 比值约 1.016，8193 为 2.503..4.102，性能未达标；本步没有接通 >1024，也没有完成 8193 验收。

## 当前入口与实现

- `flash_attn_func(q, k, v, qv=...)` 支持 FP16/BF16、Dq=64/Dv=512/Hkv=1、连续 BSND、正 batch/query 长度、KV `1..1024`、无 mask、显式 `num_splits=1`。O 为 `[B,Sq,Hq,512]`，自然对数 FP32 LSE 为 `[B,Hq,Sq]`。
- KV `1..128` 保留单 tile 路径；`129..1024` 使用 128 列 tile 的在线 softmax。最大值、分母和输出分子保留 FP32；默认 scale 为 `1/sqrt(576)`，显式有限 scale 支持零/负值。
- 每个 online task 至多 16 行，完整 Q 常驻 L0A 前半区，PV 使用后半区。新增 PV 写回合并保持四个 128 维 MMAD 的数学顺序，各段写入同一 L0C 半区中相邻的 zN 列区间；16×512 FP32 结果为 32 KiB，编译期检查半区容量。每个 KV tile 的 L0C 释放等待、M→FIX 同步和完整 partial 写回各一次，L0C 双缓冲按 PV tile 轮换。
- 继承固定 CATLASS 的资源及 Tile copy/MMAD，保留来源与版权声明。KV 的 L1 双缓冲、P/V 的 L0A/L0B 事件、score/probability/partial 双缓冲、向量累积和跨 task 排空保持。没有修改公共入口、host 调度、workspace、metadata、CATLASS 或 CANN 源码。
- host 在 KV >512 且空闲核足够时按 8 个 head 分工，否则 16 个；每个 task 遍历完整 KV，无 split/合并。仅实测 910B C1V2。
- MLA prefill、varlen/kvcache 入口、KV >1024、空 KV/零 query、外层非连续 stride、mask/window、自动/多 split、softcap、pack_gqa/sm_margin、metadata、追加 KV、out、实际长度/页表/rotary/descale 和梯度请求仍明确拒绝。MLA metadata/图执行尚未验收；没有生产 CANN 回退。

## 最近闭环证据

| 检查 | 结果 |
| --- | --- |
| 接手与续跑核对 | PASS：接手工作区干净，上轮 486 项 checksum、27 项源码/扩展/对象/依赖哈希匹配；续跑时本步 52 项证据及未提交源码/扩展哈希匹配 |
| 构建溯源 | PASS：仅 v3/910，退出码 0；四个 MLA 单元跟踪 header 依赖，只有 FP16/BF16 online 对象变化；两个 short、flash_api 和 9 个普通 FA3/AICPU 对象哈希保持 |
| 910B MLA | PASS：完整 `598 passed`，无 skip；契约 233、short 141、online 224；新增 6 项覆盖输出首尾及每个 128 列分界两侧、full/单 head task 复用、奇偶 tile 数、重复执行及输入保持 |
| 针对性检查 | PASS：14 项；包含输出列边界、Q 分量分界及连续入队复用 |
| 固定源码 CANN | PASS：53 个 shape/scale/输入模式 × FP16/BF16，共 106 组；两实现均通过原双 reference/comparator，阈值未改；106 次 FIA launch 匹配配套 vendor binary |
| 普通 FA3 | PASS：`11 passed`，无 skip；dense/varlen/kvcache/FlashDecode、metadata、编译图及 dense/varlen backward |
| CANN 性能 | PASS（用户授权的共享 3 号卡）：32 例同轮交替；kernel 比值 `0.567..0.956`、同步调用比值 `0.869..0.902`，逐例低于 1.0；kernel 配对 bootstrap 95% 区间上界最高约 `0.9695` |
| 上版对照 | PASS（同一共享设备条件）：32 例 O/LSE 逐位一致；7 例 kernel 区间上界低于 1，其余 25 例跨 1，无区间明确高于 1 的用例；7 例中位 kernel 降低约 `0.7%..1.9%` |
| 既有 CPU | 本步未改 reference/profile/provenance，继承既有记录，不记为本轮重跑 |
| 质量与完整性 | 提交前执行 `ci/code_quality.sh --staged`、文档检查、`git diff --check` 和 `final_integrity.py`；结果及文件哈希保存在本轮证据目录 |

同轮性能每实现预热 10 次，同步调用 7 轮 × 50 对 AB/BA 交替，profiler 独立 3 轮 × 10 对。每轮 1920 次计时调用各对应一个 kernel，无 Slice 转换。相同输入/dtype/scale，CANN 原生自动策略、MLA 单 split，无 metadata；逐例门槛 `<=1.0` 保持。

部分结果如下，kernel 单位微秒；完整逐例数据在 `compare-cann-final.json`。

| 用例 | CANN kernel | MLA kernel | kernel 比值 |
| --- | --- | --- | --- |
| kv1024-q1-float16 | 21.7405 | 20.4910 | 0.9425 |
| kv1024-q1-bfloat16 | 21.6400 | 20.5405 | 0.9492 |
| kv1024-q7-float16 | 34.2605 | 26.1605 | 0.7636 |
| kv1024-q7-bfloat16 | 34.3805 | 26.2605 | 0.7638 |

上版对照同进程加载 `before/` 的已验收扩展，两版使用相同 C++ 参数；profiler 单列并排除 1 个 FIA setup kernel。区间明确改善的 7 例为 513 单 query FP16/BF16、1023 单 query FP16/BF16、1024 单 query BF16、D01 显式 scale FP16/BF16。其他区间跨 1，不能宣布确定收益；这些结果不能推断 8193 已达标。

## 设备条件与异常

- 初始设备检查没有健康空闲卡。用户先授权终止 3 号卡进程；复核唯一 driver PID 为 `2939247`（136 MB）后发送 SIGTERM，但当前容器返回 `ESRCH / No such process`，未能终止该宿主 PID。随后用户明确改为使用相对空闲卡试跑，本轮按该授权完成验收。
- 实测物理 3 号 Ascend910B3，采集前健康、AICore 利用率 0%；后台 PID `2939247` 保留。CANN 对比的 13 个、旧新版对照的 11 个计时窗口快照（均含停止采集后的首个边界样本）持续健康，只出现已知后台和本轮 driver PID；全部原始快照中后台占用保持 136 MB。两份采集日志均无 profiler ERROR。
- **本轮是用户授权的共享设备测量，不是独占卡计时。** 不能排除已知后台进程造成的瞬时影响；两实现同轮交替，阈值未降低，原始快照、采集窗口和统计区间均保留。后续性能验收仍需重新检查设备并记录实际条件。
- 初次沙箱内 `--collect-only` 因无法读取 NPU 型号在导入阶段失败；沙箱外复跑收集 598 项成功。此后完成上述真实 NPU 测试，收集结果不计作精度通过。

## 证据与复现

持久证据根目录：`/home/zhangqijia/repo/flash-attention-npu/build/mla-m04b2b2b-validation/`。

- `takeover-audit.json`、`card3-resume-audit.json`、`environment-before.json`、`environment-card3.json`、`before/`：接手/续跑核对、实际加载扩展和已验收上版。
- `build-coalesced-command.json` / `.log` / `-exit.json`、`build-provenance-coalesced.json`：构建命令及源码/对象/依赖/扩展哈希。
- `mla-targeted-*`、`mla-final-*`、`regression-final-*`、`run_regression.py`：精确命令、退出码、日志/JUnit 和普通 FA3 节点。
- `cann-accuracy.py` / `.json`、`dispatch-accuracy.log`、`verify_dispatch.py` / `dispatch-proof.json`：106 组精度及 vendor binary 来源。
- `compare.py`、`compare-cann-final.json`、`performance-acceptance.json`、`profile-cann-final/`：CANN 同轮计时及逐例验收。
- `compare-versions-setup.py`、`compare-versions-final.json`、`versions-summary.json`、`profile-versions-final/`：旧新版逐位一致、逐例计时和被排除的 setup kernel。
- `*-device-monitor.json`、`*-device-proof.json`、`verify_monitor.py`：原始快照、driver PID 映射、采集窗口和共享条件；`terminate-card3.json` 保留未成功的终止操作；`before-card3/`、`exclusive-card3/` 保留修改前配置。
- `quality-staged.log`、`document-check-final.log`、`final-integrity.json`、`checksums.sha256`：质量、文档和完整性。

实机使用 `login=false`、沙箱外。脚本固定物理 3 号/进程内 npu:0、线程数和隔离 cache/plog，保留 CANN PYTHONPATH；仅 CANN/比较/分派标签加载固定 vendor。复跑前检查设备，使用新标签和输出路径，不覆盖既有证据。当前监控允许已获授权的后台 PID `2939247`；设备或授权条件变化时同步调整执行器、监控、proof 和比较报告中的设备/共享标识。

```bash
FLASH_ATTN_BUILD_VERSION=v3 FLASH_ATTN_BUILD_NPU=910 \
  FLASH_ATTN_SKIP_SUBMODULE_INIT=1 python setup.py build_ext --inplace

python build/mla-m04b2b2b-validation/run_command.py mla-repro \
  timeout 300 python -m pytest -q tests/test_flash_attn_npu_v3_mla_online.py \
  tests/test_flash_attn_npu_v3_mla_short.py tests/test_flash_attn_npu_v3_mla_contract.py

python build/mla-m04b2b2b-validation/monitored_run.py compare-repro \
  timeout 300 python build/mla-m04b2b2b-validation/compare.py \
  --output build/mla-m04b2b2b-validation/compare-repro.json \
  --profile-dir build/mla-m04b2b2b-validation/profile-repro --iterations 50 --rounds 7
```

质量工具沿用 `/tmp/mla-m01-precommit`（PYTHONPATH）与 `/tmp/mla-m01-precommit-cache`（PRE_COMMIT_HOME），执行 `ci/code_quality.sh --staged`，不绕过提交钩子。

## 环境与继承基线

| 项目 | 本轮结果（2026-09-28） |
| --- | --- |
| Python / 包 | `/usr/local/python3.11.15/bin/python` 3.11.15；torch `2.7.1+cpu`，torch_npu `2.7.1` |
| CANN / 固定 FIA 源 | `/usr/local/Ascend/cann-9.0.0-beta.2`；`ca18aee610729ca5270a44e2d84c751bc687fbf0` |
| CATLASS | `769cd40a8716b28650b6bebb08db4834eea4462f`，干净且未修改 |
| 实际扩展 | 本仓 `flash_attn_npu_3/flash_attn_npu_3.cpython-311-aarch64-linux-gnu.so`；SHA256 `8e7f1aada0ed92066c57ec1d7ff770ac5aaa976d98fbdf95fc2e1928be88509b` |
| 芯片 | Ascend910B3，物理 3 号；用户授权共享卡测量，后台 PID 2939247 |

固定源码包为 `build/mla-m02b-validation/opp-dense/`，长 KV CANN 基线在 `build/mla-m02b2-validation/`，上一步闭环在 `build/mla-m04b2b2-validation/`。本轮未修改 reference/comparator、CANN 基线执行器/包、CATLASS 或 CANN 源仓，源仓既有其他状态保留。当前闭环提交可用 `git log -1 -- docs/mla/status.md` 定位。
