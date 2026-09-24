# MLA 当前交接

## 状态与下一步

- 当前步骤：M04b2b2a 已完成，在线 QK 的完整 Q 常驻 L0A，PV 使用另一半 L0A；连续 BSND 支持范围仍为 KV `1..1024`。
- 当前分支：`feat/mla-cann-migration`；接手基点 `33f60c5`。方案已确认，本轮一个提交后停止。
- 本步产物：[缓存 Q 的 QK/PV Block](../../csrc/ascend910/flash_attn_npu_3/mla/mla_online_mmad.hpp)、[在线分派](../../csrc/ascend910/flash_attn_npu_3/mla/mla_short_kernel.hpp)、[Q 分界及复用检查](../../tests/test_flash_attn_npu_v3_mla_online.py)。
- 唯一下一步：M04b2b2b，继续 KV `>1024` 的流水与长 KV 调度优化；验收包含 `8193` 单/多 query、FP16/BF16 O/LSE、重复执行及同轮 CANN 性能，保持显式 `num_splits=1`。

M04b2b2 的限界放宽先导探测通过了 1025/8193 单/多 query × 两种 dtype 的 8 组精度，但 1025 单 query kernel 比值约 1.016，8193 为 2.503..4.102，未达性能门槛。随后在 kernel 优化实现前拆为 Q 缓存的 M04b2b2a 和长 KV 的 M04b2b2b。本步没有接通 >1024，也没有完成 8193 验收；先导扩展和补充长 KV 用例保存在证据目录，生产入口已恢复原限制。

## 当前入口与实现

- `flash_attn_func(q, k, v, qv=...)` 支持 FP16/BF16、`Dq=64/Dv=512/Hkv=1`、连续 BSND、正 batch/query 长度、KV `1..1024`、无 mask、显式 `num_splits=1`。O 为 `[B,Sq,Hq,512]`，自然对数 FP32 LSE 为 `[B,Hq,Sq]`。
- KV `1..128` 保留原单 tile 路径；`129..1024` 使用 128 列 tile 的在线 softmax。最大值、分母、输出分子保留 FP32；默认 scale 为 `1/sqrt(576)`，显式有限 scale 保持 FP32，支持零/负值。
- 新 Block 继承固定 CATLASS 的资源及 Tile copy/MMAD 组件，保留来源和版权声明。每个 task 至多 16 行，完整 Q 内容和 RoPE 从 L1 zN 一次搬入 L0A zZ 的前半区；QK 按原来的 128 维分段及最终 64 维 RoPE 顺序读取。QK 的 Q 搬运由每 KV tile 5 次减为每任务 1 次，附带同步也移出逐 tile 循环。
- PV 的 P 固定使用 L0A 后半区及对应的释放事件。L1 KV 的双缓冲、QK/PV 数学顺序、score/probability/partial 双缓冲、FP32 向量累积和跨 task 信号排空保持；task 排空后才能重载下一份 Q。编译期约束要求 Q 只有一个 16 行分形并能放入半个 L0A。
- host 调度仍在 KV >512 且空闲核充足时按 8 个 head 分工，否则 16 个；每任务遍历完整 KV，没有 split/合并。公共入口、fake/C++ 校验、workspace、host 调度及 metadata 均未修改；仅实测 910B C1V2。
- MLA prefill、varlen/kvcache 公共入口、KV >1024、空 KV/零 query、外层非连续 stride、mask/window、自动/多 split、softcap、pack_gqa/sm_margin、metadata、追加 KV、预分配 out、实际长度/页表/rotary/descale 和梯度请求仍明确拒绝。MLA metadata/图执行尚未验收；没有生产 CANN 回退。

## 最近闭环证据

| 检查 | 结果 |
| --- | --- |
| 接手核对 | PASS：工作区干净，上轮 999 项 checksum 及源码/扩展/对象/依赖哈希匹配；CATLASS 和 CANN 既有状态保持 |
| 构建溯源 | PASS：仅 v3/910；最终构建核查 4 个 MLA 单元的新 header 依赖，只有 FP16/BF16 online 对象相对接手基线变化；2 个 short、flash_api 和 9 个普通 FA3/AICPU 对象哈希保持 |
| 910B MLA | PASS：完整 `592 passed`，无 skip；契约 233、短路径 141、online 218；新增 6 项覆盖每个 128 维分段两侧、内容/RoPE 分界、单 head 尾块、跨 task Q 更新、重复执行及输入保持 |
| 针对性检查 | PASS：22 项；后续补强编译期容量约束、重建后完成上述完整检查 |
| 固定源码 CANN | PASS：50 个 shape/scale/输入模式 × FP16/BF16，共 100 组；两实现各通过原双 reference/comparator，阈值未改；100 次 FIA launch 均匹配配套 vendor binary |
| 普通 FA3 | PASS：`11 passed`，无 skip；dense/varlen/kvcache/FlashDecode、metadata、编译图及 dense/varlen backward |
| CANN 性能 | PASS：32 例同轮交替；kernel 比值 `0.583..0.975`、端到端比值 `0.825..0.861`，逐例低于 1.0；kernel 配对 bootstrap 95% 区间上界最高约 `0.9801` |
| 上版对照 | PASS：同进程加载保留的上版扩展，使用相同 C++ 参数；32 例 O/LSE 逐位一致；4 例 kernel 区间明确低于 1，其余 28 例跨 1，无区间明确高于 1 的用例 |
| 既有 CPU | 本步未改 reference/profile/provenance，继承已核对的上轮 88 项记录，不记为本轮重跑 |
| 质量与完整性 | 提交前执行 `ci/code_quality.sh --staged`、文档检查、`git diff --check` 和 `final_integrity.py`；结果与文件哈希保存在本轮证据目录 |

CANN 性能每实现预热 10 次，同步调用 7 轮 × 50 对 AB/BA 交替，profiler 独立 3 轮 × 10 对；1920 次调用各对应一个 kernel，无 Slice 转换。相同输入/dtype/scale、物理 7 号卡；CANN 原生自动策略、MLA 单 split，无 metadata。门槛保持逐例比值 `<=1.0`。

部分结果如下，kernel 单位微秒；完整逐例数据在 `compare-cann-repro.json`。

| 用例 | CANN kernel | MLA kernel | kernel 比值 | 调用比值 |
| --- | --- | --- | --- | --- |
| kv1023-q1-float16 | 21.5905 | 21.0410 | 0.9745 | 0.8460 |
| kv1023-q1-bfloat16 | 21.6205 | 20.9805 | 0.9704 | 0.8423 |
| kv1023-q7-float16 | 36.0510 | 27.0805 | 0.7512 | 0.8539 |
| kv1023-q7-bfloat16 | 36.1505 | 27.0500 | 0.7483 | 0.8558 |
| kv1024-q1-float16 | 21.4905 | 20.6905 | 0.9628 | 0.8427 |
| kv1024-q1-bfloat16 | 21.3500 | 20.7710 | 0.9729 | 0.8388 |
| kv1024-q7-float16 | 34.4310 | 26.7210 | 0.7761 | 0.8469 |
| kv1024-q7-bfloat16 | 34.5205 | 26.8405 | 0.7775 | 0.8398 |

上版对照也采用 32 例、7×50 对调用及 3×10 对 profiler，包含 1 个明确记录并排除的 FIA setup kernel。1920 次旧新版 kernel 计时有效，4 个明确受益用例为 513 单 query BF16、1023 单 query FP16/BF16、1024 单 query BF16，中位 kernel 降低约 0.4%..1.1%。其余区间跨 1，不能宣布确定收益；这些收益不能推断 8193 已达标。

## 环境异常与证据使用

- 第一轮长 KV 探测在 `torch_npu` 初始化等待设备 AICPU 子进程时超时（240 秒），尚未 launch MLA；调用栈及 plog 保留。无 vendor 初始化对照成功，随后原命令的新标签重试成功。
- 第一次最终 CANN 性能采集前快照出现另一进程（driver PID 2199065，约 24 GB），该轮仅作诊断，已排除。完整精度检查正常完成。
- 正式重测增加原始 `npu-smi` 快照，按 plog 映射驱动 PID，并用 profiler `collectionTimeEnd` 限定计时窗口，额外包含采集结束后的第一个快照。CANN 重测 14 个、旧新版对照 17 个窗口快照均健康、开始空闲且计时期间只有本轮 PID。CANN 重测的 CPU 导出/退出阶段另见短暂 PID 2542720，发生于 profiler 停止采集约 28 秒后，单列在 `post_collection_samples`。
- 最终采用 `compare-cann-repro` 和 `compare-versions-final`，两份采集日志均无 profiler ERROR。设备 proof 从原始快照重新解析 PID 列，不依赖初版监控器的派生 PID 字段。

## 证据与复现

持久证据根目录：`/home/zhangqijia/repo/flash-attention-npu/build/mla-m04b2b2-validation/`。

- `takeover-audit.json`、`environment-before.json`、`before/`：接手核对、环境及上版扩展/headers。
- `pilot-scope.patch`、`pilot/`、`compare-pilot-retry.json`、`pilot-acceptance.json`：限界放宽先导源码、二进制及 8 组未达标性能探测；不计入本步闭环性能。
- `build-*-command.json` / `.log` / `-exit.json`、`build-provenance-final.json`：构建命令及最终源码/对象/依赖/扩展哈希。
- `mla-targeted-*`、`mla-final-*`、`regression-final-*`：精确命令、退出码、日志/JUnit；`run_regression.py` 保留普通 FA3 节点。
- `cann-accuracy.py` / `.json`、`dispatch-accuracy.log`、`verify_dispatch.py` / `dispatch-proof.json`：100 组精度及 vendor binary 来源。
- `compare.py`、`compare-cann-repro.json`、`profile-cann-repro/`、`performance-acceptance.json`：正式 CANN 同轮计时及逐例验收。
- `compare-versions-setup.py`、`compare-versions-final.json`、`versions-summary.json`、`profile-versions-final/`：旧新版逐位一致性、逐例计时与被排除的 setup kernel。
- `*-device-monitor.json`、`*-device-proof.json`、`verify_monitor.py`、`performance-occupied*.json`：原始设备快照、采集窗口核对及未采用的第一轮性能。
- `quality-staged.log`、`document-check.log`、`final-integrity.json`、`checksums.sha256`：质量、文档与证据完整性。

实机使用 `login=false`、沙箱外；`run_command.py` 保留 CANN PYTHONPATH，固定线程数、物理 7 号/进程内 npu:0 和隔离 cache/plog，只有 CANN/比较/分派标签加载固定 vendor。复跑前检查设备健康/占用，更换标签和输出路径，不覆盖既有证据；性能同时保存设备快照。

```bash
FLASH_ATTN_BUILD_VERSION=v3 FLASH_ATTN_BUILD_NPU=910 \
  FLASH_ATTN_SKIP_SUBMODULE_INIT=1 python setup.py build_ext --inplace

python build/mla-m04b2b2-validation/run_command.py mla-repro \
  timeout 300 python -m pytest -q tests/test_flash_attn_npu_v3_mla_online.py \
  tests/test_flash_attn_npu_v3_mla_short.py tests/test_flash_attn_npu_v3_mla_contract.py

python build/mla-m04b2b2-validation/monitored_run.py compare-repro \
  timeout 240 python build/mla-m04b2b2-validation/compare.py \
  --output build/mla-m04b2b2-validation/compare-repro.json \
  --profile-dir build/mla-m04b2b2-validation/profile-repro --iterations 50 --rounds 7
```

质量工具沿用 `/tmp/mla-m01-precommit`（前置 PYTHONPATH）与 `/tmp/mla-m01-precommit-cache`（PRE_COMMIT_HOME），执行 `ci/code_quality.sh --staged`，不绕过提交钩子。

## 环境与继承基线

| 项目 | 本轮结果（2026-09-24） |
| --- | --- |
| Python / 包 | `/usr/local/python3.11.15/bin/python` 3.11.15；torch `2.7.1+cpu`，torch_npu `2.7.1` |
| CANN / 固定 FIA 源 | `/usr/local/Ascend/cann-9.0.0-beta.2`；`ca18aee610729ca5270a44e2d84c751bc687fbf0` |
| CATLASS | `769cd40a8716b28650b6bebb08db4834eea4462f`，干净且未修改 |
| 实际扩展 | 本仓 `flash_attn_npu_3/flash_attn_npu_3.cpython-311-aarch64-linux-gnu.so`；SHA256 `78d3ebaaf127d9f7014e9ccc13c8dd950baaceef973f92e2e90aa5aadd4c8c3b` |
| 芯片 | Ascend910B3，物理 7 号；正式性能采集窗口健康且无其他进程 |

固定源码包仍为 `build/mla-m02b-validation/opp-dense/`，长 KV CANN 基线在 `build/mla-m02b2-validation/`，上一步闭环在 `build/mla-m04b2b-validation/`。本轮未修改 reference/comparator、CANN 基线执行器/包、CATLASS 或 CANN 源仓；源仓其他目录既有状态保留。下一步仍须同轮交替比较。当前闭环提交可用 `git log -1 -- docs/mla/status.md` 定位。
