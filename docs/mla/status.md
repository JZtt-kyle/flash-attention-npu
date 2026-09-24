# MLA 当前交接

## 状态与下一步

- 当前步骤：M04b1 已完成，512+64 的连续 BSND 前向已支持 KV `1..512`。
- 当前分支：`feat/mla-cann-migration`；接手基点 `fc5a10e`。方案已确认，本轮一个提交后停止。
- 本步产物：[在线 softmax/累积 Block](../../csrc/ascend910/flash_attn_npu_3/mla/mla_online_epilogue.hpp)、[共用 dense Kernel](../../csrc/ascend910/flash_attn_npu_3/mla/mla_short_kernel.hpp)、[host 分派](../../csrc/ascend910/flash_attn_npu_3/mla/mla_host.h)、[精度与入口检查](../../tests/test_flash_attn_npu_v3_mla_online.py)。
- 唯一下一步：M04b2，扩展 KV `>512` 的流水与调度优化，验收含 `8193` 的单/多 query、FP16/BF16 O/LSE、重复执行及同轮 CANN 性能；保持显式 `num_splits=1`，其余未实现组合继续明确拒绝。

M04b 在实现前拆为 M04b1（129..512 的有限多 tile 累积）与 M04b2（更长 KV 的流水及性能）。本步没有完成完整长 KV 验收，也没有生产 CANN 回退。

## 当前入口行为

- `flash_attn_func(q, k, v, qv=...)` 支持 FP16/BF16、`Dq=64/Dv=512/Hkv=1`、连续 BSND、正 batch/query 长度、KV 长度 `1..512`、无 mask、`num_splits=1`。单/多 query 按表示统一分派。
- KV `1..128` 保留单 tile 路径；`129..512` 使用 128 列 tile 的在线 softmax。最大值、分母和输出分子保留 FP32，最终归一化并转换为输入 dtype；LSE 为自然对数 FP32 `[B,Hq,Sq]`，O 为 `[B,Sq,Hq,512]`。
- 复用固定 CATLASS 的 QK+RoPE / PV Block 和 reduction Tile。Q 留在 L1，PV 复用内容 KV；两组 score/probability 缓冲交叠 QK、softmax 与 PV，跨 tile 累积保存在 UB。每次调用独享 workspace，两个 vector lane 完成后才复用 partial；无完整输入拼接、KV 头展开、长度 D2H 或调度 H2D。
- 默认 scale 为 `1/sqrt(576)`，显式有限 scale 保持 FP32；正值、零、负值及跨 tile 最大值大幅变化均通过验证。连续非零 storage offset（含 offset=1）可用。
- dense qv 公开调用使用紧凑内部 custom op，将 scale 放在输入旁，避免通用 op 对未使用参数的额外遍历。计算和全部真实校验仍进入同一 C++ 边界；fake 复用通用校验。通用内部入口和直接 C++ 调用也已验收。
- MLA prefill、varlen/kvcache 公共入口、KV `>512`、空 KV、零 query、外层非连续 stride、mask/window、自动/多 split、softcap、pack_gqa/sm_margin、metadata、追加 KV、预分配 out、实际长度/页表/rotary/descale 和梯度请求仍明确拒绝。
- 运行时读取 Cube/Vector 资源并按 sub-block 数分工；仅实测 910B。MLA metadata 与图执行尚未验收，普通 FA3 的 byte metadata 和编译图回归通过。

## 最近闭环证据

| 检查 | 结果 |
| --- | --- |
| 接手核对 | PASS：分支/提交与交接一致、工作区干净；M04a 的 269 项 checksum 全匹配，原扩展哈希一致 |
| 构建溯源 | PASS：仅 v3/910；新增两种 online dtype 对象，两种 short 对象及 `flash_api.o` 重编，depfile 覆盖本仓 MLA headers；9 个普通 FA3/AICPU 对象哈希保持，最终扩展重新链接 |
| 910B MLA | PASS：`476 passed in 51.17s`，无 skip；契约 233 项、短路径 141 项、本步 102 项；覆盖 O/LSE、KV/头尾块、单/多 query、跨 tile 最大值、任务复用、重复执行、输入不改写、连续 storage offset、真实/fake/C++ 和紧凑公开入口 |
| 固定源码 CANN | PASS：15 个 shape/scale × FP16/BF16，共 30 组；CANN 与新实现分别通过原双 reference/comparator，阈值未改；30 次 FIA launch 全匹配配套 vendor binary |
| 普通 FA3 | PASS：`11 passed in 29.86s`，无 skip；dense/varlen/kvcache/FlashDecode、metadata 和编译图，含 dense/varlen backward |
| 既有 CPU | PASS：reference/profile/provenance 共 `88 passed in 20.64s` |
| 性能 | PASS：最终 12 例同轮交替测量，kernel 比值 `0.651..0.912`、端到端比值 `0.822..0.871`，逐例低于 1.0；没有用平均加速掩盖退化 |
| 来源与质量 | CATLASS 干净且未修改，固定 FIA 源干净；CANN 源仓其他目录的既有删除/未跟踪状态原样保留。提交前质量、文档和完整性检查见 `quality-staged.log`、`document-check.log`、`final-integrity.json` |

最终性能为 `compare-b.json` 的中位数，单位微秒。D01 为 `B=2,Sq=1,Sk=257,Hq=8`；511 用例为 `B=1,Hq=16`，Sq 为 1/7；D02 为原短路径 `B=1,Sk=65,Hq=16`，Sq 为 1/7。全部 `Hkv=1`、相同输入/dtype/scale、物理 7 号卡。

| 用例 | CANN kernel | MLA kernel | kernel 比值 | CANN 调用 | MLA 调用 | 调用比值 |
| --- | --- | --- | --- | --- | --- | --- |
| D01 默认 FP16 | 14.1405 | 11.3605 | 0.8034 | 301.650 | 258.430 | 0.8567 |
| D01 默认 BF16 | 14.5805 | 11.4800 | 0.7874 | 322.450 | 274.190 | 0.8503 |
| D01 scale=0.23 FP16 | 14.2300 | 11.3800 | 0.7997 | 313.960 | 273.315 | 0.8705 |
| D01 scale=0.23 BF16 | 14.1900 | 11.4600 | 0.8076 | 315.830 | 270.050 | 0.8550 |
| 511-q1 FP16 | 18.4905 | 16.8605 | 0.9118 | 309.630 | 258.875 | 0.8361 |
| 511-q1 BF16 | 18.8510 | 16.8300 | 0.8928 | 313.665 | 260.930 | 0.8319 |
| 511-q7 FP16 | 28.7305 | 20.4100 | 0.7104 | 312.860 | 259.475 | 0.8294 |
| 511-q7 BF16 | 28.7310 | 20.4705 | 0.7125 | 308.250 | 257.445 | 0.8352 |
| D02-q1 FP16 | 10.4300 | 7.0400 | 0.6750 | 309.070 | 258.805 | 0.8374 |
| D02-q1 BF16 | 10.2505 | 7.0100 | 0.6839 | 303.235 | 251.650 | 0.8299 |
| D02 FP16 | 16.6805 | 10.8500 | 0.6505 | 302.885 | 249.090 | 0.8224 |
| D02 BF16 | 16.7100 | 10.8800 | 0.6511 | 301.525 | 247.880 | 0.8221 |

每实现预热 10 次；同步调用 7 轮 × 50 对 AB/BA 交替；独立 profiler 3 轮 × 10 对，720 次调用均对应恰好一个 kernel，无 Slice 转换。调用时间包含输出/workspace 分配、校验和静态调度；MLA metadata、复用和图模式不在本步性能结论内。配对 bootstrap 的 95% 区间上界最高 0.918，验收门槛仍为逐例比值 `<=1.0`。

首版 129 长度精度失败，定位到复用 UB 时缺少 V→MTE2 依赖及中间累积错误使用 MTE3 完成信号；修复并重编后全部精度通过。首轮性能的 D01 显式 scale 调用比值为 1.013/1.002，判为未通过；主机 profile 发现通用 custom op 参数遍历开销，紧凑入口修复后重验达标。完整回归中另一处旧测试仍将 136 长度视为未支持，已改为大于 512 的输入并重验。原失败日志均保留。

## 证据与复现

持久证据根目录：

`/home/zhangqijia/repo/flash-attention-npu/build/mla-m04b-validation/`

- `environment-before.json`、`takeover-audit.json`：环境、既有源仓状态、旧扩展与 269 项证据校验。
- `build-{1,2}-command.json` / `.log` / `-exit.json`、`build-provenance-final.json`：构建和依赖证据，含实际扩展及 Python 入口哈希；`compare-b.json` 同时记录运行时加载扩展。
- `mla-final-2-*` / `mla-final-2.log` / `mla-final-2.xml`、`regression-final-*` / `.log` / `.xml`：最终 MLA 和普通 FA3 精确命令与结果；`cpu-*` / `cpu.log` 为 CPU 检查。
- `cann-accuracy.py` / `cann-accuracy.json`、`dispatch-accuracy.log`、`verify_dispatch.py` / `dispatch-proof.json`：30 组同输入精度及 device binary 来源；CANN 仅用于测试。
- `compare.py`、`compare-{a,b}.json`、`profile-{a,b}/`、`host_profile.py` / `compare-host.log`、`accept-performance.py` / `performance-acceptance.json`：修复前后原始交替计时、主机分析、profiler 及逐例验收。
- `mla-1.log`、`diagnose-1.log`、`mla-final.log`：数值/旧测试边界失败证据；`device-final-2.log`、`quality-staged.log`、`document-check.log`、`final-integrity.json`、`checksums.sha256`：设备、质量与证据完整性。

实机使用 shell `login=false`、沙箱外、物理 7 号/进程内 `npu:0`。`run_command.py` 保留 CANN PYTHONPATH，固定线程数、设备及隔离 cache/plog；`compare*` / `dispatch*` 才加载固定 custom FIA vendor。复跑前检查芯片健康/占用，并更换未使用的标签及输出路径，不覆盖本轮证据。

```bash
FLASH_ATTN_BUILD_VERSION=v3 FLASH_ATTN_BUILD_NPU=910 \
  FLASH_ATTN_SKIP_SUBMODULE_INIT=1 python setup.py build_ext --inplace

python build/mla-m04b-validation/run_command.py mla-repro \
  timeout 180 python -m pytest -q tests/test_flash_attn_npu_v3_mla_online.py \
  tests/test_flash_attn_npu_v3_mla_short.py tests/test_flash_attn_npu_v3_mla_contract.py

TORCH_DEVICE_BACKEND_AUTOLOAD=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m pytest -q tests/test_flash_attn_npu_v3_mla_reference.py \
  tests/test_flash_attn_npu_v3_mla_profile.py tests/test_flash_attn_npu_v3_mla_provenance.py

python build/mla-m04b-validation/run_command.py compare-repro \
  python build/mla-m04b-validation/compare.py \
  --output /tmp/mla-m04b1-repro.json --profile-dir /tmp/mla-m04b1-repro-profile \
  --iterations 50 --rounds 7
```

普通 FA3 精确节点见 `regression-final-command.json`，复跑时更换标签/JUnit 路径。质量工具沿用 `/tmp/mla-m01-precommit` 与 `/tmp/mla-m01-precommit-cache`，前者前置 PYTHONPATH，后者设为 PRE_COMMIT_HOME，执行 `ci/code_quality.sh --staged`。

## 环境与继承基线

| 项目 | 本轮结果（2026-09-24） |
| --- | --- |
| Python / 包 | `/usr/local/python3.11.15/bin/python`，3.11.15；torch `2.7.1+cpu`，torch_npu `2.7.1` |
| CANN / 固定 FIA 源 | `/usr/local/Ascend/cann-9.0.0-beta.2`；`ca18aee610729ca5270a44e2d84c751bc687fbf0` |
| CATLASS | `769cd40a8716b28650b6bebb08db4834eea4462f`，干净且未修改 |
| FA3 实际扩展 | 本仓 `flash_attn_npu_3/flash_attn_npu_3.cpython-311-aarch64-linux-gnu.so`；SHA256 `33e22e7e82f1f38668b8a74b8d99e207260026e51cb3ba14d3b9c8901d06ae36` |
| 芯片 | Ascend910B3 物理 7 号，运行前健康且空闲；没有使用告警的 5/6 号设备 |

固定源码包仍为 `build/mla-m02b-validation/opp-dense/`，完整 CANN 长 KV 基线在 `build/mla-m02b2-validation/`，M04a 证据在 `build/mla-m04a-validation/`。本轮未修改 reference、comparator、CANN 基线执行器/包、CATLASS 或 CANN 源仓。历史 wall 中位数不可跨会话直接作为门槛；M04b2 继续同轮交替比较。当前闭环提交可由 `git log -1 -- docs/mla/status.md` 定位。
