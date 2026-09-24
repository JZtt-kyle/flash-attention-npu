# MLA 当前交接

## 状态与下一步

- 当前步骤：M04b2a 已完成，512+64 的连续 BSND 前向已支持 KV `1..1024`。
- 当前分支：`feat/mla-cann-migration`；接手基点 `b19e9b9`。方案已确认，本轮一个提交后停止。
- 本步产物：[双缓冲 dense Kernel](../../csrc/ascend910/flash_attn_npu_3/mla/mla_short_kernel.hpp)、[在线 softmax/累积 Block](../../csrc/ascend910/flash_attn_npu_3/mla/mla_online_epilogue.hpp)、[head 分工](../../csrc/ascend910/flash_attn_npu_3/mla/mla_host.h)、[精度与入口检查](../../tests/test_flash_attn_npu_v3_mla_online.py)。
- 唯一下一步：M04b2b，继续 KV `>1024` 的流水与调度优化，验收含 `8193` 的单/多 query、FP16/BF16 O/LSE、重复执行及同轮 CANN 性能；保持显式 `num_splits=1`，其余未实现组合继续明确拒绝。

M04b2 在实现前拆为 M04b2a（双缓冲流水与 513..1024）和 M04b2b（更长 KV 调度）。本步未完成 8193 的长 KV 验收，没有生产 CANN 回退。

## 当前入口行为

- `flash_attn_func(q, k, v, qv=...)` 支持 FP16/BF16、`Dq=64/Dv=512/Hkv=1`、连续 BSND、正 batch/query 长度、KV `1..1024`、无 mask、`num_splits=1`。单/多 query 按表示统一分派。
- KV `1..128` 保留单 tile 路径；`129..1024` 使用 128 列 tile 的在线 softmax。最大值、分母和输出分子保留 FP32，最后归一化并转换为输入 dtype；LSE 为自然对数 FP32 `[B,Hq,Sq]`，O 为 `[B,Sq,Hq,512]`。
- 复用固定 CATLASS 的 QK+RoPE / PV Block。Q 留在 L1，PV 复用内容 KV。两组 score/probability/PV partial 缓冲允许 Cube 与 Vector 累积交叠；复用 partial 前等待最早的完成信号，每个 task 结束时排空两组信号，避免跨 task 污染。workspace 大小由所选路径和使用的核数决定。
- 在线 softmax 先保存 probability，再调用 CATLASS 原地 `ReduceSumRepeatM`，分母仍由 FP32 指数值求和。未修改 CATLASS 子模块。
- KV `>512` 且原 16-head 任务数不超过 Cube 核数一半时，每任务处理 8 个独立 head；否则保持 16 个。每个任务遍历完整 KV，直接写对应 O/LSE，没有 split-KV 或合并阶段。运行时读取 Cube/Vector 资源并按 sub-block 数分工，仅实测 910B。
- 默认 scale 为 `1/sqrt(576)`，显式有限 scale 保持 FP32；正值、零、负值、跨 tile 最大值大幅变化、连续非零 storage offset（含 offset=1）、异长调用连续入队均通过验证。
- dense qv 公开调用继续使用紧凑 custom op；真实校验统一进入 C++，fake 复用通用校验。通用内部入口、直接 C++ 调用及普通 FA3 回归通过。
- MLA prefill、varlen/kvcache 公共入口、KV `>1024`、空 KV、零 query、外层非连续 stride、mask/window、自动/多 split、softcap、pack_gqa/sm_margin、metadata、追加 KV、预分配 out、实际长度/页表/rotary/descale 和梯度请求仍明确拒绝。MLA metadata 与图执行尚未验收。

## 最近闭环证据

| 检查 | 结果 |
| --- | --- |
| 接手核对 | PASS：分支/提交与交接一致、工作区干净；M04b1 的 238 项 checksum 全匹配，原扩展及 Python 入口哈希一致 |
| 构建溯源 | PASS：仅 v3/910；两种 online、两种 short 对象及 `flash_api.o` 重编，depfile 覆盖本仓 MLA headers；9 个普通 FA3/AICPU 对象哈希保持，最终扩展重新链接 |
| 910B MLA | PASS：最终完整检查 `552 passed`，新增归约尾块专项 `12 passed`，无 skip；合计 564 项，契约 233 项、短路径 141 项、online 190 项；含 513..1024、奇偶 tile 数、head/KV 尾块、单/多 query、极值、跨任务复用、重复执行和连续入队 |
| 固定源码 CANN | PASS：36 个 shape/scale × FP16/BF16，共 72 组；CANN 与新实现分别通过原双 reference/comparator，阈值未改；72 次 FIA launch 全匹配配套 vendor binary |
| 普通 FA3 | PASS：`11 passed`，无 skip；dense/varlen/kvcache/FlashDecode、metadata 和编译图，含 dense/varlen backward |
| 既有 CPU | PASS：reference/profile/provenance 共 `88 passed in 20.62s` |
| 性能 | PASS：最终 24 例同轮交替测量，kernel 比值 `0.589..0.988`、端到端比值 `0.780..0.864`，逐例低于 1.0；配对 bootstrap 95% 区间上界最高 0.996 |
| 来源与质量 | CATLASS 干净且未修改，固定 FIA 源干净；CANN 源仓其他目录的既有删除/未跟踪状态原样保留。提交前质量、文档与完整性证据见下文 |

最终性能为 `compare-c.json` 的中位数，单位微秒。新增 513/1023/1024 均为 `B=1,Hq=16,Sq=1/7`；D01 为 `B=2,Sq=1,Sk=257,Hq=8`；511 和 D02（Sk=65）为 `B=1,Hq=16,Sq=1/7`。全部 `Hkv=1`、相同输入/dtype/scale、物理 7 号卡；CANN 使用原生自动策略，MLA 显式单 split。

| 用例 | CANN kernel | MLA kernel | kernel 比值 | CANN 调用 | MLA 调用 | 调用比值 |
| --- | --- | --- | --- | --- | --- | --- |
| kv513-q1-float16 | 21.5305 | 13.9405 | 0.6475 | 350.795 | 285.865 | 0.8149 |
| kv513-q1-bfloat16 | 21.6505 | 13.9600 | 0.6448 | 353.595 | 275.720 | 0.7798 |
| kv513-q7-float16 | 34.5810 | 20.5610 | 0.5946 | 312.990 | 264.975 | 0.8466 |
| kv513-q7-bfloat16 | 34.6810 | 20.4305 | 0.5891 | 310.170 | 258.425 | 0.8332 |
| kv1023-q1-float16 | 21.7200 | 21.2805 | 0.9798 | 334.160 | 286.915 | 0.8586 |
| kv1023-q1-bfloat16 | 21.6200 | 21.3610 | 0.9880 | 326.245 | 272.020 | 0.8338 |
| kv1023-q7-float16 | 36.1205 | 27.7510 | 0.7683 | 324.980 | 280.770 | 0.8640 |
| kv1023-q7-bfloat16 | 36.2610 | 27.7200 | 0.7645 | 320.995 | 270.070 | 0.8414 |
| kv1024-q1-float16 | 21.5405 | 21.0000 | 0.9749 | 311.355 | 263.710 | 0.8470 |
| kv1024-q1-bfloat16 | 21.4100 | 21.0800 | 0.9846 | 338.410 | 279.265 | 0.8252 |
| kv1024-q7-float16 | 34.7510 | 27.3805 | 0.7879 | 332.755 | 276.535 | 0.8310 |
| kv1024-q7-bfloat16 | 34.8310 | 27.4605 | 0.7884 | 331.030 | 278.260 | 0.8406 |
| D01-default-float16 | 14.5005 | 9.6800 | 0.6676 | 344.260 | 284.910 | 0.8276 |
| D01-default-bfloat16 | 14.4600 | 9.7000 | 0.6708 | 324.190 | 274.840 | 0.8478 |
| D01-scale-float16 | 14.3200 | 9.6900 | 0.6767 | 340.305 | 271.925 | 0.7991 |
| D01-scale-bfloat16 | 14.4005 | 9.6700 | 0.6715 | 318.405 | 269.055 | 0.8450 |
| kv511-q1-float16 | 19.0005 | 13.5505 | 0.7132 | 333.430 | 271.695 | 0.8148 |
| kv511-q1-bfloat16 | 19.3005 | 13.5110 | 0.7000 | 333.315 | 261.380 | 0.7842 |
| kv511-q7-float16 | 30.0705 | 18.1610 | 0.6039 | 337.860 | 273.080 | 0.8083 |
| kv511-q7-bfloat16 | 29.9210 | 18.2200 | 0.6089 | 316.245 | 256.525 | 0.8112 |
| D02-q1-float16 | 10.3505 | 7.1700 | 0.6927 | 357.645 | 290.515 | 0.8123 |
| D02-q1-bfloat16 | 10.2800 | 7.1605 | 0.6965 | 316.110 | 269.040 | 0.8511 |
| D02-float16 | 18.0305 | 12.0005 | 0.6656 | 295.515 | 251.225 | 0.8501 |
| D02-bfloat16 | 18.0705 | 11.9505 | 0.6613 | 325.080 | 265.485 | 0.8167 |

每实现预热 10 次；同步调用 7 轮 × 50 对 AB/BA 交替；独立 profiler 3 轮 × 10 对，1440 次调用均对应恰好一个 kernel，无 Slice 转换。调用时间包含输出/workspace 分配、校验和静态调度。门槛保持逐例比值 `<=1.0`，没有用平均加速掩盖退化；MLA metadata、复用和图模式不在本步性能结论内。

首轮完整检查为 `548 passed / 2 failed`：两项旧拒绝测试仍将 527 长度视为未支持，更新到 >1024 后针对性复验 14 项通过，最终完整重验 552 项通过，随后补充的 12 项归约尾块专项也通过。双缓冲首版及归约优化后的单 query 1023/1024 性能仍有退化（最大比值 1.024/1.016），均判为未通过；细分独立 head 分工后重验达标。失败日志、两轮不通过的性能数据和验收报告均保留。

## 证据与复现

持久证据根目录：

`/home/zhangqijia/repo/flash-attention-npu/build/mla-m04b2-validation/`

- `environment-before.json`、`takeover-audit.json`：环境、既有源仓状态、旧扩展与 238 项证据校验。
- `build-{1,2,3}-command.json` / `.log` / `-exit.json`、`build-provenance-{2,3}.json`：各版构建及依赖证据；最终版本为 3。`compare-c.json`、`cann-accuracy.json` 记录实际加载扩展，与构建哈希一致。
- `mla-final-2-command.json` / `.log` / `.xml` / `-exit.json`、`mla-reduction-tail-*`、`regression-final-*`、`cpu-*`：最终验证精确命令、返回码及结果。`mla-1*`、`mla-fix-1*`、`mla-final*` 保留中间检查。
- `cann-accuracy.py` / `cann-accuracy.json`、`dispatch-accuracy.log`、`verify_dispatch.py` / `dispatch-proof.json`：72 组同输入精度及 device binary 来源；CANN 仅用于测试。
- `compare.py`、`compare-{a,b,c}.json`、`profile-{a,b,c}/`、`accept-performance.py`、`performance-{a,b}.json`、`performance-acceptance.json`：三版交替计时、profiler 及逐例验收。
- `device-before-performance.log`、`device-final.log`、`quality-staged.log`、`document-check.log`、`final-integrity.json`、`checksums.sha256`：设备、质量、文档及证据完整性。`run_remaining.py` 串行执行普通 FA3 和来源验证。

实机使用 shell `login=false`、沙箱外、物理 7 号/进程内 `npu:0`。`run_command.py` 保留 CANN PYTHONPATH，固定线程数、设备及隔离 cache/plog；`compare*` / `dispatch*` 才加载固定 custom FIA vendor。复跑前检查芯片健康/占用，并更换未使用的标签及输出路径，不覆盖本轮证据。

```bash
FLASH_ATTN_BUILD_VERSION=v3 FLASH_ATTN_BUILD_NPU=910 \
  FLASH_ATTN_SKIP_SUBMODULE_INIT=1 python setup.py build_ext --inplace

python build/mla-m04b2-validation/run_command.py mla-repro \
  timeout 240 python -m pytest -q tests/test_flash_attn_npu_v3_mla_online.py \
  tests/test_flash_attn_npu_v3_mla_short.py tests/test_flash_attn_npu_v3_mla_contract.py

TORCH_DEVICE_BACKEND_AUTOLOAD=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m pytest -q tests/test_flash_attn_npu_v3_mla_reference.py \
  tests/test_flash_attn_npu_v3_mla_profile.py tests/test_flash_attn_npu_v3_mla_provenance.py

python build/mla-m04b2-validation/run_command.py compare-repro \
  python build/mla-m04b2-validation/compare.py \
  --output /tmp/mla-m04b2a-repro.json --profile-dir /tmp/mla-m04b2a-repro-profile \
  --iterations 50 --rounds 7
```

普通 FA3 精确节点见 `regression-final-command.json`，复跑时更换标签/JUnit 路径。质量工具沿用 `/tmp/mla-m01-precommit` 与 `/tmp/mla-m01-precommit-cache`，前者前置 PYTHONPATH，后者设为 PRE_COMMIT_HOME，执行 `ci/code_quality.sh --staged`。

## 环境与继承基线

| 项目 | 本轮结果（2026-09-24） |
| --- | --- |
| Python / 包 | `/usr/local/python3.11.15/bin/python`，3.11.15；torch `2.7.1+cpu`，torch_npu `2.7.1` |
| CANN / 固定 FIA 源 | `/usr/local/Ascend/cann-9.0.0-beta.2`；`ca18aee610729ca5270a44e2d84c751bc687fbf0` |
| CATLASS | `769cd40a8716b28650b6bebb08db4834eea4462f`，干净且未修改 |
| FA3 实际扩展 | 本仓 `flash_attn_npu_3/flash_attn_npu_3.cpython-311-aarch64-linux-gnu.so`；SHA256 `d8bb30524bf934168bcb81ac87032aa10724f98086eb79f921d61142ba1aa5cd` |
| 芯片 | Ascend910B3 物理 7 号，性能验收前健康且空闲；没有使用告警的 5/6 号设备 |

固定源码包仍为 `build/mla-m02b-validation/opp-dense/`，完整 CANN 长 KV 基线在 `build/mla-m02b2-validation/`，M04b1 证据在 `build/mla-m04b-validation/`。本轮未修改 reference、comparator、CANN 基线执行器/包、CATLASS 或 CANN 源仓。历史 wall 中位数不可跨会话直接作为门槛；M04b2b 继续同轮交替比较。当前闭环提交可由 `git log -1 -- docs/mla/status.md` 定位。
