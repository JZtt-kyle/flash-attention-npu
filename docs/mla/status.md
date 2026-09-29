# MLA 当前状态

## 当前能力与下一步

- 分支：`feat/mla-cann-migration`；接手 HEAD 为 `7b60202`，接手工作区干净。本次完成 M04b3：同范围 BF16 dense MLA 实例化及验收。
- 910B `flash_attn_func(q, k, v, qv=qv, num_splits=1)` 已支持 FP16/BF16：512+64 共享 KV、连续 BSND、B>0、`1<=Sq*Hq<=256`、KV `1..512`、Hkv=1、无 mask。O 与输入 dtype 一致，可选 LSE 为 FP32；默认/零/正/负 scale、连续视图的非零 storage offset 均已验收。
- Kernel / Block / Tile 按输入类型实例化；六个算法头去除新增模板声明后，FP16 计算代码与上步一致。BF16 保留源的 aMLA 因子往返转换、P 的 `CAST_ROUND` 和 O 的 `CAST_RINT`。两种 dtype 共用原调度、workspace 布局和同步协议，生成器分别提供 C1V2 编译单元。
- 真实/fake/C++ 对两种 dtype 一致拒绝 prefill、metadata、TND、PA、mask、非连续布局、长度、预分配 out、append/rotary/descale、非默认调度选项、梯度和形状范围外组合；公开 varlen/kvcache 仍拒绝 MLA。910C/C1V1 和完整图生命周期尚未验收。
- 唯一下一步：M05，已验收 dense 范围的 MLA metadata 及 E2E 验收，详见 [plan](plan.md)。本步提交后停止。

## 最近一次闭环证据

证据目录：`build/mla-m04b3-bf16/`；物理 7 号 Ascend910B3。

| 项目 | 结果 |
| --- | --- |
| 上步复核 | PASS：M04b2c 源码、扩展与完整证据哈希匹配，环境一致；本步新建产物重新覆盖受影响的 FP16 最小闭环 |
| 生产构建 | PASS：v3/910 强制构建 15 个 ASC 单元及 AICPU，链接 16 个对象；实际加载路径、源码、对象和动态库哈希可追溯。两种 MLA 单元的设备依赖均含全部六个算法头，无原 CANN 算法头依赖 |
| 独立组件 | PASS：358 项双 dtype dense 检查及 155 项 FP16 Cube 回归，无 skip。常规 O/LSE 与固定源逐元素一致并满足独立 FP64/低精度双 reference；另验证单任务 QK/P/PV 逐元素一致及两种 dtype 的精确中点舍入 |
| 入口契约 | PASS：582 项 eager/fake/C++、公开入口、metadata 拒绝及普通 fake shape 检查，无 skip；两种 dtype 使用相同边界 |
| 公开入口精度 | PASS：204 项，无 skip；每种 dtype 94 组记录、282 次公开重复调用与固定源逐元素一致并通过双 reference。FP16 O/LSE 最大绝对误差 `0.001953125` / `5.722046e-6`，BF16 为 `0.015625` / `7.629395e-6`；另验收舍入、交错 scale/形状、多 stream、输入释放后重分配、内部 scale 和仅返回 O |
| CPU 回归 | PASS：176 项，无 skip；包括调度对照、reference、来源、profile、生成器/依赖及 ASan/UBSan scoped launcher 检查 |
| 普通 FA3 | PASS：11 项，无 skip；覆盖 BSND、TND、分页、split、metadata 及编译图前反向。8 个普通前反向 Kernel 对象与上步逐字节一致，AICPU 由原源码重新构建 |
| 实际分派 | PASS：每种 dtype 五组 native CANN/FA3，合计 20 次 MLA launch 的核数、C1V2 和 BATCH_MODE 一致。FP16/BF16 key 分别为 `105000000010000001` / `105000000010022221`；注册二进制匹配 M02b1，303 份原生基线产物哈希未变。MLA 后普通 FA3 仍使用 mode 2 |
| 性能诊断 | 完成两种 dtype 各五组同卡交替采样，合计 2,000 个同步公开调用 E2E、1,000 条 kernel task；逐条核验原始 CANN CSV 的名称、设备、起始时间和时长。无 metadata E2E 仍慢于 CANN |
| 提交质量 | 六项适用仓库质量检查通过；源码、文档链接、差异及证据完整性核对见 `quality-staged.log`、`acceptance.json`、`checksums.sha256` |

单位 μs，中位数；shape 为 `B,Sq,Hq,Sk`：

| dtype | shape / scale | FA3 kernel | CANN kernel | FA3 公开 E2E | CANN E2E | E2E 比值 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| FP16 | 2,1,8,257 / 默认 | 12.730 | 14.130 | 508.615 | 305.040 | 1.6674 |
| FP16 | 2,1,8,257 / 0.23 | 12.820 | 14.061 | 513.760 | 306.170 | 1.6780 |
| FP16 | 1,7,16,65 / 默认 | 15.480 | 16.491 | 514.390 | 312.610 | 1.6455 |
| FP16 | 1,1,128,512 / 默认 | 27.621 | 28.651 | 511.875 | 308.285 | 1.6604 |
| FP16 | 2,2,128,511 / 默认 | 49.471 | 48.811 | 514.590 | 320.910 | 1.6035 |
| BF16 | 2,1,8,257 / 默认 | 12.650 | 14.101 | 451.205 | 274.435 | 1.6441 |
| BF16 | 2,1,8,257 / 0.23 | 12.761 | 14.060 | 450.985 | 274.915 | 1.6405 |
| BF16 | 1,7,16,65 / 默认 | 15.590 | 16.681 | 453.200 | 278.215 | 1.6290 |
| BF16 | 1,1,128,512 / 默认 | 27.811 | 28.701 | 484.645 | 297.320 | 1.6300 |
| BF16 | 2,2,128,511 / 默认 | 49.951 | 49.271 | 487.435 | 304.630 | 1.6001 |

每种 dtype 单独采样；每例每实现预热 5 次，5 轮交替顺序，每轮 E2E 20 次、profiler 10 次，每次等待设备完成。两端使用预先准备的同一输入，计时排除输入 H2D。FA3 E2E 包括公开入口校验、静态调度、输出/workspace 分配、发射及计算；CANN 包括固定原生分派和每次输出分配。

**最终性能门槛尚未达到。** BF16 无 metadata E2E 比值约 1.60–1.64，FP16 约 1.60–1.68。当前数据仅为诊断；M05 仍须逐例验收 `FA3 metadata 路径 E2E / 原 CANN E2E <= 1.0`，并按 [design](design.md) 区分 metadata 生成、复用和完整调用段成本。

## 环境、失败与限制

- Python `/usr/local/python3.11.15/bin/python3` 3.11.15；torch `2.7.1+cpu`，torch_npu `2.7.1`；CANN `/usr/local/Ascend/cann-9.0.0-beta.2`，bisheng clang 15，`dav-2201`、`-O2`。
- 7 号 910B3 初始、两轮性能采样前后和结束健康为 OK，前后 ECC 为 0；性能快照均只见一个进程，结束无进程。见 `environment.json`、`performance-*/npu-*.log`、`npu-final.json`。
- 实际扩展为本仓 `flash_attn_npu_3/flash_attn_npu_3.cpython-311-aarch64-linux-gnu.so`，SHA256 `b050af84c3163f9cda2d866adb96d5bf0465db73460a004c85e04ec2de3d3249`；本步已替换 M04b2c 产物，生产验收均使用这一新产物。
- 固定 CANN `ca18aee610729ca5270a44e2d84c751bc687fbf0`、CATLASS `769cd40a8716b28650b6bebb08db4834eea4462f`；源仓现有删除/未跟踪文件保持原样，CATLASS 子模块未修改。native vendor 仍为 `build/mla-m02b-validation/opp-dense/vendors/mla_ca18aee_transformer`。
- 首次 BF16 探针失败：tiling 字段解析覆盖了构建器的 dtype 参数，导致 BF16 输入误用 FP16 探针。已修复为独立字段变量；新探针 manifest 和编译命令确认 BF16 定义生效，完整复验通过。保留 `dense-initial.log`；定位后主动终止的重复复测见 `bf16-isolated*`，未记为通过。
- 新增的 FP16 中点构造首次触发双 reference 的零 ULP 估计失败，原 CANN 同样向离零方向舍入。该定向用例现按源 `CAST_ROUND` / `CAST_RINT` 分别作解析值的零容差检查；BF16 验证正负中点及奇偶尾数，常规精度阈值未调整。首次失败保留在 `components-final.log`，修正后的完整闭环为 `components-validated*`。
- 首次构建证据脚本按行首统计并行日志，漏计交错的编译标记；已修复并在 `provenance-final*` 核实 15 个 ASC / 16 个链接对象。分派证据脚本也已区分 FIA 与随后普通用例触发的 Cast/Fill 注册；原始日志保留全部算子。首次质量检查按钩子自动格式化，后续检查通过。
- 登录 shell 初始化仍不返回输出，使用非登录 shell；NPU 及 sanitizer 沿用上步已验证的沙箱外执行方式。CANN beta 字符串、重复 tiling 注册和普通 Kernel 自动类型推断警告保留于日志，profiler 原始数据解析通过。

## 证据与复现

- `previous-verified.json`、`environment*`：上步源码/证据及当前环境；源仓初始状态也保存在前者。
- `build-final*`、`build-source-manifest.json`、`build-provenance.json`：强制构建、实际加载、对象哈希及两种 MLA 单元的设备依赖。
- `components-validated.log/xml`、`dense-validated/`、`cube-validated/`：独立精度、舍入、中间结果和 FP16 Cube 回归；包含固定源导出、编译命令、源/库哈希和逐例记录。
- `accuracy-final.log/xml`、`accuracy-final/`、`cpu-final*`、`regression-final*`：公共入口、契约、CPU 和普通 FA3 验收；精确命令见对应 `*-command.json`。
- `performance-fp16/`、`performance-bf16/`：软件栈及加载库、墙钟样本、完整 profiler 原始数据、逐条 CSV 核验与汇总。`dispatch-*.log/json`、`dispatch-proof.json` 保存实际分派。
- `source-mapping.json`、`native-baseline-verified.json`、`acceptance.json`、`checksums.sha256`：算法对应关系、固定基线、源码和最终证据完整性。

从仓库根目录复现；实机选空闲健康设备，复验使用新的 artifact 目录，保留既有证据：

```bash
FLASH_ATTN_BUILD_VERSION=v3 FLASH_ATTN_BUILD_NPU=910 FLASH_ATTN_SKIP_SUBMODULE_INIT=1 FLASH_ATTN_FORCE_REBUILD=TRUE python3 setup.py build_ext --inplace
ASCEND_RT_VISIBLE_DEVICES=7 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MLA_DENSE_ARTIFACT_DIR=build/mla-m04b3-bf16/recheck-dense MLA_CUBE_ARTIFACT_DIR=build/mla-m04b3-bf16/recheck-cube python3 -m pytest -x -q tests/test_flash_attn_npu_v3_mla_dense.py tests/test_flash_attn_npu_v3_mla_cube.py
ASCEND_RT_VISIBLE_DEVICES=7 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MLA_DENSE_ARTIFACT_DIR=build/mla-m04b3-bf16/recheck-accuracy python3 -m pytest -x -q tests/test_flash_attn_npu_v3_mla_contract.py tests/test_flash_attn_npu_v3_mla_forward.py
TORCH_DEVICE_BACKEND_AUTOLOAD=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 -m pytest -x -q tests/test_flash_attn_npu_v3_mla_build.py tests/test_flash_attn_npu_v3_mla_reference.py tests/test_flash_attn_npu_v3_mla_provenance.py tests/test_flash_attn_npu_v3_mla_profile.py tests/test_flash_attn_npu_v3_mla_schedule.py
MLA_VENDOR_DIR="$PWD/build/mla-m02b-validation/opp-dense/vendors/mla_ca18aee_transformer"
ASCEND_RT_VISIBLE_DEVICES=7 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MLA_CANN_VENDOR="$MLA_VENDOR_DIR" ASCEND_CUSTOM_OPP_PATH="$MLA_VENDOR_DIR" LD_LIBRARY_PATH="$MLA_VENDOR_DIR/op_api/lib:$LD_LIBRARY_PATH" MLA_FORWARD_DTYPE=bf16 MLA_FORWARD_ARTIFACT_DIR=build/mla-m04b3-bf16/recheck-performance-bf16 python3 -m tests.mla_forward_profile
```

FP16 性能复验将 `MLA_FORWARD_DTYPE` 与输出目录后缀改为 `fp16`。普通 FA3 的 11 项精确节点见 `regression-final-command.json`；重跑时也使用新的 JUnit 输出路径。

最终暂存检查：

```bash
PYTHONPATH=/tmp/mla-m01-precommit PRE_COMMIT_HOME=/tmp/mla-m01-precommit-cache bash ci/code_quality.sh --staged
python3 build/mla-m04b3-bf16/verify_acceptance.py --staged
git diff --check
git diff --cached --check
```

当前闭环提交用 `git log -1 -- docs/mla/status.md` 定位。
