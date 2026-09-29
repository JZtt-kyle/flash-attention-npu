# MLA 当前状态

## 当前能力与下一步

- 分支 `feat/mla-cann-migration`，接手 HEAD `1a574b8`，接手工作区干净。本步完成 M05b2b3b：已验收 dense 范围的公开 metadata 生成与只读 kvcache 复用。
- 官方 `get_scheduler_metadata` 对 64/512 返回 v1 的 401 字 int32 设备记录。`flash_attn_with_kvcache` 接受显式记录或内部生成，LSE 模式返回 O、LSE 及两个空 accumulator；真实/fake/C++ 边界一致。`flash_attn_func` 继续内部生成。
- 范围仍为 910B、FP16/BF16、512+64 共享 KV、连续 BSND、B>0、`1<=Sq*Hq<=256`、KV 容量 `1..512`、Hkv=1、无 mask、显式 `num_splits=1`。追加 KV、TND/PA、部分/零长度、梯度、非连续输入及其他未接通选项明确拒绝。
- 生成器要求 NPU 上连续 `int32[B]` 长度；前向接受 None（整容量）、等于容量的整数或同设备连续 `int32[B]`。AICPU 读取并校验全部长度；生产消费者在 attention 前再检查当次全部长度，显式复用不能绕过长度变化。非法生成写失败状态并清零 payload，消费由正常调用/同步报告 Trap；无长度 D2H 或 steady-state 主机等待。
- metadata 可克隆或使用连续偏移视图，不依赖 Python 私有属性，scale 只由当次前向提供。显式复用跳过 AICPU 生成；生成/消费登记 allocator stream，闭包保留输入、长度和记录，每次前向独立分配 O/LSE/workspace。跨流使用仍由调用方建立正常 stream/event 依赖，消费期间记录保持只读。
- 计算 Kernel/Block/Tile、序列化、舍入和分核算法未改；设备入口只增加完整长度校验。host 暂留同一 `BuildDenseSchedule`，为启动、分配及独立校验 context 提供边界。
- **唯一下一步：M05c，优化重复 host 准备/调度及发射成本，逐例验收 metadata E2E / 同轮 CANN E2E <= 1.0。** 本步功能闭环，最终性能门槛未达到，整个 M05 尚未完成；不开始 M04c1。本步提交后停止。

## 最近一次闭环证据

证据目录：`build/mla-m05b2b3b-public/`；物理 7 号 Ascend910B3。最终必要检查共 **1,579 项 pytest 通过，无失败或 skip**；32 个预期设备异常由隔离测试明确验收。初检及中断轮不重复计入此总数。

| 项目 | 结果 |
| --- | --- |
| 上步复核 | PASS：2,013 份证据哈希、1,079 项结果、17 个生产对象及扩展一致，固定源仓状态未变，见 `previous-verified.json` |
| CPU | PASS：90 项 metadata 解码、损坏记录、启动桥和构建依赖检查，含 ASan/UBSan，见 `cpu-final*` |
| 公开 metadata/kvcache | PASS：512 项，覆盖生成记录与 host 全字段一致、真实/fake/C++ 参数边界、None/整数/tensor 长度、返回值、Torch compile、两种 dtype 源对照精度、克隆/偏移复用、运行时 scale、跨流释放/分配压力、双 host 线程及快照语义，见 `public-complete*`、`producer-complete/` |
| 生产拒绝路径 | PASS：32 项独立进程，覆盖生成失败、隐式生成与旧记录复用时的负/零/短/超容量长度，以及 magic/dtype/shape/workspace 损坏，见 `errors-final*` |
| 原 dense 前向 | PASS：208 项两种 dtype 精度、边界、舍入、scale、连续调用和生命周期回归，见 `forward-final*` |
| 契约与普通 FA3 | PASS：588 项 MLA 契约、129 项普通 metadata 签名/兼容/精度/compile、20 项普通 FA3 BSND/TND/分页/split/metadata/mask/softcap/scale/编译回归，见 `contract-final*`、`signature-final*`、`regression-final*` |
| 构建与加载 | PASS：15 个 ASC、2 个 AICPU 对象强制重建；设备头依赖、60 个构建源文件、125 个运行源码快照、对象及实际加载库哈希均核查，见 `build-initial*`、`build-provenance.json`、`production-artifacts/` |
| 性能诊断 | 完成两种 dtype 各五组与固定 CANN 的同轮采集；显式复用的 500 条 profiler 样本中 AICPU 生成任务为 0，见下述口径；未达到最终门槛 |

公开精度共 48 组形状/scale，两种 dtype 分别核对隐式生成、显式记录和奇数 offset 克隆视图三种路径，O/LSE 与固定 CANN 源逐元素一致，独立双 reference 阈值未改。运行时错误耗时 **71.95–84.27 ms**；非法长度放在第 21 个 batch，检查覆盖整个长度数组。生成失败另检查失败状态和清零 payload，最终生成/注入到消费间禁止 Python 设备标量回读和主机等待。32 个错误进程组无存活或遗留进程，见 `process-groups.json`。

## 性能口径与未完成项

每种 dtype 各五组，每组五轮 × 20 个 wall 样本，分别采集显式复用、CANN 单次、metadata 生成、隐式生成前向、一次生成加 1/4 次前向及 CANN 四次调用，共 **3,500 个 wall 样本**。R=1/4 在采集前固定，完整段内只在末尾同步，两边执行相同次数的前向。输入 H2D 不计时，输出/metadata/workspace 分配、校验、准备、发射和同步计时。另以五轮 × 10 个 profiler 样本交替比较复用前向与 CANN，共 **500 个 attention Kernel 样本**。

下表是五组逐例中位数比值的范围；逐例值、轮均值和全部样本见 `performance-fp16/`、`performance-bf16/`，不能用范围或均值代替逐例门槛。

| 指标 | FP16 | BF16 |
| --- | --- | --- |
| attention Kernel / CANN | 0.973–1.030 | 0.957–1.034 |
| 显式复用 E2E / CANN 单次 | **1.643–1.717** | **1.594–1.608** |
| 隐式生成前向 E2E / CANN 单次 | 1.609–1.655 | 1.562–1.585 |
| metadata 生成 E2E | 301.46–315.51 μs | 320.46–325.20 μs |
| 一次生成 + 1 次前向 / CANN 单次 | 2.387–2.482 | 2.318–2.374 |
| 一次生成 + 4 次前向 / CANN 四次 | 2.043–2.134 | 1.995–2.028 |

Profiler 与原始 CANN CSV 逐条核对；显式复用确认未重发 AICPU。固定 CANN 的 303 份构建文件、两轮的 PID/物理设备映射与前后独占快照已核查，见 `native-baseline-verified.json`、`performance-device-proof.json`。wall 存在离群样本（如 FP16 某轮 CANN 均值约 520 μs），原始数据全部保留。所有代表用例的复用 E2E 均高于 1.0；**本步是功能闭环和性能诊断，不是 M05c 最终验收。**

## 环境、失败与限制

- Python `/usr/local/python3.11.15/bin/python3` 3.11.15；torch `2.7.1+cpu`、torch_npu `2.7.1`；CANN `/usr/local/Ascend/cann-9.0.0-beta.2`；实际 bisheng clang 15.0.5、`dav-2201`、`-O2`，见 `compiler.json`。
- 实际扩展 `flash_attn_npu_3/flash_attn_npu_3.cpython-311-aarch64-linux-gnu.so`，SHA256 `1cde883860e34ac08ac37399b441720f7a8089816aef0b5febcca3220495c7f4`。CATLASS `769cd40a8716b28650b6bebb08db4834eea4462f`、CANN `ca18aee610729ca5270a44e2d84c751bc687fbf0`、官方 FA3 `d15f1531a460ba456f41b01a774f33ab2db8febf`。未改子模块、CANN 源仓既有改动或 950 后端。
- 物理 7 号开始/结束健康 OK、ECC 全零，结束无设备进程，见 `environment.json`、`environment-final.json`。Trap 后不承诺复用执行上下文；测试禁用 core dump，保留 tensor/stream 至进程退出并清理专属进程组。
- 首轮 `public-final` 在 479 项通过后，因同一 shape/scale 的 CPU reference 重复计算主动中断，returncode=2；未计为完整通过，日志及 `public-interrupted-source.py` 保留。将四种 scale 的 reference 各计算一次、仍核对全部 32 组跨流输出后，`public-complete` 全量重跑 512 项通过；未改范围或阈值。
- 初始格式检查由 ruff/clang-format 改写格式，后续项目检查通过。一次质量扫描包含运行期 `kernel_meta/buildPidInfo.json`，EOF 检查加了换行；随后只扫描本步源码，生成目录已归档为 `kernel-meta/`。这些非零退出检查未记为通过。
- 一个非固定版本的临时 upstream 工作树不含目标 commit，未作为依据；本步使用固定版本快照，与 M05b2 保存的官方源码哈希逐一一致，见 `upstream/`。实机及 sanitizer 沿用非登录 shell、沙箱外执行。
- 图捕获生命周期未验收，留给 M16；Torch compile 本轮只验收 eager backend/fullgraph。重复 host 调度及端到端开销留给 M05c。

## 证据与复现

`*-command.json`、`*-exit.json`、日志和 JUnit 保存精确命令、环境及结果；`build-source-manifest.json` / `runtime-source-manifest.json`、源码快照和 `production-artifacts/` 对应本轮构建与运行产物。性能命令见 `run_performance.py`，设备回归节点见 `run_device_complete.py`。提交前质量、链接、差异和证据检查见 `quality-staged.log`、`acceptance.json`、`checksums.sha256`。

从仓库根目录复现；选择空闲健康的 910B，为 artifact/JUnit 设置新路径，避免覆盖已有证据：

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 MLA_CONSUMER_SANITIZE=1 python3 -m pytest -x -q tests/test_flash_attn_npu_v3_mla_metadata_consumer.py tests/test_flash_attn_npu_v3_mla_build.py
FLASH_ATTN_BUILD_VERSION=v3 FLASH_ATTN_BUILD_NPU=910 FLASH_ATTN_SKIP_SUBMODULE_INIT=1 FLASH_ATTN_FORCE_REBUILD=TRUE python3 setup.py build_ext --inplace
ASCEND_RT_VISIBLE_DEVICES=7 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 -m pytest -x -q tests/test_flash_attn_npu_v3_mla_kvcache.py tests/test_flash_attn_npu_v3_mla_forward.py
```

提交前检查：

```bash
PYTHONPATH=/tmp/mla-m01-precommit PRE_COMMIT_HOME=/tmp/mla-m01-precommit-cache bash ci/code_quality.sh --staged
python3 build/mla-m05b2b3b-public/verify_acceptance.py --staged
git diff --check
git diff --cached --check
```

当前闭环提交用 `git log -1 -- docs/mla/status.md` 定位。
