# MLA 当前状态

## 当前能力与下一步

- 分支 `feat/mla-cann-migration`，接手 HEAD `c1ecb2e`，接手工作区干净。本步完成 M05b2b3a：现有 dense 前向的生产 AICPU 生成与校验消费链路。
- M05b2b3 已在 plan 拆为 a/b。官方 `flash_attn_func` 没有显式 metadata 参数；公开复用还需开放 `flash_attn_with_kvcache` 的只读 dense 子集及完整长度校验，因此本次先独立验收生产内部链路。
- `DenseForward` 每次分配私有 v1 metadata，由生产 `ComputeMlaDenseMetadata` 在 AICPU 生成，再经双事件依赖交给当前 stream。metadata 登记 AICPU allocator stream，提交闭包保留输入、记录、O/LSE 和 workspace；各调用不共享可变 scratch。
- 生产 FP16/BF16 launcher 使用 `LoadDenseMetadata`，所有 Cube/Vector 校验通过后才构造 TPipe、进入 attention Kernel。实际切块和分核端点来自设备记录；scale 仍由当次前向提供。host 暂留同一 `BuildDenseSchedule`，仅提供启动、分配及独立校验 context 的边界。
- 记录仍为 v1、401 个 int32 字；只将 context 声明移到无设备函数的公共私有头中，未改序列化、计算 Kernel/Block/Tile、舍入、分核算法或成功范围。
- 公开范围仍为 FP16/BF16、512+64 共享 KV、连续 BSND、B>0、`1<=Sq*Hq<=256`、KV `1..512`、Hkv=1、无 mask、显式 `num_splits=1`。真实/fake/C++ 的显式 MLA metadata、kvcache 及范围外组合继续明确拒绝。
- 唯一下一步：M05b2b3b，接通已验收 dense 范围的公开 metadata 生成与只读 kvcache 复用，补齐完整长度校验、真实/fake/C++、运行时 scale、allocator/stream 生命周期及普通 FA3 回归。之后 M05c 验收最终 E2E；整个 M05 完成前不开始 M04c1。本步提交后停止。

## 最近一次闭环证据

证据目录：`build/mla-m05b2b3-public/`；物理 7 号 Ascend910B3。最终必要检查共 **1,079 项 pytest 通过，无失败或 skip**。8 个预期设备异常由独立测试明确验收；早期失败保留且不计为通过。

| 项目 | 结果 |
| --- | --- |
| 上步复核 | PASS：1,956 份证据哈希、937 项结果、61 个源码、17 个生产对象及实际扩展一致，固定源仓状态未变，见 `previous-verified.json` |
| CPU | PASS：90 项消费者解码/损坏记录/启动桥/构建依赖检查，含 metadata 入口与消费者 ASan/UBSan，见 `cpu-final*` |
| 公开前向精度及生命周期 | PASS：208 项 FP16/BF16 检查，含 D01/D02、边界、多 batch、舍入、scale、连续调用、跨流输入释放、metadata/scratch 分配压力及双 host 线程；源对照逐元素一致，独立双 reference 阈值未改，见 `forward-final*` |
| 实际生产消费者 | PASS：32 项，测试 C 桥链接本轮扩展，不编译另一份设备实现；24 项源对照与 8 项真实设备异常，见 `production-complete*`、`producer-complete/` |
| 公共入口回归 | PASS：129 项普通 metadata 签名/兼容/精度/compile，20 项普通 FA3 BSND/TND/分页/split/metadata/mask/softcap/scale/编译图，以及 600 项 MLA 契约，见 `signature-final*`、`regression-final*`、`contract-final*` |
| 构建与加载 | PASS：15 个 ASC、2 个 AICPU 对象强制重建；新消费者设备头依赖、源/对象/实际加载库哈希均核查，见 `build-final*`、`build-provenance.json` 和 `production-artifacts/` |
| 性能诊断 | 两种 dtype 各五组，与固定原 CANN 同轮交替测量；BF16 因 wall 抖动追加一次完整五组补测，全部原始数据保留，见下述口径 |
| 提交质量 | 六项适用仓库检查、文档链接、差异及证据完整性检查，见 `quality-staged.log`、`acceptance.json`、`checksums.sha256` |

8 个错误覆盖损坏头部、生成失败、形状不匹配和 workspace 高位损坏；均由正常 event/device 同步在 **26.62–29.21 ms** 报告 trap 异常。每次先用原诊断消费者验证准确状态，再重新生成/注入记录，交给实际生产 launcher；生成到消费间禁止 Python 设备标量回读及主机等待。全状态解码拒绝仍由 CPU 与上步设备证据覆盖，本次未重复未改的全部独立组件矩阵。

## 性能口径与未完成项

- 每种 dtype 各五组，每组五轮 × 20 个 wall 样本和五轮 × 10 个 profiler 样本，FA3/CANN 交替。每轮完整采集保存 1,000 个 wall 样本、500 个 attention Kernel 样本和 250 个 AICPU 生成样本；BF16 补测再保存同样数量。
- Profiler 与原始 CANN CSV 逐条核对，并验证每次 FA3 的 AICPU 任务完成后才开始 attention。每次调用都生成 metadata，E2E 包含当前 host 调度、metadata/O/LSE/workspace 分配、事件、生成与消费；输入 H2D 不计时。CANN 同样逐次分配 O/LSE。
- attention Kernel / 同轮 CANN 中位数比值：FP16 **0.973–1.027**；BF16 首轮 **0.965–1.034**，补测 **0.976–1.036**。
- 整调用 E2E / 同轮 CANN：FP16 **1.695–1.755**；BF16 首轮 **1.237–1.732**，补测 **1.568–1.662**。BF16 首轮部分 CANN wall 轮均值约 313–898 μs，补测仍有离群样本；不能把受主机抖动影响的低比值视为改善。逐例值、轮均值与样本见 `performance-fp16/`、`performance-bf16/`、`performance-bf16-confirm/`。
- 固定 CANN 303 份构建文件重新核对；两轮性能的 PID/物理设备映射及前后快照见 `performance-device-proof.json`、`performance-confirm-device-proof.json`。物理 7 号设备在采集前后仅有本轮进程。
- **这是内部生成加一次前向的诊断，不是显式 metadata 复用验收，也未达到最终性能门槛。** 公开生成/复用尚未开放，M05 整体未完成。M05c 仍须逐例验收 `FA3 metadata E2E / 同轮 CANN E2E <= 1.0`，分别记录生成、复用和完整调用段。

## 环境、失败与限制

- Python `/usr/local/python3.11.15/bin/python3` 3.11.15；torch `2.7.1+cpu`、torch_npu `2.7.1`；CANN `/usr/local/Ascend/cann-9.0.0-beta.2`，bisheng clang 15，`dav-2201`、`-O2`。
- 实际扩展 `flash_attn_npu_3/flash_attn_npu_3.cpython-311-aarch64-linux-gnu.so`，SHA256 `5eacb702a28409899967e6de90332463d08c2c72f4e3c6273a58ecaf82e9199b`。CATLASS `769cd40a8716b28650b6bebb08db4834eea4462f`、固定 CANN `ca18aee610729ca5270a44e2d84c751bc687fbf0`、官方 FA3 `d15f1531a460ba456f41b01a774f33ab2db8febf`。未改子模块、CANN 源仓既有改动或 950 后端。
- 7 号设备开始/结束健康 OK、ECC 全零，结束无设备进程，见 `environment.json`、`environment-final.json`。8 个异常测试进程组无存活进程，见 `process-groups.json`。Trap 后不承诺复用执行上下文；错误测试保留 tensor/stream 至进程退出，父进程清理其 CANN 工作进程并禁用 core dump。
- 首次 `build-initial` 因启动头提前引入设备校验函数、缺少 `__aicore__` 定义失败；将纯 context 声明移到 `mla_metadata.h` 后修复，再对最终源码全量重建。首次 `production-final` 在测试桥编译阶段缺少 CANN profiling 头搜索路径，未运行设备断言；补齐后 `production-complete` 全部通过。两份失败日志保留。
- 初次 clang-format 改写格式，随后 ruff 指出测试 fixture 导入别名遮蔽，均已修正并复查。登录 shell 有延迟，沙箱内 npu-smi 返回初始化失败；实机及 sanitizer 沿用非登录 shell、沙箱外执行。未把这些失败计为通过。
- 本步不开放公开长度/metadata 复用，也不承诺图捕获；相关边界和生命周期分别在 M05b2b3b、M16 验收。重复 host 调度及调用开销留待 M05c。

## 证据与复现

`*-command.json`、`*-exit.json`、日志与 JUnit 保存精确命令、环境及结果；`build-source-manifest.json` / `runtime-source-manifest.json` 和 `source-snapshots/` 保存验收源码；`production-artifacts/` 保存扩展及全部链接对象。性能命令见 `run_performance.py`、`run_performance_confirm.py`，普通回归节点见 `run_regression.py`。

从仓库根目录复现；选择空闲健康设备，为 artifact/JUnit 设置新路径，避免覆盖已有证据：

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 MLA_CONSUMER_SANITIZE=1 python3 -m pytest -x -q tests/test_flash_attn_npu_v3_mla_metadata_consumer.py tests/test_flash_attn_npu_v3_mla_build.py
FLASH_ATTN_BUILD_VERSION=v3 FLASH_ATTN_BUILD_NPU=910 FLASH_ATTN_SKIP_SUBMODULE_INIT=1 FLASH_ATTN_FORCE_REBUILD=TRUE python3 setup.py build_ext --inplace
ASCEND_RT_VISIBLE_DEVICES=7 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 -m pytest -x -q tests/test_flash_attn_npu_v3_mla_forward.py tests/test_flash_attn_npu_v3_mla_production_metadata.py
```

提交前检查：

```bash
PYTHONPATH=/tmp/mla-m01-precommit PRE_COMMIT_HOME=/tmp/mla-m01-precommit-cache bash ci/code_quality.sh --staged
python3 build/mla-m05b2b3-public/verify_acceptance.py --staged
git diff --check
git diff --cached --check
```

当前闭环提交用 `git log -1 -- docs/mla/status.md` 定位。
