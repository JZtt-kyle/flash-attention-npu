# MLA 当前状态

## 当前能力与下一步

- 分支 `feat/mla-cann-migration`，接手 HEAD `4280158`，接手工作区干净。本步完成 M05c1：已验收只读 dense kvcache 的精简 C++ dispatcher。
- 公开 `flash_attn_with_kvcache` 识别 MLA、拒绝未支持选项后调用内部 `_mla_dense_forward`，返回 O/LSE；请求 LSE 时包装补两个遵循默认浮点 dtype 的空 accumulator。真实/fake 独立检查张量、scale、长度和 metadata 外形，Autograd 层透传且明确拒绝 requires_grad。公开签名、v1 的 401 字记录及设备消费校验沿用既有契约。
- 已支持范围仍为 910B、FP16/BF16、512+64 共享 KV、连续 BSND、B>0、`1<=Sq*Hq<=256`、KV 容量 `1..512`、Hkv=1、无 mask、显式 `num_splits=1`。长度只接受整容量；追加 KV、TND/PA、部分/零长度、梯度等范围外组合明确拒绝。
- 调度、Kernel/Block/Tile、metadata 生成/序列化、stream 及 allocator 生命周期保持原路径。显式复用跳过 AICPU；kvcache 无 metadata 时内部生成；`flash_attn_func` 仍走原通用入口。
- **唯一下一步：M05c2，优先优化公开 metadata 生成开销，再按测量处理剩余 host 准备/调度并验收完整 E2E。** 两种 dtype 各五组的复用前向和生成加四次前向已达标，但生成加一次前向仍高于 1.0；整个 M05 尚未完成，不开始 M04c1。本步提交后停止。

## 最近一次闭环证据

证据目录：`build/mla-m05c1-host/`；物理 7 号 Ascend910B3。最终检查集共 **1,637 次测试执行、1,595 个去重测试通过，无失败或 skip**；重复的 42 项为返回/compile 26 项和最终精度复验 16 项。初检失败轮及被最终入口测试覆盖的 `dispatch-complete` 不计入此总数。

| 项目 | 结果 |
| --- | --- |
| 上步复核 | PASS：2,153 份证据哈希、1,579 项结果、17 个生产对象、实际扩展及 303 份固定 CANN 构建文件一致；源仓状态未变，见 `previous-verified.json` |
| CPU | PASS：90 项消费者解码、启动桥及构建依赖检查，含 ASan/UBSan，见 `cpu-final*` |
| 主验证：公开 MLA | PASS：512 项，含双 dtype 48 组形状/scale 的三路径源对照、舍入、运行时 scale、跨流释放/分配压力、主机线程和 snapshot，见 `public-final*`、`producer-final/` |
| 主验证：设备拒绝 | PASS：32 个隔离进程，覆盖非法生成、隐式生成及旧记录复用时的非法长度与损坏记录；进程组均已退出，见 `errors-final*`、`process-groups.json` |
| 主验证：契约与普通 FA3 | PASS：588 项 MLA 契约、129 项普通 metadata 签名/兼容/精度/compile、20 项普通 FA3 BSND/TND/分页/split/metadata/mask/softcap/scale/compile，见对应 `*-final*` |
| 最终 scale 边界复验 | PASS：250 项独立 dispatcher/schema/fake/梯度/参数/默认 dtype/compile/返回检查；含正负 FP32 最大值及超出边界的 double、NaN/Inf；另复验双 dtype D01/D02 的 16 组源精度，见 `dispatch-guard*`、`accuracy-guard*` |
| 构建与加载 | PASS：最终增量重建 `mla_forward.o` 并重新链接，其余 16 个对象（含 Kernel/AICPU）哈希与接手时一致；源码、依赖、对象和实际加载库核查通过，见 `build-guard*`、`build-provenance.json`、`production-artifacts/` |
| 性能 | PASS：最终产物的十个代表用例复用 E2E 均 <= 1.0；metadata 生成及 R=1/4 完整调用段分别记录，R=1 尚未达标，见下表和 `performance-*/` |

主验证扩展为 `54d9e9f0…`。复核时发现 double 略超 FP32 最大值可被舍入成有限 float，随后仅将 scale 检查移到窄化前并补边界用例；合法输入的计算路径未变。最终扩展为 `697f73ed0eaf07ce6248d875791f24c9860c408e503ca7121a199b1fff15642b`，完成表中最终边界/精度复验及全部性能采集。主验证源码和产物保留于 `before-scale-guard/`；两版差异见 `scale-guard.patch`，两版的其余 16 个对象完全一致。未将旧扩展的全量结果误记为最终扩展重跑。

## 性能口径与未完成项

两种 dtype 各五组，每组五轮 × 20 个 wall 样本，比较显式复用、CANN 单次、metadata 生成、隐式生成 kvcache、生成加 1/4 次前向和 CANN 四次调用，共 **7,000 个 wall 样本**。R=1/4 在采集前固定，完整段只在末尾同步，两边前向次数一致。输入 H2D 不计时；入口检查、准备、输出/workspace 分配、发射、依赖和完成同步计时。另以五轮 × 10 个样本交替采集两边 attention Kernel，共 **1,000 个 Kernel 样本**；显式复用的 AICPU 生成任务为 0。

逐例复用 E2E / 同轮 CANN E2E：

| `(B,Sq,Hq,Sk)` / scale | FP16 | BF16 |
| --- | --- | --- |
| `(2,1,8,257)` / 默认 | 0.768274 | 0.758649 |
| `(2,1,8,257)` / 0.23 | 0.767245 | 0.745805 |
| `(1,7,16,65)` / 默认 | 0.772359 | 0.765525 |
| `(1,1,128,512)` / 默认 | 0.773832 | 0.772883 |
| `(2,2,128,511)` / 默认 | 0.774649 | 0.764895 |

其他指标的五组中位数范围如下；全部逐例值、轮均值和原始样本保留在 `profile.json`、`wall-samples.json` 及 profiler CSV 中。

| 指标 | FP16 | BF16 |
| --- | --- | --- |
| attention Kernel / CANN | 0.963–1.029 | 0.967–1.034 |
| metadata 生成 E2E | 296.60–301.96 μs | 312.46–316.85 μs |
| 隐式生成 kvcache E2E / CANN 单次 | 0.859–0.905 | 0.810–0.887 |
| 一次生成 + 1 次前向 / CANN 单次 | **1.530–1.588** | **1.525–1.608** |
| 一次生成 + 4 次前向 / CANN 四次 | 0.887–0.896 | 0.874–0.892 |

Profiler 与原始 CANN CSV 逐行核对，PID/物理 7 号映射及采集前后独占快照通过，见 `performance-device-proof.json`。没有通过增加复用次数代替 R=1 验收；**M05c1 完成，M05c2 继续处理 metadata 生成及剩余 host 成本，整个 M05 未完成。**

## 环境、修复与限制

- Python `/usr/local/python3.11.15/bin/python3` 3.11.15；torch `2.7.1+cpu`、torch_npu `2.7.1`；CANN `/usr/local/Ascend/cann-9.0.0-beta.2`；bisheng clang 15.0.5、`dav-2201`、`-O2`。实际扩展位于 `flash_attn_npu_3/flash_attn_npu_3.cpython-311-aarch64-linux-gnu.so`，哈希见上文。
- CATLASS `769cd40a8716b28650b6bebb08db4834eea4462f`，固定 CANN `ca18aee610729ca5270a44e2d84c751bc687fbf0`，官方 FA3 `d15f1531a460ba456f41b01a774f33ab2db8febf`；子模块、CANN 既有改动和 950 后端未改。设备前后健康 OK、ECC 全零，结束无设备进程，见 `environment*.json`。
- 首次诊断因覆盖 CANN PYTHONPATH 初始化失败，恢复原路径后成功。`dispatch-initial` 在 125 项通过后发现 Autograd 注册绕过 fake；改为 fallthrough。`dispatch-final` 在 184 项通过后发现公共包装的 math.isfinite 无法 fullgraph 捕获；scale 校验留在真实/fake op 内后通过。失败日志及对应源码保留，均未计为通过。
- 初次 clang-format 改写格式，随后检查通过；最终暂存质量和差异检查见 `quality-staged*`。生成的 `kernel_meta` 归档至证据目录。沿用非登录 shell、沙箱外实机/sanitizer 执行。
- 图捕获生命周期仍留给 M16；本轮 Torch compile 只验收 eager backend/fullgraph。Trap 后不承诺继续复用执行上下文；错误测试禁用 core dump 并清理专属进程组。

## 证据与复现

`*-command.json`、`*-exit.json`、日志和 JUnit 保存精确命令、环境及结果；`build-source-manifest.json` / `runtime-source-manifest.json`、源码快照和生产产物记录最终版本，`before-scale-guard/` 保留主验证版本。完整节点见 `run_device.py`、最终边界复验见 `run_guard.py`、性能命令见 `run_performance.py`；提交前综合核验见 `acceptance.json`、`checksums.sha256`。

从仓库根目录复现；选择空闲健康的 910B，为 artifact/JUnit 设置新路径，避免覆盖已有证据：

```bash
FLASH_ATTN_BUILD_VERSION=v3 FLASH_ATTN_BUILD_NPU=910 FLASH_ATTN_SKIP_SUBMODULE_INIT=1 python3 setup.py build_ext --inplace
TORCH_DEVICE_BACKEND_AUTOLOAD=0 MLA_CONSUMER_SANITIZE=1 python3 -m pytest -x -q tests/test_flash_attn_npu_v3_mla_metadata_consumer.py tests/test_flash_attn_npu_v3_mla_build.py
ASCEND_RT_VISIBLE_DEVICES=7 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 -m pytest -x -q tests/test_flash_attn_npu_v3_mla_dispatch.py tests/test_flash_attn_npu_v3_mla_kvcache.py
```

提交前检查：

```bash
PYTHONPATH=/tmp/mla-m01-precommit PRE_COMMIT_HOME=/tmp/mla-m01-precommit-cache bash ci/code_quality.sh --staged
python3 build/mla-m05c1-host/verify_acceptance.py
git diff --check
git diff --cached --check
```

当前闭环提交用 `git log -1 -- docs/mla/status.md` 定位。
