# MLA 当前状态

## 当前能力与下一步

- 分支：`feat/mla-cann-migration`；接手 HEAD 为 `61109bb`，接手工作区干净。本次将 M05b2b 拆为设备长度读取（M05b2b1）、设备错误上报（M05b2b2）和公开接入（M05b2b3），完成 M05b2b1。
- 内部 AICPU 生成器现在读取设备上的连续 int32 `cache_seqlens`。仅接受 B 个长度全部等于 dense KV 容量；零、负、较短、非均匀及超容量长度均拒绝。空地址/计数冲突、计数不匹配、不对齐和地址范围溢出在读取前拒绝。内部无长度路径继续按完整容量生成。
- 成功记录仍为 v1、401 字 int32，与无长度路径逐字一致，不保存长度地址、scale 或可变 scratch。新增失败状态为 `InvalidCacheLengths=3` / `UnsupportedCacheLength=4`，payload 全零；现有设备消费者返回 `GenerationFailed`，不触碰 O/LSE/workspace。尚未将此内部状态转成公开调用的运行时错误。
- 公开能力未扩展：FP16/BF16、512+64 共享 KV、连续 BSND、B>0、`1<=Sq*Hq<=256`、KV `1..512`、Hkv=1、无 mask、显式 `num_splits=1`；公开 MLA metadata 在真实/fake/C++ 中继续明确拒绝。普通 FA3 的官方签名及旧关键字兼容层保持 M05b2a 行为。
- 唯一下一步：M05b2b2，完成设备校验错误上报；失败记录或损坏 metadata 必须在调用/同步点报错，不能静默返回未初始化输出，也不能引入长度 D2H 或 steady-state 主机同步。之后 M05b2b3 接公开链路，M05c 验收最终 E2E；整个 M05 完成前不开始 M04c1。本步提交后停止。

## 最近一次闭环证据

证据目录：`build/mla-m05b2b1-lengths/`；物理 7 号 Ascend910B3。最终必要检查共 **1,058 项 pytest 通过，无失败或 skip**。早期环境失败和扩展检查超时单独记录如下，未计为通过。

| 项目 | 结果 |
| --- | --- |
| 上步复核 | PASS：M05b2a 的 611 份证据哈希、771 项结果、61 个源码、17 个对象、实际扩展和固定源仓状态一致 |
| CPU | PASS：175 项；本步 metadata 入口及消费者 ASan/UBSan、地址/计数/内容边界、失败覆盖与恢复、增量依赖和完整固定源调度对照；见 `cpu-complete*` |
| AICPU | PASS：52 项；保留 3,529 份设备记录与 host 比对，其中 3,494 份对固定源；新增 30 项覆盖逐样本长度、4 字节 offset、非法 buffer/内容、设备更新及双 stream；见 `producer-final*` |
| 独立 dense 消费 | PASS：60 项，FP16/BF16 的 D01/D02、边界、scale、复用、损坏记录及长度失败/恢复；生成到消费之间无 D2H/同步，失败时全部 O/LSE/workspace 字节不变；见 `device-final*` |
| 公共入口回归 | PASS：129 项签名/兼容/精度/compile、622 项 MLA 契约/精度/生命周期和 20 项普通 FA3，含 BSND/TND/分页/split/metadata/mask/softcap/scale/编译图；阈值未调整 |
| 构建与加载 | PASS：实际重编译 `mla_metadata.o` 并链接 17 个对象；其余 15 个 ASC 和 1 个普通 FA AICPU 对象哈希不变。加载扩展与构建产物一致；依赖、61 个源码和产物副本见 `build-provenance.json`、`production-artifacts/` |
| 性能诊断 | PASS：长度生成与无长度生成同轮交替测量；两种 dtype 各五组 metadata 复用与固定源/原 CANN 同轮对比。仅为内部组件诊断，见下述口径 |
| 提交质量 | 六项适用仓库质量检查、文档链接、差异和证据完整性检查；见 `quality-staged.log`、`acceptance.json`、`checksums.sha256` |

本步未修改调度算法、计算 Kernel/Block/Tile、生产 launcher 或 Python/C++ 公共入口。新增长度只限制为当前 dense 已支持容量，不扩展实际长度语义；样本内空/短 KV 等仍在 M06 验收。成功 metadata 保持旧 v1 布局，既有消费者对任意非零生成状态仍先拒绝。

## 性能口径与未完成项

- 内部 metadata 生成：双 dtype、五种 shape（含 B=41）各 5 轮×20 次。预分配输出和设备长度，计 ctypes 发射、ready/done 事件及完成等待；含长度路径中位数 66.48–100.20 µs，与同轮无长度路径比值 0.97–1.03。不含输入 H2D、分配或长度 D2H；原始样本在 `metadata-performance/`。
- 内部 metadata 消费：双 dtype 各五种代表输入，预生成 metadata 并复用 O/LSE/workspace；原 CANN 包含自身分派和分配。metadata 消费 Kernel / 原 CANN Kernel 中位数比值：FP16 0.98–1.02；BF16 0.97–1.01；每种 dtype 有 2,000 个 wall 样本和 1,000 个 profiler 样本，并核对原始 CANN CSV。固定 CANN 303 份构建文件已复核；见 `performance-fp16/`、`performance-bf16/`、`native-baseline-verified.json`。
- **公开 MLA metadata 链路和 M05 整体尚未完成，最终 E2E 门槛未验收。** 本次内部测量不代表公开 FA3 metadata E2E；M05c 仍须按 [design](design.md) 逐例验收 `FA3 metadata E2E / 同轮 CANN E2E <= 1.0`，分别记录生成、复用和完整调用段。

## 环境、失败与限制

- Python `/usr/local/python3.11.15/bin/python3` 3.11.15；torch `2.7.1+cpu`，torch_npu `2.7.1`；CANN `/usr/local/Ascend/cann-9.0.0-beta.2`，bisheng clang 15，`dav-2201`、`-O2`。实际扩展为本仓 `flash_attn_npu_3/flash_attn_npu_3.cpython-311-aarch64-linux-gnu.so`，SHA256 `34f7b616ea77c7c22107c7ea0172242dbf3dcc76d555deed95b7be7b2ba2b3d6`。
- 固定 CANN `ca18aee610729ca5270a44e2d84c751bc687fbf0`、CATLASS `769cd40a8716b28650b6bebb08db4834eea4462f`；CANN 源仓既有删除/未跟踪文件保持原样。官方 FA3 固定版本仍为 `d15f1531a460ba456f41b01a774f33ab2db8febf`。未改子模块、CANN 源仓或 950 后端。
- 7 号 910B3 开始/结束健康 OK、ECC 全零，结束无残留设备进程；见 `environment.json`、`environment-after.json`。NPU 与 sanitizer 在沙箱外运行，使用非登录 shell；最初登录 shell 和沙箱内 npu-smi 无输出，改用既有非登录/沙箱外流程后正常。
- 首次 CPU 运行被沙箱 ptrace 导致的 LeakSanitizer fatal error 阻断，见 `cpu-final*` 和 `consumer-final/build.log`。随后扩大到整个未修改调度矩阵的 sanitizer 运行在 600 秒超时（退出 124，无断言失败），见 `cpu-validated*`；已完成的 52 组边界共 47,520 份源对照保留在 `sanitized-boundaries.json`，该整体运行未记为通过。最终 `cpu-complete*` 对本步入口/消费者保留 ASan/UBSan，并完整通过全部调度源对照。
- 首次 ruff/clang-format 调整格式后复查通过；相关日志保留。实机、构建及普通 FA3 最终检查无失败；CANN beta 和重复 tiling 注册警告原样保留。设备归属证据首次误将容器 PID 与 npu-smi 宿主 PID 直接比较，已按本轮 device-7 日志核对映射及前后快照；见 `device-proof-initial.json`、`performance-device-proof.json` 和 `device-logs/`。未声称已完成公开 allocator/stream 生命周期、图捕获或运行时设备错误上报。

## 证据与复现

所有阶段的 `*-command.json`、`*-exit.json`、日志和 JUnit 保存精确命令、环境和结果。测试桥接库的 `build-manifest.json` 保存源码/依赖/实际加载库哈希；`build-source-manifest.json`、`build-provenance.json`、`production-snapshots.json` 对应生产构建。

从仓库根目录复现，选择空闲健康设备；按上述 command 文件设置新的 artifact/JUnit 路径，避免覆盖本次证据：

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 MLA_CONSUMER_SANITIZE=1 python3 -m pytest -x -q tests/test_flash_attn_npu_v3_mla_metadata_consumer.py tests/test_flash_attn_npu_v3_mla_build.py tests/test_flash_attn_npu_v3_mla_schedule.py
ASCEND_RT_VISIBLE_DEVICES=7 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 -m pytest -x -q tests/test_flash_attn_npu_v3_mla_metadata.py tests/test_flash_attn_npu_v3_mla_metadata_dense.py
FLASH_ATTN_BUILD_VERSION=v3 FLASH_ATTN_BUILD_NPU=910 FLASH_ATTN_SKIP_SUBMODULE_INIT=1 python3 setup.py build_ext --inplace
```

公共入口/普通 FA3 节点及顺序见 `run_validation.py` 与 `signature-final-command.json`、`accuracy-final-command.json`、`regression-final-command.json`。性能的精确工具和环境见 `run_performance.py` 及对应 command 文件；两类计时口径不可混用。

本步提交前检查：

```bash
PYTHONPATH=/tmp/mla-m01-precommit PRE_COMMIT_HOME=/tmp/mla-m01-precommit-cache bash ci/code_quality.sh --staged
python3 build/mla-m05b2b1-lengths/verify_acceptance.py --staged
git diff --check
git diff --cached --check
```

当前闭环提交用 `git log -1 -- docs/mla/status.md` 定位。
