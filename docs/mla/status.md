# MLA 当前状态

## 当前能力与下一步

- 分支 `feat/mla-cann-migration`，接手 HEAD `e99b4a0`，接手工作区干净。本步完成 M05b2b2 的内部设备校验错误上报。
- `kernel/mla_metadata.hpp` 新增 `LoadDenseMetadata`，复用现有解码与全部校验。失败记录或损坏 metadata 在消费时执行 `AscendC::Trap()`，由正常调用/同步点报告异常；所有 Cube/Vector 在进入 attention Kernel、分配事件、初始化 O/LSE/workspace 前完成校验。成功路径不需要状态 tensor、长度 D2H 或 steady-state 主机同步。
- 成功记录仍为 v1、401 个 int32 字；长度仍只接受 B 个值都等于 dense KV 容量，计算 Kernel/Block/Tile 与调度算法未改。status-only 诊断驱动保留，用于验证拒绝不改写任何 O/LSE/workspace 字节。
- 公开能力未扩展：FP16/BF16、512+64 共享 KV、连续 BSND、B>0、`1<=Sq*Hq<=256`、KV `1..512`、Hkv=1、无 mask、显式 `num_splits=1`。公开 MLA metadata 在真实/fake/C++ 中继续明确拒绝；生产 launcher 尚未接入新的错误适配层。
- 唯一下一步：M05b2b3，接通已验收 dense 范围的公开 metadata 生成与消费，使用 `LoadDenseMetadata`，补齐真实/fake/C++、运行时 scale、allocator/stream 生命周期、普通 FA3 回归及同轮诊断。之后 M05c 验收最终 E2E；整个 M05 完成前不开始 M04c1。本步提交后停止。

## 最近一次闭环证据

证据目录：`build/mla-m05b2b2-errors/`；物理 7 号 Ascend910B3。最终必要检查共 **937 项 pytest 通过，无失败或 skip**。36 个预期设备异常均由测试明确验收；早期失败不计为通过。

| 项目 | 结果 |
| --- | --- |
| 上步复核 | PASS：1,678 份证据哈希、1,058 项结果、61 个源码、17 个生产对象、实际扩展与固定源仓状态一致，见 `previous-verified.json` |
| CPU | PASS：90 项解码/损坏记录/构建依赖回归；metadata 入口与消费者 ASan/UBSan，见 `cpu-final*` |
| 设备精度与拒绝 | PASS：98 项 FP16/BF16 检查，含 D01/D02、边界、scale、复用和双 stream；与固定源码逐元素一致，独立双 reference 阈值未调整，见 `device-complete*` |
| 真实错误上报 | PASS：两种 dtype 各 18 个独立进程，覆盖全部 10 种解码失败状态及非法长度；`-O2 -DNDEBUG` 仍触发异常，device/stream/event 同步均覆盖，见 `device-complete/*/error-*/` |
| 公共入口回归 | PASS：129 项普通 metadata 签名/兼容/精度/compile，20 项普通 FA3 BSND/TND/分页/split/metadata/mask/softcap/scale/编译图，以及 600 项 MLA 公开契约，见 `signature-final*`、`regression-final*`、`contract-final*` |
| 构建与加载 | PASS：生产 15 个 ASC 与 2 个 AICPU 对象和扩展均复用且哈希与上步一致；新适配头尚非生产 launcher 的依赖。独立 checked/diagnostic/source 驱动重建并记录设备依赖、编译参数、源与实际库哈希，见各 `build-manifest.json`、`build-provenance.json` |
| 性能诊断 | 两种 dtype 各五组，checked 消费、旧诊断消费、静态迁移、固定源码和原 CANN 同轮交替测量；见下述口径 |
| 提交质量 | 六项适用仓库检查、文档链接、差异、证据完整性检查，见 `quality-staged.log`、`acceptance.json`、`checksums.sha256` |

36 个错误在 26.12–58.56 ms 内被同步点捕获，均有明确 trap 诊断，未依赖 15 秒算子超时。每次先用诊断路径核对准确状态，再重新生成/注入错误，直接消费到同步；这段链路禁止 Python 设备标量回读和主机等待。低层发射仍仅使用既有 stream 事件及 ACL launch，无新增主机同步。成功路径另做八次连续生成/消费检查。

## 性能口径与未完成项

- 每种 dtype 各五组代表用例，预生成 metadata，复用 O/LSE/workspace；每组五轮 × 20 个 wall 样本和五轮 × 10 个 profiler 样本。每种 dtype 保存 2,500 个 wall 样本和 1,250 个 Kernel 样本，并逐条核对原始 CANN CSV。
- checked metadata 消费 / 同轮原 CANN Kernel 中位数比值：FP16 0.96–1.01；BF16 0.97–1.02。相对旧诊断消费：FP16 0.98–1.00；BF16 0.97–1.00。这是内部组件诊断，原 CANN 调用还包括自身分派与分配，不能混用 wall 与 Kernel 口径。
- 固定 CANN 303 份构建文件重新核对，见 `native-baseline-verified.json`；`performance-device-proof.json` 用运行日志映射容器/宿主 PID，确认前后设备快照及原始 Kernel CSV 对应物理 7 号设备。
- **公开 MLA metadata 链路和 M05 整体尚未完成，最终 E2E 门槛未验收。** 本步不重测未修改的 metadata 生成性能；M05c 仍须按 [design](design.md) 逐例验收 `FA3 metadata E2E / 同轮 CANN E2E <= 1.0`，分别记录生成、复用和完整调用段。

## 环境、失败与限制

- Python `/usr/local/python3.11.15/bin/python3` 3.11.15；torch `2.7.1+cpu`，torch_npu `2.7.1`；CANN `/usr/local/Ascend/cann-9.0.0-beta.2`，bisheng clang 15，`dav-2201`、`-O2`；错误驱动额外使用 `-DNDEBUG`。
- 本仓实际扩展 `flash_attn_npu_3/flash_attn_npu_3.cpython-311-aarch64-linux-gnu.so`，SHA256 `34f7b616ea77c7c22107c7ea0172242dbf3dcc76d555deed95b7be7b2ba2b3d6`。CATLASS `769cd40a8716b28650b6bebb08db4834eea4462f`、固定 CANN `ca18aee610729ca5270a44e2d84c751bc687fbf0`、官方 FA3 `d15f1531a460ba456f41b01a774f33ab2db8febf`。未改子模块、CANN 源仓既有改动或 950 后端。
- 7 号设备开始/结束健康 OK、ECC 全零，结束无残留设备进程，见 `environment.json`、`environment-after.json`。最终 36 项错误及四项 smoke 的独立进程组无存活进程，见 `process-groups.json`。实机与 sanitizer 沿用非登录 shell、沙箱外运行；接手时登录 shell 查询有延迟。
- Trap 会使当前执行上下文进入错误状态，不承诺出错后继续复用 stream。测试为每个错误创建独立进程组，记录真正的 RuntimeError 后直接退出，父进程清理该组的 CANN 工作进程；tensor/stream 保持存活至退出，避免在错误上下文上提交 allocator 析构事件。此生命周期只用于隔离错误测试，公开 allocator/stream 与图捕获仍待后续步骤。
- 首次 `device-initial` 在 31 项通过后因错误码断言失败：混合核实际返回 AICore 的 `507015`，测试误写成 Vector 的 `507035`。已按安装的 ACL 定义修正，且要求 trap 日志与异常时限。随后 `device-final` 在 35 项通过后因正常解释器退出触发 allocator 事件 SIGABRT；`cleanup-offline` 仍发现编译子进程，`cleanup-pool` 的知识库 finalize 超时 90 秒。这些运行均未计为通过。
- 最终用父进程管理独立进程组解决清理问题，四项 `runtime-smoke-fixed` 与完整 `device-complete` 均通过。初次 smoke 的命令覆盖 CANN PYTHONPATH，已保留原路径重跑；初次 ruff 改写格式后复查通过。原始日志保留，包括 beta/重复 tiling 注册及预期异常告警。测试生成的 1.75 GB core 已删除并记录 `generated-core-cleanup.json`，后续错误子进程禁用 core dump。

## 证据与复现

各阶段 `*-command.json`、`*-exit.json`、日志及 JUnit 保存精确环境和结果；`runtime-source-manifest.json` / `source-snapshots/` 保存本步实际验收源码；独立构建的 `build-manifest.json` 保存源码、设备依赖和实际加载库。生产扩展副本见 `production-artifacts/`。

从仓库根目录复现，选择空闲健康设备，为 artifact/JUnit 设置新路径，避免覆盖证据：

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 MLA_CONSUMER_SANITIZE=1 python3 -m pytest -x -q tests/test_flash_attn_npu_v3_mla_metadata_consumer.py tests/test_flash_attn_npu_v3_mla_build.py
ASCEND_RT_VISIBLE_DEVICES=7 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 -m pytest -x -q tests/test_flash_attn_npu_v3_mla_metadata_dense.py
FLASH_ATTN_BUILD_VERSION=v3 FLASH_ATTN_BUILD_NPU=910 FLASH_ATTN_SKIP_SUBMODULE_INIT=1 python3 setup.py build_ext --inplace
```

普通 FA3/MLA 契约的精确节点见 `run_regression.py` 与对应 command 文件；性能工具及环境见 `run_performance.py`。本步只复核上步完整 AICPU/调度矩阵证据，未重跑未受修改影响的全调度源对照。

提交前检查：

```bash
PYTHONPATH=/tmp/mla-m01-precommit PRE_COMMIT_HOME=/tmp/mla-m01-precommit-cache bash ci/code_quality.sh --staged
python3 build/mla-m05b2b2-errors/verify_acceptance.py --staged
git diff --check
git diff --cached --check
```

当前闭环提交用 `git log -1 -- docs/mla/status.md` 定位。
