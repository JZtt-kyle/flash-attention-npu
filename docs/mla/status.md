# MLA 当前状态

## 当前能力与下一步

- 分支：`feat/mla-cann-migration`；接手 HEAD 为 `b952fef`，接手工作区干净。本次将 M05b 拆为设备消费与公开接线，完成 M05b1。
- 新增共用 CPU/设备的 v1 metadata 解码校验和设备读取适配；独立链路已打通 AICPU → 401 字 int32 记录 → 现有 FP16/BF16 dense Kernel。运行时 scale、输入地址及每次调用的 scratch 仍独立传入。
- 校验覆盖大小、头部、生成状态、dtype、形状、平台、分核端点、64 位 workspace 和本地内存。所有 Cube/Vector 一致拒绝无效记录，不进入 attention 同步与计算，O/LSE/workspace 保持原值；截断错误头部不读载荷。记录使用期间须只读并遵守 stream 依赖。
- 本步消费者只链接进独立测试库。公开 dense MLA 仍是 M04b3 范围：FP16/BF16、512+64 共享 KV、连续 BSND、B>0、`1<=Sq*Hq<=256`、KV `1..512`、Hkv=1、无 mask、显式 `num_splits=1`。公开 MLA metadata 及范围外组合仍明确拒绝；普通 FA3 byte metadata 沿用原协议。
- 唯一下一步：M05b2，接通已验收 dense 范围的公开 metadata 调用链，完成官方签名、fake/真实/C++ 边界、生产发射、错误传播及生命周期验收。M05c 再验收最终 E2E 门槛；M04c1 在整个 M05 完成前不开始。本步提交后停止。

## 最近一次闭环证据

证据目录：`build/mla-m05b1-consumer/`；物理 7 号 Ascend910B3。最终共 **958 项 pytest 检查通过，无失败或 skip**。

| 项目 | 结果 |
| --- | --- |
| 上步复核 | PASS：M05a 的 943 份证据、已验收源码、扩展及对象哈希匹配，软件栈一致，固定源仓状态保持原样 |
| CPU 解码与回归 | PASS：263 项；其中消费者 84 项，解码 6,864 组双 dtype 调度记录；另含固定源调度 49,251 组对照、reference、来源、profile、构建和依赖检查 |
| ASan/UBSan | PASS：真实 8 字截断头的逐字段错误、非法字数、空指针、奇数 storage offset、401 字各四种置值及 guard；失败不改写调度输出。另检验完整 64 位 workspace、溢出、容量不足及平台条件 |
| 设备消费链路 | PASS：58 项；每 dtype 28 组、84 次调用与固定源逐元素一致并通过双 reference，含 D01/D02、默认/零/正/负 scale、M/KV 尾块、M=128、batch/核数边界及非零 offset |
| 设备失败与复用 | PASS：每 dtype 65 种记录损坏、5 种缓冲错误、9 种调用条件错误，逐核状态与完整 O/LSE/workspace 均核查；另含两 stream 各自 scratch 的 8 次不同输入/scale 复用，以及同记录 12 次成功/生成失败/形状错配覆写 |
| AICPU 回归 | PASS：22 项；3,529 组设备记录与 host 全字一致，其中 3,494 组对照固定源，另有 24 次跨 stream 成功/失败覆写 |
| 公开 MLA 回归 | PASS：604 项，含 582 项真实/fake/C++ 契约和 22 项公开精度/生命周期；未开放的 MLA metadata 继续拒绝 |
| 普通 FA3 | PASS：11 项，覆盖 BSND、TND、分页、split、metadata 及编译图前反向 |
| 构建与加载 | PASS：双 dtype 源探针、无 metadata 探针、消费者探针及 host 调度库共 8 个独立库重新编译；设备依赖包含新消费者与原 Kernel。AICPU 及 CPU 解码库有独立命令和哈希；生产扩展、17 对象、59 个相关源码文件保持 M05a 哈希 |
| 性能诊断 | PASS：每 dtype 五组同轮固定 CANN/源探针/原 Kernel/消费者对照；合计 4,000 次同步调用样本和 2,000 条 kernel task，逐条核对原始 CSV 名称、设备、开始时间和时长 |
| 提交质量 | 六项适用仓库质量检查、文档链接、差异及证据完整性检查通过；见 `quality-staged.log`、`acceptance.json` 和 `checksums.sha256` |

设备消费者最大绝对误差：FP16 O/LSE 为 `0.001953125` / `5.722046e-6`，BF16 为 `0.015625` / `3.814698e-6`。阈值未调整。

下表为 kernel 中位数，单位 μs；shape 为 `B,Sq,Hq,Sk`。消费者测量包含元数据读取、校验和测试驱动的逐核状态回写：

| dtype | shape / scale | metadata 消费者 | 无 metadata Kernel | 固定源探针 | 原 CANN |
| --- | --- | ---: | ---: | ---: | ---: |
| FP16 | 2,1,8,257 / 默认 | 24.401 | 22.721 | 23.810 | 24.831 |
| FP16 | 2,1,8,257 / 0.23 | 24.210 | 22.651 | 23.620 | 24.601 |
| FP16 | 1,7,16,65 / 默认 | 20.491 | 19.270 | 19.410 | 19.770 |
| FP16 | 1,1,128,512 / 默认 | 48.691 | 47.541 | 48.301 | 48.941 |
| FP16 | 2,2,128,511 / 默认 | 66.281 | 65.401 | 65.791 | 65.421 |
| BF16 | 2,1,8,257 / 默认 | 24.440 | 23.090 | 23.940 | 25.040 |
| BF16 | 2,1,8,257 / 0.23 | 24.410 | 22.970 | 23.901 | 24.900 |
| BF16 | 1,7,16,65 / 默认 | 20.420 | 19.210 | 19.511 | 19.790 |
| BF16 | 1,1,128,512 / 默认 | 49.021 | 47.911 | 48.771 | 48.681 |
| BF16 | 2,2,128,511 / 默认 | 66.191 | 65.512 | 66.921 | 65.591 |

每 dtype 单独采样；每例每实现预热 5 次，5 轮交替顺序，每轮同步调用 20 次、profiler 10 次。消费者每例生成一次 metadata，在三次精度、五次预热、一百次调用计时及五十次 profiler 调用中复用。源码/消费者探针使用预分配输入、输出、workspace 和 host 启动预算；原 CANN 包含其公开分派及输出分配。墙钟原始数据一并保存，但这些内部调用段**不是公开 FA3 metadata E2E**，不能据此宣称达到性能门槛。303 份固定原生基线产物哈希未变。

**M05 整体尚未完成，最终性能门槛未达成。** M05b2 接通后仍须在 M05c 按 [design](design.md) 逐例验收 `FA3 metadata E2E / 同轮原 CANN E2E <= 1.0`，区分生成、复用和完整调用段。本步仍由 host 调度准备独立驱动的发射维度与分配预算，没有宣称消除该成本。

## 环境、失败与限制

- Python `/usr/local/python3.11.15/bin/python3` 3.11.15；torch `2.7.1+cpu`，torch_npu `2.7.1`；CANN `/usr/local/Ascend/cann-9.0.0-beta.2`，bisheng clang 15，`dav-2201`、`-O2`。
- 7 号 910B3 开始和结束健康为 OK、ECC 为 0；性能前后快照只见本轮采样进程，结束无残留设备进程。见 `environment.json`、`environment-after.json`、`performance-*/npu-*.log`。
- 当前实际扩展仍为本仓 `flash_attn_npu_3/flash_attn_npu_3.cpython-311-aarch64-linux-gnu.so`，SHA256 `8c883e1764b7db351276ce2ae0a2dfa762c726db9d9c51029d7a6aff69196a47`。生产依赖未变，未重建整包；新消费者的实际编译/加载证据见 `build-provenance.json`、`device-final/*/build-manifest.json`。
- 固定 CANN `ca18aee610729ca5270a44e2d84c751bc687fbf0`、CATLASS `769cd40a8716b28650b6bebb08db4834eea4462f`；源仓既有删除/未跟踪文件和子模块保持原样。native vendor 仍为 `build/mla-m02b-validation/opp-dense/vendors/mla_ca18aee_transformer`。
- 消费者只验收 910B C1V2；CPU 的合成核数不代表其他芯片的设备验收。校验保证兼容性和地址边界，不认证记录来源；不参与计算的 maxCost、FD 记账和闲置核端点不做真实性校验。并发修改 metadata 不受支持。公开 API 的版本错误传播、allocator 生命周期、metadata E2E 和图内更新尚未接通。
- 首次设备检查的 O/LSE 已通过，但测试驱动将各核状态写在相邻字，发生缓存行回写覆盖；已改为独占 64 字节缓存行并显式回写，修复后 56 项、最终扩展覆盖后 58 项均通过。保留 `device-pilot.log/xml`、`device-status.log/xml`。首次质量检查发现 fixture 别名 lint 问题并自动排版，修复后通过。性能脚本首次命名为 `profile.py`，遮蔽 Python 标准库并在 torch 导入阶段退出；重命名后重新采样，原失败保留于 `performance-fp16-initial.log`。证据检查首次误将容器 PID 与 npu-smi 宿主 PID 直接比较，已按本轮设备日志核对映射；见 `acceptance-initial.log`、`performance-device-proof.json` 及 `device-logs/`。未调整计算代码或精度阈值。
- 登录 shell 和沙箱内 npu-smi 不返回输出，沿用非登录 shell，NPU/sanitizer 在沙箱外运行。CANN beta 版本字符串及重复 tiling 注册警告保留于日志。CPU 命令中新增的 `MLA_CONSUMER_*` 环境变量按实际工具调用补录，sanitizer 原始编译与执行命令另存；详见 `cpu-final-command.json` 的注记。

## 证据与复现

- `previous-verified.json`、`environment*`、`build-provenance.json`、`native-baseline-verified.json`：接手状态、未变的生产构建及固定源证据。
- `consumer-final/`、`schedule-final/`、`cpu-final.log/xml`：解码边界、字段矩阵、sanitizer 和 CPU 回归。
- `device-final/`、`producer-final/`、`device-final.log/xml`：消费者、固定源导出、实际编译命令、双 dtype 逐例误差和错误路径。
- `producer-regression*`、`oracle-regression/`、`accuracy-final*`、`regression-final*`：AICPU、公开 MLA 与普通 FA3 回归；精确节点、环境与命令见对应 `*-command.json`。
- `performance-fp16/`、`performance-bf16/`：逐例墙钟和 profiler 原始数据；`metadata_consumer_profile.py`、`run_performance.py` 保存完整采样过程与库哈希检查。提交前根目录无遗留的 `kernel_meta/` 临时产物，检查见 `artifact-cleanup.json`。

从仓库根目录复现，选择空闲健康的设备，替换 artifact 路径以保留既有证据：

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MLA_CONSUMER_SANITIZE=1 MLA_CONSUMER_ARTIFACT_DIR=build/mla-m05b1-consumer/recheck-consumer python3 -m pytest -x -q tests/test_flash_attn_npu_v3_mla_metadata_consumer.py
ASCEND_RT_VISIBLE_DEVICES=7 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MLA_DENSE_ARTIFACT_DIR=build/mla-m05b1-consumer/recheck-device MLA_METADATA_ARTIFACT_DIR=build/mla-m05b1-consumer/recheck-producer python3 -m pytest -x -q tests/test_flash_attn_npu_v3_mla_metadata_dense.py
```

其余验收按 `cpu-final-command.json`、`producer-regression-command.json`、`accuracy-final-command.json`、`regression-final-command.json` 和 `performance-*-command.json` 的命令与环境复验，替换 artifact/JUnit 路径。性能脚本核查并复用 `device-final`/`producer-final` 已验收库，源码改变时须先重建这些库。

本步提交前检查：

```bash
PYTHONPATH=/tmp/mla-m01-precommit PRE_COMMIT_HOME=/tmp/mla-m01-precommit-cache bash ci/code_quality.sh --staged
python3 build/mla-m05b1-consumer/verify_acceptance.py --staged
git diff --check
git diff --cached --check
```

当前闭环提交用 `git log -1 -- docs/mla/status.md` 定位。
