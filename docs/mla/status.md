# MLA 当前状态

## 当前能力与下一步

- 分支：`feat/mla-cann-migration`；接手 HEAD 为 `b358ce6`，接手工作区干净。本次完成 M05a：内部 AICPU 调度组件、版本化 int32 metadata 及构建接入。
- 公开 dense MLA 仍是 M04b3 已验收范围：FP16/BF16、512+64 共享 KV、连续 BSND、B>0、`1<=Sq*Hq<=256`、KV `1..512`、Hkv=1、无 mask、显式 `num_splits=1`。O 与输入 dtype 一致，可选 LSE 为 FP32。
- AICPU 直接编译现有 `BuildDenseSchedule` 和分核源码；版本 1 metadata 为 401 个 int32 字，保存完整调度与版本/类型/状态。64 位字段按低/高字保存；生成失败清空调度载荷。metadata 不含 scale、张量地址或可变 scratch。
- 本步仅通过独立测试桥执行 AICPU 组件。公开 MLA `get_scheduler_metadata`、前向的 `scheduler_metadata` 以及原范围外组合仍明确拒绝；未接入设备消费者，未宣称完成 metadata E2E。普通 FA3 byte metadata 沿用原协议。
- 唯一下一步：M05b，接通已验收 dense 范围的公开 metadata 调用链，验收官方签名、fake/真实/C++ 边界、运行时 scale、版本/条件检查及复用；最终性能门槛在 M05c 验收。详见 [plan](plan.md)。本步提交后停止。

## 最近一次闭环证据

证据目录：`build/mla-m05a-metadata/`；物理 7 号 Ascend910B3。共 816 项 pytest 检查通过，无失败或 skip。

| 项目 | 结果 |
| --- | --- |
| 上步复核 | PASS：M04b3 源码、扩展、链接对象及 2,717 份证据哈希匹配，环境一致；固定源仓状态保持原样 |
| AICPU 组件 | PASS：22 项检查，3,529 组设备记录与 host 全字一致，其中 3,494 组逐字段对照固定源；另有 24 次跨 stream 成功/失败覆写。覆盖两种 dtype、实际平台边界、合成资源、M=128 调整、等 cost 选择、FD 字段、宽 workspace、奇数 storage offset 及 guard |
| 失败路径 | PASS：形状、平台、资源、溢出及 dtype 拒绝；设备记录使用错误头部和零载荷。CPU ASan/UBSan 另核查空指针、对齐、输出大小、guard 与成功/失败交替覆写 |
| 生产构建 | PASS：强制构建 15 个 ASC 和 2 个 AICPU 单元，链接 17 个对象。原 15 个 ASC 对象与上步逐字节一致；普通 AICPU 由原源码重编译。加载库、对象、源码和依赖哈希可追溯 |
| 增量构建 | PASS：AICPU 记录真实头文件/包含的 `.cpp` 依赖和编译选项；实际编译检查证明传递头文件变更、缺失依赖/选项记录会触发重编译。无改动整包构建复用 15+2 个对象并跳过链接，动态库哈希未变 |
| CPU 回归 | PASS：179 项；含固定源调度 49,251 组对照、reference、来源、profile、生成器/依赖及 ASan/UBSan |
| MLA 契约与精度 | PASS：604 项，含 582 项契约和 22 项公开精度/生命周期。FP16/BF16 的 D01/D02、默认/零/正/负 scale、舍入、多 stream 和输入释放后重分配通过；每 dtype 8 组、24 次公开重复调用与固定源逐元素一致并通过双 reference |
| 普通 FA3 | PASS：11 项，覆盖 BSND、TND、分页、split、metadata 及编译图前反向 |
| 性能诊断 | 两种 dtype 各五组同轮固定 CANN 对照，合计 2,000 次同步公开 E2E 和 1,000 条 kernel task；原始 CSV 名称、设备、时间与时长逐条核实。另测 8 组 metadata 生成、1,600 个 host/AICPU 耗时样本 |
| 提交质量 | 六项适用仓库质量检查、文档链接、差异及证据完整性核对通过；见 `quality-validated.log`、`acceptance.json` 和 `checksums.sha256` |

公开精度最大绝对误差：FP16 O/LSE 为 `0.001953125` / `5.722046e-6`，BF16 为 `0.015625` / `3.814698e-6`。阈值未调整。

单位 μs，中位数；shape 为 `B,Sq,Hq,Sk`，FA3 仍为**无 metadata**公开入口：

| dtype | shape / scale | FA3 kernel | CANN kernel | FA3 公开 E2E | CANN E2E | E2E 比值 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| FP16 | 2,1,8,257 / 默认 | 12.720 | 14.110 | 465.440 | 277.400 | 1.6779 |
| FP16 | 2,1,8,257 / 0.23 | 12.811 | 14.000 | 456.520 | 277.675 | 1.6441 |
| FP16 | 1,7,16,65 / 默认 | 16.851 | 17.870 | 457.735 | 280.615 | 1.6312 |
| FP16 | 1,1,128,512 / 默认 | 29.681 | 30.251 | 458.140 | 278.550 | 1.6447 |
| FP16 | 2,2,128,511 / 默认 | 51.671 | 50.861 | 466.520 | 297.950 | 1.5658 |
| BF16 | 2,1,8,257 / 默认 | 12.520 | 14.011 | 541.215 | 421.610 | 1.2837 |
| BF16 | 2,1,8,257 / 0.23 | 12.660 | 13.980 | 536.735 | 419.065 | 1.2808 |
| BF16 | 1,7,16,65 / 默认 | 16.920 | 17.880 | 535.725 | 414.585 | 1.2922 |
| BF16 | 1,1,128,512 / 默认 | 30.001 | 30.351 | 532.215 | 413.735 | 1.2864 |
| BF16 | 2,2,128,511 / 默认 | 52.131 | 51.231 | 529.995 | 431.500 | 1.2283 |

每 dtype 单独采样；每例每实现预热 5 次，5 轮交替顺序，每轮 E2E 20 次、profiler 10 次，每次等待设备完成。两端使用相同预备输入，排除输入 H2D，保留输出分配及各自公开调用成本。host E2E 较上轮有波动；原计算对象完全相同，不能把跨轮差异记为本步优化收益。303 份固定原生基线产物哈希未变。

内部 metadata 生成诊断使用预分配输出及参数，同样预热 5 次、5 轮交替、每轮 20 次：host 共用函数为 8.330–8.915 μs，AICPU 发射、ready/done 事件及等待完成为 77.080–104.435 μs。这里排除分配与 H2D，没有公开包装器或 attention 计算，**不是 metadata 路径 E2E**；逐例值与原始样本见 `metadata-performance/`。

**最终性能门槛尚未达到。** M05b 接通后仍须在 M05c 按 [design](design.md) 逐例验收 `FA3 metadata E2E / 同轮原 CANN E2E <= 1.0`，区分生成、复用和完整调用段；M04c1 在整个 M05 完成前不开始。

## 环境、失败与限制

- Python `/usr/local/python3.11.15/bin/python3` 3.11.15；torch `2.7.1+cpu`，torch_npu `2.7.1`；CANN `/usr/local/Ascend/cann-9.0.0-beta.2`，bisheng clang 15，`dav-2201`、`-O2`。
- 7 号 910B3 开始和结束健康为 OK、ECC 为 0；性能快照只见本轮进程，结束无残留进程。见 `environment.json`、`performance-*/npu-*.log`、`metadata-performance/npu-*.log`、`npu-final.json`。
- 当前实际扩展为本仓 `flash_attn_npu_3/flash_attn_npu_3.cpython-311-aarch64-linux-gnu.so`，SHA256 `8c883e1764b7db351276ce2ae0a2dfa762c726db9d9c51029d7a6aff69196a47`。本步已替换 M04b3 产物；`environment.json` 保存接手时旧库哈希，当前加载记录见 `build-provenance.json`。
- 固定 CANN `ca18aee610729ca5270a44e2d84c751bc687fbf0`、CATLASS `769cd40a8716b28650b6bebb08db4834eea4462f`；CANN 源仓既有删除/未跟踪文件和子模块保持原样。native vendor 仍为 `build/mla-m02b-validation/opp-dense/vendors/mla_ca18aee_transformer`。
- 设备上的合成核数/C1V1 输入只验收 CPU 调度计算，不代表 910C/C1V1 attention Kernel 验收。图内 MLA metadata、动态长度、公开 metadata 生命周期仍未接通；v2/v4/950 仅检查构建源选择，未做整包构建或实机验收。
- 首次质量检查按钩子自动排版，随后检查通过；证据检查首次误写暂存文件数为 14（实际 13），并在 npu-smi 解析中遗漏列分隔符；已按实际清单和表格列修正，原失败记录见 `acceptance-initial.json`，设备占用复核见 `performance-device-proof.json`。未出现需要调整精度阈值的失败。登录 shell 初始化仍不返回输出，使用非登录 shell；NPU 和 sanitizer 使用已验证的沙箱外方式。CANN beta 字符串、重复 tiling 注册和普通 Kernel 自动类型推断警告保留于日志。

## 证据与复现

- `previous-verified.json`、`environment*`、`native-baseline-verified.json`：接手状态和固定源证据。
- `build-final*`、`build-source-manifest.json`、`build-provenance.json`、`build-incremental*`、`incremental-proof.json`：编译、链接、加载与缓存核查。
- `device-final.log/xml`、`device-final/`、`oracle-final/`、`cpu-final.log/xml`、`schedule-final/`：设备字段对照、平台、固定源导出、编译命令和 CPU 检查。
- `accuracy-final.log/xml`、`accuracy-final/`、`regression-final.log/xml`：MLA 和普通 FA3 验收；精确节点、环境及命令见对应 `*-command.json`。
- `performance-fp16/`、`performance-bf16/`、`metadata-performance/`：软件栈、原始墙钟样本及 profiler 数据。`run_performance.py` 和 `metadata_profile.py` 保存完整采样过程。

从仓库根目录复现；选择空闲健康设备，使用新的 artifact 路径保留既有证据：

```bash
FLASH_ATTN_BUILD_VERSION=v3 FLASH_ATTN_BUILD_NPU=910 FLASH_ATTN_SKIP_SUBMODULE_INIT=1 FLASH_ATTN_FORCE_REBUILD=TRUE python3 setup.py build_ext --inplace
ASCEND_RT_VISIBLE_DEVICES=7 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MLA_METADATA_ARTIFACT_DIR=build/mla-m05a-metadata/recheck-device MLA_SCHEDULE_ARTIFACT_DIR=build/mla-m05a-metadata/recheck-source python3 -m pytest -x -q tests/test_flash_attn_npu_v3_mla_metadata.py
TORCH_DEVICE_BACKEND_AUTOLOAD=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MLA_SCHEDULE_ARTIFACT_DIR=build/mla-m05a-metadata/recheck-schedule python3 -m pytest -x -q tests/test_flash_attn_npu_v3_mla_build.py tests/test_flash_attn_npu_v3_mla_reference.py tests/test_flash_attn_npu_v3_mla_provenance.py tests/test_flash_attn_npu_v3_mla_profile.py tests/test_flash_attn_npu_v3_mla_schedule.py
```

公开 MLA 和普通 FA3 按 `accuracy-final-command.json` / `regression-final-command.json` 的节点复验，替换 artifact/JUnit 路径。CANN 性能使用原生 vendor 环境执行 `python3 -m tests.mla_forward_profile`，分别设置 `MLA_FORWARD_DTYPE=fp16` / `bf16` 和新的 `MLA_FORWARD_ARTIFACT_DIR`；完整环境见对应 `performance-*-command.json`。

本步提交前检查：

```bash
PYTHONPATH=/tmp/mla-m01-precommit PRE_COMMIT_HOME=/tmp/mla-m01-precommit-cache bash ci/code_quality.sh --staged
python3 build/mla-m05a-metadata/verify_acceptance.py --staged
git diff --check
git diff --cached --check
```

当前闭环提交用 `git log -1 -- docs/mla/status.md` 定位。
