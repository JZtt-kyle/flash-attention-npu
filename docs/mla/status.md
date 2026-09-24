# MLA 当前交接

## 状态与下一步

- 当前步骤：M02b2 已完成，固定源码的 BSND 性能基线采集、精度和分派核验闭环；主机计时的噪声范围单列保留。
- 当前分支：`feat/mla-cann-migration`；接手基点 M02b1 `247423f`。用户已确认现有方案，后续已确认范围不重复审批。
- 本步产物：[扩展执行器](../../tests/mla_cann_baseline.py)、[profiler 采集与解析](../../tests/mla_cann_profile.py)、[CPU 完整性检查](../../tests/test_flash_attn_npu_v3_mla_profile.py) 及持久原始数据。
- 唯一下一步：M03，识别 MLA 入口与形状/stride，明确拒绝无效及尚未接通组合，处理生产入口 `qv` 静默忽略，补 fake/真实契约及普通 FA3 回归。本次一个提交后停止。

**生产 MLA kernel/API 尚未接通。** 本次仍只执行固定 CANN 源码包；不构成迁移实现的性能通过结论。TND/PA、mask、显式 split、metadata 和图不在本步基线内。

## 最近闭环的证据

| 检查 | 结果 |
| --- | --- |
| 接手复核 | PASS：M02b1 `checksums.sha256` 全部匹配，配套库/二进制/header 哈希一致，原 CANN 源仓既有状态和 CATLASS 未变，实际 FA3 扩展路径及哈希一致 |
| 代表输入 | 原 D01 默认/0.23/0 scale、D02、P01 split/combined；新增 P02 的 8/8、8/1 头数，D/P 的 Sq=1/7、Sk=8193，以及 P 的 Sq=257、Sk=1025、8/2；均为 BSND、无 mask、CANN 原生自动 split；共 18 个输入 × 两种 dtype |
| 910B 精度/重复执行 | PASS：`36 passed in 82.74s`，无 skip；O/LSE 双 reference、重复执行和全部输入不改写通过。四次性能采集也各自完成全部 36 组 O/LSE 检查 |
| 精度口径 | comparator/reference/阈值未改；最大 O 误差 FP16 `0.001953125`、BF16 `0.0078125`；最大 LSE 误差分别 `5.7220458984375e-6`、`1.9073486328125e-6` |
| 实际分派 | PASS：36 组共 144 次 FIA launch 均匹配 M02b1 vendor 的两个 `.o` 和已编译 keys；D 为模板 5 `HIGH_PERFORMANCE_MLA`，P 为模板 3 `HIGH_PERFORMANCE_GQA`；长 KV 使用对应自动 split key，无需补编译，无系统旧 FIA 回落 |
| profiler 完整性 | PASS：每次 36 × 30 = 1080 个 FIA kernel、1440 个 Slice；四次均逐调用核对完整序列。每个 split prefill 调用有四个 Slice，combined 和压缩表示无 Slice |
| 调用计时 | 每次每例预热 10 次，5 轮 × 50 次 event span，另 5 轮 × 50 次逐调用同步 wall time；profiler 关闭时采集，保留每轮/每次原始值 |
| CPU 检查 | PASS：`17 passed in 7.10s`；含原 6 项包校验及 11 项 profiler 检查，拒绝缺失/额外 FIA、缺失/尾随拷贝、未知 kernel、非有限/零耗时、多设备、缺失列和空数据；不执行 NPU kernel |
| 普通 FA3 | 生产入口、metadata、kernel 和扩展均未改变，沿用 M02b1 同 vendor 环境的 8 项普通 FA3 回归及强制构建溯源；本步未重复运行 |
| 仓库质量 | PASS：独立索引仅含本步五个文件；`ci/code_quality.sh --staged` 的空白、末尾换行、冲突标记、Ruff check/format 通过，其余语言无适用文件；文档链接/围栏和 `git diff --check` 通过 |

## 性能口径与观测

四次独立采集为 `performance-a/b/c/d`。a/b 继承运行环境的 CPU 亲和性，c/d 使用 `taskset -c 116,117,118,120`，执行器记录了实际四个 CPU。每次 profiler 为 Level0、CPU+NPU；在一段活动区内串行执行全部准备好的调用，每例分三组、每组 10 次，每调用后同步。输入 H2D、reference 和一次性视图创建均在采集外。

profiler 的 `Duration(us)` 分别统计 FIA 和 Slice；kernel 耗时之和是设备工作量，不能当作端到端延迟。输出分配、框架/算子发射、包装器内转换和同步都计入未开启 profiler 的 wall time。当前 torch_npu 2.7.1 导出的 Step Id 有错位，解析器使用时间顺序和每例预期的 Slice→FIA 序列，并核对总数；缺失数据不会被当成零耗时。

未固定 CPU 的 a/b 中，逐例 kernel 中位数的两轮最大比值为 `1.0572`，wall 中位数最大比值为 `1.4297`。采样时设备本地 CPU 负载较高，改用较低负载的相邻 NUMA 节点四个 CPU 后，c/d 的相应比值为 `1.0504` 和 `2.6025`；其中 D01-zero BF16 的 wall 中位数为 `437.42 / 1138.37 us`，kernel 仍约 `14 us`。**固定亲和性未消除共享主机的调用计时噪声，不能宣布端到端稳定或性能不劣化。** 全部样本均保留，没有删除异常轮次或放宽验收阈值。

下表列出 c/d 两次采集的中位数范围（us，仅作本配置观测；完整 36 组和另外两轮见 JSON/CSV）：

| 输入 / dtype | FIA kernel | Slice 合计 | 同步调用 |
| --- | ---: | ---: | ---: |
| D01-default / fp16 | 13.95–14.22 | 0.00 | 424.94–552.94 |
| D01-default / bf16 | 14.06–14.29 | 0.00 | 408.02–485.29 |
| P01-split / fp16 | 16.50–16.65 | 7.04–7.14 | 309.45–365.11 |
| P01-combined / fp16 | 16.18–16.61 | 0.00 | 271.40–287.47 |
| FD01-D-q1 / bf16 | 32.31–33.94 | 0.00 | 255.54–280.81 |
| FD01-P-q7-split / bf16 | 54.19–54.57 | 13.72–13.72 | 341.12–382.17 |
| FD01-P-q7-combined / bf16 | 60.09–60.84 | 0.00 | 295.14–304.02 |
| P-long-combined / bf16 | 66.68–68.37 | 0.00 | 278.77–305.02 |

后续性能验收以同一时段交替执行固定源码 CANN 和新实现为准，使用匹配的 CPU/NPU 环境、输入及计时方法；这些历史 wall 中位数不是跨会话的固定门槛。噪声不能分辨时补测，逐例 `新实现 / CANN <= 1.0` 的约定不变。prefill 同时保留 split/combined，两路的 kernel 与必要转换分别可见。M02a 安装包仅有调用诊断值且版本对应关系未建立，不能由其 event span 与本次差异推断 kernel 退化。

## 持久证据与复现

本步证据根目录（本仓专用忽略目录）：

`/home/zhangqijia/repo/flash-attention-npu/build/mla-m02b2-validation/`

- `takeover-audit.json`、`environment-before.log`、`concurrent-commit-audit.json`、`final-integrity.json`：环境、原仓状态、CATLASS、配套包和 FA3 扩展溯源。
- `dispatch.json` / `.log`、`dispatch-command.json`、`verify_dispatch.py`、`dispatch-proof.json`：36 组逐例实际二进制/key/template，共 144 次 launch。开启调试日志的计时不作性能基线。
- `performance-a/b/c/d.json`、对应 `*-command.json`、`*.log`、`*-exit.json` 和 `*-profile/`：四次独立采集、完整原始 profiler 数据与 CSV/trace；后面三次附 `*-device-monitor.json`。
- `all-captures.json`：四次输入、计时函数与文件哈希核对及所有结果；`unbound-summary.json/csv` 汇总 a/b，`performance-summary.json/csv` 汇总 c/d。新增亲和性记录只改变 `environment()`；a/b 当时的执行器原文保存在 `diagnostic-harness.py`。
- `cpu-affinity-selection.json`、`cpu-affinity-local-sample.json`、`cpu-load-global.json`：CPU 负载采样及亲和性选择。NPU 的前后健康和复测前占用状态保存于 `npu-smi-*.log`。
- `accuracy.log` / `accuracy-command.json`、`cpu-final.log`、`quality-final.log`、`document-check.log`、`checksums.sha256`：测试、质量与证据哈希。

配套源码包继续使用 M02b1 的 `build/mla-m02b-validation/opp-dense/`；原生构建命令、依赖和实际 device binary 来源见该目录的 `build-manifest.json`、`device-build.json`、`package-audit.json`。该目录及其源码导出未重建、未修改。原 JIT-only `opp/` 会混用系统旧 device binary，不能用于验收。

复现前重新检查设备健康/占用和 CPU 条件；本轮使用物理 7 号、进程内 `npu:0`，实机命令需在沙箱外执行，shell `login=false`。保留 CANN `set_env` 的 `PYTHONPATH`。输出 JSON 和 profiler 目录必须使用新路径：

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m pytest -q tests/test_flash_attn_npu_v3_mla_profile.py \
  tests/test_flash_attn_npu_v3_mla_provenance.py

mla_work="$PWD/build/mla-m02b2-validation"
mla_vendor="$PWD/build/mla-m02b-validation/opp-dense/vendors/mla_ca18aee_transformer"
export ASCEND_RT_VISIBLE_DEVICES=7 ASCEND_CUSTOM_OPP_PATH="$mla_vendor"
export LD_LIBRARY_PATH="$mla_vendor/op_api/lib:$LD_LIBRARY_PATH"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export ASCEND_GLOBAL_LOG_LEVEL=3 ASCEND_SLOG_PRINT_TO_STDOUT=0
export ASCEND_CACHE_PATH="$mla_work/reproduce-cache"
mkdir -p "$ASCEND_CACHE_PATH"
python -m pytest -q tests/test_flash_attn_npu_v3_mla_cann.py

taskset -c 116,117,118,120 python -m tests.mla_cann_baseline \
  --output "$mla_work/reproduce.json" \
  --profile-dir "$mla_work/reproduce-profile" \
  --cann-source /home/zhangqijia/repo/ops-transformer \
  --expected-vendor "$mla_vendor" --seed 20260924 \
  --warmup 10 --iterations 50 --rounds 5 \
  --profile-iterations 10 --profile-rounds 3
```

试采第一次因临时脚本覆盖 CANN `PYTHONPATH` 而无法导入 `tbe`，未执行 kernel；修复后试采及四次正式采集通过，失败和重试日志均保留。复测前遇到设备占用时未启动计时，等待释放后才执行；固定 CPU 是噪声对照，未修改系统设置或其他进程。

## 环境与来源

| 项目 | 本轮结果（2026-09-24） |
| --- | --- |
| Python / 包 | `/usr/local/python3.11.15/bin/python`，3.11.15；torch `2.7.1+cpu`，torch_npu `2.7.1` |
| CANN / 源码 | `/usr/local/Ascend/cann-9.0.0-beta.2`；固定源码 `ca18aee610729ca5270a44e2d84c751bc687fbf0`，仅加载独立 custom vendor |
| CATLASS | `769cd40a8716b28650b6bebb08db4834eea4462f`，子模块干净，未修改 |
| 配套 API / tiling SHA256 | `5681ca7bb73a28b781deb0a55399a601416afdc91e2a655bc45d9c7e2c1a4599` / `acb5048b6633d5c7db4ab78a6e54ff520aadf8baf0228f4343315864cd2f88ef` |
| device binary SHA256 | FP16 `a941647ab85ed1c3b0d315c6bc17020fb9a96939a94a33b61e3296d27a9b8c4c`；BF16 `e26ee6feaebf64e364fb2dc2aecc985d54a081440edc621d4ffcaa80676eb798`；各 12 个 dense keys |
| 实际 FA3 扩展 | 本仓 `flash_attn_npu_3/flash_attn_npu_3.cpython-311-aarch64-linux-gnu.so`，SHA256 `fabdbb711f336e2a062e0b032fe0bdb65758da091b61c6537a0b8294a6c2677c`，继承 M02a 强制构建产物 |
| 设备 / CPU | Ascend910B3 物理 7 号；告警 5/6 未使用。Kunpeng-920；c/d 实际亲和性为 `[116,117,118,120]`，不代表 CPU 独占 |

## 下一步约束

M03 只做已确认的生产入口识别与拒绝契约，不把本测试 CANN 执行器接成生产回退。后续引入新 kernel 时，继续同时验收 O/LSE、代表性能和受影响的普通 FA3 路径。当前闭环提交可由 `git log -1 -- docs/mla/status.md` 定位。
