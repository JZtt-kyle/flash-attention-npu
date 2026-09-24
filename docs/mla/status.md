# MLA 当前交接

## 状态与下一步

- 当前步骤：M02b1 已完成；固定源码的 host/tiling/device 配套构建、加载溯源、910B 精度及回归通过。
- 当前分支：`feat/mla-cann-migration`；接手基点 M02a `8c08972`，接手时本仓仅有既有未跟踪 `atk_test/`，未改、未暂存。
- 用户已授权按现有方案继续。M02b 在实现前拆为 M02b1（构建与精度）和 M02b2（profiler 性能基线），不重复审批。
- 本步产物：[执行器 vendor 检查](../../tests/mla_cann_baseline.py)、[CPU 回归](../../tests/test_flash_attn_npu_v3_mla_provenance.py)、固定源隔离构建及持久证据；本次闭环提交由 `git log -1 -- docs/mla/status.md` 定位。
- 唯一下一步：M02b2，在下述配套源码包上扩展代表输入，用 profiler 保存 kernel 时间、调用端到端时间和转换成本，形成正式性能基线。本轮一个提交后停止。

**生产 MLA kernel/API 尚未接通。** 本步只验收测试用 CANN 源码包。编译包仅含 FP16/BF16 的选定 BSND dense keys；PA/TND 等未编译、未验收。调用计时和带调试日志的 trace 均不能作为正式 kernel 性能基线。

## 最近闭环的证据

| 检查 | 结果 |
| --- | --- |
| M02a 接手复核 | PASS：上轮 `checksums.sha256` 全部匹配；Python/torch/torch_npu/CANN/CATLASS 与实际 FA3 扩展哈希均一致 |
| 固定源码隔离 | PASS：`git archive ca18aee610729ca5270a44e2d84c751bc687fbf0` 导出至源仓外；构建后逐文件核对 13,585 个文件均未改变；原源仓完整 status 与接手时一致，CATLASS 干净 |
| host/tiling 构建 | PASS：原生 `build.sh --jit --noaicpu --soc=ascend910b --ops=fused_infer_attention_score --vendor_name=mla_ca18aee -j8 -O2`，指定隔离依赖目录；构建和自定义路径安装退出码均为 0 |
| device 构建 | PASS：使用原生生成的参数 JSON、wrapper 和 `opc`，FP16/BF16 各编译 12 个 dense keys；未改 kernel/tiling 源码。编译命令、keys、返回码见 `device-build.json` |
| 包与实际执行溯源 | PASS：296 个相关 kernel/header 与固定源码/依赖字节一致，安装的三份 custom 库与构建产物一致。完整 12 组 trace 中 48 次 FIA launch 均匹配配套 vendor 的两个 `.o` 及其已编译 keys，无 FIA device binary 回落到安装包 |
| 实际模板 | D01 默认/0.23/0 scale、D02 为模板 5 `HIGH_PERFORMANCE_MLA`；P01 split/combined 均为模板 3 `HIGH_PERFORMANCE_GQA`；FP16/BF16 均已记录实际 key。split prefill 的 Slice 等转换在 trace 中可见，成本留给 M02b2 量化 |
| 源码包精度 | PASS：`12 passed in 14.08s`，无 skip；O/LSE 双 reference、重复执行、全部内容/RoPE 输入不改写通过。最终执行器另跑全部 12 组并保存输入、原始调用计时、环境、库与二进制哈希 |
| 精度口径 | 未改 comparator/reference 或阈值。最大 O 误差 FP16 `0.001953125`、BF16 `0.0078125`；最大 LSE 误差 FP16 `5.7220458984375e-6`、BF16 `1.9073486328125e-6` |
| 包检查 CPU 回归 | PASS：`6 passed in 2.58s`；允许原生 set_env 的末尾空分隔符，拒绝 JIT-only 包、多 vendor、缺失/过期 device binary、指向包外的 kernel symlink，均不执行 NPU kernel |
| 普通 FA3 回归 | PASS：同一 custom OPP 环境下沿用上轮 8 个节点，`8 passed in 31.25s`，无 skip；dense、TND、paged cache、FP16/BF16 split-KV 和 metadata 通过；FA3 扩展未变，无需重新构建 |
| 仓库质量 | PASS：以独立索引对本步四个文件执行 `ci/code_quality.sh --staged`，空白、末尾换行、冲突标记、Ruff check/format 均通过，其余语言 hook 无适用文件；文档链接/围栏及 `git diff --check` 通过 |

本轮故障与修复：原生 **JIT-only** 包会在此 CANN 环境中将新 custom tiling 配到旧安装包 FIA `.o`，首例报 `507015` / MTE 越界。`probe-source-retry.log` 明确记录旧二进制路径；该结果不计为通过。随后从固定源码编译配套 device 二进制、生成原生 kernel 配置并组装 `opp-dense`，精度与实际路径核查通过。执行器新增 `--expected-vendor`，在执行前要求唯一配置、完整 FP16/BF16 二进制及编译 metadata 哈希一致；执行后检查 custom 库已加载、四个关键 header 与固定 commit 一致。最终 device dispatch 仍以原始日志与 `dispatch-proof.json` 为证，不将“host 库已加载”等同于“device 二进制已使用”。

初次构建的依赖下载受沙箱限制，以及首次探测遗漏创建 `ASCEND_CACHE_PATH` 的初始化失败，均保留原日志并已解决。未使用告警设备，失败进程退出后重新查询，7 号仍健康空闲。原 JIT-only `opp/` 仅保留故障证据；验收与下一步使用 **`opp-dense/`**。

## 持久证据与复现

证据根目录为本仓专用忽略目录：

`/home/zhangqijia/repo/flash-attention-npu/build/mla-m02b-validation/`

- `source-export.json`、`ops-transformer.tar`、`snapshot-audit-final.json`：固定导出、原仓既有状态及构建后核对。
- `build-command.json`、`build-incremental-command.json`、`build-initial.log`、`build-network.log`、`install-command.json`、`install.log`：原生构建/安装全过程；`deps/` 保留独立依赖，未链接回原源仓或 CATLASS 子模块。
- `package-audit.json`、`device-build.json`、`compile-fp16.log`、`compile-bf16.log`、`kernel-package-command.json`、`build-manifest.json`：源码包、二进制及构建命令/哈希。`ops-transformer/` 是导出的构建副本；`opp-dense/vendors/mla_ca18aee_transformer/` 是最终配套包。
- `paired-probe.json` / `.log`、`source-all-traced.json` / `.log`、`dispatch-proof.json`：真实 host/device 路径、逐例实际 key、48 次 launch 与编译 metadata 对应。调试日志开启时的计时不作性能解释。
- `source-final.json` / `.log`、`source-final-command.json`：关闭调试输出后的 12 组精度与调用诊断值，预热 10、5 轮 × 50 次；已核对执行器最终文件哈希。仍无 profiler kernel 时间，无性能通过结论。
- `source-pytest.log`、`provenance-pytest.log`、`fa3-regression.log`、`fa3-regression-command.json`、`environment.log`、`npu-smi-*.log`、`quality-final.log`、`checksums.sha256`：验收与质量证据。
- `probe-source.log`、`probe-source-retry.log`、`source-probe*-bindings.*`：初始化失败与版本混用故障的原始证据；不计入通过结果。

从仓库根执行；先重新检查设备健康/占用。本轮物理设备为 7、进程内为 `npu:0`。实机命令在沙箱外运行，shell 使用 `login=false`。

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m pytest -q tests/test_flash_attn_npu_v3_mla_provenance.py

mla_work="$PWD/build/mla-m02b-validation"
mla_vendor="$mla_work/opp-dense/vendors/mla_ca18aee_transformer"
ASCEND_RT_VISIBLE_DEVICES=7 ASCEND_CUSTOM_OPP_PATH="$mla_vendor" \
  LD_LIBRARY_PATH="$mla_vendor/op_api/lib:$LD_LIBRARY_PATH" \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m pytest -q tests/test_flash_attn_npu_v3_mla_cann.py

ASCEND_RT_VISIBLE_DEVICES=7 ASCEND_CUSTOM_OPP_PATH="$mla_vendor" \
  LD_LIBRARY_PATH="$mla_vendor/op_api/lib:$LD_LIBRARY_PATH" \
  ASCEND_GLOBAL_LOG_LEVEL=3 ASCEND_SLOG_PRINT_TO_STDOUT=0 \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m tests.mla_cann_baseline \
  --output "$mla_work/reproduce.json" \
  --cann-source /home/zhangqijia/repo/ops-transformer \
  --expected-vendor "$mla_vendor" \
  --seed 20260924 --warmup 10 --iterations 50 --rounds 5
```

执行器拒绝覆盖既有 JSON，复现须选新路径。普通 FA3 精确节点与参数数组保存在 `fa3-regression-command.json`；相同 vendor 环境下用 `subprocess.run(json.load(...), check=True)` 原样执行。若重新构建基线，在新目录重新导出固定 commit，按构建 JSON 执行原生 host 构建和安装，再执行 `device-build.json` 的两条 `opc` 命令和 `kernel-package-command.json` 生成配置；替换路径时同时调整 wrapper、参数、输出和 vendor，保留新一轮日志与 SHA256。原生 `--jit` 不生成配套 device binary，不能单独拿来验收。

```bash
mla_index="$PWD/build/mla-m02b-validation/mla-commit.index"
GIT_INDEX_FILE="$mla_index" PYTHONPATH=/tmp/mla-m01-precommit \
  PRE_COMMIT_HOME=/tmp/mla-m01-precommit-cache \
  ci/code_quality.sh --staged
git diff --check
GIT_INDEX_FILE="$mla_index" git diff --cached --check
```

## 环境与下一步约束

| 项目 | 本轮核实结果（2026-09-24） |
| --- | --- |
| CANN 来源 | `ca18aee610729ca5270a44e2d84c751bc687fbf0`；原仓 FIA/common 无改动，其他路径既有删除/未跟踪状态完整保存在 source-export，未处理 |
| 构建依赖 | opbase `0a83b915a6f6be49a03b907713d6f000b3dea1ea`；ops-tensor `43d554d6913d5677e31255fc51b0238d9fdaa7cd`；tensor_api `04e3587dceb4792be9351e1057bf41772b5ffcd6`；cann-cmake `master-058` → `cde19c2d5bfcbe40f51acfbd1e455d0866388884`；CATLASS `769cd40a8716b28650b6bebb08db4834eea4462f`；隔离依赖 Git 状态均干净 |
| Python / 包 | `/usr/local/python3.11.15/bin/python`，Python 3.11.15，torch `2.7.1+cpu`，torch_npu `2.7.1` |
| CANN / 系统 OPP | `/usr/local/Ascend/cann-9.0.0-beta.2` 及其 `opp`，未修改；本步仅通过进程环境加载独立 custom vendor |
| custom API / tiling | `libcust_opapi.so` SHA256 `5681ca7bb73a28b781deb0a55399a601416afdc91e2a655bc45d9c7e2c1a4599`；`libcust_opmaster_rt2.0.so` SHA256 `acb5048b6633d5c7db4ab78a6e54ff520aadf8baf0228f4343315864cd2f88ef` |
| 配套 device binary | FP16 `a941647ab85ed1c3b0d315c6bc17020fb9a96939a94a33b61e3296d27a9b8c4c`；BF16 `e26ee6feaebf64e364fb2dc2aecc985d54a081440edc621d4ffcaa80676eb798`；各 12 keys，完整路径及元数据见最终 JSON/manifest |
| FA3 扩展 | 实际加载本仓 `flash_attn_npu_3/flash_attn_npu_3.cpython-311-aarch64-linux-gnu.so`，SHA256 `fabdbb711f336e2a062e0b032fe0bdb65758da091b61c6537a0b8294a6c2677c`；继承 M02a 强制重建产物与证据，源码未变 |
| 设备 | 7 号 Ascend910B3；5、6 号 Alarm 未使用；后续必须重新查询 |

M02b2 先核对代表输入会使用的 keys，超出本包时补编译并再次核实实际加载，不能退回系统旧 FIA 二进制。扩大输入、profiler、转换成本和正式性能验收尚未完成。普通调用诊断值与 M02a 安装包存在差异，应在 M02b2 分析 host/kernel 成本，不能拿 event span 的差异直接宣布退化或性能达标。

M03 才处理生产入口 `qv` 静默忽略缺口；TND/PA、mask、split/metadata 和图的 MLA 适配仍未接通。
