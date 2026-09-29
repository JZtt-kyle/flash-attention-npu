# MLA 当前状态

## 当前能力与下一步

- 分支：`feat/mla-cann-migration`；接手 HEAD 为 `e0b3382`，接手工作区干净。本次完成 M04b2c：首步 FP16 dense MLA 的 FA3 接入与生产构建。
- 已开放 910B `flash_attn_func(q, k, v, qv=qv, num_splits=1)`：512+64 共享 KV、连续 BSND、FP16、B>0、`1<=Sq*Hq<=256`、KV `1..512`、Hkv=1、无 mask。返回 FP16 O 和可选 FP32 LSE；默认/零/正/负 scale 与连续视图的非零 storage offset 已验收。
- 生产 `mla_forward.cpp` 复用 M04b1 静态调度和实际平台资源，分配每次独立的 O/LSE/workspace，并记录输入消费 stream；生成器提供 FP16 C1V2 编译单元。M04b2b 的六个 Kernel / Block / Tile 头文件逐字节未变，保留 CANN 来源与 `LICENSE.cann`，CATLASS 子模块未修改。
- 真实/fake/C++ 一致拒绝 BF16、prefill、metadata、TND、PA、mask、非连续布局、长度、预分配 out、append/rotary/descale、非默认调度选项、梯度和首步形状范围外组合；公开 varlen/kvcache 仍拒绝 MLA。尚未开放 910C/C1V1。
- 唯一下一步：M04b3，同范围 BF16 实例化及验收，详见 [plan](plan.md)。本步提交后停止。

## 最近一次闭环证据

证据目录：`build/mla-m04b2c-integration/`；物理 7 号 Ascend910B3。

| 项目 | 结果 |
| --- | --- |
| 上步最小闭环 | PASS：核对前次源码与证据哈希，当前源码重新构建并运行 175 项独立 dense 检查，无 skip |
| 生产构建 | PASS：目标 v3/910 全量强制构建 14 个 ASC 单元及 AICPU；最终链接 15 个对象。输入生命周期补充后增量重编译；实际加载路径、源码、对象和动态库哈希可追溯 |
| 增量依赖 | PASS：独立设备预处理依赖包含全部六个算法头；触碰 `kernel/mla_dense.hpp` 时间戳实际触发 MLA 设备单元重编译，其他 12 个未变化单元复用；头文件内容和时间戳已恢复 |
| 入口契约 | PASS：339 项 eager/fake/C++、公开入口、metadata 拒绝及普通 fake shape 检查，无 skip；未接通参数逐项拒绝 |
| 公开入口精度 | PASS：101 项，无 skip；94 组记录、282 次公开重复调用的 O/LSE 与固定源逐元素一致，通过独立 CPU FP64/低精度双 reference；最大绝对误差 `0.001953125` / `5.722046e-6`。另验收交错 scale/形状、多 stream、输入释放后重分配、内部默认 scale 和仅返回 O |
| CPU 回归 | PASS：176 项，无 skip；含 reference、来源、profile、85 项调度对照，以及新增生成器/依赖和带 ASan/UBSan 的 scoped launcher 检查 |
| 普通 FA3 | PASS：11 项，无 skip；BSND、TND、分页、split、metadata 及编译图前反向。8 个普通前反向 Kernel 对象仍与 M03 逐字节一致；AICPU 由原源码重新构建 |
| 实际分派 | PASS：五组 native CANN 与 FA3 共 10 次 MLA launch 的核数、C1V2、BATCH_MODE 一致。native key `105000000010000001`，注册二进制与 M02b1 相符，303 份基线产物哈希未变。MLA 后普通 FA3 launch 保持 mode 2；CPU 检查确认普通配置、其他属性及错误码原样透传 |
| 性能诊断 | 完成五组同卡交替采样：1,000 个同步公开调用 E2E、500 条 kernel task；每条 task 与原始 CANN CSV 的名称、设备、时间、时长一致。无 metadata E2E 仍明显慢于 CANN，见下表 |
| 提交质量 | 源对应关系、文档链接、差异、产物完整性及六项适用仓库质量检查通过；提交前最终核对见 `quality-staged.log`、`acceptance.json`、`checksums.sha256` |

单位 μs，中位数；shape 为 `B,Sq,Hq,Sk`：

| shape / scale | FA3 kernel | CANN kernel | FA3 公开 E2E | CANN E2E | E2E 比值 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2,1,8,257 / 默认 | 12.540 | 14.080 | 479.545 | 293.865 | 1.6319 |
| 2,1,8,257 / 0.23 | 12.641 | 13.980 | 505.005 | 310.270 | 1.6276 |
| 1,7,16,65 / 默认 | 17.020 | 18.120 | 510.730 | 313.235 | 1.6305 |
| 1,1,128,512 / 默认 | 29.730 | 30.171 | 498.000 | 309.580 | 1.6086 |
| 2,2,128,511 / 默认 | 51.511 | 50.681 | 503.315 | 312.520 | 1.6105 |

每例每实现预热 5 次，5 轮交替顺序；每轮 E2E 20 次、profiler 10 次，每次等待设备完成。两端均使用预先准备的同一输入，计时排除输入 H2D；FA3 E2E 包括公开入口校验、静态调度、输出/workspace 分配、发射及计算。CANN 沿用固定原生分派及每次输出分配。

**最终性能门槛尚未达到。** 当前没有 MLA metadata；五组无 metadata E2E 比值约 1.61–1.63，不能用 kernel 接近 CANN 宣称性能达标。M05 仍须逐例验收 `FA3 metadata 路径 E2E / 原 CANN E2E <= 1.0`，接口、调度准备和复用成本需要继续处理。

## 环境、失败与限制

- Python `/usr/local/python3.11.15/bin/python3` 3.11.15；torch `2.7.1+cpu`，torch_npu `2.7.1`；CANN `/usr/local/Ascend/cann-9.0.0-beta.2`，bisheng clang 15，`dav-2201`、`-O2`。
- 7 号 910B3 初始、性能采样前后和结束健康为 OK；前后 ECC 为 0，性能采样快照均只见一个进程，结束无进程。见 `environment.json`、`performance-final/npu-*.log`、`npu-final.json`。
- 实际扩展：本仓 `flash_attn_npu_3/flash_attn_npu_3.cpython-311-aarch64-linux-gnu.so`；SHA256 `7f7d711a5dad675e5d136f243ec0f23b0d932b4855f188aee47a39638d1f2b6b`。本步已替换原 M03 产物；生产验收均使用这一最终产物。
- 固定 CANN `ca18aee610729ca5270a44e2d84c751bc687fbf0`、CATLASS `769cd40a8716b28650b6bebb08db4834eea4462f`；源仓现有删除/未跟踪文件保持原样。native vendor 仍为 `build/mla-m02b-validation/opp-dense/vendors/mla_ca18aee_transformer`。
- 登录 shell 初始化未返回输出，改用非登录 shell；沙箱内 NPU 驱动初始化失败，实机命令改为沙箱外执行。CPU launcher 的首次 sanitizer 检查因沙箱 ptrace 限制导致 LeakSanitizer 失败，未记为通过；沙箱外相同 ASan/UBSan 检查通过，保留 `cpu-launch.log` 与 `cpu-final.log`。
- CANN beta 版本字符串、重复 tiling 注册及普通 Kernel 自动类型推断警告保留在日志；本轮 profiler 无解析错误。`setup.py` 的额外排版由仓库质量钩子产生。
- 生产 BATCH_MODE 封装只作用于 MLA 发射线程的局部作用域，保留其他 launch 属性并传递错误码；若编译器绕过该 ACL 路径则主动失败。当前仅验收本轮 CANN/编译器；完整图生命周期和其他芯片仍按后续计划推进。

## 证据与复现

- `previous-verified.json`、`previous-dense*`：上一步来源/产物复核和重建回归。
- `build-final*`、`build-current*`、`build-source-manifest.json`、`build-provenance.json`、`incremental-proof.json`：强制构建、最终增量构建、实际加载和设备依赖。
- `accuracy-final/`、`accuracy-final.log/xml`：当前源码新构建的原/迁移探针及公开入口验收，包含源导出、编译命令、库哈希与逐例记录。
- `cpu-final.log/xml`、`regression-final.log/xml`：CPU 与普通 FA3 回归；精确命令见对应 `*-command.json`。
- `performance-final/`：环境及加载库、墙钟样本、完整 profiler 原始数据、逐条 CSV 核验与汇总；`dispatch.log/json`、`dispatch-proof.json` 为独立分派证据。
- `source-mapping.json`、`source-repositories.json`、`native-baseline-verified.json`、`acceptance.json` 和 `checksums.sha256`：来源、边界、文件及最终证据完整性。

从仓库根目录复现；使用新的 artifact 目录，保留既有证据：

```bash
FLASH_ATTN_BUILD_VERSION=v3 FLASH_ATTN_BUILD_NPU=910 FLASH_ATTN_SKIP_SUBMODULE_INIT=1 FLASH_ATTN_FORCE_REBUILD=TRUE python3 setup.py build_ext --inplace
ASCEND_RT_VISIBLE_DEVICES=7 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MLA_DENSE_ARTIFACT_DIR=build/mla-m04b2c-integration/recheck-accuracy python3 -m pytest -x -q tests/test_flash_attn_npu_v3_mla_contract.py tests/test_flash_attn_npu_v3_mla_forward.py
TORCH_DEVICE_BACKEND_AUTOLOAD=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 -m pytest -x -q tests/test_flash_attn_npu_v3_mla_build.py tests/test_flash_attn_npu_v3_mla_reference.py tests/test_flash_attn_npu_v3_mla_provenance.py tests/test_flash_attn_npu_v3_mla_profile.py tests/test_flash_attn_npu_v3_mla_schedule.py
MLA_VENDOR_DIR="$PWD/build/mla-m02b-validation/opp-dense/vendors/mla_ca18aee_transformer"
ASCEND_RT_VISIBLE_DEVICES=7 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MLA_CANN_VENDOR="$MLA_VENDOR_DIR" ASCEND_CUSTOM_OPP_PATH="$MLA_VENDOR_DIR" LD_LIBRARY_PATH="$MLA_VENDOR_DIR/op_api/lib:$LD_LIBRARY_PATH" MLA_FORWARD_ARTIFACT_DIR=build/mla-m04b2c-integration/recheck-performance python3 -m tests.mla_forward_profile
```

最终暂存检查：

```bash
PYTHONPATH=/tmp/mla-m01-precommit PRE_COMMIT_HOME=/tmp/mla-m01-precommit-cache bash ci/code_quality.sh --staged
python3 build/mla-m04b2c-integration/verify_acceptance.py --staged
git diff --check
git diff --cached --check
```

当前闭环提交用 `git log -1 -- docs/mla/status.md` 定位。
