# MLA 当前状态

## 当前能力与下一步

- 分支：`feat/mla-cann-migration`；接手 HEAD 为 `d649e93`，接手时工作区干净。本次完成 M04b2b：FP16 Vector Block 与 dense Kernel 的独立设备验收。
- 新增 `block/mla_vector.hpp`、`tile/mla_vector_ops.hpp`、`kernel/mla_dense.hpp` 和私有 `mla_dense_params.h`；复用 M04b2a Cube 与 CATLASS Tile，扩展 Cube/Vector 共用的运行信息。源码保留固定 CANN 来源、版权和 `LICENSE.cann`；CATLASS 子模块未修改。
- 独立 Kernel 范围：连续 BSND、512+64 共享 KV、FP16、B>0、`1<=Sq*Hq<=256`、KV `1..512`、Hkv=1、无 mask、显式 `num_splits=1`。驱动使用真实平台资源和 M04b1 调度，运行时传入 scale，计算 FP16 O 与 FP32 LSE。
- 保留源 SoftmaxFlashV2、aMLA 中间计算和舍入、三项任务环及排空、双槽 GM、UB 分区、跨核事件、LSE 初始化/SyncAll 和 O/LSE 写回顺序。15 个数值、资源和事件辅助函数的函数体经归一化比较一致；其余 dense 专门化与来源映射见 `source-mapping.json` 和 [design 6.1](design.md#61-首个-dense-实现的源对应关系)。
- 生产 MLA 前向与 MLA metadata 仍明确报未实现；公共入口、生产构建、普通 FA3 源码、已链接对象和实际加载扩展保持 M03 状态。保留的 C1V1 与跨 S2 nUpdate/原子加分支尚未验收，未开放 910C、BF16 或其他范围。
- 唯一下一步：M04b2c，接通首步 FP16 Kernel、静态调度、FA3 入口与生产构建，核查真实/fake/C++ 边界、加载产物、公开调用 E2E 和普通 FA3 回归。详见 [plan](plan.md)；本步提交后停止。

## 本步验证

证据目录：`build/mla-m04b2b-dense/`，实机为物理 7 号 Ascend910B3。

| 项目 | 结果 |
| --- | --- |
| 上步最小闭环 | PASS：开始前重新构建并运行 M04b2a 的 155 项 Cube 检查，无 skip；既有验收文件哈希与交接一致。M04b1 的 85 项 CPU、49,251 组调度对照和 sanitizer 证据已复核，本次未重跑未修改的 CPU 调度测试 |
| dense 精度 | PASS：当前源码新构建，175 项、无 skip；174 组正向记录、每实现合计 544 次重复调用。O/LSE 与固定源逐元素一致，独立 CPU FP64 reference 与低精度对照满足仓库 `assert_fa_close`；对已舍入 reference 的最大绝对误差为 `0.001953125` / `5.722046e-6` |
| 边界与流水 | PASS：D01/D02、默认/零/正/负 scale、M 的 1/15/16/17/127/128/129/255/256、KV 尾块到 512；多 query、非零 storage offset、内容/RoPE 四段独立扰动、aMLA 指数变化、B 到 61；覆盖 mBase=128/256、最多 20 核、任务环排空、交错形状及 scale 复用、输入/输出/workspace 保护 |
| Cube 回归 | PASS：共用结构扩展后重新构建，既有 155 项全部通过，无 skip；Cube 数值代码未修改 |
| 固定 CANN 与分派 | PASS：五组性能用例的 O/LSE 同时与固定 native CANN 逐元素一致；独立分派日志中 15 次调用的核数、C1V2、BATCH_MODE 一致。native key 为 `105000000010000001`，实际注册二进制与 M02b1 构建对应；303 份 vendor 产物哈希保持一致 |
| 构建与来源 | PASS：原 Kernel/Cube/Vector 及依赖由固定 Git blob 导出，原设备算法文件未修改；原/迁移探针分别构建并核对设备依赖。原探针仅共享私有参数 POD，不包含迁移算法；迁移探针不依赖原算法头。精度、性能和分派使用的源码/动态库可追溯 |
| 性能诊断 | 五组同卡交替采样，1,500 次调用墙钟、750 条 kernel task；kernel CSV 与原始 CANN op summary 的名称、设备、时间和时长逐条一致，详见下表 |
| 提交质量 | PASS：18 处文档链接/锚点、唯一下一步、差异、构建及二进制完整性核对通过，六项适用仓库质量检查通过；仅本步 13 个文件，最终结果见 `quality-final.log`、`acceptance.json` |
| 生产构建、普通 FA3 回归 | 未运行：本步未修改公共入口、metadata 或生产构建。M03 的 233 项 MLA 契约和 11 项普通 FA3 证据仍在 `build/mla-interface-guard-validation/`；本次重新核对生产源码、对象及扩展未变 |

最终诊断（单位 μs，中位数；shape 为 `B,Sq,Hq,Sk`）：

| shape / scale | 源探针 kernel | 迁移 kernel | native CANN kernel | 迁移 / native kernel | 源探针 / 迁移 / native 调用墙钟 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2,1,8,257（D01 默认） | 24.031 | 23.081 | 24.620 | 0.9375 | 158.795 / 158.250 / 251.045 |
| 2,1,8,257（D01 scale=0.23） | 23.800 | 22.931 | 24.661 | 0.9298 | 163.360 / 156.625 / 247.820 |
| 1,7,16,65（D02 默认） | 19.420 | 18.560 | 19.831 | 0.9359 | 160.015 / 156.655 / 247.555 |
| 1,1,128,512（默认） | 48.541 | 47.471 | 49.321 | 0.9625 | 183.245 / 182.880 / 272.810 |
| 2,2,128,511（默认） | 65.642 | 64.921 | 64.462 | 1.0071 | 206.230 / 203.840 / 294.355 |

每例每实现预热 5 次，5 轮交替顺序；每轮墙钟 20 次、profiler 10 次，均等待设备完成。两个探针预分配输入、输出、workspace 和调度；native CANN 包含公开算子分派与每次输出分配，计时不含输入 H2D。两种调用口径分别记录，不能据此认定 FA3 调用有 E2E 优势。本步尚未验收 metadata E2E；最终门槛仍为逐例 `FA3 metadata 路径 E2E / 原 CANN E2E <= 1.0`。

## 环境、失败与限制

| 项目 | 本轮复核 |
| --- | --- |
| Python / 包 | `/usr/local/python3.11.15/bin/python3` 3.11.15；torch `2.7.1+cpu`、torch_npu `2.7.1` |
| CANN / 编译器 | `/usr/local/Ascend/cann-9.0.0-beta.2`；bisheng clang 15，`dav-2201`、`-O2`；精度和性能探针均从当前源码新构建 |
| 芯片与占用 | 7 号 Ascend910B3；初始 Warning `81078603` 为端口 link down，结束时健康为 OK、无进程，前后 ECC 计数均为 0；性能采样前后均仅记录一个进程。完整快照见 `environment.json`、`npu-final.json`、`performance-final/npu-*.log` |
| 固定来源 | FIA `ca18aee610729ca5270a44e2d84c751bc687fbf0`；CATLASS `769cd40a8716b28650b6bebb08db4834eea4462f`，子模块干净；CANN 源仓已有删除/未跟踪文件保持原样 |
| native CANN | 沿用 `build/mla-m02b-validation/opp-dense/vendors/mla_ca18aee_transformer`；已核对实际加载 host 库、打包头文件、设备二进制及当前分派，未重新构建该只读基线 |
| 实际生产扩展 | 本仓 `flash_attn_npu_3/flash_attn_npu_3.cpython-311-aarch64-linux-gnu.so`；SHA256 `955ecbe4c2b199b2e1131d9c86c38ce23b15ed8704c55cf2e35de37986b84877`；直接加载核实，仍为 M03 产物，不能用于证明新 Kernel 已接入生产 |

早期原探针编译缺少源翻译单元引入的常量；补齐后发现测试 tiling 子结构未满足源 `copy_data_align64` 所需的 8 字节对齐，导致原探针设备异常。修正对照驱动声明后最小闭环通过。首次启动模式封装拦截了未被编译器使用的 runtime API，驱动计数检查返回 `-2`；改为拦截实际 ACL launcher 后，日志确认两份探针及 native CANN 均为 BATCH_MODE。失败日志保留，未计为通过。

独立探针的 ACL 封装仅存在于测试 DSO，保留其他启动属性并传递错误码；编译器绕过该路径时主动失败。生产 launcher 的接入仍属 M04b2c。

Profiler 的 500 个探针 step 缺少框架 flow，`TraceStepTimeParser`、`KernelViewParser`、`CommunicationParser` 报 `There is no flow events`，并有 `step_trace` 数据库告警。最终 750 条 kernel task 与原始 CANN CSV 核对通过；本次只使用 task 时间，不使用框架 flow、step 关联、block-dimension CSV 或硬件计数器。编译和退出时的重复 tiling 注册警告也保留在日志中。

## 证据与复现

- `previous-verified.json`、`previous-cube.xml`、`previous-cube/`：交接核对及上步最小闭环重跑；`cube-final.xml`、`cube-final/` 为共用结构变更后的回归。
- `environment.json`、`source-repositories.json`、`production-unchanged.json`、`npu-final.json`：环境、源仓初始状态、生产产物不变及设备快照。
- `accuracy-final/`、`accuracy-final.log`、`accuracy-final.xml`：固定源导出及哈希、原/迁移探针与 host 调度动态库、精确编译命令/设备依赖、174 组精度记录。
- `performance-final/`、`performance-final.log`：同轮新构建、五组 native CANN 精度对照、环境与加载库、墙钟样本、完整 profiler 原始数据和解析告警。
- `dispatch.py`、`dispatch.log`、`dispatch.json`、`dispatch-proof.json`：性能所用动态库的实际启动模式、核数、固定 native CANN 二进制及 key；诊断日志与计时分开采集。
- `source-mapping.json`、`verify_acceptance.py`、`acceptance.json`、`quality-final.log`、`checksums.sha256`：源码对应关系和提交前检查。各 `*-command.json`、`*-exit.json` 保存命令、环境与退出状态。
- `smoke*-build.log`、`smoke2-run.log`、`batch-run.log` 保存早期失败；`aligned-run.log`、`batch-acl-run.log` 为修正后的最小闭环，最终结论以 `accuracy-final` 和 `performance-final` 为准。

从仓库根目录复现（需可访问 910B 驱动；每次使用新 artifact 目录保留旧证据）：

```bash
ASCEND_RT_VISIBLE_DEVICES=7 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MLA_CANN_SOURCE_ROOT=/home/zhangqijia/repo/ops-transformer MLA_DENSE_ARTIFACT_DIR=build/mla-m04b2b-dense/recheck-dense python3 -m pytest -x -q tests/test_flash_attn_npu_v3_mla_dense.py
ASCEND_RT_VISIBLE_DEVICES=7 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MLA_CANN_SOURCE_ROOT=/home/zhangqijia/repo/ops-transformer MLA_CUBE_ARTIFACT_DIR=build/mla-m04b2b-dense/recheck-cube python3 -m pytest -x -q tests/test_flash_attn_npu_v3_mla_cube.py
MLA_VENDOR_DIR="$PWD/build/mla-m02b-validation/opp-dense/vendors/mla_ca18aee_transformer"
ASCEND_RT_VISIBLE_DEVICES=7 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MLA_CANN_VENDOR="$MLA_VENDOR_DIR" ASCEND_CUSTOM_OPP_PATH="$MLA_VENDOR_DIR" LD_LIBRARY_PATH="$MLA_VENDOR_DIR/op_api/lib:$LD_LIBRARY_PATH" MLA_DENSE_ARTIFACT_DIR=build/mla-m04b2b-dense/recheck-performance python3 -m tests.mla_dense_profile
```

最终暂存检查：

```bash
PYTHONPATH=/tmp/mla-m01-precommit PRE_COMMIT_HOME=/tmp/mla-m01-precommit-cache bash ci/code_quality.sh --staged
python3 build/mla-m04b2b-dense/verify_acceptance.py --staged
git diff --check
git diff --cached --check
```

当前闭环提交用 `git log -1 -- docs/mla/status.md` 定位。
