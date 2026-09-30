# MLA 当前状态

## 当前能力与下一步

- 分支 `feat/mla-cann-migration`，接手 HEAD `512d547`，接手工作区干净。本步完成 M04c1。
- dense S1G 多块已接通：解除 `Sq*Hq<=256` 的里程碑限制，沿用固定源 `CalcMBaseSize` / `SplitCore` 和既有 Kernel 三项任务环。metadata 消费者接受源 128..256 的 M 基块，包含 129/192/255；检查跨核端点、尾块和 workspace 后执行。
- 公开 `flash_attn_func`、只读 `flash_attn_with_kvcache`、metadata 真实/fake/C++ 边界一致；401 字版本 1 记录、scale 和生命周期契约不变。仍限定 910B、FP16/BF16、512+64 共享 KV、连续 BSND、B>0、Hkv=1、KV 容量 1..512、无 mask、显式 `num_splits=1` 和整容量长度。
- 源字段/寻址上界见 [design 6.2](design.md#62-调度字段和首步边界)：`Sq<=2^30`、`Hq<=floor((2^32-1)/512)`、`Sq*Hq<=2^32-256`、`B*Sq*Hq<=floor((2^63-1)/1024)`；LCM、总 M 任务数、平台和 workspace 仍独立检查。未接通的长 KV、追加 KV、TND/PA、部分/零长度、梯度等组合明确拒绝。
- **唯一下一步：M04c2，接通 dense KV 513..1024，显式 `num_splits=1`；保留源 aMLA nUpdate、PV 原子加与事件依赖。** 本步提交后停止。

## 最近一次闭环证据

持久证据目录：`build/mla-m04c1-s1g/`，物理 7 号 Ascend910B3。最终源码和实际扩展对应 **2,008 项测试通过，无失败或 skip**；下面的初次失败/超时均单独保留，不计入通过总数。

| 项目 | 结果 |
| --- | --- |
| 上步复核 | PASS：2,712 份证据哈希、1,789 个去重测试记录、17 个生产对象及 303 份固定 CANN 构建文件核对；CANN/CATLASS 状态未变，见 `previous-verified.json` |
| CPU | PASS：186 项；51,129 组调度与固定源逐字段一致且无 FD，覆盖 C1V1/C1V2、26 种核数、多块端点、全部 128..256 候选 M；含 metadata 解码、溢出/损坏记录、ASan/UBSan 及构建依赖，见 `cpu-final*`、`schedule-final/`、`consumer-final/` |
| 多块 910B | PASS：55 项；两种 dtype 各 22 组形状/scale/内容-RoPE 扰动，三条公开路径各重复 3 次，与独立构建的固定源 O/LSE 逐元素一致并通过独立数学 reference；覆盖 257/511/512/513 行、非零 offset、跨 query/head/batch 尾块、16,512 行、多流释放压力及 fake/compile，见 `multiblock-final*` |
| dispatcher / AICPU | PASS：428 项入口边界/schema/fake/compile；54 项设备生成器及固定源逐字对照，新增所有 M 候选和多块记录，见 `dispatch-final*`、`aicpu*` |
| 原有公开 MLA | PASS：518 项；原有形状/scale、舍入、跨流、主机线程和快照回归，见 `public-final*`、`producer-final/` |
| 设备拒绝 | PASS：36 个隔离进程，含 513 行下非法 M 基块和跨核端点；所有 Trap 均在 15 秒内上报，专属进程组无存活进程，见 `errors-final*`、`process-groups.json` |
| 契约与普通 FA3 | PASS：582 项 MLA 契约、129 项普通 metadata 签名/兼容/精度/compile、20 项普通 FA3 BSND/TND/分页/split/metadata/mask/softcap/scale/compile，见对应 `*-final*` |
| 构建与加载 | PASS：增量重编译 6 个 AscendC 单元及 MLA AICPU；6 个对象哈希变化，11 个对象与上步一致（`mla_launch.o` 重编译后字节相同）。60 份构建源码、128 份运行源码、17 个对象、36 份依赖/命令文件及实际加载扩展核对，见 `build-provenance.json`、`production-artifacts/` |
| 性能 | PASS：两种 dtype 各 11 组，显式复用、生成加一次/四次前向逐例同轮 CANN E2E 比值均 <= 1.0，见下表及 `performance-*/` |

最终扩展为 `flash_attn_npu_3/flash_attn_npu_3.cpython-311-aarch64-linux-gnu.so`，SHA256：`c5ff1b919b6ad89c14844f0f2c0237fd1ec963b9f73ccecc44807ea7c12e887b`。最终功能、性能和源码快照均对应此产物。

## 性能口径与结果

保留 M05 原五组，采样前固定增加 M04c1 六组多块代表（见 [design 6.5](design.md#65-必要的-fa3-差异与验收切面)）。两边在同一进程、同一空闲健康 910B、相同 CPU 亲和性 `68,70,76,77,84,85,87,88` 下交替测量；CPU 组按采样前 NPU 7 所在 NUMA 节点的利用率选定，见 `host-affinity-selection.json` 和 `preflight-resume.json`。

每例预热 5 次、五轮 × 20 个 wall 样本，完整段 R=1/4 预先固定且只在末尾同步，双方前向次数一致；两种 dtype 共 **15,400 个 wall 样本**。输入 H2D 不计时，入口检查、准备、输出/workspace 分配、发射、依赖和完成同步计时。另以五轮 × 10 个样本交替采集 attention Kernel，共 **2,200 个 Kernel 样本**；显式复用 AICPU 生成任务为 0。

下表为每例 100 个 wall 样本的 FA3 / 同轮 CANN E2E 中位数，逐例通过 1.0 门槛：

| `(B,Sq,Hq,Sk)` / scale | FP16 复用 | BF16 复用 | FP16 生成+1次 | BF16 生成+1次 |
| --- | --- | --- | --- | --- |
| `(2,1,8,257)` / 默认 | 0.634264 | 0.616878 | 0.892304 | 0.849966 |
| `(2,1,8,257)` / 0.23 | 0.648528 | 0.641435 | 0.892427 | 0.893025 |
| `(1,7,16,65)` / 默认 | 0.632590 | 0.629788 | 0.876189 | 0.895322 |
| `(1,1,128,512)` / 默认 | 0.643389 | 0.641313 | 0.918153 | 0.919953 |
| `(2,2,128,511)` / 默认 | 0.638831 | 0.620482 | 0.918747 | 0.906162 |
| `(1,257,1,257)` / 默认 | 0.642508 | 0.645119 | 0.900690 | 0.892332 |
| `(2,7,73,65)` / 默认 | 0.673211 | 0.640141 | 0.936696 | 0.900322 |
| `(1,4,128,512)` / 默认 | 0.657838 | 0.616821 | 0.915114 | 0.844127 |
| `(2,171,3,511)` / 默认 | 0.685741 | 0.627015 | 0.927688 | 0.863991 |
| `(10,3,86,65)` / 默认 | 0.673703 | 0.674689 | 0.917360 | 0.965597 |
| `(1,129,128,33)` / 默认 | 0.668044 | 0.654211 | 0.922643 | 0.924216 |

- FP16：生成加四次前向 / CANN 四次为 0.522–0.581；metadata 生成中位数 195.320–225.940 μs；attention Kernel / CANN 为 0.896–1.033。
- BF16：生成加四次前向 / CANN 四次为 0.522–0.594；metadata 生成中位数 194.700–219.070 μs；attention Kernel / CANN 为 0.857–1.091。

逐例值、轮均值和全部原始样本保留在 `profile.json`、`wall-samples.json` 和 profiler CSV。Profiler 与原始 CANN CSV 逐行核对；PID/物理 7 号映射与采样前后独占快照见 `performance-device-proof.json`。在 `/dev/shm/mla-m04c1-*` 采集后逐文件校验并归档到本证据目录；`performance-*-archive.json` 保存原始哈希，`profile.original.json` 保留原报告，当前报告只改为持久 CSV 路径，未修改样本。

## 环境、失败记录与限制

- Python `/usr/local/python3.11.15/bin/python3` 3.11.15，torch `2.7.1+cpu`、torch_npu `2.7.1`；CANN `/usr/local/Ascend/cann-9.0.0-beta.2`，bisheng clang 15.0.5、`dav-2201`、`-O2`。CATLASS `769cd40a8716b28650b6bebb08db4834eea4462f`、固定 CANN `ca18aee610729ca5270a44e2d84c751bc687fbf0`、官方 FA3 `d15f1531a460ba456f41b01a774f33ab2db8febf`；CANN 既有改动、子模块和 950 后端未改。最终 7 号设备健康 OK、ECC 全零、无设备进程，见 `final-device.json`。
- 首轮完成采集的性能中，FP16 大行数 R=1 为 0.994 且接近门槛；BF16 三个原有用例 R=1 为 1.050/1.079/1.144，未闭环，原始样本完整保留在 `performance-noisy/`。同轮 load 从约 76 升至 287（256 核），原 CPU 组事后利用率为 13%–82%，耗时增加主要出现在 metadata 生成，不能将这一轮记为通过。随后以三个 5 秒窗口的最坏利用率预选同 NUMA 的 8 个 CPU，双方使用同组；按原 11 组、原次数和原门槛复测，额外用 `performance-host-load.jsonl` 记录期间负载。生产代码和采样值均未为此修改。
- 首次性能预检在固定 CANN 主机入口被 `num_heads<=128` 限制拒绝，尚未进入计时。保持原五组和新增六组的 B、S1G 行数、KV 容量不变，将两组的 Sq/Hq 拆分由 `3*171`、`2*129` 改为 `171*3`、`3*86`；R=1/4、采样次数及 CPU 亲和性保持不变。最终 profiler 在计时前重新通过这两组的双 dtype 原生 CANN 和独立数学精度对照；生产源码/扩展未改。初次证据见 `performance-initial/`，固定源限制和映射见 `performance-case-correction.json`。
- 首次多块检查被测试驱动旧的 256 行 guard 拒绝，未发射 Kernel；生产实现未因此修改。测试驱动同步边界后重建，最终 55 项全部通过。初版日志/产物保留在 `multiblock*`，初版运行源码快照见 `source-initial-snapshots/`。
- 首次 CPU 全矩阵超过 600 秒进程总时限，无断言失败但不计为通过；使用 1800 秒时限完整重跑通过，见 `cpu-initial*`、`cpu-final*`。
- 接手核验脚本最初误以为三个旧证据目录各有 checksum 文件；实际由最终目录的一份总清单覆盖，修正后 2,712 份哈希全部通过。构建核验最初预期重编译的 `mla_launch.o` 哈希也变化；核对后确认该对象字节相同，修正为 6 个变化、11 个未变。两次原始断言和纠正结果均保留，见 `previous*`、`build-provenance*`。
- 沿用非登录 shell、沙箱外构建/实机/sanitizer。首轮格式检查自动调整排版后复核通过；最终 `ci/code_quality.sh --staged` 通过。本轮生成的 `kernel_meta` 已归档，未提交构建产物。
- 图捕获仍留给 M16；本轮 compile 为 eager backend/fullgraph。Trap 后不承诺继续复用执行上下文；错误检查禁用 core dump 并清理各自进程组。

## 证据与复现

精确命令、环境、退出码、日志和 JUnit 为证据目录下 `*-command.json`、`*-exit.json`、对应日志及 XML。源码/对象快照、依赖清单、综合核验和全部证据哈希见 `build-source-manifest.json`、`runtime-source-manifest.json`、`build-dependencies.json`、`acceptance.json`、`checksums.sha256`。功能入口为 `run_device.py`；性能选择、采集及归档为 `select_affinity_stable.py`、`run_monitored.py`、`run_controlled.py`、`run_performance.py`。复测使用新的 artifact/JUnit 路径，避免覆盖既有证据。

```bash
FLASH_ATTN_BUILD_VERSION=v3 FLASH_ATTN_BUILD_NPU=910 FLASH_ATTN_SKIP_SUBMODULE_INIT=1 python3 setup.py build_ext --inplace
TORCH_DEVICE_BACKEND_AUTOLOAD=0 MLA_SCHEDULE_SANITIZE=1 MLA_CONSUMER_SANITIZE=1 python3 -m pytest -x -q tests/test_flash_attn_npu_v3_mla_schedule.py tests/test_flash_attn_npu_v3_mla_metadata_consumer.py tests/test_flash_attn_npu_v3_mla_build.py
ASCEND_RT_VISIBLE_DEVICES=7 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 -m pytest -x -q tests/test_flash_attn_npu_v3_mla_multiblock.py tests/test_flash_attn_npu_v3_mla_metadata_dispatch.py tests/test_flash_attn_npu_v3_mla_dispatch.py
PYTHONPATH=/tmp/mla-m01-precommit PRE_COMMIT_HOME=/tmp/mla-m01-precommit-cache bash ci/code_quality.sh --staged
python3 build/mla-m04c1-s1g/verify_acceptance.py
git diff --check
git diff --cached --check
```

当前闭环提交用 `git log -1 -- docs/mla/status.md` 定位。
