# MLA 当前状态

## 当前能力与唯一下一步

- 分支 `feat/mla-cann-migration`，接手 HEAD `48b3905`，接手工作区干净。本步完成 M04c2，将 dense MLA 的 KV 容量扩展到 1..1024，显式 `num_splits=1`；S1G 多块与源 128..256 的 M 基块继续支持。
- 公开 `flash_attn_func`、只读 `flash_attn_with_kvcache`、metadata 的真实/fake/C++ 边界一致。仍限定 910B、FP16/BF16、512+64 共享 KV、连续 BSND、B>0、Hkv=1、无 mask 和整容量长度；401 字 v1 metadata、scale 和生命周期契约不变。字段及地址上界见 [design 6.2](design.md#62-调度字段和首步边界)，另检查总物理任务数、LCM、平台资源和 workspace。
- KV 513..1024 在每个 M 块内顺序计算两个 S2 块，保留源 aMLA nUpdate、PV 原子加和事件依赖。原 CANN 分区有 FD 时，将端点向上对齐到完整 M 块、合并空区间并按源成本重算；无 FD 的原方案保持原样。metadata 按 S2 基块计算 workspace，拒绝跨核 KV 端点。没有 CANN 转调或 CATLASS 修改。
- **唯一下一步：M04c3，按源流程扩大 dense 长 KV 范围，包含 8193。** 更长 KV、追加 KV、TND/PA、部分/零长度、自动/多 split、梯度及其他未接通组合继续明确拒绝。本次提交成功即停止。

## 最近一次闭环证据

持久证据目录为 `build/mla-m04c2-kv/`，精度、功能及普通回归使用物理 1 号 Ascend910B3，设备错误及性能使用空闲的物理 2 号 Ascend910B3。当前源码和实际扩展对应 **2,113 项去重测试通过，无失败或 skip**。首版和繁忙主机上的失败结果单独保留，不计入最终通过结果。

| 项目 | 结果 |
| --- | --- |
| 上步复核 | PASS：2,756 份证据、2,008 项去重测试、17 个生产对象及 303 份固定 CANN 构建文件核对，CANN/CATLASS 状态未变，见 `previous-verified.json` |
| CPU / sanitizer | PASS：195 项；58,689 组源调度对照，含 7,560 组双 KV 块与 C1V1/C1V2、26 种核数；原无 FD 输出不变，映射后的端点/成本由独立源驱动核对；metadata、溢出、ASan/UBSan 和构建依赖通过，见 `cpu-final*`、`schedule-final/`、`consumer-final/` |
| 双 KV 块 910B | PASS：68 项；双 dtype 各 31 组形状/scale/内容-RoPE/跨块最大值变化，三条公开路径各重复 3 次；O/LSE 与独立固定源无 FD Kernel 逐元素一致，并通过独立双数学 reference；含 aMLA 截断、尾块、多流释放压力及 fake/compile，见 `kvblocks-complete*` |
| AICPU / dispatcher | PASS：60 项设备生成器和源记录对照，含 513/1024 的设备长度；428 项入口/schema/fake/compile，见 `aicpu*`、`dispatch-final*` |
| 原有 MLA | PASS：55 项 S1G 多块，518 项公开精度/复用/跨流/线程/快照，12+4 项独立边界，见对应日志/XML |
| 设备拒绝 | PASS：42 个隔离进程，含 Sk=1024 下损坏的 S2 基块、跨核 KV 端点及 workspace；全部 Trap 在 15 秒内上报，专属进程组无存活进程，见 `errors-final*`、`process-groups.json` |
| 契约与普通 FA3 | PASS：582 项 MLA 契约、129 项普通 metadata 签名/兼容/精度/compile、20 项普通 FA3 BSND/TND/分页/split/metadata/mask/softcap/scale/compile |
| 构建与加载 | PASS：首版重建 6 个 AscendC 单元及 MLA AICPU；端点修复后重建调度和 MLA AICPU。最终相对接手为 6 个对象变化、11 个不变；60 份构建源码、129 份运行源码、17 个对象和 36 份依赖/命令文件核对，见 `build-provenance.json`、`production-artifacts/` |
| 性能 | PASS：双 dtype 各 18 组，复用、生成加一次/四次前向逐例同轮 CANN E2E 比值均 <=1.0，见下表 |

实际加载扩展为 `flash_attn_npu_3/flash_attn_npu_3.cpython-311-aarch64-linux-gnu.so`，SHA256：`e0903e9d4a2351a8c7a822058d2948285ae5bfaf204ca2a71cf7f7ffb0a31f07`。最终功能、性能和源码快照均对应此产物。

## 性能口径与结果

保留原 11 组，采样前固定增加 7 组 KV 513..1024 代表，见 [design 6.5](design.md#65-必要的-fa3-差异与验收切面)。两边同一进程、同一空闲健康的物理 2 号 910B、相同 CPU 亲和性 `139,140,141,142,148,149,150,152`；全部位于 NPU 本地 NUMA 节点。预检要求连续三个 5 秒窗口中所选 CPU 最坏利用率低于 15%，选择及采样期间负载见 `host-affinity-selection.json`、`affinity-attempts/`、`performance-host-load.jsonl`。

每例预热 5 次、五轮各 20 个 wall 样本，完整段 R=1/4 预先固定且只在末尾同步，双方前向次数一致；双 dtype 共 **25,200 个 wall 样本**。输入 H2D 不计时；入口检查、准备、分配、发射、依赖和完成同步计时。另以五轮各 10 个样本交替采集 attention Kernel，共 **3,600 个 Kernel 样本**；显式复用 AICPU 生成任务为 0。原生 CANN 保留自动 FD 分派，两边分别通过既定数学精度检查；无 FD 源 Kernel 的逐元素一致性另由上述精度套件验证。

下表为每例 100 个 wall 样本的 FA3 / 同轮 CANN E2E 中位数，逐例通过 1.0 门槛：

| `(B,Sq,Hq,Sk)` / scale | FP16 复用 | BF16 复用 | FP16 生成+1次 | BF16 生成+1次 |
| --- | --- | --- | --- | --- |
| `(2,1,8,257)` / 默认 | 0.635710 | 0.600859 | 0.886405 | 0.876008 |
| `(2,1,8,257)` / 0.23 | 0.626253 | 0.692953 | 0.884911 | 0.898979 |
| `(1,7,16,65)` / 默认 | 0.655097 | 0.624894 | 0.881038 | 0.862055 |
| `(1,1,128,512)` / 默认 | 0.630262 | 0.619383 | 0.897572 | 0.858883 |
| `(2,2,128,511)` / 默认 | 0.673411 | 0.624779 | 0.913815 | 0.851429 |
| `(1,257,1,257)` / 默认 | 0.633114 | 0.623255 | 0.866721 | 0.859824 |
| `(2,7,73,65)` / 默认 | 0.622110 | 0.663396 | 0.868586 | 0.908296 |
| `(1,4,128,512)` / 默认 | 0.609421 | 0.667708 | 0.866631 | 0.884970 |
| `(2,171,3,511)` / 默认 | 0.620676 | 0.677596 | 0.870418 | 0.906146 |
| `(10,3,86,65)` / 默认 | 0.666479 | 0.675914 | 0.910481 | 0.921204 |
| `(1,129,128,33)` / 默认 | 0.703791 | 0.680932 | 0.949544 | 0.927529 |
| `(2,1,8,513)` / 默认 | 0.685896 | 0.671327 | 0.903072 | 0.899555 |
| `(2,1,8,513)` / 0.23 | 0.679986 | 0.676784 | 0.919532 | 0.921353 |
| `(1,7,16,769)` / 默认 | 0.671561 | 0.667186 | 0.923803 | 0.900798 |
| `(1,1,128,1024)` / 默认 | 0.670231 | 0.663207 | 0.913565 | 0.907330 |
| `(2,2,128,1023)` / 默认 | 0.642629 | 0.662659 | 0.879369 | 0.894985 |
| `(10,3,86,769)` / 默认 | 0.667792 | 0.689367 | 0.887754 | 0.919649 |
| `(21,171,3,1024)` / 默认 | 0.760634 | 0.774599 | 0.884755 | 0.896055 |

- FP16：生成加四次前向 / CANN 四次为 0.524–0.916；metadata 生成中位数 198.090–252.885 μs；attention Kernel / CANN 为 0.938–1.097。
- BF16：生成加四次前向 / CANN 四次为 0.529–0.785；metadata 生成中位数 191.260–252.465 μs；attention Kernel / CANN 为 0.927–1.104。

逐例值、轮均值和全部原始样本保留在 `profile.json`、`wall-samples.json` 和 profiler CSV。PID/物理 2 号映射及采样前后独占快照见 `performance-device-proof.json`；源分区到无 FD 的映射见 `fd-schedule-mapping.json`。共享内存采集后逐文件校验并归档，原报告保留为 `profile.original.json`，当前报告只替换为持久 CSV 路径，未修改样本；归档确认后释放临时副本。

## 环境、失败记录与限制

- Python `/usr/local/python3.11.15/bin/python3` 3.11.15，torch `2.7.1+cpu`、torch_npu `2.7.1`；CANN `/usr/local/Ascend/cann-9.0.0-beta.2`，bisheng clang 15.0.5、`dav-2201`、`-O2`。CATLASS `769cd40a8716b28650b6bebb08db4834eea4462f`、固定 CANN `ca18aee610729ca5270a44e2d84c751bc687fbf0`、官方 FA3 `d15f1531a460ba456f41b01a774f33ab2db8febf`。CANN 既有改动、CATLASS 和 950 后端未改。最终设备健康 OK、ECC 全零、无设备进程，见 `final-device.json`。
- 首版仅筛除源 FD 候选，使最大新增用例只用 7 核，FP16 复用/R=1/R=4 为 1.109/1.253/1.412，未通过；首版 2,113 项功能检查、源码、构建和全部原始结果保存在 `initial-filter/`。现按完整 M 块重构端点和成本，保留该例的 18 核并行度；未修改设备计算、用例、采样次数或验收门槛。
- 最初 profiler 因可用空间不足 20 MB 停止设备采集，不能计为通过；主文件系统 `f_bavail=0`，共享内存约余 4.5 MB。核对 189 份临时文件与持久归档后仅释放临时副本，见 `shm-reclaim.json`。单例诊断的 R=1 和 CSV 检查仍失败，保留在 `diagnostic-fp16*`。随后完整双 dtype 采集时 CPU 最坏利用率 99.6%–100%、load 约 259 起，多组原有用例 R=1 失败，见 `initial-busy-host/`。这些失败均保留；最终在上述空闲预检后完整复测原矩阵。
- 首版精度首轮在 1200 秒时限退出，无断言失败；原始记录保留，完整重跑采用 8 个 CPU 线程和 2400 秒时限。接手环境首次检查因覆盖 PYTHONPATH 缺失 `tbe` 失败，source 官方环境并保留 PYTHONPATH 后通过；见 `initial-filter/`。
- 当前映射曾在物理 7 号卡上通过 64 项精度/多流检查后，于 compile 用例的输入初始化 `ZerosLike` 报 507034。其后 7 号卡有与本次测试不同的进程占用；原始诊断保留在 `initial-device7-timeout/`，不将该轮计为通过。改用空闲健康的物理 1 号 910B，复核环境并完成全部精度、功能及普通回归。
- 1 号卡设备错误首轮在 27 项通过后，BF16 implicit-short 于 CANN 初始化阶段超过 90 秒时限；单例复测通过后，完整重跑在 38 项通过后又于 BF16 m_end 初始化超时。两次日志均停在 `wait subprocess start response`，未产生 MLA 拒绝报告；同期出现其他进程，但不能仅据此确定根因。原始结果及 28/39 个进程组清理证据保留在 `initial-errors-device1-init/`、`second-errors-device1-init/`。换至空闲健康的物理 2 号 910B，保持原 90 秒子进程时限及 15 秒 Trap 断言，复测失败案例并完整重跑 42 项通过，见 `environment-device2*`、`error-device2-recheck*`、`errors-final*`。
- 使用非登录 shell、沙箱外构建/实机/sanitizer。提交前的 staged 质量检查记录为 `quality-final.log`；生成的 `kernel_meta`、`extra-info` 已归档，未提交构建产物。图捕获留给 M16，本轮 compile 为 eager backend/fullgraph；Trap 后不承诺继续复用执行上下文。

## 证据与复现

精确命令、环境、退出码、日志和 JUnit 为 `*-command.json`、`*-exit.json`、对应日志及 XML。源码/对象快照、依赖、综合核验和证据哈希见 `build-source-manifest.json`、`runtime-source-manifest.json`、`build-dependencies.json`、`acceptance.json`、`checksums.sha256`。旧轮次原始报告的路径按归档目录内 `archive-location.json` 映射。复测使用新的 artifact/JUnit 路径，避免覆盖证据。

```bash
source /usr/local/Ascend/cann-9.0.0-beta.2/set_env.sh
FLASH_ATTN_BUILD_VERSION=v3 FLASH_ATTN_BUILD_NPU=910 FLASH_ATTN_SKIP_SUBMODULE_INIT=1 python3 setup.py build_ext --inplace
TORCH_DEVICE_BACKEND_AUTOLOAD=0 MLA_SCHEDULE_SANITIZE=1 MLA_CONSUMER_SANITIZE=1 python3 -m pytest -x -q tests/test_flash_attn_npu_v3_mla_schedule.py tests/test_flash_attn_npu_v3_mla_metadata_consumer.py tests/test_flash_attn_npu_v3_mla_build.py
ASCEND_RT_VISIBLE_DEVICES=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 python3 -m pytest -x -q tests/test_flash_attn_npu_v3_mla_kvblocks.py
PYTHONPATH=/tmp/mla-m01-precommit PRE_COMMIT_HOME=/tmp/mla-m01-precommit-cache bash ci/code_quality.sh --staged
python3 build/mla-m04c2-kv/verify_acceptance.py
git diff --check
git diff --cached --check
```

当前闭环提交用 `git log -1 -- docs/mla/status.md` 定位。
