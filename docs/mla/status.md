# MLA 当前状态

当前闭环为 **M06d2b：used 记录设备校验消费者**，2026-10-10 基于 `feat/mla-cann-migration` 的 `6140228` 完成。唯一下一步为 **[M06d2c：used 的独立 Kernel 链路](plan.md#下一步m06d2c)**，尚未开始。本步提交用 `git log -1 -- docs/mla/status.md` 定位。

目标契约与实现结构见 [design](design.md)，步骤及完成条件见 [plan](plan.md)。下表只描述当前可调用能力，独立校验组件不代表公开 used 已开放。

## 当前可调用能力

共同边界为 **910B C1V2、FP16/BF16、512+64 共享 KV、Hkv=1、连续布局、无梯度**；允许非零 storage offset。Q 使用完整分配段。BSND 每样本 KV 容量与紧凑 TND 的 `max_seqlen_k` 均为 0..16384；紧凑 TND 总 Q/KV 各不超过 `INT32_MAX`，总 KV 可以超过 16384。混合 BSND cache 总容量为 64 位 `B*Sk`。完整字段、地址和资源界限见 [design 6.2](design.md#62-调度字段与边界)。

| 入口 | 输入与长度 | 调度记录 |
| --- | --- | --- |
| `flash_attn_func` | BSND 整容量 | 内部 v1：`int32[401]` |
| `flash_attn_varlen_func` | 紧凑 TND Q/KV；两组同设备连续 `int32[B+1]` 前缀，max 参数为每段上界 | 每次内部 v3：`int32[405+4B]`；无公开显式复用参数 |
| `flash_attn_with_kvcache`，BSND Q | 只读 BSND cache；KV 长度为 None、`0..Sk` 整数或同设备连续 `int32[B]` | 无长度且无记录走 v1；否则内部 v2 或显式 v2：`int32[401+B]` |
| `flash_attn_with_kvcache`，TND Q | TND Q/qv + BSND K/V；必须给出 Q 前缀和 `max_seqlen_q` 上界；KV 长度同上 | 内部 v4 或显式 v4：`int32[406+5B]` |
| `get_scheduler_metadata` | 必须给设备连续 `int32[B]` KV 长度；不接受 None/整数 | 无 Q 前缀生成 v2，有 Q 前缀生成混合 v4 |

- **调用选项：** 仅 `num_splits=1`；func/varlen 默认 1，kvcache/metadata 默认 0，须显式传 1。无 mask，`softcap=0`、`pack_gqa=None`、`sm_margin=0`、`attention_chunk=0`；metadata 的 `has_softcap=False`。scale 默认 `1/sqrt(576)`，显式值须在 FP32 有限范围，记录不固化 scale。
- **输出：** O 与输入同 dtype，LSE 为 FP32。BSND O/LSE 为 `[B,Sq,Hq,512]` / `[B,Hq,Sq]`，TND 为 `[Tq,Hq,512]` / `[Hq,Tq]`。请求 LSE 时 func/varlen 返回二项，kvcache 返回四项；未 split 的两个 accumulator 为同设备空 tensor。
- **记录校验：** 前缀/长度值留在设备检查；显式 v2/v4 逐项匹配当次长度，v4 还匹配 Q 前缀。记录可克隆或使用连续偏移视图，损坏/过期记录由消费者 Trap 拒绝。空输出仍校验记录和长度。
- **空值：** 空 batch/query 保留空 shape；非零容量中的零 KV 样本 `O=0、LSE=-inf`，全局空 KV 为 `O=0、LSE=+inf`。

**仍拒绝：** varlen `seqused_q/k`、Q-used、非连续外层 stride、PA、causal/window、自动或多 split、192/128 prefill，以及 KV 追加、内置 rotary 等范围外组合。已有 BSND cache 的实际 KV 长度通过 `cache_seqlens` 支持，内部通用前向用 `seqused_k` 传递。内部通用混合入口仍拒绝 `max_seqlen_k`。fake/compile 已覆盖，完整 graph replay 留在 M16；目标包含 910C，生产平台检查仍只开放 910B。

以上入口与拒绝边界对应 [Python 包装/fake](../../flash_attn_npu_3/flash_attn_npu_interface.py)、[C++ 通用入口](../../csrc/ascend910/flash_attn_npu_3/flash_api.cpp)、[专用前向](../../csrc/ascend910/flash_attn_npu_3/mla/mla_forward.cpp) 及 [静态校验](../../csrc/ascend910/flash_attn_npu_3/mla/mla_validation.h)；完整生产分派表见 [design 6.5](design.md#65-生产接入与构建)。

## 独立 used 生成与校验

v5 为 **`int32[410+8B]`、version/kind=5/5**，保存完整调度、两轴布局/总容量/max 上界、显式 used 标记，以及每样本的 64 位物理起点、分配长度和实际长度。三种布局均保留物理分配，缩短 used 不移动后续样本；协议见 [design 5.7](design.md#57-v5分配与-used-记录)。既有 AICPU 生成器直接读取设备前缀/used，失败清零或拒绝不安全写入的规则不变。

新 [`ReadUsedMetadata`](../../csrc/ascend910/flash_attn_npu_3/mla/mla_used_metadata_consumer.h) 使用独立 context、当次四数组描述符及返回结构。校验头部后核对布局、容量、used 标记、数组计数/对齐/地址范围和逐样本快照；空 batch/容量仍完整检查，显式空数组保留存在标记。端点按实际长度检查，零成本后缀逐样本复核；复用平台/workspace/局部资源检查，另检查未用端点、非 FD 记账及总 tile 溢出。`maxCost` 仅作诊断，不验证调度最优性或记录真实性。

[`LoadUsedMetadata`](../../csrc/ascend910/flash_attn_npu_3/mla/kernel/mla_used_metadata.hpp) 在每个 Cube/Vector 核上执行校验，失败真实 Trap，调用方在成功后才写输出或分配事件。CPU 解码失败不改写返回结构；release 设备诊断路径验证拒绝不写 O/LSE/workspace 哨兵。四组输入可只读别名，跨 stream 的事件依赖和 allocator 登记由独立 probe 验收；没有前缀/used D2H。

**当前仅独立生成与校验 probe 使用 v5，尚无 used attention 输出或公开接入。** 生成器已链接生产扩展，消费者未接入生产 launcher；公开能力仍为上表的 v1–v4，未迁移组合继续明确拒绝。

## 最近闭环证据

归档目录为 **`build/mla-m06d2b-used-consumer/`**，总索引 `acceptance.json`；`*-command.json` / `*-exit.json` 保存命令、环境与退出码，`source-snapshots/` 和 `production-final/` 保存最终源码及构建产物。验收共 **1,627 项去重 pytest 通过**，没有跳过项：

| 验收面 | 本次结果 | 证据（相对归档目录） |
| --- | --- | --- |
| 上一步最小复查 | CPU 6 项、910B AICPU 7 项；生成失败/更新/空值/别名/生命周期，ASan/UBSan/LSan | `baseline-cpu.xml`、`baseline-device.xml`、`baseline-cpu-driver-exits.json` |
| v5 CPU 消费者 | 140 项；三布局、U01/T01、缺省/显式 used、零容量/零 used、全部当前长度、损坏字段、计数/对齐、资源、端点、64 位起点、INT32 与 tile 溢出边界 | `cpu-final.xml`、`cpu-final/`、`cpu-final-driver-exits.json` |
| v5 910B release 消费者 | 68 项；双 dtype、源调度、所有 Cube/Vector 解码一致、拒绝不写哨兵、四数组更新、空值、偏移视图/别名/clone/stream/allocator、64 位起点、组件计时 | `device-verified.xml`、`device-verified/`、`device-producer-verified/` |
| 真实 Trap（包含在上述 68 项中） | 两 dtype 各 23 项；旧协议、布局/标记/跨度、四数组过期、错计数/对齐、端点/资源、生成失败与空容量；设备/stream/event 同步均取得 runtime Trap | `device-verified/error-*/runtime-error.json`、同目录 `command.json` / `process.log` |
| 固定源差分 | CPU 3,168 条：原生非 FD 2,622 / 禁止 split 映射 546；910B 900 条：718 / 182；完整字段、跨度和任务一致 | `differential-summary.json`、`cpu-final/source-*`、`device-verified/source-*` |
| v1–v4 metadata 回归 | 548 项；419 项 CPU 消费者及 129 项 AICPU 生成，含源字段、错误状态、空值和生命周期 | `legacy-metadata.xml`、`legacy-metadata-tmp/`、`legacy-metadata-driver-exits.json` |
| 普通 FA3 / 接口边界 | 839 项；普通 metadata、compile、固定官方签名、MLA 契约和未开放 used 拒绝 | `ordinary-regressions.xml`、`ordinary-nodes.json` |
| 构建 / 质量 | 19 项构建测试；新 probe 双 dtype 的真实设备依赖；生产增量重建、质量钩子和 diff 检查通过 | `build-tests.xml`、`device-verified/*-device.d`、`production-build-manifest.json`、`quality-staged.log` |

CPU 最终验收的四个持续驱动及旧 metadata 的十七个驱动均正常退出，ASan/UBSan/LSan 无报告。首轮 `device-smoke` 的 blockDim 拒绝测试同时改变了实际发射核数，导致哨兵检查按旧核数失败；已固定实际发射核数以检查独立 context 不匹配，完整补跑上述 68 项。失败轮次保留在 `device-smoke.*`，不计作通过；本步没有遗留失败。

校验组件双 dtype 各四例，预热 5 次、20 次测量，中位数 FP16 **196.87–226.05 μs**、BF16 **202.58–222.95 μs**。另在同卡同进程对匹配逻辑输入交替比较 v4/v5，预热 5 次、各 40 次测量：FP16 v4 **184.95–203.35 μs** / v5 **208.25–245.59 μs**，BF16 v4 **183.57–189.23 μs** / v5 **200.35–232.81 μs**；各例 v5 比 v4 增加 **16.78–47.75 μs**。原始样本见 `device-verified/timing-{1,2}.json` 和 `comparison-{1,2}.json`，包含 ctypes、stream 登记、诊断写出和同步，buffer 预先分配。该结果反映新增校验及 probe 开销，**不是 attention 或公开 E2E，也不声称不劣化**。未修改 attention Kernel / Block / Tile；最近公开混合 E2E 仍为 `build/mla-m06c3c-kvcache/performance-acceptance.json`，不作为 used 输出能力证据。

## 环境与构建

接手时核验上一闭环 **2,056 个归档文件、242 个源码依赖、25 个对象和扩展哈希**。沙箱内 Python/NPU 初始化未完成，开发前检查及构建/设备/完整 sanitizer 改按既有方式在沙箱外执行；未把沙箱尝试记为通过。CATLASS 和只读 CANN 源工作区前后一致。

| 项目 | 本次实测记录 |
| --- | --- |
| Python / PyTorch | `/usr/local/python3.11.15/bin/python3` 3.11.15；torch `2.7.1+cpu`、torch_npu `2.7.1` |
| CANN / 编译器 | `/usr/local/Ascend/cann-9.0.0-beta.2`；bisheng clang 15.0.5 |
| 芯片 | 八张 Ascend910B3；v5 用 7，普通 FA3 用 4，旧 metadata 用 5 |
| CATLASS / 只读 CANN 源 | `769cd40a8716b28650b6bebb08db4834eea4462f` / `ca18aee610729ca5270a44e2d84c751bc687fbf0`；CATLASS 干净，CANN 原有 67 条删除/未跟踪记录保留 |
| 生产重建 | 共享状态枚举追加 `InvalidUsedLengths`，既有值不变；六个 MLA 实例按依赖重编译，25 个对象内容哈希均不变；242 个源码依赖归档 |
| 实际加载扩展 | `flash_attn_npu_3/flash_attn_npu_3.cpython-311-aarch64-linux-gnu.so`；SHA256 `278e9ed1be9b2214d57a506b7d0c8f16d760becc1599ec8cf4c3806f595d3dd8`；与本次链接产物一致 |

## 最小复现与交接

完整命令见归档的 `cpu-final-command.json`、`device-verified-command.json`、`legacy-metadata-command.json`、`ordinary-regressions-command.json` 和 `build-tests-command.json`。下列命令使用新的输出目录；实机前检查健康空闲卡并设置 `MLA_DEVICE`，Trap 必须使用独立子进程：

```bash
MLA_CONSUMER_SANITIZE=1 MLA_SCHEDULE_SANITIZE=1 ASAN_OPTIONS=detect_leaks=1 UBSAN_OPTIONS=halt_on_error=1 MLA_USED_CONSUMER_CPU_DIR=build/mla-m06d2b-used-consumer/recheck-cpu MLA_USED_METADATA_CPU_DIR=build/mla-m06d2b-used-consumer/recheck-builder MLA_USED_SCHEDULE_ARTIFACT_DIR=build/mla-m06d2b-used-consumer/recheck-source MLA_PERSISTENT_LABEL=recheck-cpu python3 build/mla-m06d2b-used-consumer/pytest_persistent.py -x -q tests/test_flash_attn_npu_v3_mla_used_metadata_consumer.py -k 'u01_t01 or descriptors_headers or endpoint_holes or allocation_bounds' --basetemp=build/mla-m06d2b-used-consumer/recheck-cpu-tmp

ASCEND_RT_VISIBLE_DEVICES=${MLA_DEVICE:?set MLA_DEVICE} OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 MLA_USED_CONSUMER_DEVICE_DIR=build/mla-m06d2b-used-consumer/recheck-device MLA_USED_METADATA_DEVICE_DIR=build/mla-m06d2b-used-consumer/recheck-producer python3 -m pytest -x -q tests/test_flash_attn_npu_v3_mla_used_metadata_device.py -k 'u01_t01 or each_current or cross_stream or (runtime_trap and stale_uq)' --basetemp=build/mla-m06d2b-used-consumer/recheck-device-tmp
```

源码/构建变化时先重建：`FLASH_ATTN_BUILD_VERSION=v3 FLASH_ATTN_BUILD_NPU=910 python3 setup.py build_ext --inplace`。质量命令为 `PYTHONPATH=/tmp/mla-m01-precommit:$PYTHONPATH PRE_COMMIT_HOME=/tmp/mla-m01-precommit-cache ci/code_quality.sh --staged`，保留 CANN PYTHONPATH。只暂存本步文件，保留 `kernel_meta/` 未跟踪产物。

下次先核验源码、构建与运行环境，按受影响范围复查后推进 M06d2c；不要把 v5 校验通过视为 used attention 已可执行。910C 和完整 graph replay 仍未实机验收。
