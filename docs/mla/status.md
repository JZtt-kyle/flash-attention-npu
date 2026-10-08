# MLA 当前状态

实现完成 **M06b2b**，公开能力仍为 M04c4 加 M05 的整容量 dense 前向与 metadata 复用。本次基于分支 `feat/mla-cann-migration` 的 `26cd650`，接手时工作区干净；实现 v2 设备校验，复用 v1 的形状/平台/调度/资源校验，并完成旧路径回归。

唯一下一步为 **M06b3：按样本实际 KV 长度执行的 BSND Kernel / Block / Tile，独立驱动**。范围与完成条件见 [plan 的 M06 表](plan.md#m06非分页长度tnd-与-stride)：双 dtype 短 KV、跨 S2 块、混合/全零长度、端点外空行和 poison padding 的精度，以及源组件性能和 dense 回归。公开短长度接入留在 b4；后续在已确认范围内推进，无需重复确认。

## 当前支持边界

| 项目 | 代码实际行为 |
| --- | --- |
| 设备与表示 | 生产只开放 **910B C1V2**；FP16/BF16、512+64 共享 KV、Hkv=1。目标仍含 910C，当前 C1V1 合成调度检查不代表 910C 前向已接通 |
| 输入 | 连续 BSND，允许连续视图的非零 storage offset；B/Sq/Sk 可为零，Hq>0，KV 容量 `0..16384`；S1G 已支持多块，不再限于 256 行；字段/资源上界见 [design 6.2](design.md#62-调度字段与边界) |
| 前向入口 | `flash_attn_func` 内部生成 v1；`flash_attn_with_kvcache` 支持只读 dense 的内部生成或显式复用。必须显式 `num_splits=1`，无 mask，`softcap=0`、`pack_gqa=None`、`sm_margin=0`；scale 由本次前向提供 |
| 公开长度与记录 | `get_scheduler_metadata` 的 64/512 分支返回连续 `int32[401]` v1，要求设备 `int32[B]` 长度逐项等于 KV 容量。只读前向接受 None、等于容量的整数或同设备连续 int32 tensor；消费时重新检查当次长度 |
| 空容量与返回 | 空 batch/query 保留空 shape；全局空 KV 为 `O=0、LSE=+inf`，仍校验记录/长度。dense 请求 LSE 返回二元组，kvcache 返回四项，未 split 的两个 accumulator 为空 tensor |
| 独立组件 | `BuildBsndKvLengthSchedule`、v2 `int32[401+B]` 序列化/AICPU 生成及设备校验消费者已验收；消费者限 C1V2。v2 生成已编入扩展，生成/校验由独立驱动发射，**尚未接实际长度 attention 或公开短长度前向** |
| 尚未开放 | 实际短长度/非均匀长度、TND、非连续外层 stride、PA、causal/window、自动或多 split、192/128 prefill、softcap/其他调度选项、梯度。追加 KV、内置 rotary 等范围外选项同样明确拒绝；完整图 replay 生命周期留在 M16 |

核验入口为 [Python 真实/fake 检查](../../flash_attn_npu_3/flash_attn_npu_interface.py)、[C++ 边界](../../csrc/ascend910/flash_attn_npu_3/mla/mla_validation.h)、[生产生成/发射](../../csrc/ascend910/flash_attn_npu_3/mla/mla_forward.cpp) 与 [设备校验](../../csrc/ascend910/flash_attn_npu_3/mla/kernel/mla_metadata.hpp)。v1 生成遇到非法设备长度时写失败状态并清零 payload；前向消费失败记录或发现当次长度非法才触发 Trap，生成调用本身不立即报该内容错误。

## 最近闭环与证据

证据目录 **`build/mla-m06b2b-consumer/`**，索引 `acceptance.json`。原始日志、JUnit、命令/退出码、固定源清单、编译依赖和 profiler 数据均保存在该忽略目录，未随 Git 分发；交接到其他环境时需一并保留。历史结果用 Git 中对应版本的 status 查看，不计作本次通过。

新增 [`ReadBsndKvLengthMetadata` / `LoadBsndKvLengthMetadata`](../../csrc/ascend910/flash_attn_npu_3/mla/kernel/mla_bsnd_metadata.hpp)：按静态 B 核对精确字数、v2 头部和生成状态，逐项检查快照范围及与当次长度一致，再校验调度和资源。缺省长度表示整容量；全局空容量同样检查长度。省略尾部必须从样本边界开始，且每个省略样本均为零 KV。失败在 release 构建中真实 Trap，发生在任何 attention 输出/workspace 写入之前；诊断读取失败不改写输出调度。协议详见 [design 5.3](design.md#53-v2bsnd-实际-kv-长度记录与消费者)。

| 验收面 | 本次结果 | 证据（相对上述目录） |
| --- | --- | --- |
| CPU v2 消费者及 v1 回归 | 208 项通过，含 ASan/UBSan、缓冲/字数/64 位边界、损坏记录、逐项过期长度、空容量及零尾部；3,240 条固定源记录（2,330 原生非 FD、910 无 split 映射）消费一致 | `cpu-host.xml`、`cpu/`、`v1-cpu/`、`cpu-oracle/` |
| 旧 v2 生成协议及构建 | 47 项通过，含 sanitizer、v1/v2 拒绝、AICPU 依赖；6,484 条固定源记录（4,660 原生非 FD、1,824 映射），含 4 条零尾部 | `cpu-regression.xml`、`generator-regression/`、`generator-oracle/` |
| 910B v2 消费者 | 双 dtype 共 54 项通过；每个 Cube/Vector 解码一致，损坏/过期/缺省长度、零尾部洞、实际 blockDim/独立 context、拒绝不写缓冲、克隆/偏移视图/跨流/allocator 生命周期正确 | `device-full.xml`（52）、`device-extra.xml` 中 2 项；`device-final/`、`device-extra/` |
| v2 真实错误 | 上述 54 项内含 22 个独立进程真实 Trap；覆盖头部/字数、快照/过期/缺省/计数、非零及不完整样本尾部、workspace、生成失败及空 Q 长度变更；设备/stream/event 同步，最长约 0.077 秒 | `device-final/error-*/runtime-error.json` |
| 旧 dense / 普通 FA3 | 812 项通过；含普通 FA3 metadata/compile 20 项，生产消费者双 dtype 的固定源 O/LSE 与独立数学 reference 24 项 | `regressions.xml`、`regression-nodes.json`、`dense-regression/` |
| 公开短长度拒绝 | 双 dtype 的显式生成后消费、内部生成后消费、复用前向三种路径共 6 项真实 Trap 通过 | `device-extra.xml`、`public-short-errors/` |

合计 **1,127 项去重 pytest 通过，无失败/跳过**，含 **28 项真实 Trap**；按范围未选入的旧错误测试不计为覆盖。v2 probe 只验证记录，不计算 attention；本次没有短长度前向精度或公开 E2E 验收。

相关既有 dense 组件性能：双 dtype 各 7 组（固定 CASES 五组，加 `Sk=8193` 的 D01 和 `Sk=16384,Hq=128`），同轮交替比较生产 v1 消费者与固定 CANN 非 FD Kernel，相同调度、预分配缓冲。O/LSE 逐元素源一致且独立 reference 通过；Kernel 中位数比值 FP16 为 **0.992–1.056**，BF16 为 **0.987–1.008**。FP16 D02 为 20.81 / 19.71 微秒（约高 5.6%），保留原始结果，未宣称逐例不劣化。证据为 `performance/{fp16,bf16}/summary.json`、`kernel-samples.json`、`wall.json` 和 `profiler/`；这是组件调用，不是公开 metadata E2E。profiler 的框架 ACL flow / `fwk_api_db` 解析报错，未记为通过；每 dtype 的 700 条原始 CANN `op_summary` 与 `kernel_details` 的名称、设备、起点和耗时逐项一致，Kernel 统计有效。

最近公开 E2E 闭环仍为 M04c4（`85ce13d`）的 `build/mla-m04c4-empty/acceptance.json`；本次未重测公开 E2E。完整矩阵及比较路径见 [design 7.1](design.md#71-固定-dense-矩阵与阶段边界)。

初始失败保留在 `cpu-initial.*` / `cpu-sandbox/`（沙箱 ptrace 阻止 LeakSanitizer）、`device-build-initial/` / `probe-build*.log`（设备不支持初始化列表、依赖路径未归一化）及 `device-initial.*`（重生成后测试驱动沿用旧 blockDim，被消费者正确 Trap）。分别在可运行 sanitizer 的环境重跑、修正探针/依赖检查、固定全核发射后完整通过。格式检查曾重写文件并碰到运行时缓存；最终只检查本步文件。构建脚本格式化后重新构建的两个消费者二进制与完整设备验收时逐字节一致，见 `formatted-probe-equivalence.json`；不存在以旧源码产物冒充当前验证的情况。

## 环境与复现

Python `/usr/local/python3.11.15/bin/python3` 3.11.15、torch `2.7.1+cpu`、torch_npu `2.7.1`、CANN `/usr/local/Ascend/cann-9.0.0-beta.2`、bisheng clang 15.0.5、g++ 11.4.0。接手核实 8 张 Ascend910B3 健康，设备 5/6 空闲；v2/异常测试使用 5，生产回归/性能使用 6，CPU 亲和性分别为 8–15 / 16–23。CANN 原 PYTHONPATH 保留；沙箱内驱动初始化失败，实机与完整 sanitizer 使用沙箱外执行。

CATLASS 固定 `769cd40a8716b28650b6bebb08db4834eea4462f`，子模块干净；CANN 固定 `ca18aee610729ca5270a44e2d84c751bc687fbf0`，原有 67 条删除/未跟踪记录未变且未修改。环境详情 `environment.json`。

本次增量构建重编双 dtype MLA 分派和 v2 AICPU 对象；其余对象哈希未变。18 个链接对象、210 个源码/依赖哈希及双 dtype 设备依赖已核对。实际加载扩展仍为 `flash_attn_npu_3/flash_attn_npu_3.cpython-311-aarch64-linux-gnu.so`，新 SHA256 为 `b5264bed17331ccc10207de144d0249ad03e286bf050466533fc1626ce4310d8`；详见 `production-build-manifest.json`。

最小消费者检查如下；用新的 `MLA_BSND_CONSUMER_CPU_DIR`、`MLA_BSND_CONSUMER_DEVICE_DIR`、`MLA_BSND_METADATA_CPU_DIR` / `MLA_BSND_METADATA_DEVICE_DIR` 指定证据目录，避免覆盖旧产物。设备号重跑前按占用调整；全部确切命令、环境和退出码见本目录的 `*-command.json` / `*-exit.json`。

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 MLA_CONSUMER_SANITIZE=1 MLA_SCHEDULE_SANITIZE=1 python3 -m pytest -x -q tests/test_flash_attn_npu_v3_mla_bsnd_metadata_consumer.py tests/test_flash_attn_npu_v3_mla_metadata_consumer.py
ASCEND_RT_VISIBLE_DEVICES=5 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 taskset -c 8-15 python3 -m pytest -x -q tests/test_flash_attn_npu_v3_mla_bsnd_metadata_device.py
FLASH_ATTN_BUILD_VERSION=v3 FLASH_ATTN_BUILD_NPU=910 FLASH_ATTN_SKIP_SUBMODULE_INIT=1 python3 setup.py build_ext --inplace
```

本次完整回归、组件性能及证据核验脚本为 `regressions.py`、`performance.py`、`audit.py`、`check_docs.py`（位于上述证据目录）。60 个本地链接/锚点、18 张表、51 个步骤状态和官方公开签名检查通过；`git diff --check` 通过，仓库质量钩子结果见 `quality-staged.log`。本次闭环提交可用 `git log -1 -- docs/mla/status.md` 定位。
