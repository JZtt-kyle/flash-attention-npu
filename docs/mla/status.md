# MLA 当前状态

公开能力停在 **M06b4：BSND 实际 KV 长度**；最近实现闭环为 **M06c2b2d：混合布局设备寻址与独立 attention 驱动**。分支为 `feat/mla-cann-migration`，验收日期 **2026-10-10**。

本步完成 TND Q + BSND KV 的独立 attention，公开 TND/混合入口仍拒绝。唯一下一步为 **[M06c3：公开 varlen 及 TND query 只读 kvcache / metadata](plan.md#下一步m06c3)**，尚未开始；实施前按每提交一个可验收小步拆分。目标契约与迁移范围见 [design](design.md)。当前闭环提交用 `git log -1 -- docs/mla/status.md` 定位。

## 当前可调用能力

以下均指公开 MLA 入口。共同边界为 **910B C1V2、FP16/BF16、512+64 共享 KV、Hkv=1、连续 BSND**，允许非零 storage offset。Q 使用整容量，支持 S1G/S2 多块；B/Sq/Sk 可为零，Hq>0，单样本 KV 容量 `Sk=0..16384`。其他字段、地址及资源上界见 [design 6.2.4](design.md#624-平台与溢出)。

| 入口或条件 | 当前行为 |
| --- | --- |
| `flash_attn_func` | 整容量前向，内部生成 v1 `int32[401]`；没有公开长度或显式 metadata 参数 |
| `flash_attn_with_kvcache` | 只读 BSND；`cache_seqlens` 接受 None（整容量）、`0..Sk` 整数或同设备连续 `int32[B]`，支持短/非均匀/全零长度。无长度且无显式记录时走 v1，否则走 v2 |
| `get_scheduler_metadata` | 64/512 分支生成 v2 `int32[401+B]`；必须传同设备连续 `int32[B]` 的 `cache_seqlens`，不接受 None/整数。真实/fake/C++ 外形一致 |
| 显式记录复用 | 只读 kvcache 只接受 v2，逐项复核当次长度；省略长度表示整容量。旧 v1、损坏或过期记录拒绝。可克隆或用连续偏移视图复用；长度变化须重新生成，v2 要求 `B<=2^32-1-401` |
| split 与选项 | 仅支持 `num_splits=1`；`flash_attn_func` 默认即为 1，kvcache/metadata 生成器默认是 0，须显式传 1。无 mask、`softcap=0`、`pack_gqa=None`、`sm_margin=0`、`attention_chunk=0` |
| scale | 当次前向提供，缺省 `1/sqrt(576)`，显式值须在 FP32 有限范围内；记录不固化 scale |
| 输出与返回 | O 与输入同 dtype，LSE 为 FP32 `[B,Hq,Sq]`；func 请求 LSE 返回二项，kvcache 请求 LSE 返回四项，未 split 的两个 accumulator 为同设备空 tensor |
| 空行与空容量 | 非零容量中的零 KV 样本 `O=0、LSE=-inf`，含端点外零尾部；全局空 KV 为 `O=0、LSE=+inf`。空 batch/query 保留空 shape，仍校验记录 |

**公开入口仍拒绝** TND 前向、Q-used、非连续外层 stride、PA、causal/window、自动或多 split、192/128 prefill 及其他未接通选项；梯度、追加 KV、内置 rotary 等范围外组合也明确拒绝。fake/compile 已有覆盖，完整 graph replay 生命周期留在 M16。目标仍包含 910C，生产平台当前只开放 910B；合成 C1V1 调度不代表 910C 实机验收。

核验入口：[Python 校验与包装](../../flash_attn_npu_3/flash_attn_npu_interface.py)、[C++ 输入边界](../../csrc/ascend910/flash_attn_npu_3/mla/mla_validation.h)、[生产生成与发射](../../csrc/ascend910/flash_attn_npu_3/mla/mla_forward.cpp)、[设备消费](../../csrc/ascend910/flash_attn_npu_3/mla/mla_launch_impl.hpp)。v2 在 TPipe、attention 事件与输出写入前校验；非空容量按平台全核发射，工作核数来自设备记录，workspace 按最大 M 基块 256 预算。长度与记录无 D2H，沿用双事件和 allocator stream 生命周期。

## 独立组件边界

“已验收”仅指该列的组件；生产构建包含生成器不等于公开入口能调用。

| 布局 / 协议 | CPU 调度 | AICPU 生成 | 设备校验 | 独立 attention | 公开入口 |
| --- | --- | --- | --- | --- | --- |
| BSND Q/KV，v1/v2 | 已验收 | 已验收 | 已验收 | 已验收 | 已开放上述子集 |
| 紧凑 TND Q/KV，v3 | 已验收 | 已验收 | 已验收 | 已验收（M06c2b1） | 拒绝，待 M06c3 |
| TND Q + BSND KV，v4 | 已验收 | 已验收 | 已验收（M06c2b2c） | 已验收（M06c2b2d） | 拒绝，待 M06c3 |

紧凑 TND 复用与 BSND 相同的 `MlaNonPaged` Kernel / Block / Tile 数值路径，其 O 为 `[Tq,Hq,512]`，LSE 为 `[Hq,Tq]`，计算前校验当次前缀。混合布局现已复用同一数值路径：先校验 v4、Q 前缀和 KV 实际长度，再读取五字 span；KV 起点固定为 64 位 `b*Sk` 并贯穿任务环，短 KV 不重排后续 cache。O/LSE 采用同一 TND 布局；非空物理容量中的全零 KV 为 `O=0、LSE=-inf`，空物理容量为 `O=0、LSE=+inf`。协议字段与失败行为见 [v3](design.md#54-v3紧凑-tnd-记录与设备消费)、[v4](design.md#55-v4tnd-q-与-bsnd-kv-混合记录)，源码和独立驱动见 [架构表](design.md#6-代码结构)。

## 最近闭环证据

归档为 **`build/mla-m06c2b2d-mixed-attention/`**，总索引 `acceptance.json`；`*-command.json` / `*-exit.json` 保存完整命令、环境与退出码。接手时先核对上一步文档/实现归档及固定官方 FA3 快照，未发现哈希或未解决失败差异；本节均为本步新运行结果。

| 验收面 | 结果 | 相对证据路径 |
| --- | --- | --- |
| CPU 调度与 v4 消费者 | 220 项通过；完整 ASan/UBSan/LeakSanitizer | `cpu-sanitizer.xml`、`cpu-artifacts/` |
| 910B 混合 attention | 95 项通过；FP16/BF16、固定源/双 reference、变长/跨块/空容量、poison/offset、记录更新/复用/跨流与 allocator 生命周期 | `mixed-attention.xml`、`attention/`、`producer/`、`oracle/` |
| 既有 TND/BSND 与 metadata | 204 项通过；双 dtype attention 与设备消费者；已有 Trap 重复项未选，本步混合 Trap 单独验收 | `layouts.xml`、`tnd-attention/`、`bsnd-attention/`、`layout-artifacts/` |
| 普通 FA3 与公开 BSND | 923 项通过；普通 metadata/compile/签名、MLA 契约与公开 BSND | `regressions.xml`、`regression-nodes.json` |
| 构建 | 13 项通过；AICPU 隔离、依赖与 sanitizer；生产重建及加载路径/对象核验通过 | `build-tests.xml`、`production-build-manifest.json` |

合计 **1,455 项去重 pytest 通过**。混合 FP16/BF16 共 **36 个独立进程真实 Trap**，最长报错 **0.050 秒**；每例先用同一错误的诊断调用确认 O/LSE/workspace 哨兵不变，再走实际 attention launcher 的 release Trap。所有 core 均在 TPipe、attention 事件和写入前校验记录，未使用 D2H 长度或 CANN 算子替代迁移计算。

固定源同调度 O 与非空行 LSE 逐元素一致，零 KV 行另按源 `+inf` / FA3 `-inf` 检查；独立数学 reference 按逻辑 Q 段和物理 `b*Sk` 切片。源 probe 的 Q=TND、KV=BSH（物理 BSND），其 arch22 KV 解析器使用测试侧数值构造的 int64 累计尾值，源 header 不改；CPU 调度 oracle 仍用非累计 KV 长度。适配依据见 [design 7.2](design.md#72-参考工具与对照边界)。

## 组件性能与边界

同一空闲健康 910B 卡 3、CPU 168..175；每例预热 5 次，5 轮交替采样，每轮双方各 20 个独立调用 wall 样本及 10 个 profiler Kernel 样本。双方使用相同调度和预分配缓冲。完整矩阵、Kernel 名、原始 `op_summary` / `kernel_details`、调用段样本与设备状态保存在下列目录，未删选慢例。

| 布局 | 每 dtype 用例数 | FP16 Kernel 比值范围 | BF16 Kernel 比值范围 | 相对证据目录 |
| --- | --- | --- | --- | --- |
| 混合 | 12 | 0.900..1.670 | 0.922..1.656 | `mixed-performance-fp16/`、`mixed-performance-bf16/` |
| 既有 BSND | 12 | 0.999..1.423 | 1.003..1.418 | `bsnd-performance-fp16/`、`bsnd-performance-bf16/` |
| 既有 TND | 10 | 0.907..1.585 | 0.916..1.573 | `tnd-performance-fp16/`、`tnd-performance-bf16/` |

比值为迁移 Kernel / 固定源 Kernel。混合两组小型零计算用例约 1.36..1.67 倍，包含完整 v4 校验的成本；本步只完成组件性能对照，**不宣称公开 metadata E2E 不劣化**，公开 TND/混合入口须在 M06c3 另行验收。已有公开 BSND E2E 历史证据仍在 `build/mla-m06b4-public/performance-acceptance.json`，本步未重测该公开调用段。

独立 ctypes 发射的 profiler 仍报告缺少 ACL→NPU flow / `fwk_api_db` 解析失败；这是此次未取得的 framework 关联信息，未记为通过。用于本步计时的原始 `op_summary` 与 `kernel_details` 在条数、名称、设备、起始时间和耗时上逐行核对一致，总计 6,800 个 Kernel、13,600 个 wall 样本。该限制及原日志均保留，不用缺失的 flow 信息证明公开 E2E。

## 环境与构建

| 项目 | 本步核验 |
| --- | --- |
| Python / PyTorch | `/usr/local/python3.11.15/bin/python3` 3.11.15；torch `2.7.1+cpu`、torch_npu `2.7.1` |
| CANN / 编译器 | `/usr/local/Ascend/cann-9.0.0-beta.2`；bisheng clang 15.0.5 |
| 芯片 | 八张 Ascend910B3；卡 7 混合 attention，卡 6 既有布局，卡 4 普通/公开回归，卡 3 性能；沙箱内驱动查询失败，设备与完整 sanitizer 在沙箱外执行 |
| CATLASS | `769cd40a8716b28650b6bebb08db4834eea4462f`，工作区干净 |
| 只读 CANN 源仓 | `ca18aee610729ca5270a44e2d84c751bc687fbf0`，原有 67 条删除/未跟踪记录未改变 |
| 生产重建 | 20 个链接对象中仅 FP16/BF16 两个 MLA dispatcher 改变；其余 18 个哈希不变，223 个生产源码依赖已核验；前后产物及重建后依赖文件归档 |
| 实际加载扩展 | `flash_attn_npu_3/flash_attn_npu_3.cpython-311-aarch64-linux-gnu.so`；SHA256 `32d1a2cc34b0e95dcb7f3ae044af0b21bd812c0311777f880a5de395259d7249` |

构建审计初版错误沿用 host launcher 对象名；对照实际 `.device.d` 修正为两个 device dispatcher 后复查通过，初次失败保留在 `build-audit.log`，最终结果见 `build-audit-final.log`。格式钩子首次改写后已复核并通过。64 位 KV 起点贯穿设备寻址，但没有分配多 TB cache 做超 uint32 token 的实算；上一步的大跨度 metadata 校验不能代替该实算。910C 未实机验收。

## 最小复现与交接

完整验收选择、环境及性能命令以本目录 `*-command.json` 为准。以下是最小重查，使用新目录，先确认健康空闲设备并设置 `MLA_DEVICE`；不能把最小重查计作完整 1,455 项验收。

```bash
MLA_CONSUMER_SANITIZE=1 MLA_SCHEDULE_SANITIZE=1 ASAN_OPTIONS=detect_leaks=1 UBSAN_OPTIONS=halt_on_error=1 python3 -m pytest -x -q tests/test_flash_attn_npu_v3_mla_tnd_bsnd_metadata_consumer.py --basetemp=build/mla-mixed-attention-cpu-recheck
FLASH_ATTN_BUILD_VERSION=v3 FLASH_ATTN_BUILD_NPU=910 python3 setup.py build_ext --inplace
ASCEND_RT_VISIBLE_DEVICES=${MLA_DEVICE:?set MLA_DEVICE} OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 MLA_TND_BSND_ATTENTION_DIR=build/mla-mixed-attention-recheck MLA_TND_BSND_METADATA_DEVICE_DIR=build/mla-mixed-attention-producer-recheck MLA_TND_BSND_SCHEDULE_ARTIFACT_DIR=build/mla-mixed-attention-oracle-recheck python3 -m pytest -x -q tests/test_flash_attn_npu_v3_mla_tnd_bsnd_attention.py tests/test_flash_attn_npu_v3_metadata_signature.py tests/test_flash_attn_npu_v3_mla_bsnd_public.py
```

组件性能入口为 `python3 -m tests.mla_tnd_bsnd_profile <attention/fp16 或 bf16> <producer/build-manifest.json> <新输出目录>`。驱动按 manifest 核验源码与库哈希，不能用旧 `.so` 代替当前构建。

质量命令为 `PYTHONPATH=/tmp/mla-m01-precommit:$PYTHONPATH PRE_COMMIT_HOME=/tmp/mla-m01-precommit-cache ci/code_quality.sh --staged`，保留 CANN PYTHONPATH；`git diff --check`、文档链接/锚点与暂存钩子通过。下一次接手先检查环境与产物差异，再按 plan 推进唯一下一步。
