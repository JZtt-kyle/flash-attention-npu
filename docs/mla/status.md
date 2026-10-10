# MLA 当前状态

最近闭环为 **M06c3c：公开只读混合布局 kvcache，内部 v4 与显式复用**。分支为 `feat/mla-cann-migration`，验收日期 **2026-10-10**。唯一下一步为 **[M06d1：实际 Q/KV 使用长度 CPU 归一与调度](plan.md#下一步m06d1)**，尚未开始。范围与契约见 [design](design.md)，当前闭环提交用 `git log -1 -- docs/mla/status.md` 定位。

## 当前可调用能力

共同边界为 **910B C1V2、FP16/BF16、512+64 共享 KV、Hkv=1、连续布局**，允许非零 storage offset。前向拒绝梯度，Q 使用完整分配段。BSND 每样本 KV 容量及紧凑 TND 的 `max_seqlen_k` 为 0..16384；紧凑 TND 总 Q/KV 各不超过 `INT32_MAX`，总 KV 可以超过 16384。混合记录的 BSND 总容量为 64 位 `B*Sk`。其他字段、地址及资源界限见 [design 6.2](design.md#62-调度字段与边界)。

| 入口或条件 | 当前实现 |
| --- | --- |
| `flash_attn_func` | BSND 整容量前向，内部生成 v1 `int32[401]` |
| `flash_attn_varlen_func` | 紧凑 TND Q/KV，连续同设备 `int32[B+1]` 前缀；每次内部生成并消费 v3 `int32[405+4B]`，max 参数为上界，前缀值在设备校验 |
| `flash_attn_with_kvcache`，BSND Q | 只读 BSND；`cache_seqlens` 接受 None、`0..Sk` 整数或同设备连续 `int32[B]`。无长度且无记录时走 v1，否则走 v2；显式 v2 逐项核对当次长度 |
| `flash_attn_with_kvcache`，TND Q | 已接通 TND Q/qv + BSND K/V，必须给出同设备连续 `int32[B+1]` Q 前缀及 `max_seqlen_q` 上界。KV 长度接受 None、整数或 tensor；内部生成 v4，或消费显式 `int32[406+5B]`，逐项核对当次 Q 前缀和 KV 长度 |
| `get_scheduler_metadata` | 无 Q 前缀时生成 v2；有 `cu_seqlens_q` 时生成混合 v4。两者都要求设备连续 `int32[B]` KV 长度，不接受 None/整数便利输入 |
| split 与选项 | 仅 `num_splits=1`；func/varlen 默认 1，kvcache/metadata 默认 0，须显式传 1。无 mask、`softcap=0`、`pack_gqa=None`、`sm_margin=0`、`attention_chunk=0` |
| scale 与返回 | scale 缺省 `1/sqrt(576)`，显式值须在 FP32 有限范围；记录不固化 scale。O 与输入同 dtype，LSE 为 FP32；BSND O/LSE 为 `[B,Sq,Hq,512]` / `[B,Hq,Sq]`，TND 为 `[Tq,Hq,512]` / `[Hq,Tq]`。func/varlen 请求 LSE 返回二项，kvcache 返回四项，未 split 的 accumulator 为同设备空 tensor |
| 空容量与空行 | 空 batch/query 保留空 shape，仍校验记录和前缀。非零容量中的零 KV 样本 `O=0、LSE=-inf`，全局空 KV 为 `O=0、LSE=+inf` |

**仍拒绝** used、非连续外层 stride、PA、causal/window、自动或多 split、192/128 prefill，以及追加 KV、内置 rotary 等范围外组合。varlen 不增加显式 v3 复用参数；内部通用混合入口不接受 `max_seqlen_k`。v2/v4 可克隆或使用连续偏移视图；损坏/过期记录由设备拒绝。fake/compile 已覆盖，完整 graph replay 生命周期留在 M16。目标包含 910C，生产平台仍仅开放 910B。

## 本步实现边界

生产 [TndBsndForward](../../csrc/ascend910/flash_attn_npu_3/mla/mla_forward.cpp) 与公开 metadata 共用 `PrepareTndBsndMetadata` 和既有 `ComputeMlaTndBsndMetadataFromPrefix`。None/整数 KV 长度物化为设备向量，None 表示整容量；显式记录直接消费，不启动生成器。无前缀/长度 D2H、`.item()` 或稳态主机同步，复用双事件及 allocator stream 登记。

Q 总量来自 Q shape，batch 来自物理 cache，不以矩形 `B*maxQ*Hq` 限制紧凑 Q。host 只按静态容量预算 workspace；非空容量全核发射，空 batch/query/KV 为单校验核与零 workspace。新增 FP16/BF16 [生产 launcher](../../csrc/ascend910/flash_attn_npu_3/mla/mla_tnd_bsnd_launch_impl.hpp) 在 TPipe、同步和写输出前校验完整 v4 与当前长度，再调用既有 `MlaTndBsnd`。64 位 `b*Sk` 物理 KV 起点及 Kernel / Block / Tile 数值路径不变，短 KV 不重排后续 cache。

| 布局 / 协议 | 独立链路 | 公开入口 |
| --- | --- | --- |
| BSND Q/KV，v1/v2 | 已验收 | 上述 BSND 子集 |
| 紧凑 TND Q/KV，v3 | 已验收 | 上述 varlen 子集 |
| TND Q + BSND KV，v4 | 已验收 | 上述混合 metadata 与 kvcache 子集 |

## 最近闭环证据

归档为 **`build/mla-m06c3c-kvcache/`**，总索引 `acceptance.json`；`*-command.json` / `*-exit.json` 保存完整命令、环境及退出码。接手核验上一步 2,002 个归档文件、229 个生产源码依赖和 22 个对象；最小复查为 88 项公开 metadata 和 1 项 sanitizer 缓冲检查，均通过。源码与实际加载扩展一致，无未解决的前步失败。

| 验收面 | 当前结果 | 相对证据路径 |
| --- | --- | --- |
| 公开真实/fake/C++ 契约与生命周期 | 681 项通过；None/整数/tensor、空容量、offset、静态拒绝、64 位 fake 边界、更新/跨流/allocator/compile/别名 | `public-contracts.xml` |
| 公开双 dtype 源/数学精度 | 120 项通过；内部生成、克隆记录、偏移视图，默认/零/负 scale、poison padding、输入只读及空行 | `public-accuracy.xml`、`mixed-attention/` |
| 真实 Trap | 完整 42 项通过，最长 0.152 秒；非法当前长度、损坏/过期记录、None/整数的过期记录及空容量 | `public-traps-final.xml`、`public-traps-final/` |
| CPU v4 生成与消费者 | 183 项通过，ASan/UBSan/LeakSanitizer；固定源差分、字段/缓冲/64 位边界、记录及当前长度校验 | `cpu-consumer.xml`、`cpu-source/`、`cpu-consumer/` |
| 普通 FA3 与已有 MLA | 1,031 项通过；普通 metadata/compile、官方签名、BSND 精度/生命周期及 dense dispatcher | `regressions.xml`、`regression-nodes.json` |
| 混合 metadata 与紧凑 TND 回归 | 616 项通过，含 580 项公开 v4 和 36 项紧凑 TND 源精度/生命周期最小检查 | `metadata-regressions.xml`、`metadata-regression-nodes.json` |
| 构建 | 13 项通过；实例、依赖、AICPU 隔离及 sanitizer。24 个链接对象与加载扩展审计通过 | `build-tests.xml`、`production-build-manifest.json` |
| 原生固定 CANN 分派 | 双 dtype 共 24 例精度及 runtime tiling key 核验通过，保留非 FD、FD、空 KV 分类 | `native-dispatch-proof.json`、`native-verified-fp16/`、`native-verified-bf16/` |

合计 **2,686 项去重 pytest 通过**，接手复查和首轮重复用例不重复计数。初次 host 构建因包含设备注解失败，将未改变字段的 context POD 移到普通 metadata 头后重建通过。原生测试包首次缺空 KV 分派 key，补编固定原始源码并核验加载库/二进制；打包权限、配置工具添加 `filePath` 后的哈希及日志 marker 交错问题均已处理，原始失败日志保留。首轮 Trap 在第 32 例已捕获真实错误，但 torch_npu 的事件清理导致进程 abort；沿用既有故障驱动，在保存并校验错误后退出独立进程，完整 42 项重跑通过。该修正不改变生产错误处理。

## 公开 E2E 验收

固定十二例、双 dtype、预热 5 次、5 轮各 20 个 wall 样本，显式复用、公开生成加 R=1/4、逐次内部生成分别计时，另采 Kernel、metadata 生成和转换耗时。驱动为 [mla_mixed_profile.py](../../tests/mla_mixed_profile.py)，命令为归档 `performance-{fp16,bf16}-final-command.json`。构建及验证进程退出后，在健康空闲卡 6、CPU 168..175，同进程交替测量新实现与固定 CANN；同步至 O/LSE 完成。

固定 CANN 原生连续入口要求 Q/KV 布局一致；对照每次将 BSND K/V gather 为紧凑 TND，两次 gather 及分配均计入原生 E2E，索引/H2D/一次性视图准备在计时外，详见 [design 7.2](design.md#72-参考工具与对照边界)。固定矩阵非 FD 为 0/4/5/7/9/11，原生空 KV 为 10，全部七例纳入本步 E2E 门槛；FD 为 1/2/3/6/8，保留为能力差距。分类由采样前固定源调度和实际分派确认，不按耗时筛选。既有独立混合组件证据在 `build/mla-m06c2b2d-mixed-attention/`，不代替本步公开 E2E。

下表为适用七例的逐例中位数比值范围（FA3 / 含必要转换的原生 CANN）。两种 dtype、三种调用段、R=1/4 **均逐例满足 <=1.0**；包含一次生成的摊销值可由完整调用段除以 R 得到。

| dtype | 调用段 | R=1 E2E 比值 | R=4 E2E 比值 |
| --- | --- | --- | --- |
| FP16 | 预生成后复用 | 0.426..0.448 | 0.304..0.326 |
| FP16 | 一次公开生成 + R 次前向 | 0.603..0.640 | 0.360..0.384 |
| FP16 | R 次前向各自内部生成 | 0.516..0.590 | 0.426..0.490 |
| BF16 | 预生成后复用 | 0.425..0.442 | 0.309..0.328 |
| BF16 | 一次公开生成 + R 次前向 | 0.610..0.642 | 0.366..0.391 |
| BF16 | R 次前向各自内部生成 | 0.534..0.594 | 0.443..0.491 |

十二例均保留，原生 FD 长 KV 例 3 仍有差距：包含一次生成的 R=1 比值 FP16 为 1.210、BF16 为 1.220；逐次内部生成分别为 1.241、1.236。该分派尚未迁移，按既定口径单列，未修改门槛或提前开放 split。

生成 wall 为 FP16 199.860..215.825 µs、BF16 202.315..222.905 µs；对应 AICPU Kernel 为 19.000..33.471 µs / 18.830..33.351 µs。原生 KV 转换 wall 为 113.340..145.590 µs / 116.900..148.215 µs。新 attention Kernel 与原生 FIA Kernel 的逐例值、转换设备时间及全部调用段见 `performance-{fp16,bf16}-final/summary.json`，不可用这些诊断值替代公开 E2E。

最终两组共 24,000 个 wall 样本、3,600 次带标签的 profiler 调用和 5,800 条 Kernel 记录。CANN `op_summary` 与 `kernel_details` 的数量、名称、设备、起始时间和耗时逐行一致；逐调用核对完整 Kernel 序列，显式复用不含生成器。首轮 FP16 的 240 条 `Step Id` 为空导致汇总失败，但 2,900 条原始数据完整；按每次同步后的时间顺序、名称和数量核验 1,800 次调用后，修正汇总并完整重跑。原始数据及匹配当时 SHA256 的脚本保留在 `performance-fp16/`，最终核验见 `performance-acceptance.json`。

## 环境与构建

| 项目 | 本步核验 |
| --- | --- |
| Python / PyTorch | `/usr/local/python3.11.15/bin/python3` 3.11.15；torch `2.7.1+cpu`、torch_npu `2.7.1` |
| CANN / 编译器 | `/usr/local/Ascend/cann-9.0.0-beta.2`；bisheng clang 15.0.5 |
| 芯片 | 八张 Ascend910B3；卡 7 公开/Trap，卡 3 源精度，卡 4 普通回归，卡 6 原生分派及性能。沙箱内 NPU 初始化不可用，设备/构建及完整 sanitizer 在沙箱外执行 |
| CATLASS | `769cd40a8716b28650b6bebb08db4834eea4462f`，工作区干净 |
| 只读 CANN 源仓 | `ca18aee610729ca5270a44e2d84c751bc687fbf0`，原有 67 条删除/未跟踪记录未改变 |
| 生产重建 | 24 个对象；`flash_api.o`、`mla_forward.o`、`mla_tnd_bsnd_metadata.o` 改变，新增两种 dtype 的 mixed launcher；其余 19 个哈希不变。237 个生产源码依赖核验通过 |
| 实际加载扩展 | `flash_attn_npu_3/flash_attn_npu_3.cpython-311-aarch64-linux-gnu.so`；SHA256 `b962d57031c342264ea0a8ba0f1aacb879db1229980931aecd52bac76f8f371d` |
| 固定原生 CANN 包 | 本步 `vendor/mla_ca18aee_transformer/`；原有固定 host/opAPI 库不变，双 dtype TND device binary 补齐固定源空张量 key；`native-package.json` 和 `native-dispatch-proof.json` 核对实际加载路径与哈希 |

## 最小复现与交接

完整选择和环境见归档 `*-command.json` / `*-nodes.json`。下列仅作最小复查；先检查健康空闲设备，设置 `MLA_DEVICE` 并使用新目录。

```bash
MLA_SCHEDULE_SANITIZE=1 ASAN_OPTIONS=detect_leaks=1 UBSAN_OPTIONS=halt_on_error=1 python3 -m pytest -x -q tests/test_flash_attn_npu_v3_mla_tnd_bsnd_metadata.py -k buffer_guards --basetemp=build/mla-mixed-kvcache-cpu-recheck
FLASH_ATTN_BUILD_VERSION=v3 FLASH_ATTN_BUILD_NPU=910 python3 setup.py build_ext --inplace
ASCEND_RT_VISIBLE_DEVICES=${MLA_DEVICE:?set MLA_DEVICE} OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 MLA_TND_BSND_METADATA_DEVICE_DIR=build/mla-mixed-kvcache-producer-recheck MLA_TND_BSND_ATTENTION_DIR=build/mla-mixed-kvcache-attention-recheck python3 -m pytest -x -q tests/test_flash_attn_npu_v3_mla_mixed_public.py -k 'entries_empty_and_returns or none_integer_lengths or updates_streams_allocator_and_aliases or (source_accuracy and None and spec0)' --basetemp=build/mla-mixed-kvcache-public-recheck
```

质量命令为 `PYTHONPATH=/tmp/mla-m01-precommit:$PYTHONPATH PRE_COMMIT_HOME=/tmp/mla-m01-precommit-cache ci/code_quality.sh --staged`，保留 CANN PYTHONPATH。提交前检查源码/扩展/对象、证据总索引、文档链接/锚点、`git diff --check` 及质量钩子；只暂存本步文件，保留 `kernel_meta/` 未跟踪产物。下次先核验本步证据与最小命令，再推进唯一下一步 M06d1。
