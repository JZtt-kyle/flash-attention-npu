# MLA 当前状态

最近实现闭环为 **M06c2b1：紧凑 TND 设备寻址与独立 attention 驱动**；分支 `feat/mla-cann-migration`。公开能力仍为 **M06b4：BSND 实际 KV 长度**。

唯一下一步为 **M06c2b2：TND Q + BSND KV 设备寻址与独立驱动**，本轮未开始。M06c2b 已按布局拆小，完成条件见 [plan 的下一步](plan.md#下一步m06c2b2)；在已确认范围内推进，无需重复确认。契约与迁移范围查 [design](design.md)。

## 当前可调用能力

| 入口或条件 | 当前行为 |
| --- | --- |
| 共同输入边界 | 910B C1V2、FP16/BF16、512+64 共享 KV、Hkv=1；连续 BSND，允许非零 storage offset；B/Sq/Sk 可为零，Hq>0，单样本 KV 容量 `Sk=0..16384`；Q 使用整容量，支持 S1G/S2 多块 |
| `flash_attn_func` | 整容量前向，内部生成 v1 `int32[401]`；没有公开长度或显式 metadata 参数 |
| `flash_attn_with_kvcache` | 只读 BSND；`cache_seqlens` 接受 None（整容量）、`0..Sk` 整数或同设备连续 `int32[B]`，支持短/非均匀/混合与全零长度；无长度、无显式记录时走 v1，否则走 v2 |
| `get_scheduler_metadata` | 64/512 分支生成 v2 `int32[401+B]`，必须传同设备连续 `int32[B]` 的 `cache_seqlens`；不接受 kvcache 的 None/整数便利输入。真实/fake/C++ 外形一致 |
| 显式记录复用 | 只读 kvcache 接受 v2；复核当次长度，省略长度表示整容量。旧 v1、损坏或过期记录拒绝；记录可克隆或用连续偏移视图复用，长度变化须重新生成 |
| 空行与返回 | 非零容量中的零 KV 样本 `O=0、LSE=-inf`，含端点外零尾部；全局空 KV 为 `O=0、LSE=+inf`。空 batch/query 保留空 shape，仍校验记录。kvcache 请求 LSE 返回四项，未 split 的两个 accumulator 为空 tensor |
| 调度选项 | 必须显式 `num_splits=1`；无 mask、`softcap=0`、`pack_gqa=None`、`sm_margin=0`。scale 由当次前向提供；静态界限见 [design 6.2.4](design.md#624-平台与溢出)，v2 另要求 `B<=2^32-1-401` |

尚未接通的公开能力：TND 前向、Q-used、非连续外层 stride、PA、causal/window、自动或多 split、192/128 prefill 及其他调度选项。梯度、追加 KV、内置 rotary 等范围外组合继续明确拒绝。完整 graph replay 生命周期留在 M16；fake/compile 通过不代表 replay 已验收。目标仍包含 910C，当前生产平台只开放 910B；合成 C1V1 调度不代表 910C 实机验收。

v2 在 TPipe、attention 事件和输出写入前完成设备校验；非空容量使用平台全核 blockDim，工作核数来自设备记录，workspace 按最大 M 基块 256 预算。长度与记录无 D2H，沿用双事件与 allocator stream 生命周期。核验入口：[Python 校验与包装](../../flash_attn_npu_3/flash_attn_npu_interface.py)、[生产生成与发射](../../csrc/ascend910/flash_attn_npu_3/mla/mla_forward.cpp)、[设备消费](../../csrc/ascend910/flash_attn_npu_3/mla/mla_launch_impl.hpp)。

## 紧凑 TND 独立链路

[`kernel/mla_tnd.hpp`](../../csrc/ascend910/flash_attn_npu_3/mla/kernel/mla_tnd.hpp) 接受独立 TND 调度类型，与 BSND 复用 `MlaNonPaged` 任务环及 Cube/Vector 数值路径。每个任务保存 Q/KV token 起点和实际段长，支持可变 M 块与跨样本流水；Tile 使用绝对 token 起点，O 为 `[Tq,Hq,512]`，LSE 直接写 `[Hq,Tq]`。保留 64 位寻址及超大 LSE 头跨度的逐头写回分支。

v3 仍为 `int32[405+4B]`，生成、当前前缀校验和生命周期沿用前步协议。独立 [attention 驱动](../../tests/mla_tnd_attention_probe.cpp) 在 TPipe、同步和 O/LSE/workspace 写入前校验；计算只消费设备记录，传入无效 host 调度也不影响迁移端结果。零 Q/KV 样本跳过计算，零成本尾部照常初始化；全局空 KV 为 `O=0、LSE=+inf`，非零总容量中的零 KV 行为 `O=0、LSE=-inf`。

**已验收的是紧凑 TND 独立 attention，公开 varlen/kvcache TND 仍拒绝。** 混合 BSND KV、used、外层 stride 尚未接通；v1/v2 不接受 v3。固定源独立驱动用数值构造的 int64 累计尾值，不重解释 FA3 int32 前缀，不读取迁移端起点快照。

## 本轮闭环证据

证据目录 **`build/mla-m06c2b1-tnd-attention/`**，总索引 `acceptance.json`。各 `*-command.json` / `*-exit.json` 保存命令、环境、退出码；JUnit、构建依赖、库哈希、原始异常、源码快照和性能原始数据均在该目录，未覆盖上轮归档。

| 验收面 | 结果 | 相对证据路径 |
| --- | --- | --- |
| 交接与 CPU 消费者 | 上轮 2,829 份证据、218 个源码依赖及 19 个对象哈希复核；CPU 消费者 100 项通过 | `precheck.json`、`precheck-cpu.xml` |
| 910B 紧凑 TND | 73 项通过；双 dtype T01、多 query、跨样本尾块、零成本后缀、总空容量、offset、总 KV 超过 16K、scale、记录复用/重生成及跨流；独立双 reference 与非空容量固定源逐元素对照 | `device-final.xml`、`attention-final/`、`producer/`、`device-final-artifacts/` |
| 真实错误上报 | 包含于上述 73 项：26 个独立进程 release Trap，最长约 52 ms；损坏/过期记录、失败生成及空容量拒绝均先验证不写输出/workspace | `attention-final/*/trap-*/` |
| BSND 精度与 v2 设备消费 | 147 项通过：attention 93、设备消费者 54；双 dtype 短/混合/全零/长 KV、源对照、生命周期及旧协议拒绝 | `bsnd-regression.xml`、`bsnd-attention/`、`bsnd-consumer/` |
| 普通 FA3 与公开 BSND | 923 项通过；普通 metadata/compile/签名、MLA 契约与公开 BSND | `regressions.xml`、`regression-nodes.json` |
| 生产 v1 前向 | 208 项通过；双 dtype 源/数学精度、M/KV 边界、头次序、内容/RoPE、scale 与流生命周期 | `v1-forward.xml`、`v1-artifacts/` |
| 构建与 sanitizer | 9 项通过，保留完整 ASan/UBSan/LeakSanitizer；新扩展及实际依赖核验通过 | `build-tests.xml`、`production-build-manifest.json` |

合计 **1,460 项去重 pytest 通过**，最终选入项无失败或跳过。BSND 套件显式排除了 6 项既有 attention runtime_rejection，不计为通过；本轮新增实际 TND attention Trap 有独立完整证据。

## 性能与限制

- **TND 源组件**：FP16/BF16 各 10 个预定代表用例，卡 7、CPU 亲和性 176..183；迁移/源 Kernel 比值分别为 **0.927–1.658** / **0.896–1.597**。最大相对差异来自零计算量的小用例，FP16 为 6.07 / 3.66 µs；不能据此宣称全例不劣化。见 `tnd-performance-fp16/summary.json` 与 `tnd-performance-bf16/summary.json`。
- **BSND 源组件回归**：双 dtype 各 12 个既定用例，卡 4、CPU 亲和性 168..175；Kernel 比值分别为 **0.999–1.424** / **0.997–1.441**，见 `bsnd-performance-*/summary.json`。
- 两组均先验精度、交替预热 5 次、5 轮各 20 个 wall 样本与 10 个 profiler 样本。TND 每 dtype 1,000 条、BSND 每 dtype 1,200 条 kernel 记录与原始 `op_summary` 逐项核对名称、时间和耗时。Profiler framework flow/DB 解析仍报错，完整 trace 不记为通过；原始错误保留，仅采用已核对的 kernel CSV。
- 以上为预分配组件对比，未计入公开入口完整生成/分配路径；公开 TND 与公开 E2E 留给后续接入。本轮未重测公开 BSND metadata E2E，最近公开闭环仍为 `276b7dc` 的 `build/mla-m06b4-public/performance-acceptance.json`。910C 仍未实机验收。

## 环境、构建与已处理失败

Python `/usr/local/python3.11.15/bin/python3` 3.11.15、torch `2.7.1+cpu`、torch_npu `2.7.1`、CANN `/usr/local/Ascend/cann-9.0.0-beta.2`、bisheng clang 15.0.5；八张 Ascend910B3。本轮卡 7 跑 TND、卡 4 跑 BSND、卡 5 跑普通回归、卡 3 跑生产 v1；使用前已核验健康与空闲。沙箱 NPU 查询仍返回 `-8005`，设备及完整 sanitizer 检查在沙箱外完成。

CATLASS 固定为 `769cd40a8716b28650b6bebb08db4834eea4462f` 且干净；CANN 源仓仍为 `ca18aee610729ca5270a44e2d84c751bc687fbf0`，原有 67 条删除/未跟踪记录保持不变。FA3/910 定向构建只改变两个 MLA BSND 实例，其余 17 个链接对象哈希不变；旧/新产物保存在 `production-before/`、`production-after/`。实际加载扩展为 `flash_attn_npu_3/flash_attn_npu_3.cpython-311-aarch64-linux-gnu.so`，SHA256 **`469dc8fed68708b0b09d290d2c8b14e8d2cf9dae25c01194f8b0261735a60147`**。TND 独立库另有 device 依赖与哈希。本轮生成的 `kernel_meta` 已归档到 `kernel-meta-artifacts/`。

初次源驱动漏设 TND `outputLayout`，以及全局空 KV 初始化缺少 MTE3→Vector 依赖，分别在 `initial-device.log` / `device-accuracy.log` 留有失败；修复后完整 73 项通过。默认全后端构建已停止，定向构建枚举由 `3` 修正为 `v3`；证据脚本与标准库 `profile` 同名导致的加载失败也已修正，见对应构建和 audit 日志。格式工具自动改写后重新检查通过，所有失败/中止记录保留，不计为通过。

## 最小复现与交接

新验收使用新目录，先确认设备健康、空闲并设置 `MLA_DEVICE`。完整测试选择与性能命令以本轮 `*-command.json` 为准。

```bash
FLASH_ATTN_BUILD_VERSION=v3 FLASH_ATTN_BUILD_NPU=910 python3 setup.py build_ext --inplace
ASCEND_RT_VISIBLE_DEVICES=${MLA_DEVICE:?set MLA_DEVICE} OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 MLA_TND_ATTENTION_DIR=build/mla-tnd-attention-recheck MLA_TND_METADATA_DEVICE_DIR=build/mla-tnd-producer-recheck MLA_TND_SCHEDULE_ARTIFACT_DIR=build/mla-tnd-oracle-recheck python3 -m pytest -x -q tests/test_flash_attn_npu_v3_mla_tnd_attention.py
ASCEND_RT_VISIBLE_DEVICES=${MLA_DEVICE:?set MLA_DEVICE} OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 python3 -m pytest -x -q tests/test_flash_attn_npu_v3_metadata_signature.py tests/test_flash_attn_npu_v3_mla_bsnd_public.py
```

质量命令为 `PYTHONPATH=/tmp/mla-m01-precommit:$PYTHONPATH PRE_COMMIT_HOME=/tmp/mla-m01-precommit-cache ci/code_quality.sh --staged`，保留 CANN PYTHONPATH。本步提交可用 `git log -1 -- docs/mla/status.md` 定位；旧归档验收脚本绑定各自基准和暂存列表，不直接作为新提交验收脚本。
