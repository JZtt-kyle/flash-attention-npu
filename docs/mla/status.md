# MLA 当前状态

本轮完成 **M06e2c2：混合 stride 生产接入**（2026-10-10），基线 `1e46ff5`，分支 `feat/mla-cann-migration`。公开混合只读 kvcache v4 与内部混合 Q-used v5 已支持合法外层 stride，M06 完成。唯一下一步为 **[M07：ND paged cache](plan.md#下一步m07)**，先核对固定源 PA 边界并拆为可独立验收的小步。

本页维护当前能力、证据、环境差异和下一步；目标契约与架构见 [design](design.md)，步骤与完成条件见 [plan](plan.md)。当前闭环提交可用 `git log -1 -- docs/mla/status.md` 定位，功能闭环提交见下方证据表。

## 当前可调用能力

共同边界为 **910B C1V2、FP16/BF16、512+64 共享 KV、Hkv=1、无梯度**。Q/K 末维为 64，qv/V 为 512；四路输入须同 dtype、同 NPU 设备。BSND 每样本 KV 容量与 TND 的 `max_seqlen_k` 均为 0..16384；TND 总 Q/KV 各不超过 `INT32_MAX`，总 KV 可以超过 16384。其他字段、地址与资源限制见 design [6.2](design.md#62-调度字段与边界)、[6.3](design.md#63-长度与布局调度)、[6.4](design.md#64-输入-stride-与地址校验)。

### 布局与 stride

| 输入布局 | 生产支持 | 独立组件边界 |
| --- | --- | --- |
| BSND Q/KV | 四路末维连续、外层非负 stride；含 BNSD、切片、广播、重叠、独立 storage offset；完整 storage/字节界限在发射前校验 | CPU 校验、设备 attention 和生产接入均完成，覆盖 v1/v2/v5 |
| TND Q/KV | 四路末维连续、外层非负 stride；含 NTD、切片、广播、重叠、独立 storage offset；完整 storage/字节界限在发射前校验 | CPU、独立 attention 和生产 v3/v5 接入均完成 |
| TND Q + BSND KV | 四路末维连续、外层非负 stride；Q/qv 使用 TND 描述，K/V 各自使用 BSND 描述；完整 storage/字节界限在发射前校验 | CPU、独立 attention 和生产 v4/v5 接入均完成；v4 metadata 可跨合法视图复用 |

stride 与 storage offset 不进入 metadata；`data_ptr()` 已包含 offset。CPU 地址成立、metadata 能表达布局、生产入口可调用是不同层次。

### 公开入口

长度 tensor 均须同设备、连续、int32；B 为 batch，分配前缀为 `[B+1]`，实际长度为 `[B]`。公开显式生成/复用只开放 v2/v4，精确字数见 [metadata 协议](design.md#5-metadata-设计)。

| 入口 | 输入与长度 | metadata 路径 |
| --- | --- | --- |
| `flash_attn_func` | BSND Q/KV，使用整容量 | 每次内部 v1 |
| `flash_attn_varlen_func` | TND Q/KV，两组分配前缀；max 参数是每段分配上界；可选 `seqused_q/k` | 任一 used 存在时每次内部 v5，否则 v3；无显式复用参数 |
| `flash_attn_with_kvcache`，BSND Q | Q 整容量，只读 BSND cache；`cache_seqlens` 为 None（整容量）、`0..Sk` 整数或长度 tensor | 无长度且无记录走 v1，否则内部 v2 或显式 v2 |
| `flash_attn_with_kvcache`，TND Q | Q 使用完整分配段，须给 Q 前缀和 `max_seqlen_q`；只读 BSND cache，KV 长度同上 | 内部 v4 或显式 v4 |
| `get_scheduler_metadata` | 必须给 KV 长度 tensor，不接受 None/整数；MLA 维度为 `headdim=64, headdim_v=512` | 无 Q 前缀生成 v2，有 Q 前缀生成混合 v4 |

### 内部入口

以下为 `_flash_attn_forward` / C++ `fwd`，不增加公共签名。三种布局的 used 分支共用 `UsedForward`、v5 生成/校验与 `MlaUsed`。

| 输入 | 记录选择与边界 |
| --- | --- |
| BSND Q/KV | 有 `seqused_q` 时每次内部 v5，可选 `seqused_k`；无 Q-used 沿用 v1/v2；Q-used 与显式 metadata 组合拒绝 |
| TND Q/KV | 前缀、max、可选 used 及 v3/v5 选择与公开 varlen 相同；拒绝显式 metadata |
| TND Q + BSND KV | 无 Q-used 沿用 v4；有 `seqused_q` 时每次内部 v5，可选 `seqused_k`，复用 TND used launcher；Q-used 与显式 metadata 组合拒绝，混合入口始终拒绝 `max_seqlen_k` |

核验入口为 [Python 包装/fake](../../flash_attn_npu_3/flash_attn_npu_interface.py)、[C++ 通用分派](../../csrc/ascend910/flash_attn_npu_3/flash_api.cpp)、[专用前向](../../csrc/ascend910/flash_attn_npu_3/mla/mla_forward.cpp) 和 [静态校验](../../csrc/ascend910/flash_attn_npu_3/mla/mla_validation.h)；launcher 与 Kernel 映射见 [design 第 6 节](design.md#6-代码结构)。

### 调用约束与输出

- **选项：** 只支持 `num_splits=1`。func/varlen/内部前向默认 1；kvcache/metadata 默认 0，调用时须显式传 1。仅无 mask、`softcap=0`、`pack_gqa=None`、`sm_margin=0`、`attention_chunk=0`；metadata 的 `has_softcap=False`，不接受数值 `softcap/softmax_scale` 关键字。前向 scale 默认 `1/sqrt(576)`，显式值须在 FP32 有限范围，记录不固化 scale。
- **输出：** O 与输入同 dtype，LSE 为 FP32。BSND O/LSE 为 `[B,Sq,Hq,512]` / `[B,Hq,Sq]`，TND 为 `[Tq,Hq,512]` / `[Hq,Tq]`。请求 LSE 时 func/varlen 返回二项，kvcache 返回四项；后两项是同设备空 accumulator，dtype 随 PyTorch 默认浮点 dtype。
- **长度与复用：** 长度值在设备检查，不做 D2H。显式 v2/v4 须匹配当次 KV 长度，v4 还匹配 Q 前缀；损坏/过期记录由消费者 Trap 拒绝，空输出仍校验。v5 完整检查两轴分配与 used，即使另一轴 used 为零；缺省 used 使用整段，缩短 used 不移动后续样本起点。生命周期见 [design 5.7](design.md#57-stream-与内存生命周期)。
- **空值与有效区：** 空 batch/query 保留空 shape；非零容量中的零 KV 样本有效行 `O=0、LSE=-inf`；全局空 KV 初始化整个输出分配区为 `O=0、LSE=+inf`。其他未使用 query 槽位不承诺输出值。

**尚未开放：** PA、causal/window、自动或多 split、192/128 prefill、非零 softcap 与非默认调度选项。KV 追加、内置 rotary 等范围外组合也明确拒绝，完整范围见 [design 第 4 节](design.md#4-功能边界)。fake/compile 已有覆盖，完整 graph replay 留在 M16；910C 尚未开放或实机验收。

## 最近闭环与证据

本轮证据位于忽略目录 `build/mla-m06e2c2-mixed-stride-production/`，路径相对仓库根目录；源码快照、构建产物、原始性能数据和哈希保存在该目录，不随 Git 分发。`acceptance.json` 汇总验收，当前闭环提交可用 `git log -1 -- docs/mla/status.md` 定位。

| 检查 | 结果 | 证据入口 |
| --- | --- | --- |
| 混合 stride 生产，双 dtype 910B | 448 项通过：公开 v4／内部 v5／raw C++／op／fake／compile，NTD／四路 offset／广播／重叠，65535/65536/65537、24 项真实 >uint32 地址、scale、完整 storage、跨视图 metadata、长度更新、空值、流生命周期及 20 项真实 Trap | `production.xml`、`production-tests/accuracy.jsonl`、`production-tests/*/trap-*/` |
| 连续 TND／混合、混合 metadata 与 BSND stride 回归 | 2,684 项通过；132 项既有进程级 Trap 未选入，本步相关混合 Trap 另验，不计为通过或跳过 | `affected-final.xml`、`affected-final-command.json`、`run_affected.py` |
| 普通 FA3／接口／metadata／compile | 833 项通过 | `ordinary-regressions.xml`、`run_regressions.py` |
| 已开放 TND stride 与独立 TND／混合组件 | 当前构建 694 项通过，其中生产 374、独立 320；14 项 TND Trap 未重复选入。前置原扩展完整最小闭环另复跑 388 项，不重复计数 | `stride-regressions.xml`、`baseline.xml` |
| 构建与加载来源 | 255 份源码依赖、31 个对象；仅混合 FP16/BF16 launcher、MLA 前向与公共 C++ 分派共 4 个对象改变，其余 27 个哈希未变；新 probe 从当前源码和固定 CANN Git blob 构建 | `production-build-manifest.json`、`production-dependencies/`、`production-final/`、`probes/*/build-manifest.json` |
| 原生分派与公开 E2E | 双 dtype 各 11 例；8 例适用、3 例原生 FD 单列；复用、生成一次加 R=1/4、内部生成，共 96 个适用调用段逐例达标 | `native-package.json`、`native-dispatch-proof.json`、`performance-acceptance.json` |
| 质量、文档与证据核验 | 适用质量钩子、链接/锚点/表格、`git diff --check`、源码/对象/库/运行证据哈希通过 | `quality-staged.log`、`docs-check.json`、`environment-final.json`、`acceptance.json` |

共 **4,659 个去重 pytest 用例通过，无失败或跳过**；前置复跑与未选入的 Trap 不计入该数字。精度同时对照固定源和既有双数学 reference，未改精度阈值。CPU 地址实现、Kernel/Block/Tile 数值与同步流程未改；未重跑 ASan/UBSan，最近地址组件闭环仍为 M06e2a `2a8cd38` 的 `build/mla-m06e2a-tnd-layout/sanitized/report.json`。

### 性能与限制

生产组件矩阵双 dtype 各 6 例，固定源非 FD、相同逻辑输入/调度及 20 核发射。源连续物化和 Q-used 压紧在计时前完成，迁移端直接读原 stride 视图；生产内部 wall 含 metadata 生成，源 probe 预分配，不能作为公开 E2E。每 dtype 预热 5 次、5 轮交替采样，保留 600 个 Kernel 样本及 1,200 个 wall 样本。

公开矩阵双 dtype 各 11 例，双方接收相同物理 tensor；原生每次从 BSND cache gather 紧凑 KV，Q/qv 的原生布局转换也计入 CANN E2E。分别采集 metadata 生成、显式复用、生成一次加 R=1/4、逐次内部生成及转换调用；两端包含 O/LSE 完成同步。CPU 亲和性固定 128–135，预热 5 次、5 轮各 20 个 wall 样本，每 dtype 保留 11,000 个 wall 样本和 1,650 个 profiler 调用样本。设备 CSV 与原始 CANN `op_summary` 的名称、设备、开始时间、耗时逐项一致；attention 与转换设备耗时单列。

| dtype | 8 个适用公开用例各调用段的 FA3 / CANN E2E | 6 例生产 / 固定源 Kernel 范围 | Kernel 慢于源端 |
| --- | --- | --- | --- |
| FP16 | 0.284–0.623，逐例均 ≤1.0 | 0.898–1.888 | 4/6 |
| BF16 | 0.284–0.675，逐例均 ≤1.0 | 0.901–2.458 | 4/6 |

公开证据为 `public-fp16/`、`public-bf16/`，组件为 `component-fp16/`、`component-bf16/`。**组件仍有退化，不能宣称 Kernel 全部不劣化。** 原生 FD 三例按固定源调度和实际 tiling key 判定，完整数据保留；其差距留待 M10，没有按耗时筛选或放宽门槛。组件 profiler 仍报告 `fwk_api_db` 解析错误；设备样本核验通过，不据此声称 host 关联可用。

BSND 公开 E2E 最近闭环仍为 M06e1c `4e83d32` 的 `build/mla-m06e1c-bsnd-stride/performance-acceptance.json`；TND Q/KV 公开 E2E 为 M06e2c1 `1e46ff5` 的 `build/mla-m06e2c1-tnd-stride-production/performance-acceptance.json`。本轮没有重复采样这两套矩阵。

## 环境与构建

`environment.json` 核验 Python `/usr/local/python3.11.15/bin/python3` 3.11.15、torch `2.7.1+cpu`、torch_npu `2.7.1`、CANN `/usr/local/Ascend/cann-9.0.0-beta.2`、bisheng clang 15.0.5、g++ 11.4、8 张健康 Ascend910B3。构建与 NPU 验证在沙箱外执行。前置/生产/Trap 使用物理 7 号卡，受影响回归使用 5 号卡与 CPU 0–15，普通 FA3、TND/独立回归及随后性能使用 6 号卡；性能双方同卡、同亲和性。各命令保存实际环境变量，设备健康与占用由初始环境及性能前后快照记录。

生产构建设置 `FLASH_ATTN_SKIP_SUBMODULE_INIT=1 FLASH_ATTN_BUILD_VERSION=v3 FLASH_ATTN_BUILD_NPU=910`。实际加载 `flash_attn_npu_3/flash_attn_npu_3.cpython-311-aarch64-linux-gnu.so`，SHA256 为 `38582e6e4deb2eed122a38e5c1da06cd6d5386b4d00bc405f21e1cf469a4a00e`；加载库与链接产物一致。混合 v4 按四路连续性选择连续或 stride 实例，混合 v5 复用既有 TND used launcher；metadata 字节协议不变。

额外进程查询及独立环境重载的自动审批超时，均未执行；沙箱内额外 `npu-smi` 查询也以 187 退出（dcmi 初始化 -8005），不记为通过。最终 `environment-final.json` 采用已成功完成的双 dtype 公开采样进程环境和末次设备快照，另在沙箱内复核当前源码、对象、扩展及源仓状态；这不是单独再加载 NPU 扩展。全部必要实机验收已有对应成功进程的证据。

CATLASS `769cd40a8716b28650b6bebb08db4834eea4462f` 干净；只读 CANN 源 `ca18aee610729ca5270a44e2d84c751bc687fbf0` 的已有差异及本仓未跟踪 `kernel_meta/` 均保留。910C 尚未实机验收，完整 graph replay 仍留在 M16。

## 唯一下一步与复现

下一步仅为 **[M07](plan.md#下一步m07)**：固定源 ND paged cache 边界核对与实施拆分；本轮提交后停止，不开始 PA 实现。再次开发前重新核查设备健康、占用与扩展来源，输出使用新目录保留本轮证据。

```bash
ASCEND_RT_VISIBLE_DEVICES=7 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
MLA_TND_STRIDE_DIR=build/mla-m06e2c2-recheck/probes \
MLA_TND_STRIDE_PRODUCER_DIR=build/mla-m06e2c2-recheck/producer \
MLA_MIXED_STRIDE_DIR=build/mla-m06e2c2-recheck/production \
python3 -m pytest -x -q tests/test_flash_attn_npu_v3_mla_mixed_stride_production.py
```

该命令默认重建 probe；仅在源码、库和 manifest 哈希一致时才设置 `MLA_TND_STRIDE_REUSE=1`。构建和加载核验见 `production-build-command.json`、`production-audit-command.json`。双 dtype 组件与公开性能由 `performance.py component`、`performance.py public` 归档命令复现；公开驱动为 `python3 -m tests.mla_mixed_profile OUTPUT VENDOR PLATFORM DTYPE --stride-probes PROBES --stride-producer PRODUCER_MANIFEST`。其余检查使用上表对应命令文件，均另设输出目录。
