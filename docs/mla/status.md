# MLA 当前状态

本轮完成 **M07a：ND paged cache 固定源核对与实施拆分（纯文档）**（2026-10-10），基线 `538b8e5`，分支 `feat/mla-cann-migration`。M06 已完成，PA 仍明确拒绝；M07 的目标契约见 [design 6.4.3](design.md#643-nd-paged-cache-契约与来源)。唯一下一步为 **[M07b1：PA CPU 地址组件](plan.md#下一步m07b1)**。

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

本轮只修改 design、plan、status。证据位于忽略目录 `build/mla-m07a-paged-plan/`，路径相对仓库根目录；包含固定源快照、来源/哈希核对、文档检查脚本与结果，不随 Git 分发。没有实现或运行 PA，没有重建或重新加载扩展，也没有运行 NPU 精度/性能测试。

| 检查 | 结果与证据 |
| --- | --- |
| 固定 CANN 来源 | 16 份源文件/许可证逐一从固定 Git blob 提取，工作树内容一致；`cann-source/`、`source-audit.json`。源仓原有 67 项差异保留，核对前后相同 |
| 固定官方 FA3 | Python/C++ 两份归档与既有固定提交 manifest 哈希一致；`official/`。本轮使用已验证归档，未从网络刷新 |
| 上一步最小闭环复核 | M06e2c2 的 acceptance、生产命令/448 项 XML、性能与环境证据哈希一致；269 份非文档源码及当前扩展哈希未变；`source-audit.json`。历史 4,659 项通过不是本轮测试数 |
| 文档与范围 | 本地链接/锚点、表格、步骤唯一性、官方签名、固定源函数/分支、当前 PA 拒绝条件与三文档一致性通过；`docs-check.json` |
| 提交质量 | `git diff --check` 与适用仓库质量钩子通过；`quality-staged.log`、`acceptance.json` |

核对确定：页大小交集是 16..1024 的全部 64 个 16 倍数；P=112 会改变源 S2 基块，P=528 仍需块内跨页。PA 的 KV 长度始终按样本，TND Q 不使其变为累计长度。metadata 生成器没有页表/物理 cache 参数，故页号与物理 storage 校验属于当次消费者，记录可跨合法页表映射复用。PA 的空值须按官方 `total_k=B*P` 分支核对，不能按物理 cache 的空 shape 套用非分页 `+inf`。这些是源码/契约结论，实机验收分别列入 M07c–e。

### 最近功能闭环与限制

最近功能提交为 **M06e2c2 `538b8e5`**，证据仍在 `build/mla-m06e2c2-mixed-stride-production/`：

- `acceptance.json` 汇总 4,659 个去重 pytest 通过、无失败或跳过；生产混合 stride 448、受影响回归 2,684、普通 FA3 833、TND/独立 stride 回归 694。前置复跑与未选入的进程级 Trap 不计入。
- `performance-acceptance.json`：双 dtype 各 8 个适用公开用例，metadata 复用、生成加 R=1/4、逐次内部生成共 96 段逐例达标；FA3/CANN E2E 为 FP16 0.284–0.623、BF16 0.284–0.675。各 3 个原生 FD 用例完整保留并单列，待 M10。
- 组件 Kernel 比值仍为 FP16 0.898–1.888、BF16 0.901–2.458，各有 4/6 例慢于固定源；**不能宣称 Kernel 全部不劣化**。组件 profiler 的 `fwk_api_db` 解析错误保留，设备 CSV 已核验，不据此声称 host 关联可用。
- BSND 与 TND stride 的公开 E2E 证据分别在 `build/mla-m06e1c-bsnd-stride/` 与 `build/mla-m06e2c1-tnd-stride-production/`；最近 CPU 地址 sanitizer 为 `build/mla-m06e2a-tnd-layout/sanitized/report.json`，本轮未重跑。

## 环境与构建

本轮通过静态来源核验 Python `/usr/local/python3.11.15/bin/python3` 3.11.15、已安装 torch/torch_npu distribution 均为 2.7.1、CANN 路径 `/usr/local/Ascend/cann-9.0.0-beta.2`，CATLASS 为 `769cd40a8716b28650b6bebb08db4834eea4462f` 且干净。此前成功进程的实际 torch 版本为 `2.7.1+cpu`，不能把 distribution 的字符串差别写成环境升级。

磁盘扩展为 `flash_attn_npu_3/flash_attn_npu_3.cpython-311-aarch64-linux-gnu.so`，SHA256 `38582e6e4deb2eed122a38e5c1da06cd6d5386b4d00bc405f21e1cf469a4a00e`，与上一步实际加载路径及构建记录一致。这里只复核磁盘文件与源码，**本轮未重新加载扩展、未查询芯片健康/占用**。此前实机环境为 8 张 Ascend910B3、bisheng clang 15.0.5、g++ 11.4；进入 M07b1 开发前重新核查当前环境，设备步骤须在实际验证进程中确认加载来源和可用卡。

只读 CANN 源仍为 `ca18aee610729ca5270a44e2d84c751bc687fbf0`，原有差异和本仓未跟踪 `kernel_meta/` 保留。文档检查使用非 login shell 和已有 `/tmp/mla-m01-precommit`、`/tmp/mla-m01-precommit-cache`；默认 Python 环境没有安装 pre-commit，无需改动系统安装。910C 尚未开放或实机验收，完整 graph replay 留在 M16。

## 唯一下一步与复现

下一步仅为 **[M07b1](plan.md#下一步m07b1)**：独立 PA CPU 地址/页内分段组件及固定源差分、ASan/UBSan；本轮提交后停止，不开始实现。

本轮最小闭环为以下文档检查，第二条同时复核来源、上一步证据和未变源码/扩展；质量检查在只暂存本轮三份文档后运行：

```bash
python3 build/mla-m07a-paged-plan/audit_sources.py
python3 build/mla-m07a-paged-plan/check_docs.py
PYTHONPATH=/tmp/mla-m01-precommit PRE_COMMIT_HOME=/tmp/mla-m01-precommit-cache \
bash ci/code_quality.sh --staged
git diff --cached --check
```

上一功能步的最小实机命令保留如下，**本轮未运行**；后续修改源码/构建时，按影响范围重跑，使用新输出目录，probe 默认重建：

```bash
ASCEND_RT_VISIBLE_DEVICES=7 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
MLA_TND_STRIDE_DIR=build/mla-m07-recheck/probes \
MLA_TND_STRIDE_PRODUCER_DIR=build/mla-m07-recheck/producer \
MLA_MIXED_STRIDE_DIR=build/mla-m07-recheck/production \
python3 -m pytest -x -q tests/test_flash_attn_npu_v3_mla_mixed_stride_production.py
```

设备编号须根据当次健康/占用重新选择。`MLA_TND_STRIDE_REUSE=1` 仅在源码、库和 manifest 哈希一致时使用；生产构建、加载与性能复现继续参考上一步目录的 `production-build-command.json`、`production-audit-command.json`、`performance.py` 及原始 command JSON。
