# MLA 当前状态

本轮完成 **M07b1：独立 PA CPU 地址与页内分段组件**（2026-10-10），基线 `f93b9b5`，分支 `feat/mla-cann-migration`。M06 已完成，PA 仍明确拒绝；M07 的目标契约见 [design 6.4.3](design.md#643-nd-paged-cache-契约与来源)。唯一下一步为 **[M07b2：PA CPU 调度](plan.md#下一步m07b2)**。

本页维护当前能力、证据、环境差异和下一步；目标契约与架构见 [design](design.md)，步骤与完成条件见 [plan](plan.md)。当前闭环提交可用 `git log -1 -- docs/mla/status.md` 定位，功能闭环提交见下方证据表。

## 当前可调用能力

共同边界为 **910B C1V2、FP16/BF16、512+64 共享 KV、Hkv=1、无梯度**。Q/K 末维为 64，qv/V 为 512；四路输入须同 dtype、同 NPU 设备。BSND 每样本 KV 容量与 TND 的 `max_seqlen_k` 均为 0..16384；TND 总 Q/KV 各不超过 `INT32_MAX`，总 KV 可以超过 16384。其他字段、地址与资源限制见 design [6.2](design.md#62-调度字段与边界)、[6.3](design.md#63-长度与布局调度)、[6.4](design.md#64-输入-stride-与地址校验)。

### 布局与 stride

| 输入布局 | 生产支持 | 独立组件边界 |
| --- | --- | --- |
| BSND Q/KV | 四路末维连续、外层非负 stride；含 BNSD、切片、广播、重叠、独立 storage offset；完整 storage/字节界限在发射前校验 | CPU 校验、设备 attention 和生产接入均完成，覆盖 v1/v2/v5 |
| TND Q/KV | 四路末维连续、外层非负 stride；含 NTD、切片、广播、重叠、独立 storage offset；完整 storage/字节界限在发射前校验 | CPU、独立 attention 和生产 v3/v5 接入均完成 |
| TND Q + BSND KV | 四路末维连续、外层非负 stride；Q/qv 使用 TND 描述，K/V 各自使用 BSND 描述；完整 storage/字节界限在发射前校验 | CPU、独立 attention 和生产 v4/v5 接入均完成；v4 metadata 可跨合法视图复用 |

新增的 [PA CPU 地址组件](../../csrc/ascend910/flash_attn_npu_3/mla/mla_paged_layout.h) 独立描述逻辑 B/页表宽度/容量、物理页池、页大小、页表行距和 K/V storage，复用 BSND 完整 storage 校验；只检查有效页号，按页边界返回 K/V 独立地址。它只处理 CPU host buffer，不读取设备指针或执行 D2H，也没有接入调度、metadata、Kernel 或公开入口。Q/qv 继续使用既有 BSND/TND 地址组件；CPU 地址层不承担生产调度的 16384 容量上限。

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

本轮新增 `mla_paged_layout.h/.cpp`、两个独立 CPU 驱动及 [PA 地址测试](../../tests/test_flash_attn_npu_v3_mla_paged_layout.py)，更新 plan/status。没有修改既有生产源码、构建配置、metadata 或 Kernel；没有重建生产扩展或运行 NPU 精度/性能测试。证据位于忽略目录 `build/mla-m07b1-paged-layout/`，路径相对仓库根目录，保留编译命令、固定源快照、哈希、全部驱动输入/输出、XML 和检查脚本，不随 Git 分发。

| 检查 | 结果与证据 |
| --- | --- |
| 上一步最小闭环 | 修改源码前复制并复跑 M07a 检查，通过 16 份固定 CANN 源、官方归档、197 个文档链接、27 个表格和 46 个源锚点检查；269 份既有非文档源码与生产扩展哈希一致。`prerequisite.log`、`source-audit.json`、`docs-check.json`；历史 4,659 项不是本轮测试数 |
| PA CPU 地址 | 171 个 pytest 通过；全部 64 种 P、乱序/重复页、尾页、padding poison、K/V 独立 storage offset/stride、真实 FP16/BF16 视图、页表广播/重叠、65535/65536/65537、>uint32 地址、空值、整 storage 与字节溢出、失败输出不变。`cpu.xml`、`normal/` |
| 固定源差分 | 从固定 Git blob 原样提取 `BlockTableParser`、`GM_KV_PA_BNBD` offset、QK 联合 Copy 和 PV Copy 类，独立编译为 CPU trace 驱动；与迁移组件及独立标量 oracle 比较地址、分段、L1 偏移、D 子区间和每段页查询次数。`normal/source_helpers.inc`、`normal/manifest.json` |
| 受影响 CPU 回归 | BSND 地址 82、TND 地址及分配/used 组合 101 项通过；与 PA 合计 **354 个去重 pytest 通过，无失败或跳过**。`cpu-command.json`、`cpu.log`、`cpu.xml` |
| ASan/UBSan | 批量重放迁移驱动 2,953 条输入、固定源驱动 2,270 条输入，输出与已通过 oracle 的原始结果逐字一致；启用 leak 检查，两进程退出 0 且 stderr 为空。`sanitized-replay/report.json`、`sanitizer-replay-command.json` |
| 文档与提交质量 | 当前链接/锚点、步骤唯一性、来源和未改生产路径核对见 `step-check.json`；适用仓库钩子见 `quality-staged.log`，最终汇总见 `acceptance.json` |

固定源差分中的 Copy 操作替换为 CPU 参数记录；FA3 外层 stride 直接注入源 layout getter。它证明地址与页内分段，不证明原生 FIA 接受全部视图，也不代表设备 Copy、L1 数值、同步、attention 或性能已验收。超过 uint32 的 cache 地址本轮只作算术检查，未分配对应大内存。

Sanitizer 首次逐用例运行受沙箱 ptrace 限制，LeakSanitizer 退出 1；沙箱外逐进程重跑通过 2 项后主动中断，改用既有 CPU 验收方式批量重放完整输入，保留 ASan/UBSan/LeakSanitizer 全部检查。两次未完成记录保留在 `sanitized.log` / `sanitized-host.log` 及对应 exit JSON，**不计为通过**；闭环依据为最终 `sanitized-replay/report.json`。

### 最近生产闭环与限制

最近生产功能提交仍为 **M06e2c2 `538b8e5`**，证据在 `build/mla-m06e2c2-mixed-stride-production/`：

- `acceptance.json` 汇总 4,659 个去重 pytest 通过、无失败或跳过；本轮仅复核来源/哈希，没有复跑生产测试。
- `performance-acceptance.json`：双 dtype 各 8 个适用公开用例，共 96 段逐例达标；FA3/CANN E2E 为 FP16 0.284–0.623、BF16 0.284–0.675。各 3 个原生 FD 用例单列，待 M10。
- 组件 Kernel 比值仍为 FP16 0.898–1.888、BF16 0.901–2.458，各有 4/6 例慢于固定源；**不能宣称 Kernel 全部不劣化**。组件 profiler 的 `fwk_api_db` 解析错误保留，设备 CSV 已核验，不据此声称 host 关联可用。
- BSND/TND stride 的公开 E2E 证据分别在 `build/mla-m06e1c-bsnd-stride/`、`build/mla-m06e2c1-tnd-stride-production/`。PA 的生成器、消费者、设备 Copy/attention 与生产 E2E 仍按 M07c–e 分别验收。

## 环境与构建

本轮实际导入确认 Python `/usr/local/python3.11.15/bin/python3` 3.11.15、torch `2.7.1+cpu`、torch_npu `2.7.1`、CANN `/usr/local/Ascend/cann-9.0.0-beta.2`。CATLASS 为 `769cd40a8716b28650b6bebb08db4834eea4462f` 且干净；CPU 驱动使用 g++ 11.4。

实际加载扩展为 `flash_attn_npu_3/flash_attn_npu_3.cpython-311-aarch64-linux-gnu.so`，SHA256 `38582e6e4deb2eed122a38e5c1da06cd6d5386b4d00bc405f21e1cf469a4a00e`，与上一步构建/加载记录一致；**本轮新增 CPU 组件没有进入该扩展**。环境记录见 `environment.json`。`npu-smi.log` 显示 8 张 Ascend910B3 健康、AICore 利用率为 0，1–7 卡未列出运行进程；本轮没有发射 NPU 测试。

沙箱内驱动查询/扩展加载失败，改为沙箱外只读查询后通过；CPU pytest 设置 `TORCH_DEVICE_BACKEND_AUTOLOAD=0`，避免导入 NPU 后端。ASan/UBSan 的最终批量重放也在沙箱外运行，以保留 LeakSanitizer 检查。使用非 login shell；质量检查沿用 `/tmp/mla-m01-precommit` 与 `/tmp/mla-m01-precommit-cache`，没有修改系统安装。

只读 CANN 源仍为 `ca18aee610729ca5270a44e2d84c751bc687fbf0`，原有 67 项差异和本仓未跟踪 `kernel_meta/` 保留。910C 尚未开放或实机验收，完整 graph replay 留在 M16。

## 唯一下一步与复现

下一步仅为 **[M07b2](plan.md#下一步m07b2)**：独立 PA CPU 调度、固定源完整字段差分、sanitizer 与旧 CPU 路径回归；本轮提交后停止。

本轮最小闭环（sanitizer 使用前一条生成的完整驱动输入，须在允许 LeakSanitizer ptrace 的环境运行）：

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
MLA_PAGED_LAYOUT_DIR=build/mla-m07b1-paged-layout/normal \
MLA_BSND_LAYOUT_DIR=build/mla-m07b1-paged-layout/bsnd \
MLA_TND_LAYOUT_DIR=build/mla-m07b1-paged-layout/tnd \
python3 build/mla-m07b1-paged-layout/run.py cpu python3 -m pytest -x -q \
  tests/test_flash_attn_npu_v3_mla_paged_layout.py \
  tests/test_flash_attn_npu_v3_mla_bsnd_layout.py \
  tests/test_flash_attn_npu_v3_mla_tnd_layout.py \
  --junitxml=build/mla-m07b1-paged-layout/cpu.xml
python3 build/mla-m07b1-paged-layout/replay_sanitized.py
python3 build/mla-m07b1-paged-layout/check_step.py
PYTHONPATH=/tmp/mla-m01-precommit PRE_COMMIT_HOME=/tmp/mla-m01-precommit-cache \
bash ci/code_quality.sh --staged
git diff --cached --check
```

固定源通过 `MLA_CANN_SOURCE_ROOT` 选择只读 checkout，默认 `../ops-transformer`。后续变更使用新的证据目录；生产步骤重新核查设备占用、构建与实际加载来源，并按影响范围重跑相关实机检查。M06e2c2 的生产/性能复现仍见其证据目录中的 `production-build-command.json`、`production-audit-command.json`、`performance.py` 和原始 command JSON。
