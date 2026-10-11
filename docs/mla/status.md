# MLA 当前状态

核对日期：2026-10-11；分支：`feat/mla-cann-migration`；代码基线：`9f9aeb9`（M07c3）。**本轮仅核验和整理文档，没有新增功能或重跑 CPU/NPU 功能测试。** 最近生产接入为 M06e2c2，PA 完成到独立设备消费者；唯一下一步仍是 [M07d1：独立 PA Copy Tile](plan.md#下一步m07d1)。

本页回答“现在能用什么、证据在哪里、下一步是什么”。接口、协议和验收方法查 [design](design.md)，步骤与完成条件查 [plan](plan.md)。本轮文档提交用 `git log -1 -- docs/mla/status.md` 定位。

## 当前可调用能力

共同边界：**910B C1V2、FP16/BF16、512+64 共享 KV、Hkv=1、无梯度、无 mask、`num_splits=1`**。Q/K 末维为 64，qv/V 为 512，四路同 dtype、同 NPU 设备。BSND 单样本 KV 容量及 TND 的 `max_seqlen_k` 为 0..16384；TND 总 Q/KV 各不超过 `INT32_MAX`，总 KV 可超过 16384。其他字段和资源界限见 [design 6.2–6.3](design.md#62-调度字段与边界)。

| 层次 | 已接通能力 | 记录与边界 |
| --- | --- | --- |
| 公开前向 | `flash_attn_func`，BSND Q/KV | 整容量，内部 v1 |
| 公开前向 | `flash_attn_varlen_func`，TND Q/KV | 两轴分配前缀；无 used 时内部 v3，任一 `seqused_q/k` 存在时内部 v5 |
| 公开只读 cache | `flash_attn_with_kvcache`，BSND Q + BSND KV | 整容量或实际 KV 长度；无长度且无记录走 v1，否则 v2；支持显式 v2 复用 |
| 公开只读 cache | `flash_attn_with_kvcache`，TND Q + BSND KV | Q 使用完整分配段；内部生成或显式复用 v4 |
| 公开 metadata | `get_scheduler_metadata` | 仅 v2（BSND）和 v4（混合）；生成器须接收 KV 长度 tensor |
| 内部前向 | BSND、TND、混合布局的 Q-used | 内部 v5；BSND/混合 Q-used 无公开等价入口，不开放显式 v5 复用 |
| 独立 PA 组件 | CPU 地址/调度、v6 CPU/AICPU 生成、CPU/910B 设备消费校验 | **PA 仍明确拒绝**；尚无设备 Copy、attention 或公开调用 |

三种非分页布局均支持四路独立的末维连续、外层非负 stride，包括 BNSD/NTD 归一视图、切片、广播、重叠及 offset；发射前检查完整 storage。显式 v2/v4 可跨合法视图复用，仍须匹配当次长度和 v4 Q 前缀。完整入口分派见 [design 6.7.1](design.md#671-非分页入口与记录选择)。

调用时最容易混淆的约束：

- kvcache/metadata 的 `num_splits` 默认 0，MLA 须显式传 **1**；metadata 的 `qkv_dtype` 默认 BF16，须与前向输入一致。func/varlen 默认 `num_splits=1`。
- 仅接受默认 mask/调度选项和 `softcap=0`；生成器使用 `has_softcap=False`，拒绝数值 `softcap/softmax_scale` 兼容关键字。前向 scale 默认 `1/sqrt(576)`，显式值须在 FP32 有限范围；完整限制见 [design 6.7.2](design.md#672-校验输出与发射)。
- 长度 tensor 须同设备、连续、int32；kvcache 另接受 None/整数。动态值由设备检查，无长度 D2H；空输出也不绕过记录校验，损坏或过期记录由消费者 Trap 拒绝。
- 请求 LSE 时 func/varlen 返回二项，kvcache 返回四项。非零容量中的零 KV 有效行为 `O=0、LSE=-inf`；全局空 KV 初始化分配区为 `O=0、LSE=+inf`；其他未使用 query 槽位不承诺值。shape、accumulator 与生命周期见 [design 6.7.2](design.md#672-校验输出与发射) 和 [5.8](design.md#58-stream-与内存生命周期)。

计划内尚未开放：PA、causal/window、自动或多 split、192/128 prefill、非零 softcap 与非默认调度选项。fake/compile 已有覆盖，完整 graph replay 留在 M16；910C 尚未开放或实机验收。KV 追加、内置 rotary、量化、backward 等排除项见 [design 第 4 节](design.md#4-功能边界)。

## 本轮文档核验

证据目录为 `build/mla-docs-audit-9f9aeb9/`。只修改三个 MLA 文档；本仓原有未跟踪 `kernel_meta/`、CANN 的 67 项已有差异和 CATLASS 均保留。

| 核验 | 结果与依据 |
| --- | --- |
| 上一步闭环 | 编辑前运行 M07c3 的 `check_acceptance.py` 通过；校验历史源码/产物哈希、XML 与验收汇总，未执行其中的功能测试。结果保存为 `prerequisite-acceptance.json` |
| 文档与源码 | Python/fake/C++ 入口、v1–v6 字数与版本、PA 调用链、构建依赖及固定性能清单核对通过；官方 FA3 使用固定快照及既有来源哈希，CANN 使用固定 Git blob。见 `source-audit.json`、`signatures.json`、`official/` |
| 结构与契约 | 明确目标契约、生产入口和独立组件的区别；修正 PA 静态描述与 CPU buffer 检查的边界；合并历史步骤索引，性能矩阵直接链接定义与采样器。范围、协议、验收门槛和 M07d1 顺序不变；链接、锚点、表格及命令语法检查见 `docs-check.json` |
| 提交质量 | `git diff --check`、暂存差异检查及仓库适用质量钩子通过；仅三个文档进入本步提交。见 `quality-staged.log` |

## 最近功能闭环与限制

以下均是**历史运行结果，本轮只核验证据**。路径相对仓库根目录；这些忽略目录不随 Git 分发，迁机复现须保留或恢复相应证据目录。

| 功能闭环 | 结果与证据 |
| --- | --- |
| M07c3 独立 v6 消费者，`9f9aeb9` | `build/mla-m07c3-paged-consumer/acceptance.json`：2,273 项去重 pytest，无失败/跳过；新 CPU 425、新设备 334、旧 CPU 559、旧设备 122、普通 FA3/入口 833 |
| v6 精确解码与拒绝 | 同目录：CPU/910B 各 2,552 组固定源差分，双 dtype、BSND/TND、64 种 P；58 次 release 真实 Trap；2,622,720 次 sanitizer 调用。覆盖当前长度、有效页、资源、空值、拒绝不写输出及流生命周期；见 `cpu.xml`、`device.xml`、`sanitizer-report.json`、`device/error-*/runtime-error.json` |
| v6 校验耗时 | 双 dtype 各 12 例，v6/v5 wall 比为 FP16 1.000–1.717、BF16 1.029–1.702；含诊断写入、ctypes、事件及同步，生成/上传在计时外。固定源没有同类消费者；这不是 attention Kernel 或公开 E2E。见 `device/comparison-*.json` |
| 最近生产接入 M06e2c2，`538b8e5` | `build/mla-m06e2c2-mixed-stride-production/acceptance.json`：4,659 项去重 pytest，完成三种非分页布局的 stride 生产接入 |
| M06e2c2 公开性能 | 同目录 `performance-acceptance.json`：双 dtype 各 8 个适用公开用例，R=1/4 的复用、生成加前向、内部生成共 96 段逐例达标；FA3/CANN E2E 为 FP16 0.284–0.623、BF16 0.284–0.675。各 3 个原生 FD 用例留待 M10 |

**仍保留的限制：** M06e2c2 组件 Kernel 比值为 FP16 0.898–1.888、BF16 0.901–2.458，两种 dtype 均有 6 例中的 4 例慢于固定源；不能宣称 Kernel 全部不劣化。profiler 的 `fwk_api_db` 解析错误仍保留，设备 CSV 不证明 host 关联可用。

PA 消费者只读调度、长度、页表，不读 cache 内容，不定义 attention 空行 O/LSE。空表/空池只通过校验层；Copy、attention 精度、真实 cache 大地址及公开 PA E2E 待后续验收。v6 的 >uint32 cache 地址只有算术证据，64 位 Q 起点也不代表分配了相应 attention tensor。

M07c3 已处理的开发期失败保留在原目录 `initial-full-device*`、`aicpu-comparison-failure-*`，处理经过查该提交的 status；早期失败不计入通过数量。旧设备 Trap 和旧耗时对比未在 M07c3 重跑。

## 环境与构建

最近功能运行环境（M07c3）：Python `/usr/local/python3.11.15/bin/python3` 3.11.15，torch `2.7.1+cpu`、torch_npu `2.7.1`，CANN `/usr/local/Ascend/cann-9.0.0-beta.2`，卡 6 的 Ascend910B3（进程内设备 0）。CATLASS 为 `769cd40a8716b28650b6bebb08db4834eea4462f` 且干净，CANN 源为 `ca18aee610729ca5270a44e2d84c751bc687fbf0`。版本/tiling 注册警告保留在原日志。

M07c3 实际加载 `flash_attn_npu_3/flash_attn_npu_3.cpython-311-aarch64-linux-gnu.so`，SHA256 为 `0e8f3c6e343fe7aac542622845b65519cb505c55f03d177f3d672d72ba30e74f`，见原目录 `build-provenance.json` / `production-load.log`；生产扩展编入 v6 生成器，但没有 PA 发射调用。v6 消费者只由独立 FP16/BF16 release 探针调用。

**本轮未查询 NPU 健康/占用、未加载扩展或重建。** 仅复核磁盘产物哈希、源码与静态环境信息；上述卡号不作为下一次运行的可用性保证。继续使用非 login shell；CPU pytest 设置 `TORCH_DEVICE_BACKEND_AUTOLOAD=0`，质量工具沿用 `/tmp/mla-m01-precommit` 及其 cache。开发或实机测试前按 [AGENTS.md](../../AGENTS.md) 重新检查环境、设备和实际加载路径。

## 唯一下一步与复现

下一步仅为 **[M07d1](plan.md#下一步m07d1)**：独立 PA Copy Tile，按页拆段搬运内容/RoPE 和 PV，完成全部页大小的双 dtype 搬运精度及固定源代表性 Copy 性能。本轮文档提交后停止。

本轮最小复核（仓库根目录，非 login shell；不导入 torch 或发射 NPU）：

```bash
python3 build/mla-docs-audit-9f9aeb9/check_docs.py
PYTHONPATH=/tmp/mla-m01-precommit PRE_COMMIT_HOME=/tmp/mla-m01-precommit-cache \
bash ci/code_quality.sh --staged
git diff --check
git diff --cached --check
```

M07c3 的原 `check_acceptance.py` 绑定当时三个文档及步骤表的哈希/结构；本轮编辑前已通过，编辑后不应直接用它检查新文档。上面的文档检查按 `9f9aeb9` Git 内容核对历史快照，再独立核验当前文档与未变代码，不改写历史验收规则。

功能重跑从原证据目录的 `run_checks.py` 及各 `*-command.json` 取完整命令：`cpu/device/old-cpu/ordinary/old-device-bsnd/old-device-tnd/old-device-tnd_bsnd/old-device-used`；构建见 `build-command.json`、`collect_production.py` 和独立探针清单。源码、环境或二进制变化后须在新证据目录重跑受影响检查，历史通过不能代替新运行结果。
