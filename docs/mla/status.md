# MLA 当前状态

核对日期：2026-10-11；分支：`feat/mla-cann-migration`；接手基线：`ea3be44`。
本轮完成 **M07e2c：BSND 只读 PA kvcache 公开接入**，8,280 项去重 pytest 通过。910B 双 dtype 精度、生产 Kernel 对照和 **72 个适用公开 R=1/4 调用段逐例 E2E** 完成；TND PA 前向仍明确拒绝。唯一下一步为 [M07e3：TND Q 只读 PA kvcache 接入](plan.md#下一步m07e3)。

本页记录当前能力、最近闭环证据和唯一下一步；契约见 [design](design.md)，迭代边界见 [plan](plan.md)。本轮闭环提交用 `git log -1 -- docs/mla/status.md` 定位。

## 当前可调用能力

共同边界：**910B C1V2、FP16/BF16、512+64 共享 KV、Hkv=1、无梯度、无 mask、`num_splits=1`**。Q/K 末维为 64，qv/V 为 512，四路同 dtype、同 NPU 设备。单样本逻辑 KV 容量上界 16384；TND 总 KV 可超过 16384。完整边界见 [design 6.2–6.4](design.md#62-调度字段与边界)。

| 层次 | 已接通能力 | 记录与边界 |
| --- | --- | --- |
| 公开前向 | `flash_attn_func`，BSND Q/KV | 整容量，内部 v1 |
| 公开前向 | `flash_attn_varlen_func`，TND Q/KV | 两轴分配前缀；无 used 时内部 v3，任一 used 存在时内部 v5 |
| 公开只读 cache | `flash_attn_with_kvcache`，BSND Q + BSND KV | 无长度且无记录走 v1，否则 v2；支持显式 v2 复用 |
| 公开只读 cache | `flash_attn_with_kvcache`，TND Q + BSND KV | Q 使用完整分配段，生成或显式复用 v4 |
| **公开只读 PA cache** | `flash_attn_with_kvcache(page_table=...)`，BSND Q + ND PA KV | **内部生成或显式复用 v6；None/整数/tensor KV 长度；完整 Q 分配段** |
| 公开 metadata | `get_scheduler_metadata` | 非 PA：BSND v2、TND v4；指定 `page_size`：BSND/TND v6 `int32[408+5B]` |
| 内部前向 | 非 PA 三布局 Q-used；BSND PA Q-used | 非 PA v5、PA v6；Q-used 每次生成记录，不开放显式复用 |
| 独立 TND PA attention | TND Q 分配/used、独立 stride | 尚无 TND PA 生产前向；独立 probe 不代表公开能力 |

调用时须注意：

- kvcache/metadata 的 `num_splits` 默认 0，MLA 须显式传 **1**；生成器的 `qkv_dtype` 默认 BF16，须匹配输入。生成器只接受长度 tensor，None/整数便利输入属于公开 kvcache。
- PA 页大小 P 为 **16..1024 的 16 倍数**；逻辑容量 **C=页表列数×P，在 0..16384**。None 表示 C，整数须在 0..C，按 Q 的逻辑 B 物化；物理页数不决定 B 或 C。生成器的 `max_seqlen_k` 须等于 C。
- v6 不绑定 cache 地址、stride、页表或 scale。每次前向都在设备核对当次长度及全部有效页；未用列允许 poison。PA 与非 PA 协议互斥。四路末维连续、外层非负独立 stride、offset、广播和重叠按完整 storage 校验。
- 请求 LSE 时 func/varlen 返回二项，kvcache 返回四项；未 split 的两个 accumulator 为同设备空 tensor。PA 所有有效零 KV 行均为 `O=0、LSE=-inf`，含 C=0/空物理池；非 PA 全局空 KV 的 `+inf` 分支保持不变。
- BSND PA 的专用/通用 fake、公开 fullgraph compile 与真实路径校验一致；编译后的真实调用仍检查 storage。完整 graph replay 留在 M16。公开没有 Q-used 参数；内部通用 `fwd` 可传 Q-used，但不能显式复用其记录。

尚未开放：TND PA 生产前向、causal/window、自动或多 split、192/128 prefill、非零 softcap 和非默认调度选项；910C 未开放或实机验收。追加 KV、内置 rotary、量化、backward 等排除项见 [design 第 4 节](design.md#4-功能边界)。

## 本轮闭环与证据

公开包装按逻辑容量处理整数长度并直接调用既有 PA dispatcher；通用 dispatcher/raw C++ 也汇入 `PagedBsndForward`。新增 fake 静态几何、完整 storage、页表/长度/记录描述校验；复用既有 AICPU、消费者及 Kernel/Block/Tile，没有设备数值路径修改、动态值读回、完整 cache gather 或 CANN attention 转调。

证据位于忽略目录 `build/mla-m07e2c-bsnd-paged-public/`，下表路径相对此目录。包含命令、日志、XML、输入/输出、固定源快照、实际库及逐样本性能数据；本机证据不随 Git 分发，迁机须复制该目录和 manifest 引用的历史固定源产物。

| 核验 | 结果与依据 |
| --- | --- |
| 前置闭环 | 编辑前 M07e2b 的 6,392 项历史证据及最小质量检查复核通过，不计本轮测试；`prerequisite.json` |
| 功能与回归 | **8,280 项去重 pytest，无失败/跳过**：CPU 893、公开功能 1882、公开真实 Trap 32、普通 FA3/入口 822、非分页及 TND PA 共享组件 1098、公开 PA metadata 2840、内部 PA 713；各模式 XML、日志及 `*-command.json` |
| 公开精度与记录 | 双 dtype 全部 64 种 P×4 个 cache profile，共 512 组源/双数学 O/LSE；另有 36 组公开/raw C++ 跨视图复用与 stride/真实大地址检查。固定源 O/非空行 LSE 逐元素一致，零 KV 符号单验；`probes/*/public-accuracy/` |
| 入口与生命周期 | None/整数/tensor、offset 记录、空表/池、完整 storage、fake/fullgraph compile、当前流长度原位更新、多 stream 释放后复用均通过；非法页/过期长度/v1–v5/损坏 v6 共 32 项真实 Trap，独立进程先正常预热；`public.xml`、`public-trap.xml`、`kvcache/` |
| 构建与加载 | 最终扩展 36 对象、271 个仓库源、6,197 个依赖核验；仅 `flash_api.o` / `mla_forward.o` 两个 host 对象变化，34 个对象含 PA 设备实例保持原哈希；`production-load.json`、`production-object-delta.json`、`production-final/` |
| 独立探针复用 | 逐项核对 M07e2b 探针依赖，双 dtype 各 55 源/953 依赖与 v6 producer 22 源未变，复用库但隔离新输出；`probe-reuse-audit.json` |
| 固定源原生 PA | 从既有固定源快照补编 FP16/BF16 BSND ND PA 非 FD/FD 实例，不改 CANN 源仓；24 例实际 Kernel、tiling key 与新包对应，每 dtype 各 6 例非 FD/FD；`native-package.json`、`native-dispatch-proof.json`、`native-*/` |
| 性能 | 双 dtype 24 例生产 Kernel 组件对照，另有 24 例公开生成/复用/内部生成 R=1/4 E2E；先验源/双数学精度，性能驱动用例不计 pytest；`profile-*/`、`public-profile-*/` |
| 提交质量 | 文档链接/锚点、命令与表格、差异及 staged 质量钩子通过；只暂存本步文件；`docs-check.json`、`quality-staged-result.json`、`source-snapshots.json`、`acceptance.json` |

初轮 Dynamo 参数化测试达到缓存上限，已在编译测试间 reset；大地址测试的 11 页误需 80 GiB，已缩为仍跨 >32 位地址的 2 页。失败日志和旧测试源码分别保存在 `failed-compile-test-cache/`、`failed-wide-pool/`，最终完整公开测试重跑通过。原生 OPC 首次受沙箱网络接口枚举限制，已在沙箱外完成；`failed-native-sandbox/` 保留原错误。一次重复生产构建被主动停止，随后串行重建所有受影响 host 对象，最终链接/加载审计为准；早期日志保留在 `preformat-build/`、`duplicate-build-stopped/`。初轮 1,334 项 smoke 不重复计数。

## 性能范围与保留限制

两种 dtype 使用预先固定的 P=16/112/128/256/528/1024 × 短/长 KV 矩阵，每种 12 例。组件保留既有内部 Q-used 几何；公开矩阵使用完整 BSND Q。每例预热 5 次、5 轮交替采样，每轮 20 个 wall、10 个 profiler 样本；公开 R=1/4，双方同进程固定同一 CPU，seed=20261011。

| dtype | 生产/固定源组件 Kernel 比值 | Kernel 较慢例数 | 适用公开 E2E 比值范围 | 适用调用段 |
| --- | --- | --- | --- | --- |
| FP16 | 0.9947–1.2360 | 9/12 | 0.3581–0.6988 | 36/36 ≤1.0 |
| BF16 | 1.0108–1.2097 | 12/12 | 0.3552–0.7238 | 36/36 ≤1.0 |

E2E 比值为 FA3/CANN，含 O/LSE 完成、入口校验、输出/workspace 分配、必要转换和同步；生成加 R 次、既有记录复用及每次内部生成分别计量。CANN 保留可原生表达的独立页 stride，其余 Q/cache/页表转换计入每次调用。**72 个适用调用段逐例达标；原生 FD 另列能力差距，不纳入当前单 split 门槛。** 组件 Kernel 含 v6/当次长度/全部有效页校验，不能据公开 E2E 达标宣称 Kernel 逐例不劣化；probe wall 不作公开 E2E。

Profiler 框架 API 数据库导出状态由 `acceptance.json` 的 `profiler_exports` 逐模式记录；本轮发现 2 条已知 `Failed to find node launch str id` 错误。计时仅使用与 CANN `op_summary` 逐行核对的文本 Kernel CSV，不依赖失败的框架关联表；解析器源码/哈希保存在 `profiler-parser-sources/`。

未重跑普通 FA3 全套、完整 TND PA attention 矩阵、独立 Copy 性能、全部历史 Trap、NPU sanitizer 或全布局公开 E2E；相关共享路径回归已补。非分页公开性能仍为 M06e2c2 的历史结果，不能记作本轮新测；TND PA、mask 和 FD 尚未接入生产，910C 未验收。

## 环境与构建

Python 3.11.15，torch 2.7.1+cpu、torch_npu 2.7.1，CANN 9.0.0-beta.2；8 张 910B3，使用物理卡 6、进程内设备 0，设备任务串行。CATLASS `769cd40a8716b28650b6bebb08db4834eea4462f` 干净；只读 CANN `ca18aee610729ca5270a44e2d84c751bc687fbf0` 原有 67 项差异及本仓未跟踪 `kernel_meta/` 保留。环境与前后状态见 `environment.json`、`*-status-before.txt` / `*-status-final.txt`。

最终扩展实际路径为 `flash_attn_npu_3/flash_attn_npu_3.cpython-311-aarch64-linux-gnu.so`，SHA256 `4844118428c071d40512259ea6a0bf2770c869b315bb0b9e925d6cf2eab0193f`。固定源原生包、host/device 库与实际分派分别见 `native-package.json`、`public-profile-*/environment.json`、`native-dispatch-proof.json`。

## 唯一下一步与复现

下一步仅为 **[M07e3](plan.md#下一步m07e3)**：TND Q 只读 PA kvcache 与内部 TND PA Q-used 的生产接入及相应验收。本轮只提交一次，成功后停止。

最小闭环复核（仓库根目录，非 login shell，不发射 NPU）：

```bash
python3 build/mla-m07e2c-bsnd-paged-public/check_acceptance.py
PYTHONPATH=/tmp/mla-m01-precommit PRE_COMMIT_HOME=/tmp/mla-m01-precommit-cache \
bash ci/code_quality.sh --staged
git diff --check
git diff --cached --check
```

`run_checks.py` 支持 `cpu`、`public`、`public-trap`、`ordinary`、`nonpaged`、`metadata`、`device`、`native-fp16/bf16`、`profile-fp16/bf16`、`public-profile-fp16/bf16`；命令与环境保存在相应 `*-command.json`。生产构建/审计为 `build_production.py` / `audit_production.py`，固定源补编/打包为 `build_native.py` / `package_native.py`。重跑先复制脚本到新证据目录，不覆盖本轮数据；设备任务须在沙箱外串行执行，CPU 禁用 torch 自动加载 NPU。
