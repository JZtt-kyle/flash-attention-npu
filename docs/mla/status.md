# MLA 当前状态

核对日期：2026-10-11；分支：`feat/mla-cann-migration`；接手基线：`7618939`。本轮完成 **M07d1：独立 PA Copy Tile**，相关 1,401 项 pytest 通过，完成固定源 Copy 性能采样。最近生产接入仍为 M06e2c2；唯一下一步为 [M07d2：BSND Q 的独立 PA attention](plan.md#下一步m07d2)。

本页回答“现在能用什么、证据在哪里、下一步是什么”。接口、协议和验收方法查 [design](design.md)，步骤与完成条件查 [plan](plan.md)。本轮闭环提交用 `git log -1 -- docs/mla/status.md` 定位。

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
| 独立 PA 组件 | CPU 地址/调度、v6 CPU/AICPU 生成、CPU/910B 消费者、910B PA Copy Tile | **PA 仍明确拒绝**；Copy 只由独立探针调用，尚无 PA attention 或公开调用 |

三种非分页布局均支持四路独立的末维连续、外层非负 stride，包括 BNSD/NTD 归一视图、切片、广播、重叠及 offset；发射前检查完整 storage。显式 v2/v4 可跨合法视图复用，仍须匹配当次长度和 v4 Q 前缀。完整入口分派见 [design 6.7.1](design.md#671-非分页入口与记录选择)。

调用时最容易混淆的约束：

- kvcache/metadata 的 `num_splits` 默认 0，MLA 须显式传 **1**；metadata 的 `qkv_dtype` 默认 BF16，须与前向输入一致。func/varlen 默认 `num_splits=1`。
- 仅接受默认 mask/调度选项和 `softcap=0`；生成器使用 `has_softcap=False`，拒绝数值 `softcap/softmax_scale` 兼容关键字。前向 scale 默认 `1/sqrt(576)`，显式值须在 FP32 有限范围；完整限制见 [design 6.7.2](design.md#672-校验输出与发射)。
- 长度 tensor 须同设备、连续、int32；kvcache 另接受 None/整数。动态值由设备检查，无长度 D2H；空输出也不绕过记录校验，损坏或过期记录由消费者 Trap 拒绝。
- 请求 LSE 时 func/varlen 返回二项，kvcache 返回四项。非零容量中的零 KV 有效行为 `O=0、LSE=-inf`；全局空 KV 初始化分配区为 `O=0、LSE=+inf`；其他未使用 query 槽位不承诺值。shape、accumulator 与生命周期见 [design 6.7.2](design.md#672-校验输出与发射) 和 [5.8](design.md#58-stream-与内存生命周期)。

计划内尚未开放：PA、causal/window、自动或多 split、192/128 prefill、非零 softcap 与非默认调度选项。fake/compile 已有覆盖，完整 graph replay 留在 M16；910C 尚未开放或实机验收。KV 追加、内置 rotary、量化、backward 等排除项见 [design 第 4 节](design.md#4-功能边界)。

## 本轮闭环与证据

新增 [PA Copy Tile](../../csrc/ascend910/flash_attn_npu_3/mla/tile/mla_paged_copy.hpp)，沿固定 CANN `CopyKKropePAGmToL1` / `CopyKvGmToL1` 按页拆段，QK 每段共用一次页号读取，K/V 地址各用自己的 64 位 stride，复用既有 ND→NZ 回退。调用方仍负责前置完整校验和事件/L1 生命周期；本步没有接入 attention 或公共入口。协议及探针边界见 [design](design.md#独立设备-copy)。

证据保存在忽略目录 `build/mla-m07d1-paged-copy/`，以下路径相对此目录。保留命令、日志、XML、源码快照、固定源、实际加载库、L1 导出和性能原始数据；汇总为 `acceptance.json`，完整性由 `check_acceptance.py` 检查。这些本机证据不随 Git 分发，迁机须复制该目录。

| 核验 | 结果与依据 |
| --- | --- |
| 前置闭环 | 编辑前重跑上一文档步 `check_docs.py`、差异和适用质量检查通过；历史证据核验不计作本轮功能测试。`prerequisite.log` |
| 本轮功能 | **1,401 项去重 pytest 通过，无失败/跳过**：新 Copy 300、CPU 地址/消费者 596、设备 v6 消费者 274、既有 Cube 155、双 dtype 非分页 stride 76；`*-command.json`、五套 `*.xml` 与日志 |
| 全页大小搬运 | 双 dtype、全部 64 种 P，各覆盖连续与两路独立 stride；L1 NZ 有效元素与未改写的固定源 Copy、独立 host 标量索引逐元素一致。覆盖 QK 两半、PV 四个 D 块、尾页/跨页、重复/乱序页和 offset；`probes/accuracy.jsonl`、`probes/accuracy/*.pt` 保存输入及实际导出 |
| 地址与空值 | 覆盖 65535/65536/65537、广播/重叠、页表行广播/重叠、零 count 和空表/池；K/V 各自真实 >uint32 页偏移/行地址及 >uint32 页表行距，实际 storage 约 8–16 GiB。零 count 只证明 Copy 不读 cache，不定义 attention 空行 O/LSE；`device.xml`、上述精度文件 |
| 缓冲和 stream | 两个 L1 槽位跨多轮复用、MTE2→MTE1 就绪/反向释放和两条 worker stream 的显式事件依赖通过。诊断 L1→GM 使用额外 MTE3 同步；Vector 分支不执行 Copy。`device.xml`、`tests/mla_paged_copy_probe.cpp` |
| 构建与来源 | FP16/BF16 各自固定源/迁移 release 库重新构建，核验设备依赖、626 项文件哈希和实际加载库；固定源读取 Git blob，不依赖 CANN 工作区改动。`probes/build-manifest.json`、`*-loaded.json`、`oracle-source-manifest.json` |
| Copy 性能 | 双 dtype 各 12 例，P=16/112/128/256/528/1024，短/长 KV、单/多 Q 块工作量、乱序页和零长度；5 次预热、五轮各 20 对 wall 与 10 对设备样本。迁移/源 Kernel 中位比为 FP16 **0.9985–1.0036**、BF16 **0.9949–1.0041**；wall 比为 **0.9807–1.0278**、**0.9873–1.0134**。`performance/comparison.json`、双 dtype `wall.json`、原始 profiler CSV |
| 受影响回归 | v6 当前长度/页表/资源/空值/流，以及既有非分页 Copy/Cube 和双 dtype stride 通过；消费者 release Trap、旧耗时对比、普通 FA3 全套未重跑。公共入口、metadata 和生产实现未改动；相关 XML 见上表 |
| 提交质量 | 文档链接/锚点、表格和命令语法、源码/产物哈希、暂存差异、`git diff --check` 及仓库适用质量钩子通过；仅本步 8 个文件进入提交。`docs-check.json`、`quality-staged.log` |

**性能口径：** Copy 计时含页表/任务读取、GM→L1、每 tile 512 字节 MTE1 消费及双缓冲事件，初始化和诊断导出在计时外。BSND/TND 标签仅描述复制工作量，未执行 Q 读取、metadata 或 attention；不是公开 PA E2E。FP16 的 7 例、BF16 的 4 例中位比大于 1，差异很小但未据此宣称逐例不劣化。固定源使用测试侧 layout stride 注入；广播及超宽页表行距的精度对照先物化源输入，不能宣称原生 FIA host 接受相同视图。

profiler 仍报告 `fwk_api_db` / host flow 解析错误，保留在 `profile.log`；采样器逐条核对 `kernel_details.csv` 与原始 `op_summary_*.csv` 的名称、时间、设备和数量，因此只使用设备任务耗时，不声称 host 关联解析通过。

开发期两次探针失败均已修复并保留：`initial-vector-launch-failure/` 记录未显式 C1V2 发射导致 Vector 核内存异常；`initial-mixed-launch-mismatch/` 和 `debug-*.pt` 记录 L1 导出缺少 MTE3 同步、读到初始化哨兵。修正后小测和完整 300 项设备套件通过；早期失败不计入通过数量。

## 最近功能闭环与限制

以下保留此前功能闭环的完整结果。本轮新运行的回归子集见上表，不能以子集代替历史全套。路径相对仓库根目录，迁机须保留相应证据目录。

| 功能闭环 | 结果与证据 |
| --- | --- |
| M07c3 独立 v6 消费者，`9f9aeb9` | `build/mla-m07c3-paged-consumer/acceptance.json`：2,273 项去重 pytest，无失败/跳过；新 CPU 425、新设备 334、旧 CPU 559、旧设备 122、普通 FA3/入口 833 |
| v6 精确解码与拒绝 | 同目录：CPU/910B 各 2,552 组固定源差分，双 dtype、BSND/TND、64 种 P；58 次 release 真实 Trap；2,622,720 次 sanitizer 调用。覆盖当前长度、有效页、资源、空值、拒绝不写输出及流生命周期；见 `cpu.xml`、`device.xml`、`sanitizer-report.json`、`device/error-*/runtime-error.json` |
| v6 校验耗时 | 双 dtype 各 12 例，v6/v5 wall 比为 FP16 1.000–1.717、BF16 1.029–1.702；含诊断写入、ctypes、事件及同步，生成/上传在计时外。固定源没有同类消费者；这不是 attention Kernel 或公开 E2E。见 `device/comparison-*.json` |
| 最近生产接入 M06e2c2，`538b8e5` | `build/mla-m06e2c2-mixed-stride-production/acceptance.json`：4,659 项去重 pytest，完成三种非分页布局的 stride 生产接入 |
| M06e2c2 公开性能 | 同目录 `performance-acceptance.json`：双 dtype 各 8 个适用公开用例，R=1/4 的复用、生成加前向、内部生成共 96 段逐例达标；FA3/CANN E2E 为 FP16 0.284–0.623、BF16 0.284–0.675。各 3 个原生 FD 用例留待 M10 |

**仍保留的限制：** M06e2c2 组件 Kernel 比值为 FP16 0.898–1.888、BF16 0.901–2.458，两种 dtype 均有 6 例中的 4 例慢于固定源；不能宣称 Kernel 全部不劣化。profiler 的 `fwk_api_db` 解析错误仍保留，设备 CSV 不证明 host 关联可用。

PA 消费者仍只读调度、长度和页表；本轮 Copy 已搬运真实 cache 并验证大地址，但 PA attention 的 O/LSE、完整任务环和公开 E2E 尚待后续验收。Copy 零 count/空池通过不代表 attention 空行处理已接通，历史 64 位 Q 起点测试也不代表分配了相应 attention tensor。

M07c3 已处理的开发期失败保留在原目录 `initial-full-device*`、`aicpu-comparison-failure-*`，处理经过查该提交的 status；早期失败不计入通过数量。旧设备 Trap 和旧耗时对比未在 M07c3 重跑。

## 环境与构建

本轮使用 Python `/usr/local/python3.11.15/bin/python3` 3.11.15，torch `2.7.1+cpu`、torch_npu `2.7.1`，CANN `/usr/local/Ascend/cann-9.0.0-beta.2`，物理卡 6 的 Ascend910B3（进程内设备 0）。8 张卡健康，卡 6 在接手查询时无运行进程；采样前后健康/占用见 `npu-smi.log` 与 `performance/npu-before.log` / `npu-after.log`。

CATLASS `769cd40a8716b28650b6bebb08db4834eea4462f` 且干净；只读 CANN `ca18aee610729ca5270a44e2d84c751bc687fbf0` 的原有 67 项差异、本仓原有未跟踪 `kernel_meta/` 均保留。详见 `environment.json` 和 `cann-status-before.txt`。

本轮实际导入的生产扩展为 `flash_attn_npu_3/flash_attn_npu_3.cpython-311-aarch64-linux-gnu.so`，SHA256 仍为 `0e8f3c6e343fe7aac542622845b65519cb505c55f03d177f3d672d72ba30e74f`，见 `environment-load.log` / `environment.json`。新增 header 只由独立 Copy probe 使用，不在生产依赖中，生产扩展没有重建。四个新库的源码/依赖/实际加载哈希见 `probes/build-manifest.json` 与 `fp16/bf16-loaded.json`；各回归探针也由本轮源码构建。

继续使用非 login shell；CPU pytest 设置 `TORCH_DEVICE_BACKEND_AUTOLOAD=0`，NPU 在沙箱外执行，质量工具使用 `/tmp/mla-m01-precommit` 及其 cache。版本/tiling 注册警告保留在日志中。下次开发/实机验证前仍须重新检查设备和加载产物，不能把本轮卡号视为持续可用保证。

## 唯一下一步与复现

下一步仅为 **[M07d2](plan.md#下一步m07d2)**：BSND Q 的独立 PA attention，接通 v6 消费与 PA Copy，保留源数值/事件/任务环，完成固定源及双数学 O/LSE、同调度非 FD Kernel 性能。TND attention 和公开 PA 分别留在后续步骤；本轮提交后停止。

本轮最小闭环复核（仓库根目录，非 login shell，不发射 NPU）：

```bash
python3 build/mla-m07d1-paged-copy/check_acceptance.py
PYTHONPATH=/tmp/mla-m01-precommit PRE_COMMIT_HOME=/tmp/mla-m01-precommit-cache \
bash ci/code_quality.sh --staged
git diff --check
git diff --cached --check
```

完整构建/功能复现从本步 `run_checks.py` 与各 `*-command.json` 取命令：`build`、`device`、`cpu-regression`、`consumer-regression`、`copy-regression`、`stride-regression`、`profile`。设备测试固定卡 6；`MLA_PAGED_COPY_REUSE=1` 只加载哈希一致的已构建 probe。重新执行须改用新证据目录，保留既有失败与通过产物；源码、环境或构建变化后重跑受影响检查，历史证据核验不替代新运行。
