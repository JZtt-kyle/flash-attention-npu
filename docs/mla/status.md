# MLA 当前状态

本轮完成 **M07c3：独立 PA 设备消费者**，基线 `602ae5a`，分支 `feat/mla-cann-migration`，日期 2026-10-11。M06 已完成，M07 完成到 v6 独立设备校验；**PA 仍明确拒绝**。唯一下一步是 **[M07d1：独立 PA Copy Tile](plan.md#下一步m07d1)**。

本页只维护当前能力、最近证据、环境差异和下一步。接口、协议与架构查 [design](design.md)，步骤与完成条件查 [plan](plan.md)；当前文档闭环提交用 `git log -1 -- docs/mla/status.md` 定位。

## 当前可调用能力

共同边界为 **910B C1V2、FP16/BF16、512+64 共享 KV、Hkv=1、无梯度、无 mask、`num_splits=1`**。Q/K 末维为 64，qv/V 为 512；四路输入同 dtype、同 NPU 设备。BSND 每样本 KV 容量与 TND 的 `max_seqlen_k` 均为 0..16384；TND 总 Q/KV 各不超过 `INT32_MAX`，总 KV 可以超过 16384。其他字段及资源上限见 [design 6.2–6.3](design.md#62-调度字段与边界)。

| 能力 | 当前接通范围 |
| --- | --- |
| BSND Q/KV | 公开 `flash_attn_func`；只读 `flash_attn_with_kvcache` 支持整容量或实际 KV 长度；内部 Q-used 走 v5 |
| TND Q/KV | 公开 varlen 支持两轴分配前缀及可选 `seqused_q/k`，每次内部生成 v3/v5，无显式 metadata 复用 |
| TND Q + BSND KV | 公开只读 kvcache 使用完整 Q 分配段，支持 v4 生成/复用；内部混合 Q-used 走 v5 |
| 三种布局的 stride | 均已生产接入；四路末维连续、外层非负 stride，含 BNSD/NTD、切片、广播、重叠和独立 offset；发射前校验完整 storage |
| 显式 metadata | 公开生成/复用仅 v2（BSND）和 v4（混合）；可跨合法视图复用，仍须匹配当次长度及 v4 的 Q 前缀 |
| PA | 独立 CPU 地址、CPU/AICPU 调度与 v6 生成，以及 BSND/TND Q-used 的独立设备消费者；未接入 Copy、attention 或公开入口 |

入口参数与 v1–v5 选择统一见 [design 6.7.1](design.md#671-非分页入口与记录选择)，stride 与 PA CPU 组件见 [design 6.4](design.md#64-输入-stride-与地址校验)。调用时还须注意：

- **默认值：** func/varlen/内部前向的 `num_splits` 默认 1；kvcache/metadata 默认 0，MLA 调用须显式传 1。只接受 `causal=False`、`window_size=(-1,-1)`、`softcap=0`、`pack_gqa=None`、`sm_margin=0`、`attention_chunk=0` 和 `rotary_interleaved=True`。metadata 的 `has_softcap=False`，不接受数值 `softcap/softmax_scale` 兼容关键字。
- **长度与 scale：** 长度 tensor 必须同设备、连续、int32，值在设备检查，无 D2H。kvcache 另接受 None/整数长度；生成器必须接收 tensor。前向 scale 默认 `1/sqrt(576)`，显式值须在 FP32 有限范围，metadata 不保存 scale。
- **输出与空值：** O 同输入 dtype，LSE 为 FP32；请求 LSE 时 func/varlen 返回二项，kvcache 返回四项。非零容量中的零 KV 有效行 `O=0、LSE=-inf`；全局空 KV 初始化整个分配区为 `O=0、LSE=+inf`；其他未使用 query 槽位不承诺输出值。空输出仍校验记录和长度，损坏或过期记录由消费者 Trap 拒绝。shape、空 accumulator 和生命周期见 [design 6.7.2](design.md#672-校验输出与发射)、[5.7](design.md#57-stream-与内存生命周期)。

**计划内尚未开放：** PA、causal/window、自动或多 split、192/128 prefill、非零 softcap 与非默认调度选项。fake/compile 已有覆盖，完整 graph replay 留在 M16；910C 尚未开放或实机验收。KV 追加、内置 rotary、量化与 backward 等范围外组合也明确拒绝，完整范围见 [design 第 4 节](design.md#4-功能边界)。

## 本轮核验与证据

新增共用 CPU/设备的 `ReadPagedMetadata` 与 release `LoadPagedMetadata`。使用独立调用 context 和当前三组长度，检查 v6 精确字数、几何、P/C 与源 S2、64 位 Q 起点、分核端点和资源；复用 CPU 地址描述，另验当次页表与 K/V 完整跨度、全部有效页。Q 分配/used 为空仍验有效页，padding 哨兵不影响合法复用；错误在返回调度或写诊断输出前拒绝，见 [design 5.8](design.md#58-v6nd-pa-的独立记录)。

证据位于忽略目录 `build/mla-m07c3-paged-consumer/`，下表路径均相对此目录。保留编译/执行命令、源码快照、固定源、输入/输出、XML、对象与哈希清单；汇总为 `acceptance.json`。这些本机证据不随 Git 分发。

| 核验项 | 结论与证据 |
| --- | --- |
| 上一步交接 | 编辑前重跑 `602ae5a` 的最小闭环证据核验及差异/质量检查；通过。历史 1,754 项 pytest 等仅核验证据，不计作本轮重跑。`prerequisite-command.json` / `prerequisite.log` |
| 本轮功能 | **2,273 项去重 pytest 通过，无失败或跳过**：新 CPU 425、新设备 334、旧 CPU 消费者 559、旧设备消费者 122、普通 FA3/入口契约 833。各套件 `*.xml` / `*-command.json` / 日志 |
| 固定源与精确解码 | CPU 与 910B **各 2,552 组**，双 dtype、BSND/TND、全部 64 种 P、可选 Q-used、非均匀/零尾部；完整 v6 记录与固定源逐字一致，设备各 Cube/Vector 导出的 96 字调度一致。设备原生非 FD 896、无 split 映射 1,656。`cpu-producer/consumer-*.json`、`device/source-*.json`、`cpu-source/`、`device-source/` |
| 页表、长度与资源 | 损坏记录、过期 Q 前缀/used/KV、P/C/S2、编译 dtype、实际 blockDim、端点/零成本后缀、FD 未用槽位与资源拒绝；不同物理池/页映射/独立 stride 复用、页表行广播/重叠、负页/越界页、空池/空表、Q=0/used=0、64 位 Q 起点通过。拒绝时记录与输出/诊断 LSE/workspace 哨兵保持不变。`cpu.xml`、`device.xml` |
| release 真实 Trap | 双 dtype 各 **29 类，共 58 次**隔离进程验收；先用返回状态入口核对拒绝不写输出，再由同一校验的 `LoadPagedMetadata` 执行真实 Trap，经 device/stream/event 同步点报告。`device/error-*/runtime-error.json` / `process.log`；探针 `-DNDEBUG` 见构建清单 |
| 流与生命周期 | AICPU 双事件依赖、两个 worker、跨 current stream、长度只读别名、metadata clone、释放输入与 allocator 重用通过；enqueue/消费期间禁止 host readback。`device.xml` |
| CPU 与 sanitizer | 新 CPU 425 项含 `BuildPagedCacheLayout` 与调用描述一致性、完整 storage/算术边界、64 位 Q/workspace、tile 溢出及失败不改输出；同一校验器 **2,622,720 次 ASan/UBSan/LeakSanitizer 调用通过**。`cpu/build.log`、`cpu/commands.json`、`sanitizer-report.json` |
| 构建与实际加载 | 10 个 MLA ASC 对象及 v6 AICPU 按依赖重编，15 个 ASC 与其余 6 个 AICPU 对象复用；32 个链接对象、host/device 依赖及实际扩展哈希核验通过。`build.log`、`build-provenance.json`、`production-artifacts/`；独立消费者/生成器分别为 `device/` 与 `producer/`，消费者未链接到生产 attention |
| 旧路径回归 | v1–v5 CPU 校验、v2–v5 设备成功/损坏/当前长度/空值/生命周期、普通 metadata、func/varlen/kvcache、双 dtype、softcap/scale、真实/fake/C++/compile 通过；普通路径使用本轮重建扩展。`old-cpu.xml`、`old-device-*.xml`、`ordinary.xml`。旧设备 Trap 与旧耗时对比明确未重跑，本轮 Trap 针对 v6 |
| 校验耗时对比 | 双 dtype 各 12 个固定用例，P=16/112/128/256/528/1024、BSND/TND、短/长 KV、单/多 Q 和零尾部；5 次预热、40 对交替样本。v6/v5 wall 中位数比为 FP16 **1.000–1.717**、BF16 **1.029–1.702**，新增页扫描有开销。`device/comparison-*.json` 保存 wall 与设备事件区间；包含诊断写入、ctypes、stream、事件及同步，不能当作独立 Kernel 或公开 E2E |
| 提交质量与来源 | 本步源码/证据快照与哈希、CANN 固定 blob/原有差异、CATLASS、三文档链接/表格/步骤、差异及适用质量钩子核验。`source-snapshot/`、`source-hashes.json`、`evidence-hashes.json`、`check_acceptance.py`、`quality-staged.log` |

开发期两次中断均已处理：格式化改变头文件哈希后，旧探针被加载核验拒绝，随后重编；同进程第二个 AICPU 生成模块报 `get kernel failed` 后，将 v5 对照记录改为计时前 CPU 生成/上传，避免在消费校验对比中加载第二个生成模块。最终完整设备套件 334 项通过；原始 `initial-full-device*`、`aicpu-comparison-failure-*` 与修复先行 `comparison-smoke-*` 均保留，不重复计数或将失败记为通过。固定 CANN 没有对应的 v6 校验消费者，耗时对照使用既有 v5；双方生成/上传均在计时之外。

## 最近功能闭环与限制

| 闭环 | 历史结果与证据 |
| --- | --- |
| M07c2 独立 AICPU，`602ae5a` | `build/mla-m07c2-paged-aicpu/acceptance.json`：1,754 项 pytest、7,264 组设备固定源差分；本轮生成/消费使用本轮核验的独立生成产物 |
| 最近生产接入：M06e2c2，`538b8e5` | `build/mla-m06e2c2-mixed-stride-production/acceptance.json`：4,659 项去重 pytest；完成三种非分页布局的 stride 生产接入 |
| M06e2c2 公开性能 | 同目录 `performance-acceptance.json`：双 dtype 各 8 个适用公开用例，R=1/4 的复用、生成加前向、内部生成共 96 段逐例达标；FA3/CANN E2E 为 FP16 0.284–0.623、BF16 0.284–0.675。各 3 个原生 FD 用例留待 M10 |

**保留的限制：** M06e2c2 组件 Kernel 比值为 FP16 0.898–1.888、BF16 0.901–2.458，两种 dtype 均有 6 例中的 4 例慢于固定源，不能宣称 Kernel 全部不劣化；profiler 的 `fwk_api_db` 解析错误仍保留，设备 CSV 不代表 host 关联可用。

PA 消费者只读取调度、长度与页表，不读取 cache 内容，不定义 attention 空行 O/LSE。空表/空池仅通过校验层；Copy、attention 精度、真实 cache 大地址及公开 PA E2E 仍待后续步骤。>uint32 cache 地址本轮仅作算术检查，64 位 Q 起点测试也不代表分配了相应 attention tensor。

## 环境与构建

Python `/usr/local/python3.11.15/bin/python3` 3.11.15，torch `2.7.1+cpu`、torch_npu `2.7.1`，CANN `/usr/local/Ascend/cann-9.0.0-beta.2`；CATLASS `769cd40a8716b28650b6bebb08db4834eea4462f` 且干净。`environment.json` / `npu-smi.log` 确认 8 张 Ascend910B3 健康，查询时卡 0 有进程，其余空闲；本轮使用卡 6（进程内设备 0）。

接手扩展 SHA256 为 `a3bb9de632561cf57ffde79163db16d29266d4bb0ea2918a3db1ebfd8e6cbb0b`；本轮重建后实际加载 `flash_attn_npu_3/flash_attn_npu_3.cpython-311-aarch64-linux-gnu.so` 的 SHA256 为 **`0e8f3c6e343fe7aac542622845b65519cb505c55f03d177f3d672d72ba30e74f`**，见 `build-provenance.json` / `production-load.log`。生产扩展含既有 v6 生成器但不发射 PA；消费者只由独立 FP16/BF16 release 探针调用，不能用编译或链接代替公开能力验收。

沿用非 login shell；CPU pytest 设 `TORCH_DEVICE_BACKEND_AUTOLOAD=0`，硬件查询、NPU 和 LeakSanitizer 在沙箱外执行。未修改系统安装；CANN 版本/tiling 注册警告保留在日志中，质量检查使用 `/tmp/mla-m01-precommit` 与 `/tmp/mla-m01-precommit-cache`。

只读 CANN 源仍为 `ca18aee610729ca5270a44e2d84c751bc687fbf0`，原有 67 项差异及本仓原有未跟踪 `kernel_meta/` 保留；没有修改 CATLASS、attention Kernel 数值路径或公开入口。

## 唯一下一步与复现

下一步仅为 **[M07d1](plan.md#下一步m07d1)**：独立 PA Copy Tile，复用当前地址/消费者与既有 Tile 组件，完成全部页大小的双 dtype 搬运精度及固定源代表性 Copy 性能。本轮提交后停止。

最小闭环复核（不加载扩展或发射 NPU）：

```bash
python3 build/mla-m07c3-paged-consumer/check_acceptance.py
PYTHONPATH=/tmp/mla-m01-precommit PRE_COMMIT_HOME=/tmp/mla-m01-precommit-cache \
bash ci/code_quality.sh --staged
git diff --check
git diff --cached --check
```

完整复现见 `run_checks.py` 的 `cpu/device/old-cpu/ordinary/old-device-bsnd/old-device-tnd/old-device-tnd_bsnd/old-device-used` 及对应 `*-command.json`；CPU 本轮设置 `MLA_CONSUMER_SANITIZE=1`。构建与加载见 `build-command.json`、`collect_production.py` 及独立探针清单。后续先核对源码/二进制/输入哈希并使用新的证据目录；本轮没有 PA attention 精度或公开 E2E 结论。
