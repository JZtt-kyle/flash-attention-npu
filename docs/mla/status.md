# MLA 当前状态

本轮完成 **M07c2：独立 PA AICPU 生成器**，基线 `41ff582`，分支 `feat/mla-cann-migration`，日期 2026-10-11。M06 已完成，M07 完成到独立 v6 AICPU 生成；**PA 仍明确拒绝**。唯一下一步是 **[M07c3：PA 设备消费者](plan.md#下一步m07c3)**。

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
| PA | 独立 CPU 地址、CPU/AICPU 调度与 v6 生成（含 BSND/TND Q-used）；未接入设备消费者、Copy、attention 或公开入口 |

入口参数与 v1–v5 选择统一见 [design 6.7.1](design.md#671-非分页入口与记录选择)，stride 与 PA CPU 组件见 [design 6.4](design.md#64-输入-stride-与地址校验)。调用时还须注意：

- **默认值：** func/varlen/内部前向的 `num_splits` 默认 1；kvcache/metadata 默认 0，MLA 调用须显式传 1。只接受 `causal=False`、`window_size=(-1,-1)`、`softcap=0`、`pack_gqa=None`、`sm_margin=0`、`attention_chunk=0` 和 `rotary_interleaved=True`。metadata 的 `has_softcap=False`，不接受数值 `softcap/softmax_scale` 兼容关键字。
- **长度与 scale：** 长度 tensor 必须同设备、连续、int32，值在设备检查，无 D2H。kvcache 另接受 None/整数长度；生成器必须接收 tensor。前向 scale 默认 `1/sqrt(576)`，显式值须在 FP32 有限范围，metadata 不保存 scale。
- **输出与空值：** O 同输入 dtype，LSE 为 FP32；请求 LSE 时 func/varlen 返回二项，kvcache 返回四项。非零容量中的零 KV 有效行 `O=0、LSE=-inf`；全局空 KV 初始化整个分配区为 `O=0、LSE=+inf`；其他未使用 query 槽位不承诺输出值。空输出仍校验记录和长度，损坏或过期记录由消费者 Trap 拒绝。shape、空 accumulator 和生命周期见 [design 6.7.2](design.md#672-校验输出与发射)、[5.7](design.md#57-stream-与内存生命周期)。

**计划内尚未开放：** PA、causal/window、自动或多 split、192/128 prefill、非零 softcap 与非默认调度选项。fake/compile 已有覆盖，完整 graph replay 留在 M16；910C 尚未开放或实机验收。KV 追加、内置 rotary、量化与 backward 等范围外组合也明确拒绝，完整范围见 [design 第 4 节](design.md#4-功能边界)。

## 本轮核验与证据

新增 `mla_paged_metadata.aicpu` 与独立 launch ABI，共用 v6 的 `408+5B` 字序列化和既有源调度。TND 总 Q 在检查输出、dtype 和三组长度描述符后从设备前缀尾值推导；BSND 仍校验显式分配总量。失败清零、非零跨度输入/输出别名拒绝、零计数 used 存在标记和设备写屏障均已验证，见 [design 5.8](design.md#58-v6nd-pa-的独立记录)。没有 PA 设备消费者或公开调用方。

证据位于忽略目录 `build/mla-m07c2-paged-aicpu/`，下表未注明目录的文件均指此处。保留固定源、源码快照、编译/执行命令、输入/输出、XML、链接对象和哈希清单，汇总为 `acceptance.json`；路径相对仓库根目录，不随 Git 分发。

| 核验项 | 结论与证据 |
| --- | --- |
| 上一步交接 | 开发前复跑 `41ff582` 最小闭环检查，核对历史 2,522 项 pytest、32,506 条 sanitizer、固定源、文档、扩展与 CATLASS；通过，不将历史结果计为本轮重跑。`prerequisite-command.json` / `prerequisite.log` |
| 本轮功能 | **1,754 项去重 pytest 通过，无失败或跳过**：新 AICPU 379、v6 CPU 协议 353、构建/入口 sanitizer 25、旧 v1–v5 AICPU 164、普通 FA3/入口契约 833。`device.xml`、`cpu.xml`、`old-metadata.xml`、`ordinary.xml` 及对应 `*-command.json` / 日志 |
| v6 设备固定源差分 | **7,264 组**完整记录与 CPU/固定源逐字一致：原生非 FD 3,648、FA3 无 split 映射 3,616。FP16/BF16 编码、全部 64 种 P、BSND/TND Q、可选 used、非均匀与零尾部；实际平台及 1..26 AIC、C1V1/C1V2 合成描述符。`device/pages-*.json`、`device/cores-*.json`、`device-source/source-manifest.json` |
| 空值、错误及边界 | 空分配/全零 KV、Q-used=0 仍验 KV、P/C/S2、TND 尾值/used/KV 更新、64 位 Q 起点/workspace、BSND 分配总量及零计数别名；42 次非法描述符、6 次非法 dtype 和 66 组调度/长度错误均检查失败清零。`device/empty-*.json`、`descriptors-*.json`、`failures-*.json`；完整用例见 XML |
| 危险输出与生命周期 | 70 组设备诊断用例调用相同入口，返回 1 且输出 guard/输入不变，含空参数、字数/地址错误及三数组重叠；两个 worker、多个 current stream、跨流消费、输入释放和 allocator 重用通过。`device/unsafe-output-*.json`、`device.xml`；这是生成入口返回码验收，不是消费者 release Trap |
| CPU 与 sanitizer | v6 默认 CPU 行为保持兼容，重跑 **8,097 组**固定源差分；真实 v1/v6 AICPU 入口仅替换设备屏障的两组 ASan/UBSan/LeakSanitizer 测试通过。`cpu/`、`cpu-source/`、`cpu.xml`、`sanitizer-report.json` |
| 构建与实际加载 | `setup.py` 纳入 v6；5 组新增传递依赖测试覆盖头文件、ABI、序列化和调度，含缓存命中及缺失依赖/选项证据时重编。生产构建 25 个 ASC 对象复用，v3–v6 AICPU 重编并重链；32 个链接对象、依赖、符号与实际扩展核验通过。`build-command.json` / `build.log`、`build-provenance.json`、`production-artifacts/`；独立探针为 `device/build-manifest.json` |
| 旧路径回归 | 空闲 910B 卡 6 上 v1–v5 独立生成及生命周期、普通 metadata、func/varlen/kvcache、FP16/BF16、softcap/scale、真实/fake/C++ 签名及 compile 通过。`old-metadata.xml`、`ordinary.xml`；普通路径使用本轮重建扩展 |
| 耗时诊断 | 双 dtype 各 12 组固定生成用例：BSND/TND、P=16/112/128/256/528/1024，5 次预热及 20 个 wall 样本。`device/timing-*.json`；包含 ctypes、事件和同步，仅为独立生成诊断，不代表 PA attention Kernel 或公开 E2E |
| 提交质量与来源 | 本步源码/证据哈希、CANN 固定 blob/原有差异、CATLASS、三文档链接/表格/步骤、`git diff --check` 与适用质量钩子核对。`source-snapshot/`、`source-hashes.json`、`evidence-hashes.json`、`check_acceptance.py`、`quality-staged.log` |

初轮设备 377 项通过后补齐 BSND 分配错误及 B=0 零计数别名边界，最终完整重跑 379 项通过；初轮保留在 `initial-device*`，不重复计数。C1V1 仅是同一 910B AICPU 上的合成平台描述符，不表示在 C1V1 芯片完成 attention 验收。所有设备长度均直接传给 AICPU，enqueue 路径禁止 Python D2H/主机等待；CPU oracle 的独立 host 数组不属于生产生成路径。

## 最近功能闭环与限制

以下为历史证据；本轮实际重跑项目以[上表](#本轮核验与证据)为准。

| 闭环 | 结果与可复现证据 |
| --- | --- |
| M07c1 CPU 协议，`41ff582` | `build/mla-m07c1-paged-metadata/acceptance.json`：2,522 项去重 pytest、8,097 组源字段差分、32,506 条 ASan/UBSan/LeakSanitizer；本轮重跑 v6 CPU 协议 353 项 |
| M07b2 CPU 调度，`8fb46d7` | `build/mla-m07b2-paged-schedule/acceptance.json`：1,131 项去重 pytest、7,897 组源字段差分、31,836 条 ASan/UBSan/LeakSanitizer |
| M07b1 CPU 地址，`ccc5fe5` | `build/mla-m07b1-paged-layout/acceptance.json`：354 项 pytest；`sanitized-replay/report.json`：迁移 2,953、固定源 2,270 条 sanitizer 输入/输出一致 |
| 最近生产接入：M06e2c2，`538b8e5` | `build/mla-m06e2c2-mixed-stride-production/acceptance.json`：4,659 项去重 pytest；生产 448、受影响回归 2,684、普通 FA3 833、TND/独立 stride 694 |
| M06e2c2 公开性能 | 同目录 `performance-acceptance.json`：双 dtype 各 8 个适用公开用例，R=1/4 的复用、生成加前向、内部生成共 96 段逐例达标；FA3/CANN E2E 为 FP16 0.284–0.623、BF16 0.284–0.675。各 3 个原生 FD 用例单列，待 M10 |

**保留的限制：** M06e2c2 组件 Kernel 比值为 FP16 0.898–1.888、BF16 0.901–2.458，两种 dtype 均有 6 例中的 4 例慢于固定源，不能宣称 Kernel 全部不劣化。组件 profiler 的 `fwk_api_db` 解析错误保留，设备 CSV 已核验，不据此声称 host 关联可用。BSND/TND stride 的公开 E2E 证据分别在 `build/mla-m06e1c-bsnd-stride/`、`build/mla-m06e2c1-tnd-stride-production/`。

PA 的 CPU 地址/Copy trace、调度和 AICPU 生成只证明对应层，不代替设备消费者、Copy、attention 或 E2E；>uint32 cache 地址仍仅作算术检查。全零 KV 的单校验核/零 workspace 不定义 PA 空行 O/LSE，空表/空池及设备消费行为留在后续步骤。

## 环境与构建

Python `/usr/local/python3.11.15/bin/python3` 3.11.15、torch `2.7.1+cpu`、torch_npu `2.7.1`，CANN `/usr/local/Ascend/cann-9.0.0-beta.2`；CATLASS `769cd40a8716b28650b6bebb08db4834eea4462f` 且干净。`environment.json` / `npu-smi.log` 确认 8 张 Ascend910B3 健康，查询时 0、2 卡有进程，其余卡空闲；本步使用卡 6（进程内可见设备 0）。

接手时扩展 SHA256 为 `38582e6e4deb2eed122a38e5c1da06cd6d5386b4d00bc405f21e1cf469a4a00e`；本轮按真实依赖重建，实际加载 `flash_attn_npu_3/flash_attn_npu_3.cpython-311-aarch64-linux-gnu.so` 的 SHA256 为 **`a3bb9de632561cf57ffde79163db16d29266d4bb0ea2918a3db1ebfd8e6cbb0b`**。v6 AICPU 对象已链接，但生产入口没有发射它；设备验收使用从本轮源码独立编译的 `device/metadata.so`，对象/源码/编译策略与加载路径各自保存，不能用扩展含有符号代替设备功能结论。

硬件查询、设备测试和需要 LeakSanitizer ptrace 的检查在沙箱外执行；CPU pytest 设 `TORCH_DEVICE_BACKEND_AUTOLOAD=0`。沿用非 login shell；CANN 的版本/tiling 注册警告保留在日志中。未修改系统安装，质量检查继续使用 `/tmp/mla-m01-precommit` 与 `/tmp/mla-m01-precommit-cache`。

只读 CANN 源仍为 `ca18aee610729ca5270a44e2d84c751bc687fbf0`，原有 67 项差异及本仓未跟踪 `kernel_meta/` 保留；没有修改 CATLASS 或 attention Kernel/实例。

## 唯一下一步与复现

下一步仅为 **[M07c3](plan.md#下一步m07c3)**：独立 PA 设备消费者，校验记录与当次页表/物理 cache，完成拒绝不写输出、release 真实 Trap 和流生命周期验收；本轮提交后停止。

本轮最小闭环复核（依赖本机证据，不重新导入扩展或发射 NPU）：

```bash
python3 build/mla-m07c2-paged-aicpu/check_acceptance.py
PYTHONPATH=/tmp/mla-m01-precommit PRE_COMMIT_HOME=/tmp/mla-m01-precommit-cache \
bash ci/code_quality.sh --staged
git diff --check
git diff --cached --check
```

完整功能复现使用 `run_checks.py cpu/device/old-metadata/ordinary`，实际命令、环境及退出码见对应 `*-command.json`；构建见 `build-command.json` 和 `device/probe-command.json`。后续使用新的证据目录，先核对源码/二进制/输入哈希，避免覆盖本轮结果。生产 attention 与公开 E2E 性能仍以 M06e2c2 为准，本轮没有新的 PA attention 精度或性能结论。
