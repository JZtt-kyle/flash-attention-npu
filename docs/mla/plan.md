# MLA 迭代计划

本页只记录实施顺序、每步产物与完成条件。**开发流程 W01–W05 及功能 M00–M07 已完成；唯一下一步是 M08a。** 当前可调用能力与最近证据见 [status](status.md)，范围、契约及验收口径见 [design](design.md)。流程工具已经接通，后续按影响范围验收，不改变迁移范围。

## 开发流程：W01–W05 已完成

用户已确认先把开发流程完善完整。W01 已落地规则；此前将 W02–W04 合为一个可独立验收的流程工具提交，入口、复用/恢复、计划依次验证后收尾；提交后停止，不开始 M08。

| 步骤 | 产物 | 完成条件 |
| --- | --- | --- |
| W01（已完成） | 按影响范围验收、证据继承及收尾规则 | AGENTS/design/plan/status 一致，文档与质量检查通过；原有 M08 工作保留 |
| W02（已完成） | 版本管理中的统一验收入口，接入既有八个验收组 | 复用已有构建/测试工具；支持预览命令、执行及结构化结果；与 M07e3c 的 `public` 组（历史 2,145 项，不含 Trap）核对用例集合、参数、成功/失败传播和实际产物身份；保留旧证据且不减少测试集合 |
| W03（已完成） | 在已接通的驱动中消除昂贵重复计算 | 优先复用同输入的双数学 reference；已实现分别校验的构建/reference 缓存和按组证据恢复；命中/失效/结果一致性有检查，待测路径仍实测，记录改进前后耗时 |
| W04（已完成） | 调整 M08 提交边界和当前步验收清单 | 核对保留的草稿，以独立组件或可调用行为组织步骤，优先 BSND 非 PA 闭环，再扩展 TND/used 和 PA；每步标明影响路径、测试组、继承依据及完成条件 |

W02–W04 影响测试驱动、PA 数学参考的缓存适配和验收安排，不改生产接口或数值核。统一入口放在 `tests/mla_validation.py`，复用已有 setup、probe builder、pytest 和 golden cache；保持原 public 组 2,145 项及全部参数。按测试组恢复，只有成功记录、用例清单、源码、环境和产物身份均有效时才继承；失败或中断整组重跑，性能采样仍实测。缓存键覆盖实际输入、布局/used/页表/scale 和参考/适配源码，提供关闭与强制刷新。

必跑命令为 `TORCH_DEVICE_BACKEND_AUTOLOAD=0 python3 -m pytest -q tests/test_mla_validation.py tests/test_mla_paged_reference_cache.py tests/common/test_golden_cache.py`，以及 `python3 -m tests.mla_validation --output build/mla-workflow-tools-20261012/run --suite tools cpu public ordinary --device 6`。该步因重建扩展，同时运行原构建基础检查和相关普通 FA3 回归。先对照旧清单与参数，再执行完整 public；检查缓存关闭/命中结果一致性、输入/源码/产物变化失效、失败/中断恢复、并发互斥和实际加载。结果与耗时见 status。缓存适配已覆盖 BSND/TND、used、空 KV 与页寻址，并记录冷/热耗时。M07e3c 仅作为集合和不受影响功能的历史基线，不计本次通过。已有 M08 十二步草稿先存入忽略目录的 `preexisting.patch`，本次 W04 明确重排它；未跟踪代码与原始数据保持原样。

### W05：产物清理与无头接续（已完成）

本步按用户要求清理无用中间产物，并完善仓库外的 `../mla_codex_loop.py`，脚本及其测试不纳入仓库版本管理。影响范围仅为外部开发驱动、忽略规则和流程文档；取消固定五轮文档审计，增加只读启动自检、过期轮次显式归档与恢复基线检查。旧会话和未提交 causal 草稿保留，已完成开发计数不变。

必跑命令为 `MLA_CODEX_LOOP_PATH=/home/zhangqijia/repo/mla_codex_loop.py python3 -m unittest discover -s build/mla-cleanup-20261012 -p 'test_mla_codex_loop.py' -v`：用临时 Git 仓库和假 CLI 覆盖新轮/恢复、单提交门槛、失败传播、计数迁移、锁、归档和只读预览；再检查本机 CLI 登录/参数及原入口 `--check`、`--dry-run --max-dev-runs 1`。清理前记录文件清单，排除当前扩展、活动构建依赖、共享缓存、验收原始数据及源码快照；清理后按哈希核对构建/恢复证据。保留 W02–W04 与 M07 的有效结果，不重新执行未受影响的 NPU 数值和性能矩阵。已完成清理清单与 7,840 个受保护文件哈希核对，16 项外部脚本回归及本机启动自检通过；下一轮从当前状态推进 M08a。本步没有实际启动 M08 开发，结果见 status。

## 后续功能：M08

W04 将原有十二个层次子步整理为下列独立算法或可调用行为。原草稿保存在本轮证据的 `preexisting.patch`；未跟踪 `mla_causal.h` 仍只是待复核草稿。**本轮不实现 M08；流程提交后的唯一下一步为 M08a。**

仅推进 **512+64 右下 causal、num_splits=1**。位置中心为 `i + Sk - Sq`，Sq/Sk 使用逐样本有效长度；分配前缀、真实 stride 和 PA 页寻址沿用现有契约。每项完成实现、适用验收、文档后提交一次并停止。

| 子步 | 产物与影响路径 | 本步必跑与完成条件 | 可继承的证据 |
| --- | --- | --- | --- |
| M08a | 独立 CPU causal 块范围、全遮罩行描述与分核计划；不改生产分派 | 固定源 `CalcS2Range`/mask 成本/分核差分，独立逐行枚举；BSND/TND/used/PA 逻辑长度、不等长/头跨块/尾块/空样本/溢出，ASan/UBSan；新增 `tests/test_flash_attn_npu_v3_mla_causal_schedule.py` 后运行其完整组 | 无 mask 的设备精度和公开性能；前提是公共调用、数值核及其依赖未变 |
| M08b | 首个 BSND 非 PA causal 可调用闭环：Kernel/Block/Tile、所需 metadata/AICPU、生产及 func/只读 cache 入口 | 新 causal 精度/Trap/真实/fake/compile、全遮罩 O=0/LSE=-inf、长度/stride/显式记录与复用签名；双 dtype 固定源/双数学、组件性能和公开 R=1/4 E2E；`ordinary` 及受共享协议影响的 metadata 回归 | M08a 已验证的 CPU 算法及未变化 PA/非 causal 数值组件；协议或公共分派变更覆盖其受影响组合 |
| M08c | 非 PA TND、混合布局及 used causal 完整行为；包含 varlen 和 TND 只读 cache 接入 | allocation/used 决定的独立地址与逐样本对齐、尾块/空行/Trap/生命周期、真实/fake/compile、双 dtype 精度及组件/公开性能；`ordinary`，BSND 受影响回归 | M08b 中未变化的 mask 算法与 BSND 专属路径；新入口与参数传递必须实测 |
| M08d | BSND PA causal：页 Copy、metadata、生产和公开入口一起接通 | 全部合法 P、乱序/重复页、边界/零 KV、记录重用/真实 Trap；双 dtype 精度、普通回归和 R=1/4 逐例公开 E2E | 未变化的非 PA 数值与地址组件；共用头文件/调度变化补受影响组合 |
| M08e | TND PA causal 完整行为，并完成 M08 阶段验收 | TND 前缀和有效长度、全 P、stride/空值/复用/生命周期/Trap/compile；双 dtype 精度及公开 E2E；最后按 design 实测全部已开放 causal 组合、相关普通 FA3 回归及性能矩阵 | 可复用有效构建与数学 reference；阶段验收不继承历史运行结果冒充当次实测 |

M08a 的预定最小命令为 `TORCH_DEVICE_BACKEND_AUTOLOAD=0 python3 -m pytest -q tests/test_flash_attn_npu_v3_mla_causal_schedule.py`；该文件尚未实现，不能提前计为通过。实现前先复核草稿，并把固定源差分、逐行枚举和 sanitizer 编译/运行接入该组。当前已有 `cpu` 组仅检查生成输入和依赖跟踪，不能代替 causal 算法测试。

后续进入每步时，用 `python3 -m tests.mla_validation --output build/mla-<step> --suite <已实现组> --dry-run` 核对命令，并将新 causal 用例注册为有明确集合的组。`public`/`bsnd`/`device` 等现有名称仍表示原 M07 无 mask 矩阵，不能拿它们代替新能力验收。每步新增场景先跑代表用例及最长 reference/最大内存边界，再跑约定集合；矩阵通过后立即收尾。只有独立算法或共享协议确实不能在该步验收时再拆分，并复用前一步未变组件的证据，不恢复逐层重复全矩阵的做法。

所有设备数值步覆盖 Sq<Sk、Sq=Sk、Sq>Sk、多 query、部分/全部遮罩行；未接通组合继续明确拒绝。CPU 描述不绑定数据地址、mask tensor 或 scale，不表示前向已经开放。

共同数学契约见 [design 3.3](design.md#33-共同契约)，代表输入见 [MS01/MS02](design.md#42-代表用例与参考实现边界)。window、split 和 prefill 仍按下表顺序实施。

## 后续顺序

进入每项前，将过大的步骤拆成可独立验收的提交，不提前开放未验收组合。

| 步骤 | 产物 | 完成条件 |
| --- | --- | --- |
| M09 | 512+64 局部窗口 | 有限/单侧窗口、causal 叠加、FA3 归一化和页边界；精度及 E2E 通过 |
| M10 | 512+64 split-KV 与合并 | 源分割/合并流程，自动/1/显式 split、dense/TND/PA/空 split、O/LSE/accumulator/metadata；长 KV E2E 通过 |
| M11 | 192/128 prefill BSND 前向与 metadata | 源通用非量化路径，qv=None、独立 K/V、分组头/真实 stride；O/LSE 及 E2E 通过 |
| M12 | prefill TND 与实际长度 | 非均匀长度、seqused、边界/GQA 与 reference 一致，精度及 E2E 通过 |
| M13 | prefill ND paged cache | 页寻址、内容/RoPE 真实 stride、精度及 PA E2E 通过 |
| M14 | prefill mask 与 split 组合 | 右下 causal/window、split 合并及 metadata 组合；拆分后逐项验收 |
| M15 | MLA softcap 与调度选项 | softcap/has_softcap、pack_gqa、sm_margin 的有效语义与资源校验；适用 E2E 通过 |
| M16 | 图执行与 metadata 生命周期 | eager/图一致，图内更新长度并重算，跨层复用、多 stream、fake/真实输出一致 |
| M17 | 全范围验收与用户示例 | 必迁能力均有用例，精度/逐例 metadata E2E/普通 FA3 回归通过；核对 910B/C 实现及资源兼容，实机验收仍只要求 910B；README 说明调用与边界 |

## 每步完成条件

遵循根目录 [AGENTS.md 的提交门槛](../../AGENTS.md#按影响范围验收) 和 [design 的证据继承规则](design.md#73-增量验收与证据继承)。实现前在当前步补齐改动与影响路径、必跑测试组/命令及选择依据、拟继承证据及依赖、完成条件；结果与耗时只在 status 和证据目录维护。

一次一个独立组件或可调用行为，必跑检查通过后完成文档、质量检查和一个提交。新改动、失败或具体未解决风险才触发补测；阶段结束实测约定完整组合。新增公开路径、设备数值变化及公共入口/metadata 的普通 FA3 回归不能用历史记录替代。失败、未运行、独立 probe 或仅已编译的对象不能记为已开放能力。

## 已完成步骤

这里只保留已交付边界。完整子步拆分、测试数量、性能及排障证据从对应提交的 status 查询。M04c 实际在 M05 后执行，编号保留。

| 阶段 | 已交付产物 |
| --- | --- |
| M00–M03 | 固定迁移交集和 FA3 契约；独立双 reference、固定源执行器/性能基线；MLA 识别、校验及未实现拦截 |
| M04a–M04b3 | dense 来源与分层；无 gert CPU 调度；FP16/BF16 Cube、Vector、非分页 Kernel 与 BSND 生产接入 |
| M05a–M05c2 | v1 生成/设备校验及 release Trap；官方 metadata 签名、普通 FA byte 兼容；公开只读 cache/metadata 复用及 dispatcher 验收 |
| M04c1–M04c4 | S1G 多块、KV 上界 16384、无 split 端点映射及全局空容量 |
| M06a–M06b4 | BSND 实际 KV 长度；v2 CPU/AICPU/消费者/attention 及公开只读 cache |
| M06c1–M06c3c | 紧凑 TND v3、TND Q + BSND KV v4；公开 varlen/混合 cache 和 v4 复用 |
| M06d1–M06d3b2 | 分配/used 分离的 v5；三布局生成/消费/attention，公开 varlen used、内部 BSND/混合 Q-used |
| M06e1a–M06e2c2 | 三种非分页布局的完整 storage/stride、生产接入、真实/fake/compile/流生命周期及适用公开 E2E |
| M07a–M07c3 | ND PA 来源/契约、地址与源 Copy trace、调度、v6 CPU/AICPU 及设备消费者 |
| M07d1–M07d3 | PA Copy Tile、独立 BSND/TND PA attention；双 dtype 全页大小精度、真实 Trap、stride/空值及固定源组件性能 |
| M07e1 | 公开 PA metadata；BSND/TND Q 的 v6 生成、fake/compile、长度更新与生命周期 |
| M07e2a–M07e2c | BSND PA 静态计划、生产前向、只读公开 kvcache 及 R=1/4 公开 E2E |
| M07e3a–M07e3c | TND PA 静态计划、生产前向、只读公开 kvcache 及 R=1/4 公开 E2E；M07 完成 |

M07 的页大小精度覆盖全部 16..1024 的 16 倍数；性能矩阵及原生非 FD/FD 的边界见 [design 7.1](design.md#71-固定性能矩阵与阶段边界)。公开接入包含 None/整数/tensor KV 长度、v6 复用及相关 MLA/普通 FA3 回归，结果只在 status 维护。

历史查询示例：

```bash
git log --oneline -- docs/mla/status.md
git log -G 'M06d3b2' -- docs/mla/plan.md
git show 29ed04f:docs/mla/status.md
```
