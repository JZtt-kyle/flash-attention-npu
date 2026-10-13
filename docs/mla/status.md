# MLA 当前状态

**M08c2 已闭环：非 PA TND Q + BSND KV causal cache、公开 metadata 生成/复用及内部混合 Q-used；唯一下一步是 M08d。** 分支 `feat/mla-cann-migration`，本步基线 `83244ab74690cb5e6d9691fa23cfab3dcc80e6d1`。本步只提交一次，提交后立即停止；当前闭环提交可用 `git log -1 -- docs/mla/status.md` 定位。

## 当前可调用能力

- 非 PA BSND、TND Q/KV varlen、TND Q + BSND KV 只读 cache 均支持 512+64、FP16/BF16、Hkv=1、右下 causal 和 `num_splits=1`，保留原合法四路 stride。无 mask 非分页/PA 能力保留；PA causal 仍明确拒绝。
- 混合 causal 使用独立 v9（`version=kind=9`、`410+8B` 个 int32）。公开生成器在 AICPU 校验描述符后从 Q 前缀推导总分配量；内部前向使用 tensor 静态总量。Q 分配起点与有效长度分离，KV 按物理容量定位、实际长度决定 mask。
- 公开 cache 支持 None/整数/tensor KV 长度和显式 v9 复用；前缀或长度改变后须重建。记录不绑定输入地址、合法 stride 或 scale。内部 Q-used 每次生成 v9，拒绝与显式记录组合；公开 cache/metadata 不增加 Q-used 参数。

协议及生命周期见 [design 5.11](design.md#511-v9混合-causal-记录)，影响路径与验收选择见 [plan 的本步产物](plan.md#最近完成-m08c2非-pa-混合-causal-cache)。CATLASS、Block/Tile、causal 数值任务环和 CANN 源代码未修改。

## 最近闭环证据

证据根 `build/mla-m08c2/`。下表为本次实际执行的 18 个 pytest 组，所选节点全部通过，无跳过或正式验收失败。CPU/工具、代表、设备及性能批次各自保留完整 command、候选/选中/未选中、XML、源/数学结果和依赖身份。

| 检查 | 本次通过数 | 耗时（秒） | 证据目录（相对证据根） |
| --- | ---: | ---: | --- |
| v9 CPU / ASan+UBSan（含退出 LSan） | 76 / 76 | 25.8 / 90.9 | `cpu-final/attempt-1791886737466459190` |
| v7/v5 CPU / ASan+UBSan | 100 / 100 | 55.3 / 163.9 | 同上 |
| v8 CPU / ASan+UBSan | 41 / 41 | 24.0 / 91.8 | 同上 |
| 构建 CPU / 混合性能汇总专项 | 2 / 7 | 3.5 / 3.3 | 同上 |
| 最终工具组 | 100 | 15.1 | `final-tools/attempt-1791887468713825000` |
| 混合双 dtype 代表 | 6 | 101.0 | `representative/attempt-1791886737832350310` |
| 混合公开行为 | 80 | 168.0 | `final-devices/attempt-1791887725272648300` |
| 未开放边界 / 隔离 Trap | 10 / 26 | 20.0 / 362.1 | 同上 |
| BSND / TND causal 代表回归 | 56 / 6 | 106.3 / 98.2 | 同上 |
| 固定非分页 / 普通 FA3 回归 | 86 / 100 | 734.5 / 25.6 | 同上 |
| 公开平均 E2E / 双 dtype 代表组件 | 1 / 2 | 合计 512.5 | `profile-final/attempt-1791889402618531670` |

工具专项与通用工具组有重复节点，不据上表累加唯一用例总数；早期 tools 100 项由最终工具组替代。CPU 候选/选择见 `cpu-collection-final/`；设备集合见 `final-collection/`、`selection-audit.json`。混合公开 80、边界 10、Trap 26、BSND/TND 代表 56/6、非分页回归 86 均全选；ordinary 816 选 100。未选中组合及 M08e 阶段矩阵未运行，不计通过。

`precision.json` 汇总代表、公开和组件：FP16/BF16 各 **593 次 O、593 次 LSE** 同调度固定源比较逐元素一致，双数学全部通过；本步源精度限制为 0。原生公开对照每 dtype 47 组 O/LSE 双数学通过；非逐元素一致仅出现在原生 FD 配置，未将其冒充同调度源一致。

数值与性能组数学缓存合计命中 **357** 次，新增 **156** 个不同内容的 reference；性能组 100 次及非分页回归 130 次全部命中。工具组的故意损坏/失效缓存测试单列，不混入该统计。缓存固定为 `build/mla-validation-artifacts/golden`，实现及固定源仍实际执行。

## 公开性能与组件采样

固定双 dtype 共 **24** 个配置；**16** 个适用配置的平均耗时比 **0.407973 ≤ 1**，退化配置 **0**、退化分段 **0**。其余 **8** 个配置实际启用原生 FD，保留全部结果并单列。输入不按耗时筛选；每配置先等权平均六个入口/R 分段，再等权平均适用配置。

| 分段 | 平均 FA3 / 原生 CANN 耗时比 |
| --- | ---: |
| 隐式生成，R=1 / R=4 | 0.503140 / 0.413498 |
| 一次公开生成加前向，R=1 / R=4 | 0.612117 / 0.315715 |
| 预生成复用，R=1 / R=4 | 0.358435 / 0.244930 |

公开 metadata 生成中位数范围：FP16 221.835–224.800 μs，BF16 210.915–220.155 μs。原生混合 KV gather、必要布局转换、分配与完成同步计入 E2E；真实 bool causal mask、固定 vendor 二进制/加载路径/tiling key 已验证。双方固定 CPU 0，预热 5 次、5 轮×20 个 wall 样本、R=1/4 不变。

三个固定代表组件的生产/同调度源 Kernel 比值：FP16 **1.321114 / 1.317673 / 0.372543**，BF16 **1.167427 / 1.320919 / 0.360091**；小配置 Kernel 较慢如实保留。组件耗时用于诊断，公开平均 E2E 为验收门槛，二者不互代。

原始数据位于 profile 批次 `causal-mixed-profile/public-profile/` 的 `average.json`、逐 dtype `profile.json`、`wall.json`、`metadata-wall.json`、dispatch 证明与 profiler；组件数据在 `probes/<dtype>/mixed-component-profile/`。采样前两次第 6 卡 AICore 为 0，保留 136 MiB 常驻上下文记录；无遗留验证进程，采样前/后设备及进程快照在证据根和各 dtype 目录。

## 证据身份、继承与失败记录

- **本步结果身份：**`final-evidence-audit.json` 核验 18 组的 **3579** 个原始产物哈希。`frozen-sources.json` 为限量验收时 844 个源码文件的身份；最终 `final-sources.json` 另保存质量整理后的身份。公开边界补例前已完成的 CPU/代表，经 `boundary-inheritance.json` 核对既有函数 AST 未变而保留，不重复执行。
- **质量整理后的同一步证据：**首次质量检查发现性能驱动局部变量 F841，将生命周期保留用的 `record` 改名为 `_record`。`quality-inheritance.json` 核对归一 AST 及全部 **26** 个 code object 的执行字节码、常量、参数、全局引用和闭包完全一致，仅一个局部绑定名称改变；性能/精度执行仍适用，未重跑等价采样。原始失败日志 `quality-initial.log` 保留，修复后质量检查通过。
- **继承历史组件 326 项：**M08b2 的 causal attention 324 项与组件性能 2 项，仅用于未变 causal 组件、Block/Tile、任务环及最长 reference/最大内存边界。`baseline-audit.json` 核对 M08c1 的 837 个冻结文件、完整历史原始产物和原扩展；`component-inheritance.json` 核对 15 个算法依赖。新混合协议、地址、传参与公开 E2E 均为本步实测。
- **开发失败：**首轮 v9 CPU 在 66 项通过后，驱动因文本 `2147483648` 无法读入 int32 退出；改用对应 int32 位模式后普通/sanitizer 各 76 项通过。原始日志及 core 位于 `cpu-dev/`、`cpu-dev-core.79`，不计闭环通过。
- **失效尝试：**首轮构建期间格式整理改变源码，执行器拒绝产物身份（`prepare.log`）；随后重建通过。早期 `device-collection/` 在补组/驱动收尾时拒绝 Trap 集合身份变化，最终集合已在冻结实现上重新核对；两者不记为数值回归或有效证据。

复现统一使用 `python3 -m tests.mla_validation --output build/mla-m08c2/<新批次> --device 6 --max-cases 100 --suite <组>`，实际组见上表及 [plan](plan.md#最近完成-m08c2非-pa-混合-causal-cache)，先核对 dry-run/collect。恢复时先核对进程、依赖和已完成记录，只续跑未完成/失效组；性能组不自动继承历史采样。

## 环境、构建与质量

Python 3.11.15、torch 2.7.1+cpu、torch_npu 2.7.1、CANN 9.0.0-beta.2；第 6 卡 Ascend910B3，20 AIC/40 AIV。CATLASS `769cd40a8716b28650b6bebb08db4834eea4462f`；只读 CANN 固定 `ca18aee610729ca5270a44e2d84c751bc687fbf0`，源 oracle 从固定 Git blob 提取。接手工作树干净，无上步遗留验证进程。

实际加载本仓 `flash_attn_npu_3/flash_attn_npu_3.cpython-311-aarch64-linux-gnu.so`，SHA256 `b02bb84a4f0883f907ed1ced4adb50522bc2008acd45eaa1e0632c8c1fca05ee`。代表批次重建 473.2 秒、加载核验 22.1 秒；后续构建缓存逐次核验，完整环境在各批次 `runtime.json`，manifest 在 `build/mla-validation-artifacts/prepare.json`。原生混合使用固定 TND vendor `build/mla-m06c3a-varlen/vendor/mla_ca18aee_transformer`，二进制核验见 `native-vendor-audit.json`，本步实际分派见 profile 证明。

设备与 LSan 需要主机可见性；shell 使用 `login=false`，不 source CANN set_env 或覆盖原 PYTHONPATH。质量入口 `ci/code_quality.sh --staged`，pre-commit 包 `/tmp/mla-m01-precommit` 附加至原 PYTHONPATH 前。质量日志与耗时见 `quality-final.log/json`；三份文档的本步契约、链接和表格及 `git diff --check` 均检查，不增加文档文件。

## 唯一下一步

[M08d：BSND PA causal](plan.md#唯一下一步-m08dbsnd-pa-causal)。本次到 M08c2 提交即停止，不开始 M08d；M08 全组合阶段实测仍在 M08e。
