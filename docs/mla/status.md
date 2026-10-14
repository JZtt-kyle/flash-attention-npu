# MLA 当前状态

当前分支 `feat/mla-cann-migration`，本轮起点 `ff0a64d908ccf943665ef5d958c0f8a9213f3438`。**M10c1a 显式 split 任务归属与持久 partial 布局已闭环。** 当前闭环提交用 `git log -1 -- docs/mla/status.md` 定位。生产能力仍为 M09e 的五布局 mask，公开 `num_splits` 仍只接受 1。

## 本步产物与边界

接手发现独立 partial 每核只处理一片并复用原生 FD 临时槽，独立归并只处理一个 M 块；生产接入尚缺跨波所有权和持久地址。按 plan 已允许的新算法拆分，M10c1a 先闭环此 CPU 协议，M10c1b 再整体接入 metadata、CPU/AICPU、设备调度、生产及公开入口。

新增 `mla_split_kv_plan.h/.cpp`：静态容量给出内存上界，实际 Q 长度给出紧凑 M 前缀；partial/merge 分别按 AIC/AIV 循环归属。每片拥有独立的持久 O/sum/max 槽，不随核复用覆盖；最终 O/LSE/accumulator 保持原分配 Q 地址。复用固定 CANN 块范围/workspace 和 FA3 显式分片/accumulator，原始 Q/KV 长度决定 mask。新增区在源 host reserve 后按 32 字节对齐，检查完整 int64 字节预算、64 位任务编号及失败原子性。协议见 [design 6.2.11](design.md#6211-bsnd-显式-split-任务归属与持久-partial-布局)。

模块归 `tests/mla/unit/split_kv_plan/`。CPU 计划不分配数值 tensor，不生成设备 metadata，不证明设备同步或 Tile 资源可运行；全局空容量的公开初始化仍待接入。未形成新的公开行为，因此不注册伪 split E2E。CANN/CATLASS、已有 Kernel/生产分派和公开参数未修改。

## 接手、环境与证据

接手工作树干净，无遗留 MLA 测试/构建，第 6 张 910B3 空闲。M10b2 累计 E2E 的 13 个原始文件、852 个源码、生产 6,333 个依赖和 12 个产物完整，见 `/tmp/mla-m10c1a/takeover.json`。新增源码/驱动改变保守依赖身份，本步累计 E2E 实跑，继承 0；历史单元/性能继承 0。

Python 3.11.15、torch 2.7.1+cpu、torch_npu 2.7.1、CANN 9.0.0-beta.2；CATLASS `769cd40a8716b28650b6bebb08db4834eea4462f`。初始加载扩展与上步一致。本步统一入口生产重建 487.0 秒、运行时核对 29.3 秒；实际加载仓库 `flash_attn_npu_3/flash_attn_npu_3.cpython-311-aarch64-linux-gnu.so`，新 SHA-256 为 `6fa057a1d524d1a0f610c54f1787e4625a83a89b6d0e662eb7dec3b39c16842d`，与 E2E `loaded.json` 和 `runtime.json` 一致。

证据根 `/tmp/mla-m10c1a/`；构建/reference 沿用 `/tmp/flash-attention-npu-mla-46928a793876/mla-validation-artifacts/`。`build/` 仅有 `mla-codex-loop/`。所有 `/tmp` 原始文件须保留，清理后不能继续继承。

## 本轮验收

所有组均先 dry-run/collect-only。`selection-check.json` 确认普通/sanitizer 的 43 项完全一致，累计 E2E 与上步同为 119 项且清单未变；不设数量截断。

| 本轮实际执行 | 结果 | 耗时 |
| --- | --- | --- |
| `split-kv-plan-tools` | 12/12；选择/恢复、源身份、失败传播和 CPU 退出协议 | 7.7 秒 |
| `unit-split_kv_plan` | 43/43；固定源、任务/地址/空值/溢出与输入所有权 | 24.0 秒 |
| `split-kv-plan-sanitize` | 同一 43/43，ASan/UBSan 及进程退出/泄漏检查通过 | 77.4 秒 |
| 累计小输入 `e2e` | 119/119；本次实跑，继承 0 | 33.2 秒 |

普通组使用 `development-aligned/attempt-1791986123997534050/` 的最后一次实跑；随后实现未变，源码/原始证据逐项核对一致，不重复执行。最终工具为 `cpu-final/attempt-1791986211554909940/`，sanitizer 为 `sanitize-final/attempt-1791986312881279360/`。CPU 证据核对见 `cpu-evidence-check.json`。每组保存固定 Git blob/许可证、编译命令、二进制哈希和完整请求/响应；sanitizer 三个实际执行器分别处理 3,651 / 41 / 1,109 个请求并以 0 退出。

本模块覆盖无 mask/causal/window、有效 Q/KV、零 Q 前缀、空片/全遮罩、2/8/192/193/255/256、C1V1/C1V2 和超核波数。`2^40` 任务编号和接近 int64 的最大字节预算只构造描述符并查询边界，不枚举或分配巨量缓冲。193..256 只对照 CPU 分片公式，不声称执行过该范围的源 FD 归并。CPU 测试不计算数学 reference；E2E 命中既有缓存 266 次、重算 0，待测设备路径实际执行。

最小复现命令（sanitizer 与 E2E 需可访问宿主机进程/设备的环境）：

```bash
python3 -m tests.mla --suite split-kv-plan-tools --output /tmp/mla-m10c1a/tools-repro
python3 -m tests.mla --unit split_kv_plan --output /tmp/mla-m10c1a/cpu-repro
python3 -m tests.mla --suite split-kv-plan-sanitize --output /tmp/mla-m10c1a/sanitize-repro
python3 -m tests.mla --suite e2e --device 6 --output /tmp/mla-m10c1a/e2e-repro
```

最终 E2E attempt 为 `e2e-final/attempt-1791986228937415880/`。`final-evidence-check.json` 核对四组全部原始证据、当前源码、生产构建依赖/产物和实际加载库，未发现缺失或变化；源码冻结后只修改文档。最终差异/链接与 staged 质量检查见 `doc-links-final.json` 和 `quality-final.log`。

## 已处理的失败与未运行项

首次边界检查发现奇数 M 的源 host reserve 不满足新增统计缓冲对齐，已补齐 32 字节对齐并重跑本模块全部 43 项。首次 E2E 收集在沙箱无法读取 NPU，宿主机重做收集已成功。首次 sanitizer 43 项执行后，LeakSanitizer 因沙箱 ptrace 限制在退出时报错，整组记为失败；宿主机完整重跑、保留泄漏检查后通过。失败记录在 `development/`、`collect/`、`cpu-final/`；没有关闭检查来绕过失败。首次质量工具使用缺少 actionlint 的临时缓存无法联网，改用完整已有缓存；实现冻结质量检查见 `quality-freeze.log`。

普通 FA3 专项、历史单元/性能、公开性能、full-matrix 和 910C 未运行，不列为本独立 CPU 组件的通过或继承。设备 partial 的多任务/stride 执行、归并阶段同步、metadata Trap/生命周期、新公开 E2E 和适用平均公开性能均留在 M10c1b，不能以本步 CPU 计划验收替代。

## 唯一下一步

[M10c1b：BSND 显式 split 生产接入](plan.md#唯一下一步-m10c1bbsnd-显式-split-生产接入)。整体完成所需 metadata、CPU/AICPU、设备调度/同步/stride、生产及公开显式 split，新增该能力的日常 E2E 并验收适用平均公开性能；自动 split 和其他布局后续分别闭环。本轮只提交 M10c1a，提交后立即停止。
