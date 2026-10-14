# MLA 当前状态

当前分支 `feat/mla-cann-migration`，本轮起点 `d14eb492258bdf07647c04a93bf82959a382c4ac`。**M10b2 独立 FD 归并与 BSND FA3 accumulator 适配已闭环。** 当前闭环提交用 `git log -1 -- docs/mla/status.md` 定位。生产能力仍为 M09e 的五布局 mask 能力，公开 `num_splits` 仍只接受 1；本步没有开放 metadata 或公开 split。

## 本步产物与边界

新增独立 Kernel / Block / Tile 归并和 FP32 accumulator 写回，保留源 max/sum 权重、按片顺序乘加、O 双缓冲及最终 dtype 舍入。按原始片区间与逐行 mask 排除无贡献片，空行 O=0、LSE=-inf；有效片 O 已归一化，不重复除 sum。1 片不写 accumulator，2..256 按 BSND `[S,B,H,Q,512]` / `[S,B,H,Q]` 写回。具体协议和来源见 [design 6.2.10](design.md#6210-独立-bsnd-fd-归并与-accumulator-适配)。

模块归 `tests/mla/unit/split_kv_merge/`，新构建且实际执行双方 partial 产生输入，再运行固定源 FD Block 与迁移归并。源 6 KiB 缓冲在一行时最多表示 192 片，超域 oracle 明确拒绝；193..256 的迁移扩展使用双数学验收。全空行独立检查源有限负 LSE 与迁移 -inf 的适配。未接通公开路径，不注册伪公开 E2E；CANN/CATLASS、既有 partial 算法与公开分派未修改。

## 接手、环境与实际加载

接手工作树干净，无遗留 MLA 测试/构建，第 6 张 910B3 空闲。M10b1 累计 E2E 的 13 个原始证据文件与源码哈希完整，见 `/tmp/mla-m10b2/takeover.json`；本步保守依赖身份改变，累计 E2E 实跑，继承 0，历史单元/性能继承 0。

Python 3.11.15、torch 2.7.1+cpu、torch_npu 2.7.1、CANN 9.0.0-beta.2；CATLASS `769cd40a8716b28650b6bebb08db4834eea4462f`。初始扩展与上步吻合，记录在 `environment.json`。本次生产重建 505.8 秒，运行时加载检查 20.8 秒；实际路径为仓库 `flash_attn_npu_3/flash_attn_npu_3.cpython-311-aarch64-linux-gnu.so`，新 SHA-256 为 `b630d616a60c93b38b68a9979faea3aad401572734a80685d78b58389637428c`，与各设备组 `loaded.json` 一致。固定 CANN 源仍从 `ca18aee610729ca5270a44e2d84c751bc687fbf0` Git 对象提取，不用源仓当前 HEAD 代替。

证据根 `/tmp/mla-m10b2/`；构建/reference 工作目录 `/tmp/flash-attention-npu-mla-46928a793876/mla-validation-artifacts/`。生产构建身份在该目录 `prepare.json`，各模块 probe 的 source、dependency、library 和 loaded 身份随组保存。`build/` 只保留 `mla-codex-loop/`。这些 `/tmp` 证据清理后不能继续继承。

## 最近一次闭环验收

所有组先 dry-run/collect-only，最终清单见 `final-dry-run.json` 与 `final-collect/attempt-1791983640907410700/`。最终工具 attempt 为 `tools-final/attempt-1791983634854243700/`，设备 attempt 为 `acceptance/attempt-1791983761761732450/`。

| 本次实际执行 | 结果 | 耗时 |
| --- | --- | --- |
| `split-kv-merge-tools` | 9/9；选择/恢复、失败传播、数学及构建/加载身份 | 5.1 秒 |
| `split-kv-merge-check` | 10/10；双 dtype 代表、最长 KV 和 256 片成本边界 | 127.2 秒 |
| `split-kv-merge` | 36/36；其余真实 partial、合成统计量、空值/guard/重复调用和失败原子性 | 127.9 秒 |
| `split-kv-merge-profile` | 2/2；双 dtype 共 6 个固定配置采样及原始 profiler 对应检查 | 137.1 秒 |
| 累计小输入 `e2e` | 119/119；本次实跑，继承 0 | 24.7 秒 |

模块正确性共 46 项：44 个数值配置及双 dtype host 拒绝检查。38 个数值配置运行可比固定源 FD，另 6 个 255/256 片配置明确拒绝源归并并通过双数学；真实 partial 均由双方实际执行。原始输入的完整 attention、独立归并和 accumulator LSE 数学检查全部通过，源精度限制例外为 0。原始输入、partial、结果、源、有效性和双 reference 保存在各组 `probes/<dtype>/accuracy/*.{pt,json}`。

最终代表 reference 命中 36，边界组命中 62、首次计算并写入 74，累计 E2E 命中 266；相同输入在开发/代表/验收间复用，没有重复计算 reference，设备仍实际执行。最长 KV 的首次 FP16/BF16 reference 检查为 29.2/36.5 秒，最终命中后检查为 0.23/0.22 秒；最终 256 片检查约 0.30 秒，正确性进程峰值 RSS 2,134,792 KiB。这里的单例时间不含生成 partial；上表为完整组时间，含模块冷构建。

`final-evidence-check.json` 逐项核对 2,039 个原始证据条目、各组当前源码、生产 6,333 个依赖和 12 个产物，以及 probe/扩展实际加载哈希；无缺失或变化。最终实现冻结后仅更新文档。源码质量检查见 `final-freeze-quality.log`，提交前 staged 质量检查见 `quality-final.log`。

复现命令：

```bash
python3 -m tests.mla --suite split-kv-merge-tools --output /tmp/mla-m10b2/tools-final
python3 -m tests.mla --suite split-kv-merge-check split-kv-merge split-kv-merge-profile e2e --device 6 --output /tmp/mla-m10b2/acceptance
```

## 组件性能与未开放项

性能在第 6 张卡隔离采样，CPU affinity 为 0；每配置预热 5，五轮各 20 wall/10 profiler 样本。比较同一规范化 FP32 O/sum/max 的首个 M 块，partial、行规范化、分配及 H2D 在计时外；wall 包含 ctypes、发射与完成同步。原始 CSV、逐轮样本、输入和调度位于 `split-kv-merge-profile/probes/<dtype>/profile/`。

| 固定配置 | FP16 kernel / wall 比 | BF16 kernel / wall 比 | FP16 / BF16 适配调用中位数 |
| --- | --- | --- | --- |
| merge-tail，S=2、M=16 | 1.7918 / 1.0170 | 1.7993 / 1.0037 | 140.030 / 145.335 μs |
| band-partial-empty，S=8、M=17 | 1.4340 / 0.9911 | 1.4245 / 1.0002 | 163.470 / 162.025 μs |
| longest-kv，S=2、M=256 | 1.8993 / 1.1387 | 1.8680 / 1.1518 | 216.730 / 212.290 μs |

六配置等权平均 kernel 比 **1.7028**、wall 比 **1.0504**，分别退化 6/6、5/6，组件仍存在性能差距。最后一列单列迁移原始行适配加 accumulator 写回的调用成本，每项 100 样本；源没有 FA3 accumulator，不填充对照比值。本步完成组件采样，不宣称公开平均性能通过；公开接入时仍须满足 [design 第 7 节](design.md#7-验收口径) 的适用门槛。

普通 FA3、历史单元/性能、公开性能、full-matrix 和 910C 实机未运行，不列为本独立组件的通过或继承。公开自动/显式 split、其他布局归并、metadata 和全局零 KV 的公开快速路径留待接入验收，不能将独立 probe 当成完整 M10 阶段完成。

## 已处理的开发失败

首次 probe 缺少 CATLASS 标量 helper 和固定源头文件依赖；随后发现按值端点数组超过编译器 1536 字节限制，改为在 KV≤16384 契约内使用 uint16 并加入静态检查。一次 runtime 检查因源码仍在修改而被身份检查正确拒绝。失败记录保留在 `development/`，后续冻结验收已解决。开发代表通过后复核补齐源 O 双缓冲、按 query 写回和 M10a 行宽/字节溢出检查，因此最终重跑代表；复用原数学缓存。没有未解决的验证失败。

## 唯一下一步

[M10c1：BSND 显式 split 生产接入](plan.md#唯一下一步-m10c1bsnd-显式-split-生产接入)。在非 PA BSND 同时完成所需 metadata、CPU/AICPU、生产及公开显式 split，新增该能力的日常 E2E 并验收适用公开性能；自动 split 和其他布局后续分别闭环。本轮只提交 M10b2，提交后立即停止。
