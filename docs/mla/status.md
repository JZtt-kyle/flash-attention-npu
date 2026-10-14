# MLA 当前状态

当前分支 `feat/mla-cann-migration`，本步起点 `720fe2d`。**M10c2b2 BSND 自动 split 生产接入已闭环。** 本轮只完成该可调用行为并提交一次，提交后停止；当前闭环提交用 `git log -1 -- docs/mla/status.md` 定位。

## 本步产物与边界

新增 v18 AICPU 与 FP16/BF16 自动 launcher，接通非 PA BSND `num_splits=0` 的 func、只读 cache、metadata、内部 Q-used 和 real/fake/compile。设备顺序执行 init/partial/merge，保留原生 AIC 变长分区与 Vector 8 行 FD 归约。FD 使用紧凑持久槽，Direct 使用独立每核临时槽并写回真实 FP32 第 0 片。共享 partial 增加默认关闭的绝对偏移特化，避免紧凑槽号小于物理核号时的中间减法下溢；共享 merge 增加单片 accumulator 特化。

自动请求返回静态 FP32 `[26,B,H,Sq,512]` / `[26,B,H,Sq]`，实际每 M 片数由原生端点决定；未使用片为 0/-inf，真实/fake shape 一致，不读回长度决定分配。默认 BSND cache/metadata 因此进入自动路径。其他布局 split、负 split 和尚未迁移选项仍明确拒绝，未用占位结果或原 CANN 转调。契约见 [design 6.2.15](design.md#6215-bsnd-自动-split-生产适配)。CANN 源仓、CATLASS 未改。

新增模块单元 `unit/auto_split_bsnd/`、公开/Trap/性能 `e2e/auto_split_bsnd/`、原生片段独立双数学 reference 与工具检查。日常新增 15 个小输入节点，累计 149 项，覆盖自动行为的双 dtype、三种 mask、长度/空值/stride/复用和 fake/compile；未将旧无 mask 测试作为新能力验收。

## 环境与证据身份

接手时分支与交接一致、工作树干净，无遗留 MLA 构建/测试。第 6 张 910B3 空闲，其他任务未触碰。上步 E2E 的 13 个原始文件、915 个源码、6,349 个生产依赖/12 个产物及 230 个加载库完整，7,208 个唯一文件逐项核验，见 `/tmp/mla-m10c2b2/takeover.json`。本步共享 Kernel/构建/入口改变，累计 E2E 本次实跑、继承 0；未累计历史单元或性能。

Python 3.11.15、torch 2.7.1+cpu、torch_npu 2.7.1、CANN 9.0.0-beta.2；CATLASS `769cd40a8716b28650b6bebb08db4834eea4462f`。生产构建 501.8 秒，之后经源/依赖/产物哈希验证复用。实际加载仓库 `flash_attn_npu_3/flash_attn_npu_3.cpython-311-aarch64-linux-gnu.so`，SHA-256 `0eddf65b9508e04f42b1473546bb795c2e6efd31393ebb9f8e6d345da74923c9`；最终 runtime、各组 loaded 与 Trap 报告一致。最终正确性运行时核对 21.3 秒。

证据根 `/tmp/mla-m10c2b2/`；构建/reference 沿用仓库外 `/tmp/flash-attention-npu-mla-46928a793876/mla-validation-artifacts/`。`build/` 仅保留 `mla-codex-loop/`。原始证据清理后不可继续继承。

## 最终验收

全部集合先 dry-run/collect-only；最终实际节点与收集清单逐项一致，普通契约重命名节点另行收集核对。未使用数量截断。代表/最长 KV/跨 M 成本预检 10/10 通过、182.4 秒；最终模块组重新执行同一批代表节点并复用 reference，不另继承旧代表身份。

| 最终集合 | 结果 | 证据目录（相对证据根） |
| --- | --- | --- |
| `auto-split-bsnd-tools` | 19/19，8.2 秒 | `acceptance/attempt-1791999261419447740/` |
| `auto-split-bsnd` | 69/69，160.4 秒 | `final-correctness/attempt-1792000676568266080/` |
| `auto-split-bsnd-trap` | 32/32，423.2 秒 | 同上 |
| 累计 `e2e` | 149/149，27.9 秒，继承 0 | 同上 |
| `ordinary` | 516/516，27.3 秒 | `contract-profile/attempt-1792000326292484930/` |
| `auto-split-bsnd-profile` | 18 个双 dtype 配置全部适用，通过，262.6 秒 | 同上，`auto-split-bsnd-profile/public-profile/average.json` |

10 个固定源数值配置的 O/LSE 均逐元素一致且双数学通过，无本步源精度限制例外；真实 FP32 部分 O、LSE 和 unused 槽另验。设备 v18 AIC/FD 字段与新编译的固定源 SplitCore 比较，partial/FD 实际执行未修改源代码；内部 Q-used、公开 real/fake/compile、复用/生命周期及 32 种 dtype/错误组合均通过。Trap 的 32 份原始 event 错误报告与加载身份均进入正式证据哈希清单。

公开性能平均比值 **0.94816356 ≤ 1**，12 个原生 FD、6 个非 FD，排除 0；6 个配置整体退化、56 个入口/R 分段退化均保留。配置内按适用入口/R 等权，再按配置等权；R=1/4、5 次预热、5×20 wall 和完成同步不变。分段均值：复用 R1/R4 为 0.8974/0.7831，生成加前向 R1/R4 为 1.1704/0.8620，隐式 R1/R4 为 1.0137/0.9245，适用 func R1/R4 为 1.1872/1.0525。三阶段 Kernel 明细、原生分派/库身份与各单例原始数据全部保留，未按耗时筛选。

最终模块 60 次与累计 E2E 303 次 reference 查询全部命中；六组共 460 次命中、8 次 miss/成功写入，新增计算仅为不同的性能输入，无 refresh 或读写错误。片段 reference 按实际 key 区间计算，独立标量对照通过；待测实现始终实际执行。

`final-evidence-check.json` 复核六组 949 个原始文件、当前源码及 probe 依赖/库、生产 6,362 个依赖/12 个产物、230 个实际加载库，共 8,200 个唯一文件，错误 0。文档按本步契约变化核对；链接、仓库质量检查和最终差异记录见 `doc-links-final.json`、`quality-final.log`、`final.patch`。

复现使用新输出目录，先附加 `--dry-run` 和 `--collect-only` 核对，再实跑：

```bash
python3 -m tests.mla --suite auto-split-bsnd-tools --output /tmp/mla-m10c2b2/repro-tools
python3 -m tests.mla --suite ordinary auto-split-bsnd-profile --device 6 --output /tmp/mla-m10c2b2/repro-public
python3 -m tests.mla --suite auto-split-bsnd auto-split-bsnd-trap e2e --device 6 --output /tmp/mla-m10c2b2/repro-correctness
```

## 失败处理与未运行

开发期修复局部变量 `half` 与 dtype 名冲突的编译错误；另一次构建虽成功，但准备期间新增测试改变源码快照，驱动拒绝验收。旧代表运行在 6 项通过后因片段 reference 重复全 KV 计算成本主动中断，优化后新公式身份重新计算并完成代表/完整集合；旧结果不计最终通过。

Trap 初轮在 6 项通过后因原始报告位于 pytest tmp、未进入哈希清单而主动中断（退出 2，108.2 秒），改为正式结果目录后重跑。普通回归随后发现默认自动 metadata 的旧拒绝断言（146 通过、1 失败），改为验证默认 v18 与负 split 拒绝。该测试文件也是模块/Trap/E2E 的辅助依赖，因此此前三组通过身份失效，最终全部重跑；仅无该依赖的本步工具组保留实际结果。生产与 reference 未因这次测试修正改变。所有必要失败均已处理，最终集合无失败或跳过。

历史单元/性能、full-matrix、TND/混合/PA split、其他未开放选项及 910C 未运行，不计通过；本步不是整个 M10 阶段收尾。

## 唯一下一步

[M10c3 TND Q/KV 显式 split 生产接入](plan.md#唯一下一步-m10c3tnd-qkv-显式-split-生产接入)。先核对本步进程、完整证据及实际依赖，再明确紧凑 TND 分配/used、任务/持久地址与 accumulator 的适配边界。本轮提交后立即停止，不开始下一步。
