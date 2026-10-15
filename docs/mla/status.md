# MLA 当前状态

**M11a：192/128 数值组件已完成。** 分支 `feat/mla-cann-migration`，接手基线 `feb189b`。固定 CANN 通用非量化算法已在 910B 上完成独立 K/V、GQA、三 mask、direct/partial/merge 和双 dtype 组件验收；公开 prefill 仍明确拒绝。唯一下一步是 [M11b：完整 BSND 接入](plan.md#唯一下一步-m11b)，本轮提交后停止。

## 本轮改动与边界

910 目录新增 `MlaPrefill` Kernel、通用 Cube/Vector Block、源 Tile 操作及组件发射参数，保留来源和版权。与既有 512+64 aMLA 不同，该路径使用通用 online softmax、独立 Q/K 192 和 V/O 128，复用已有 stride Copy、window mask 和 FD 归并；共享归并仅新增默认仍为 512 的 `HeadDim=128` 特化。参数是单次组件发射数据，不是新 metadata 协议；资源、事件和源映射见 [design 6.5.1](design.md#651-192128-通用非量化数值组件)。CATLASS 与 CANN 源仓未改。

新增模块独占单元、原生通用分派预检查、固定源 probe、双数学缓存及性能夹具，复用统一驱动和既有采样协议。构建缓存校验源码、头依赖、编译器、dtype/CANN 路径及实际加载库。组件当前支持 M≤512、BRC N≤8192、非 BRC N≤1024；归并测试以 KV head 映射虚拟 batch，BSND 公开头布局和完整生命周期由 M11b 接入。未更改公开 real/fake/metadata 行为，未注册尚未接通的 prefill E2E。

## 本次闭环证据

证据根 `/tmp/mla-m11a/`，最终组件为 `acceptance-final/summary.json`，回归为 `regression/summary.json`。全部先 dry-run/collect-only，固定实现后执行；未使用 `--resume`、`--max-cases` 或完整矩阵扩展，继承通过结果为 0。

| 检查 | 结果 | 用时 |
| --- | --- | --- |
| prefill 工具/缓存/失败传播 | 22 passed | 7.8 s |
| prefill 模块单元 | 56 passed：27 数值配置 × 双 dtype、2 发射失败原子性 | 253.7 s，含组件重建 |
| 性能廉价夹具 | 2 passed，双 dtype 的 attention/merge 全部固定输入 | 26.3 s |
| 组件性能 | 2 passed，12 attention + 6 merge 配置及迁移适配成本 | 67.0 s |
| 累计日常 MLA E2E | 314 passed | 68.0 s |
| 普通 FA3 回归 | 516 passed | 27.5 s |

各组均 0 failed/skipped。54 份数值报告共 1860 个双数学对照状态全部 passed，无源精度限制豁免；固定源逐元素检查、空行 inf 符号、raw sum/max 八路重复、accumulator、重复执行、输入只读和 guard 均通过。193..256 片超过原生归并缓冲域，具名 256 片用例按数学与空行契约验收，不冒充原生逐元素覆盖。最终组件 reference 54 次全部命中；E2E 281 次命中、119 次生成，普通 FA3 31 次命中，待测实现每次实际执行。

代表及最大成本检查在 `check-platform/`：10 项通过后才运行完整模块。成本输入 `B=1,Sq=257,Hq=8,Sk=4097,Hkv=2,M=512,N=512,S=8` 首次 reference 检查约 14.5 s；最终缓存后的 FP16/BF16 用例分别 1.95/2.56 s，进程最大 RSS 约 2.31 GiB。未因耗时截断覆盖。

以下比值为迁移/固定源、配置等权平均；所有单例、退化和原始 profiler/wall 样本保存在 `acceptance-final/` 下 `prefill-profile/probes/<dtype>/profile/`。

| 组件 | dtype | Kernel 均值 | probe wall 均值 | Kernel 退化数 |
| --- | --- | --- | --- | --- |
| direct/partial attention | FP16 | 0.9998 | 0.9973 | 4/6 |
| direct/partial attention | BF16 | 0.9827 | 1.0019 | 2/6 |
| 128 维 merge | FP16 | 1.8487 | 1.0013 | 3/3 |
| 128 维 merge | BF16 | 1.8526 | 1.0048 | 3/3 |

组件使用 5 次预热、5×20 wall、5×10 Kernel，同片段、预分配、输入转换在计时外，保留源输出初始化。merge 成对采样使用相同规范统计；迁移行适配及 accumulator 写入另以 5×20 wall 记录在 `adaptation-wall.json`，各输入中位数为 156.9–166.5 μs。merge Kernel 退化完整保留；本轮没有公开 metadata E2E 性能声明，M11b 必须实测适用公开平均门槛。

`final-check.json` 复用统一驱动函数核对最终六组原始证据、源码、依赖、收集/实测清单和结果，以及组件/生产构建、实际加载库和 native 包，全部有效。接手 M10 的 13 组正确性及构建证据核对在 `handoff-check.json`；共享归并和构建输入改变后，本轮重新执行累计 E2E，不继承历史单元或性能。

## 失败修正与未运行项

首次扩大代表集合在 `check-expanded/` 的 FP16 成本用例触发源 `InitOutput/SyncAll` 超时：夹具将协议容量 26 当成发射核数，卡 6 实际只有 20 AIC。改为既有平台查询并按物理核数分波后，`check-platform/` 及最终全部数值检查通过；没有改数值算法绕过失败。早期工具 mock 误取设备锁、`profile.py` 遮蔽标准库的收集失败，以及开发期编译错误均已修正；旧日志保留在本轮证据根，不计为通过。

初次性能 `performance/` 把迁移行适配和 accumulator 写入混入 merge 成对采样；最终按既有组件边界分离，并重新执行廉价夹具与完整采样。`unit/`、`tools-full-retry/`、`tools-cache-final/` 等开发检查不替代最终身份的证据。

本轮未运行历史数值/协议单元、历史公开专项/性能或阶段矩阵；共享路径回归走累计 E2E 与普通 FA3。尚未开放 192/128 公开前向、softcap、非默认调度；完整 graph replay 和 910C 实机仍未验收。M11b 须一次接通 BSND 的 host/AICPU/Kernel/metadata、real/fake/compile、三 mask 和自动/1/显式 split，新增公开 E2E 及适用平均性能。

## 环境与复现

Python 3.11.15、torch 2.7.1+cpu、torch_npu 2.7.1；CANN `/usr/local/Ascend/cann-9.0.0-beta.2`，CATLASS `769cd40a8716b28650b6bebb08db4834eea4462f`。卡 6 为 Ascend910B3、20 AIC，性能进程持有统一驱动卡锁并保存设备快照。固定源始终为 Git blob `ca18aee610729ca5270a44e2d84c751bc687fbf0`，不依赖 CANN 源仓当前 HEAD。

生产扩展已重建并实际加载 `/home/zhangqijia/repo/flash-attention-npu/flash_attn_npu_3/flash_attn_npu_3.cpython-311-aarch64-linux-gnu.so`，SHA256 `23f3468d4c302c579a7eb122520fa2ba76181f4f67b5dfefe7c372fb8761abfb`；准备 596.8 s、runtime 核验 21.6 s，回归复用构建 1.2 s、runtime 21.7 s。

仓库外 `MLA_WORK_DIR` 为默认 `/tmp/flash-attention-npu-mla-46928a793876`：生产/reference 在 `mla-validation-artifacts/`，组件构建在 `prefill-probes/`。native 包复用 `/tmp/mla-m10c3b/native/opp/vendors/mla_ca18aee_transformer`，既有 103/104 配方足够；`native-fp16-dispatch.json` / `native-bf16-dispatch.json` 证明六个 combined/split/causal 请求实际执行固定通用 template 3，绑定原始日志和二进制哈希。`build/` 仅保留无头日志/状态，未改循环计数；临时证据清理后不可继续继承。

```bash
python3 -m tests.mla --suite prefill-tools --output /tmp/mla-m11a/repro-tools
python3 -m tests.mla --suite prefill-check --device 6 --output /tmp/mla-m11a/repro-check
python3 -m tests.mla --unit prefill_attention --device 6 --output /tmp/mla-m11a/repro-unit
python3 -m tests.mla --suite prefill-profile-check prefill-profile \
  --device 6 --output /tmp/mla-m11a/repro-profile
MLA_NATIVE_ROOT=/tmp/mla-m10c3b/native python3 -m tests.mla \
  --suite e2e ordinary --device 6 --output /tmp/mla-m11a/repro-regression
```

复现先同参数 dry-run/collect-only；实机及需查询芯片的收集在宿主运行。原生通用分派检查入口为 `python3 -m tests.mla.support.mla_prefill_native --help`。日常仍为 `python3 -m tests.mla --device 6`。本次三份文档的契约、表格及变更行 16 个链接/锚点已核对；`ci/code_quality.sh --staged` 和 `git diff --check` 均通过，质量日志为 `quality-final.log`。
