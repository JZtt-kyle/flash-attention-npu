# MLA 当前状态

**M15b BSND softcap 完整公开接入已完成。唯一下一步是 [M15c 非 PA TND Q/KV softcap](plan.md#唯一下一步-m15c)，本轮提交后停止。** 本轮从 `feat/mla-cann-migration` 的 `581acbf` 干净工作树开始。

## 当前能力与边界

512+64 和 192/128 的非 PA BSND 支持正 softcap，包含三 mask、自动/1/显式 split、真实 accumulator、metadata、公开 real/fake/compile 与 raw C++ 前向。压缩使用 v32/v33，prefill 使用 v34；记录只绑定是否启用，正 cap 可改值复用。压缩单片复用 partial/merge，公开 accumulator 仍为空。数值 Tile/Block 沿用 M15a，CANN 源仓和 CATLASS 未修改。

其他布局的正 cap、非默认调度和 graph replay 仍未开放；910C 未实机验证。契约见 [design 2.2](design.md#22-后端选项与返回值) 和 [5.16](design.md#516-bsnd-softcap-记录)。

## 最近一次闭环证据

证据根 `/tmp/mla-m15b/`，测试均通过 `python3 -m tests.mla`。每个组保留实际用例、来源/依赖/构建身份和原始证据；收尾使用驱动既有 `suite_sources`、`reusable`、`matching_files` 等检查，20 组当前来源与依赖一致，生产输入、依赖和产物有效，结果见 `closeout-validation.log`。

| summary / attempt | 本轮结果与耗时 |
| --- | --- |
| `final-cpu/summary.json` / `1792138609433064070` | 工具 27 passed / 28.0 s；softcap CPU 56 passed / 77.6 s；ASan/UBSan 56 passed / 253.4 s |
| `plan-boundary/summary.json` / `1792138433395706410` | 共享 split plan 43 passed / 25.3 s；ASan/UBSan 43 passed / 80.2 s，含 S=1 分片/任务/持久工作区及公开空 accumulator 边界 |
| `final/attempt-1792135769830128610/summary.json` | 公开专项 168 passed / 309.3 s，含 16 个隔离设备 Trap；累计小输入 E2E 515 passed / 102.7 s。该 attempt 随后的旧拒绝断言失败已另行修正，不把整个 attempt 记为通过 |
| `final/summary.json` / `1792136350004320400` | 驱动校验并继承上述本轮 168/515 项；ordinary 实跑 508 passed / 26.0 s；native BSND 代表夹具 1 passed / 209.7 s；正 cap 性能 4 passed / 143.0 s，覆盖 12 配置 |
| `perf-prefill/summary.json` / `1792136835730694640` | 五布局各 12 配置全部实测，分组耗时 250.1–271.0 s；既有汇总器通过 |
| `perf-compressed/summary.json` / `1792138991141287020` | 四组 26 配置全部实测，共 954.5 s；指定组汇总通过 |
| `subset-tools/summary.json` / `1792140184435410790` | split 汇总工具 213 passed / 113.3 s；覆盖旧阶段与指定组权重、成功/失败传播、缺失/损坏证据 |

只继承了本轮身份及原始证据均有效的 E2E。历史 M15a 单元/性能没有累计继承；新增包文件使早期 CPU/工具身份变化后已在最终源码重跑。代表实机和性能夹具先完成，随后完整公开专项及正式采样覆盖最终身份。数学 reference 使用内容缓存，待测实现实际执行；最终 ordinary 命中 31 项、正 cap 性能命中 12 项。

ordinary 从 516 项变为 508 项：移除 8 个已过期的 BSND softcap 拒绝用例，pairwise 代表随选项轴调整，实际集合已核对。新能力有独立的三 mask/split 专项及日常 E2E。早期 helper 参数名、pytest 同名模块、旧拒绝断言和单片计划测试假设均已修正；sandbox 的设备不可见/LeakSanitizer ptrace 限制通过沙箱外实际运行解决，未关闭 sanitizer。首次编译后的准备记录因期间测试源码变更被拒绝，随后按 manifest 校验复用并核对实际加载；失败日志仍保留。

## 公开性能

采样前固定 98 个目标配置（含 dtype）：正 cap 12 项、受共用归并影响的 prefill 五布局 60 项、受显式 BSND/共用自动 launcher 影响的压缩四组 26 项。R=1/4、预热 5 次、wall 5×20、kernel 5×10 保持不变，未按耗时筛选或截断。

| 集合 | 实测 / 同义适用 / 单列 | 配置平均耗时比 | 退化配置 |
| --- | --- | --- | --- |
| prefill 五布局 | 60 / 30 / 30 | 0.802877 | 0 / 30 |
| 压缩显式 BSND | 8 / 2 / 6 | 0.696575 | 0 / 2 |
| 压缩自动 BSND | 6 / 6 / 0 | 1.025036 | 2 / 6 |
| 压缩自动 TND | 6 / 6 / 0 | 0.789339 | 2 / 6 |
| 压缩自动混合 | 6 / 6 / 0 | 0.670635 | 2 / 6 |
| **压缩四组整体** | **26 / 20 / 6** | **0.815160** | **6 / 20** |

`prefill-performance.json` 和 `compressed-performance.json` 从绑定证据重算。压缩使用 `selected-groups` 明确标识本步固定子集，不能称为 M10 阶段收尾；组均值仅作诊断，按配置等权整体平均≤1 验收。prefill/压缩分别有 35/28 个退化入口分段，所有单例、分段和原生 FD 单列结果保留。prefill 的 60 个 kernel 对照均退化，公开平均通过不代表 kernel 全部提速。

正 cap 没有同义 CANN attention。12 个配置的 cap=15/cap=0 公开调用平均比为 1.049269，9 个高于 1，仅作已迁移实现的开销参照，不计同义 CANN 门槛；显式复用 kernel 为 72.46–205.92 μs。metadata、各入口/R、wall 原样本和 profiler 分段在 `final/attempt-1792136350004320400/softcap-bsnd-profile/` 保留。

## 复现与收尾

```bash
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export MLA_NATIVE_ROOT=/tmp/mla-m13b/native
python3 -m tests.mla --suite softcap-bsnd-tools unit-softcap_bsnd softcap-bsnd-sanitize --output /tmp/mla-m15b/final-cpu
python3 -m tests.mla --suite unit-split_kv_plan split-kv-plan-sanitize --output /tmp/mla-m15b/plan-boundary
python3 -m tests.mla --suite detail-softcap_bsnd e2e ordinary prefill-bsnd-profile-check softcap-bsnd-profile --resume --output /tmp/mla-m15b/final
python3 -m tests.mla --prefill-stage-performance --suite prefill-bsnd-profile prefill-tnd-profile prefill-mixed-profile prefill-paged-bsnd-profile prefill-paged-tnd-profile --output /tmp/mla-m15b/perf-prefill
python3 -m tests.mla.support.mla_prefill_stage_performance --evidence /tmp/mla-m15b/perf-prefill/summary.json --output /tmp/mla-m15b/prefill-performance.json
python3 -m tests.mla --split-stage-performance --suite split-bsnd-profile auto-split-bsnd-profile auto-split-tnd-profile auto-split-mixed-profile --output /tmp/mla-m15b/perf-compressed
python3 -m tests.mla --suite split-stage-tools --output /tmp/mla-m15b/subset-tools
python3 -m tests.mla.support.mla_split_stage_performance --evidence /tmp/mla-m15b/perf-compressed/summary.json --suite split-bsnd-profile auto-split-bsnd-profile auto-split-tnd-profile auto-split-mixed-profile --output /tmp/mla-m15b/compressed-performance.json
```

必跑集合已 dry-run/collect 核对，native 包和代表夹具先行。提交收尾使用 `git diff --check` 与 `PYTHONPATH=/tmp/mla-m10c8a/quality ci/code_quality.sh --staged`，质量输出保留在 `quality-final.log`；本次影响的文档链接记录在 `design-links.log` / `handoff-links.log`。当前闭环提交可由 `git log -1 -- docs/mla/status.md` 定位。

## 环境与产物

Python 3.11.15、torch 2.7.1+cpu、torch_npu 2.7.1、CANN `/usr/local/Ascend/cann-9.0.0-beta.2`、CATLASS `769cd40a8716b28650b6bebb08db4834eea4462f`，卡 6 Ascend910B3。固定 CANN 源仓为 `ca18aee610729ca5270a44e2d84c751bc687fbf0`。生产扩展已重建，驱动实际加载本仓 `flash_attn_npu_3/flash_attn_npu_3.cpython-311-aarch64-linux-gnu.so`，SHA256 `bda4b48f9ddbe340d98b907d0e8096e214239aacfbce71a7a2ea213b05ea81aa`。

默认工作根 `/tmp/flash-attention-npu-mla-46928a793876/` 保存生产 `prepare.json` 与数学缓存；native 包使用 `/tmp/mla-m13b/native`，默认旧包配方已过期，不能误用。`build/` 仅保留无头循环日志/状态。`/tmp` 非持久存储，证据缺失或内容变化后不可继承。
