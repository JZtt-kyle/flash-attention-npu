# MLA 当前状态

**M17 全范围收尾完成，图性能差距按本轮用户授权保留为已知限制。** 用户于 2026-10-18 确认“完成 M17 全范围收尾，合并本次文档核对”，在完整实测与质量检查后确认“接受已知限制并提交”。分支 `feat/mla-cann-migration`，基线 `91a1aaa`。本轮合并 M16 后的专门文档核对，完成条件见 [plan](plan.md#m17-全范围收尾)，最终覆盖映射见 [design 7.7](design.md#77-全范围收尾映射)。

## 本步改动

`DensePlatform` 接受 CANN 的 `Ascend910_93`（910C）系列名称，保留既有 910B 分支与所有 C1V2、核数、内存和 blockDim 校验。没有改数值 Kernel/Block/Tile、metadata 协议、CANN 源仓或 CATLASS。新增平台检查编译实际生产查询函数，使用安装 CANN 的 12 种 B/C 配置验证资源及普通/全核发射，并覆盖非法名称和资源拒绝；910C 仍不作实机验收承诺。

中英文 README 同步两种表示、五布局、softcap、调度、图与明确拒绝边界。可运行示例 `python3 -m tests.mla.support.mla_examples --device 0 --dtype fp16`（另跑 bf16）覆盖压缩 dense、prefill varlen、分页 metadata 复用和动态长度图重放；对应双 dtype E2E 已注册日常显式节点。累计日常集合从 871 增为 873，增加的两项仅为示例验收。设计新增最终能力到验收组的映射；历史运行细节保留在 Git。

## 本次验收结果

证据根 `/tmp/mla-m17/`。`ci/code_quality.sh --staged` 的 14 文件质量检查和 `git diff --check` 通过，日志为 `quality-final.log`；本次涉及的本地链接/锚点和两份 README 示例语法核对通过（`docs-review.log`）。正式构建进程 599.2 s 成功退出；期间工具身份补强使外层准备记录失效，冻结后按内容复核构建 1.3 s、实际加载核验 21.1 s，均通过，没有重复编译或沿用旧 `.so`。

- `final-tools/attempt-1792345331586356370`：平台工具 3、通用工具 308、平台 CPU 41、ASan/UBSan 41 项通过，分别 5.9/148.8/6.4/196.0 s。
- `dry-run.json`、`collect/attempt-1792345291606484870` 与 `collect-frozen.log`：14 个正式组命令及收集核对完成；`selection-check.log` 与 M16 清单对照，没有删除或更换旧用例，仅日常增加双 dtype 示例。性能集合仍为图 40 + 调度 56，共 96 配置。
- `final/attempt-1792345651138391150`：构建与实际加载核验通过；双 dtype 示例 2、公开代表 90、固定源 32、边界 100、图代表 13 项通过，分别 21.2/59.8/40.0/21.8/26.2 s；累计 E2E 873、调度专项 402、图专项 233、普通 FA3 504、普通图 2 项全部通过，分别 216.4/475.8/285.8/25.8/20.7 s。
- `profile-preflight-retry/attempt-1792347313944188790`：图/调度预检均通过，分别 226.0/106.2 s，277 个文件归档并逐文件校验。此前 `profile-preflight/attempt-1792346987814086890` 的图 native 数值与真实重放完成，但 profiler 原始 `op_summary` 有一条任务开始时间重复、持续时间溢出的记录，既有模型分组正确拒绝，图预检失败（239.2 s）、后续调度预检未运行。失败原始记录与 `profiler-preflight-diagnosis.log` 保留；重采没有修改实现、集合或阈值。
- `correctness-evidence-check.log` 与 `performance-evidence-check.log`：统一驱动复核上述工具、正确性及性能结果的源码、依赖、实际加载库和原始证据；重算性能汇总与保存结果一致，固定 native 包身份有效，图失败状态保留。本次继承 E2E 为 0。
- `performance/attempt-1792347712004987370`：96 配置全部实测，调度 696.4 s、图/eager 1190.0 s。1,316 个文件归档并逐文件校验；图组退出 1，仅因平均性能门槛失败。以下耗时比为本实现 / 固定源 CANN，均按配置等权、配置内入口/R 等权；全部单例、分段与原始采样保留在各组 `public-profile/`。

| 性能集合 | 实测 / 适用配置 | 平均比 | 配置 / 分段退化 | 结论 |
| --- | --- | --- | --- | --- |
| 调度 | 56 / 20 | 0.7267397642563352 | 0 / 12 | 通过 |
| eager | 40 / 40 | 0.7967466565403458 | 6 / 31 | 通过 |
| graph | 40 / 40 | 1.484216023946693 | 40 / 205 | 未达到 ≤1 门槛 |

调度的另外 28 个单/显式 split 配置因原生 FD 策略不同、8 个正 softcap 配置因无同义源，按既定边界单列。图/eager 无排除项，分别见 `graph-average.json` / `eager-average.json`；调度见 `average.json`。未运行 910C 实机，按约定仅作配置兼容；约定的正确性组无未运行项。

准备期间已处理的失败：平台夹具补齐静态 CANN tiling 库的日志库依赖和实际配置路径；解析时区分日志与 JSON，并以 CANN 查询的可用内存检查预算。沙箱收集因设备不可见失败，改用真实设备环境。示例最初按文件注册，被显式日常清单检查拒绝，已改为节点。复核平台工具时复用既有完整编译器/ASan/UBSan 身份，并启用 sanitizer 失败退出；期间改变源码使并行验收身份失效，相关检查在最终源码上重跑。早期通过数或收集结果不冒充冻结版本的新通过。

## 环境与复现

接手实测 Python 3.11.15、torch 2.7.1+cpu、torch_npu 2.7.1，CANN `/usr/local/Ascend/cann-9.0.0-beta.2`；设备 7 为健康、空闲的 Ascend910B3（20 AIC/40 AIV）。CATLASS 为 `769cd40a8716b28650b6bebb08db4834eea4462f`；只读 CANN checkout HEAD 为 `2060a3db064c3352fec6262faed8a2decaf3a233`，native 构建继续使用固定 `ca18aee610729ca5270a44e2d84c751bc687fbf0` 的 archive。

生产扩展为 `flash_attn_npu_3/flash_attn_npu_3.cpython-311-aarch64-linux-gnu.so`，新 SHA256 `14edf068b305b840c5991b5daaef221618e1628ff626722574ac643de317373a`，与 final 的 build/runtime manifest 一致。接手旧扩展身份保留于 `runtime-before.json`。构建/数学缓存为 `/tmp/flash-attention-npu-mla-46928a793876/mla-validation-artifacts`，native 为 `/tmp/mla-m13b/native`。

使用 `/tmp/mla-m16-schedule/run.py` 恢复上一轮核验过的库路径；直接调用当前会话的 `npu-smi` 曾因 devlib 占位库缺符号失败，恢复环境后设备及实际扩展加载检查通过。性能输出沿用上一轮任务私有 tmpfs 包装器以避免磁盘普通空间不足，采样和汇总实现仍全在 `tests/mla/`，原始证据归档后保持原绝对路径。

```bash
python3 /tmp/mla-m16-schedule/run.py python3 -m tests.mla --suite platform-tools tools unit-platform platform-sanitize --output /tmp/mla-m17/final-tools
python3 /tmp/mla-m16-schedule/run.py python3 -m tests.mla --device 7 --suite examples-check gqa-options-check gqa-options-source gqa-options-boundaries graph-check e2e detail-gqa_options detail-graph ordinary graph-ordinary --output /tmp/mla-m17/final
python3 /tmp/mla-m16-schedule/run.py python3 -m tests.mla --device 7 --suite graph-profile-check gqa-options-profile-check --output /tmp/mla-m17/profile-preflight-retry
python3 /tmp/mla-m16-schedule/run.py python3 -m tests.mla --device 7 --suite gqa-options-schedule-profile graph-profile --output /tmp/mla-m17/performance
```

本次后两条命令实际经 `unshare -Ur -m python3 /tmp/mla-m16-schedule/run_private_storage.py <输出目录> <上述命令>` 执行，包装器要求新输出目录；复跑使用新目录以保留当前证据。数学 reference 允许内容缓存，待测实现和性能均实际执行。

用户接受的是本次 M17 的图性能差距，工具失败结论和默认门槛保持不变；正确性、eager 与调度性能均通过。本轮以一个提交结束，已确认迁移范围没有下一开发步，不启动新的开发轮或无头循环。当前闭环提交用 `git log -1 -- docs/mla/status.md` 定位。
