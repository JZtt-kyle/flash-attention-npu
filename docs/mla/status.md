# MLA 当前状态

当前分支 `feat/mla-cann-migration`，本轮起点 `4a8b9eb02eee7c62c77b5423d86d17ce334ad91c`。**M10a 独立 CPU split-KV 分区与 workspace 布局已闭环。** 本轮只完成该独立组件，不接通设备 partial attention/合并或公开 split。

## 当前能力与本步边界

M09e 及五布局 window 阶段的已开放能力保持：512+64、FP16/BF16，BSND、TND、混合及 BSND/TND PA 的无 mask、causal、window。生产仍仅接受 `num_splits=1`，没有新的 metadata 版本或占位 accumulator；CANN 源仓与 CATLASS 未修改。

本步新增 `mla_split_kv.h/.cpp`，仅由 CPU 驱动编译：固定源自动 FD 分区/归约记账、FA3 `<=0`/1/2..256 请求解析、mask 区间内显式分片、源内部 workspace 和 FA3 BSND/TND FP32 accumulator 布局/地址。复用既有窗口、长度输入约定和 SplitCore，不修改共享实现。协议和未开放边界见 [design 6.2.8](design.md#628-独立-cpu-split-kv-分区与布局)。

模块单元属于 `tests/mla/unit/split_kv/`；新增统一入口 `--unit split_kv`、`--suite split-kv-sanitize split-kv-tools`。CPU 组件未接通公开入口，因此本步没有 split 数值 E2E；后续接通时必须增加自己的日常 E2E。旧无 mask 回归不替代本模块的 mask/split 验收。

## 本次验收证据

证据根目录 `/tmp/mla-m10a/`；构建/reference 缓存沿用 `/tmp/flash-attention-npu-mla-46928a793876/mla-validation-artifacts/`。`build/` 只保留 `mla-codex-loop/` 日志与状态。**继承 E2E 为 0，历史单元/性能继承为 0。** 上次累计 119 项及窗口专项原始证据哈希完整；新增 CPU 文件/测试驱动进入保守依赖身份，本步实际重跑累计 E2E，核对结果在 `takeover.json`。

| 检查 | 本次结果与耗时 | 证据（相对根目录） |
| --- | --- | --- |
| CPU 模块固定源/独立边界 | 27/27，17.0 秒 | `unit/summary.json` |
| 同集合 ASan/UBSan | 27/27，54.2 秒 | `sanitize/summary.json` |
| 验收工具 | 12/12，7.5 秒 | `tools/summary.json` |
| 命令与收集 | CPU 27、sanitizer 27、工具 12、累计 E2E 119；只计清单核对 | `*-dry-run.json`、`unit-collect/`、`final-collect/`、`e2e-collect/summary.json` |
| 累计小输入 E2E | 119/119，25.0 秒；reference 命中 266、首次计算 0 | `e2e/summary.json` |
| 构建与实际加载 | 构建 490.9 秒、runtime 核验 20.6 秒；固定源探针和生产扩展通过 | `e2e/summary.json`、最终 attempt 的 `build-manifest.json`/`runtime.json` |
| 源码/原始证据与加载身份 | 四组原始证据完整且源码未变；6332 项构建依赖、12 项产物及实际加载哈希一致 | `final-evidence-check.json` |
| 文档与仓库质量 | 涉及的本地链接/锚点、`git diff --check` 与 staged 质量检查通过 | `doc-links-final.json`、`quality-final.log` |

CPU 普通与 sanitizer 各执行 adapter 25835 条、native oracle 252 条、FA3/workspace oracle 25423 条请求，完整 stdin/stdout 与退出码保存于各 attempt 的 `schedule/`。覆盖全部原生端点和 FD 字段、独立 tile 归属及归约账目，1/2/8/256 显式片、负值自动、无 mask/causal/window、尾块/尾零/空片/全空、C1V1/C1V2、buffer 对齐/不重叠及 uint64/int64 溢出。源只读固定 Git blob 与固定 FA3 哈希，CPU shape/stride 另对照 torch meta tensor。

本步 CPU 检查不计算数学 attention reference；E2E 全部复用既有内容缓存，待测实现实际执行。新增 CPU 文件/驱动进入保守构建身份，触发一次生产扩展和固定源探针重建；该构建成本与 25.0 秒日常回归耗时分开记录。普通 FA3、历史单元、性能与 full-matrix 未运行，不属于本独立 CPU 组件的必要范围；不将它们写作继承或本次通过。

### 复现与恢复

```bash
python3 -m tests.mla --unit split_kv --output /tmp/mla-m10a/unit
python3 -m tests.mla --suite split-kv-sanitize --output /tmp/mla-m10a/sanitize
python3 -m tests.mla --suite split-kv-tools --output /tmp/mla-m10a/tools
python3 -m tests.mla --suite e2e --device 6 --output /tmp/mla-m10a/e2e
```

各组先 dry-run/collect-only；sanitizer 与设备检查在可访问设备/LSAN 的环境执行。当前最终 CPU attempt 为 `unit/attempt-1791978027226787670`，sanitizer 为 `sanitize/attempt-1791978043707327180`，E2E 为 `e2e/attempt-1791978042247573420`。恢复先核对进程、summary、原始证据和依赖身份，不能只看已存在 `.so`。

## 环境差异与失败记录

接手时 HEAD/分支与上一闭环一致，工作树干净。Python 3.11.15、torch 2.7.1+cpu、torch_npu 2.7.1、CANN 9.0.0-beta.2；CATLASS `769cd40a8716b28650b6bebb08db4834eea4462f`。接手时第 6 张 910B3 空闲，无遗留 MLA 测试/构建进程；仅有本次无头控制器。原实际加载扩展为仓库 `flash_attn_npu_3/flash_attn_npu_3.cpython-311-aarch64-linux-gnu.so`，SHA-256 与 M09e 的 `23c1d9992ec9dae5eb9d1a9d2b7a04f586995313c8831c7307fe20888e7ebbe4` 一致；最终 E2E 实际加载同一路径的新构建，SHA-256 为 `f221a81a97b2524c6aba78620e710e7855ce4ee3718cc356ed1f0b12cd1b15de`，与原始 `loaded.json`、runtime 和构建 manifest 一致。

CANN 源仓当前 HEAD 已为 `2060a3db064c3352fec6262faed8a2decaf3a233`，与迁移固定版本不同；`ca18aee610729ca5270a44e2d84c751bc687fbf0` Git 对象仍可读，所有 oracle 均从该固定对象提取，未读取当前源树替代固定对照。FA3 新增读取同一固定提交的 `block.h`，内容哈希和准备要求见 design 6.2.8。

早期失败不计通过：首次 CPU 对照 shim 将源 uint32 字段写成 int64，导致原函数 `std::min` 编译失败，修正 shim 后同集合通过；E2E collect-only 在沙箱内因设备不可见失败，同清单在可访问第 6 卡环境重收集成功。开发格式检查的分号已由 formatter 修正，最终 ruff 与 `git diff --check` 通过。原始失败日志保留在 `development/` 和 `final-collect/`。受影响检查已重跑，无未解决失败；实现固定后只更新文档。仓库质量使用 `PYTHONPATH=/tmp/mla-m09e/quality-deps ci/code_quality.sh --staged`，该变量不用于设备测试，格式检查未改写验收源码。

## 唯一下一步

[M10b1：独立 BSND split-KV partial attention](plan.md#唯一下一步-m10b1独立-bsnd-split-kv-partial-attention)。本轮只完成 M10a，提交成功后立即停止，不开始 M10b1。当前闭环提交由 `git log -1 -- docs/mla/status.md` 定位，无需追加自身提交号。
