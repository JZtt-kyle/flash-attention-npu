# MLA 当前状态

当前分支 `feat/mla-cann-migration`，本步起点 `42cb965`。**M10c5b 混合布局显式 split 生产接入已完成验收。** 唯一下一步为 [M10c6a 混合原生自动 split 计划与安全协议](plan.md#唯一下一步-m10c6a混合原生自动-split-计划与安全协议)；本次只提交 M10c5b，提交后停止。当前闭环提交可用 `git log -1 -- docs/mla/status.md` 定位。

## 已开放能力与边界

非 PA TND Q + BSND KV、512+64、FP16/BF16 接通显式 `num_splits=2..256`，覆盖无 mask、右下 causal、窗口与 causal 叠加、Q allocation/used、实际 KV 长度和四路 stride。v21 公开生成只在设备读取 Q 前缀尾值；AICPU 与独立消费者验证当前长度、窗口和分片，混合 launcher 复用已有 partial/merge，KV 保留 BSND batch/token stride。公开 cache、Python/C++ raw 及私有入口返回真实 FP32 `[S,H,totalQ,512]` / `[S,H,totalQ]` accumulator，公开 real/fake/compile 已验收。

新增独占 `unit/split_mixed/` 与 `e2e/split_mixed/`；日常清单新增 12 项，原有 173 项逐项保留。CPU 协议只新增安全 prefix 模式，未修改既有计划/消费者算法。混合自动及 PA split 仍明确拒绝；未修改 CANN 源仓或 CATLASS。接口、空值、来源及构建契约见 [design 6.2.22](design.md#6222-混合显式-split-生产适配)。

## 环境与证据身份

Python 3.11.15、torch 2.7.1+cpu、torch_npu 2.7.1、CANN `/usr/local/Ascend/cann-9.0.0-beta.2`；CATLASS `769cd40a8716b28650b6bebb08db4834eea4462f`。实机为 6 号 910B3，其他卡的原有任务未改动；验收结束后本轮进程全部退出，6 号卡空闲，见 `processes-final.json` 与 `npu-final.log`。

证据根 `/tmp/mla-m10c5b/`；构建和数学缓存使用仓库外 `/tmp/flash-attention-npu-mla-46928a793876/mla-validation-artifacts/`，`build/` 仅有 `mla-codex-loop`。固定源码后的完整构建 533.4 秒、实际加载检查 21.1 秒通过，见 `check-fixed/attempt-1792016400668343840/`。实际扩展为仓库 `flash_attn_npu_3/flash_attn_npu_3.cpython-311-aarch64-linux-gnu.so`，SHA-256 `854c9a98b28e742ba099194897690a91b4b03cc98c1d81433432e5f2db25b5aa`；6,392 个生产依赖和 12 个产物与当前源码匹配。后续正式组复用内容核验后的构建，实际执行待测算子。

接手时前步 E2E 的 7,437 个唯一文件核验无差异，见 `takeover.json`。本步生产及验收身份改变，累计 E2E 全部实跑，**继承 0**；不累计其他模块历史单元。最终 `evidence-final.json` 核验 10,397 个唯一文件、252 个固定 CANN Git blob、所有集合/原始结果、probe、构建、加载库、原生包与性能归档，差异为 0。

## 本次通过

以下均为固定实现和测试源码后的实际运行，无失败或跳过；收集清单与最终节点逐项一致，未用数量截断或完整穷举。正式集合数学缓存共 438 次命中、18 次新计算并写入；开发与代表检查的缓存保留，同输入不重复计算 reference。

| 组 | 通过 | 耗时 | 证据目录（相对证据根） |
| --- | --- | --- | --- |
| `split-mixed-tools` | 23/23 | 7.9 秒 | `cpu-frozen/attempt-1792017533527093900/` |
| `unit-split_mixed_protocol` | 245/245 | 36.9 秒 | 同上 |
| `split-mixed-protocol-sanitize` | 245/245 | 114.9 秒 | 同上 |
| `unit-split_mixed` | 10/10 | 129.6 秒 | `device-frozen/attempt-1792017508433424620/` |
| `split-mixed-public` | 89/89 | 69.3 秒 | 同上 |
| `split-mixed-trap` | 30/30 | 418.4 秒 | 同上 |
| `e2e` | 185/185 | 31.1 秒 | 同上 |
| `ordinary` | 516/516 | 26.1 秒 | 同上 |
| `split-mixed-profile` | 1/1，12 个固定配置 | 302.3 秒 | `profile-native/attempt-1792018380737433220/` |

CPU 普通/sanitizer 使用同一集合，四个驱动 stderr 为空，另验公开 prefix 模式与已知 totalQ 记录逐字一致及坏前缀失败原子性。设备固定源使用独立原始 Q 段与矩形 KV 地址，双 dtype 的 O/LSE 与源逐元素一致，partial/accumulator 及双数学通过，本步无新增源精度例外。

公开组包含 8 个代表/成本边界：最长 KV 16384、最大真实 accumulator 138,682,368 字节；开发代表组先通过，最终公开组再次实际执行并复用 reference。专项覆盖三 mask、S=193/255/256、used 间隙、空容量、独立 stride、真实/fake/compile storage 边界、metadata 复用/失败恢复及流生命周期。30 个 Trap 各在独立进程先完成合法调用，再于设备 event 阶段捕获预期 507015/Trap 错误，包含空容量和旧版本/损坏/陈旧记录。

## 公开性能

固定六种输入、双 dtype，共 12 配置全部实测；每配置等权平均隐式/生成/复用各 R=1/4 的六段，再对适用配置等权平均。保持预热 5、五轮各 20 wall/10 profiler 样本，同一 CPU affinity `[0]`，计入原生 gather/分配及公开输出成本。原生包 `/tmp/mla-m10c3b/native` 的配方与 2,027 个文件核验通过；原生分派、双方数学及 AICPU→partial→merge 阶段校验通过。

| 原生分派类别 | 配置数 | 平均耗时比 FA3/CANN | 退化配置 / 分段 | 本步门槛 |
| --- | --- | --- | --- | --- |
| 模板 5、非 FD | 2 | **0.412013** | 0/2、0/12 | ≤1，通过 |
| 原生自动 FD | 8 | 0.842440 | 2/8、12/48 | 按显式 split 约定单列 |
| 空 KV 模板 0 | 2 | 0.491953 | 0/2、0/12 | 独立模板，单列 |

适用六段均值依次为 implicit R1/R4 `0.478623/0.393228`、generate `0.600409/0.319992`、reuse `0.413004/0.266821`。没有按耗时筛选输入；10 个不参与门槛的配置也保留全部样本和分段结果。原始证据在性能组的 `public-profile/`，门槛汇总 `average.json`，分类与退化计数另见证据根 `performance-groups.json`。这不是混合原生自动 split 的完成声明。

## 复现与已处理问题

以下命令均已先 dry-run/collect-only 核对；重新执行时使用新的 output 目录。性能单独运行，不与构建或其他设备测试并行。

```bash
python3 -m tests.mla --suite split-mixed-tools unit-split_mixed_protocol split-mixed-protocol-sanitize --output /tmp/mla-m10c5b/cpu-frozen
python3 -m tests.mla --suite unit-split_mixed split-mixed-public split-mixed-trap e2e ordinary --device 6 --output /tmp/mla-m10c5b/device-frozen
MLA_NATIVE_ROOT=/tmp/mla-m10c3b/native python3 -m tests.mla --suite split-mixed-profile --device 6 --output /tmp/mla-m10c5b/profile-native
```

开发失败和失效证据未计入最终通过：工具故障注入缺少 mock 曾误启动自身构建，已仅停止其进程并修正；CPU prefix 驱动编号与旧故障模式冲突，已改独立编号 25；沙箱内设备不可见，改宿主机验收。首次构建在驱动修正后以输入身份变化被拒绝，重建后通过。公开 storage 测试原选无间隙转置 Q，缩到 compact 大小仍合法；已改四路带行间隙并断言跨度前提，real/fake/compile 均通过。性能驱动导入/对照接口修正使 `device-storage-fixed/` 的旧运行按源码身份失效，最终冻结组全部重跑。首次性能启动因默认原生包配方过期而在计时前被拒绝，切换内容核验完整的既有包后通过。原始失败分别保留在 `tools-dev/`、`cpu-final/`、`check-dev/`、`device-final/`、`device-storage-fixed/` 和 `profile-frozen/`。

最终相关文档、链接/表格、差异及仓库质量检查记录在 `doc-links-final.json`、`quality-final.log`；使用既有完整 pre-commit 环境，未绕过钩子。未运行混合自动、PA split 或未开放功能的数值/性能，也未累计历史模块单元；这些不计通过或继承。下一步仅按 plan 的 M10c6a 接手，本次提交后不继续开发。
