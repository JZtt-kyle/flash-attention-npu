# MLA 当前状态

**R1 前向入口收敛已完成，回归与公开性能验收通过。** 功能基线仍为 M10c9b（`d42a7ff`），流程基线为 `dd9ed54`；分支 `feat/mla-cann-migration`。本次闭环提交由 `git log -1 -- docs/mla/status.md` 定位。唯一下一步为 [R2 测试公共依赖整理](plan.md#唯一下一步-r2测试公共依赖整理)，随后按固定 CANN/CATLASS 的方式整理内核配置，再恢复 M10c10；本轮提交后停止。

## 本步改动与来源

- 参照固定 FA3 `d15f1531a460ba456f41b01a774f33ab2db8febf` 的 `hopper/flash_attn_interface.py`：三个公开 MLA 入口共用 `_mla_forward`，公开与内部 real/fake 的 split 共用 `_mla_split_forward`，保留已注册算子的校验和 cache 四返回值优化。默认无 mask MLA 也直接调用算子；普通 attention 保留 Autograd 路径。
- 四个公开函数及内部 forward/fake 的签名与接手基线相同，参数顺序与固定源一致。未改 CANN/CATLASS、C++、metadata 协议、设备数值或构建参数；原生产扩展经身份检查后复用。
- 不支持的选项明确指出参数名；中英文 README 补齐 qv 含义、默认 `1/sqrt(576)`、V 输出维度、只读 cache、返回开关及可运行示例。接口文件由 4286 行收敛为 4080 行，主要收益是公开 API 不再各自维护布局/mask/split 分支。
- 新增 `e2e/interface/test_forward.py` 并注册日常 E2E：8 个具名布局/mask/split/scale 组合×双 dtype×real/fake，三入口默认 compile、错误参数名两两覆盖及四布局整数 cache 长度。数值与既有原始 C++ 前向对照，继续由各能力 E2E 覆盖数学、stride/空值/复用。

## 当次证据

证据根 `/tmp/mla-interface-consolidation-20261015/`。统一驱动先 dry-run/collect-only；所有必要组均不使用 `--resume`，本步继承通过结果 **0**。以下为当前源码实际执行，最终必要组无失败或跳过。

| 检查 | 结果 | 用时 | 证据 |
| --- | --- | --- | --- |
| 新入口代表集合 | 57 passed，0 failed/skipped | 22.2 s | `preflight-2/summary.json` |
| 累计日常 E2E | 300 passed，0 failed/skipped（原 243 + 新 57） | 55.8 s | `acceptance/summary.json` |
| 普通 FA3 回归 | 516 passed，0 failed/skipped | 27.1 s | `acceptance/summary.json` |
| 测试归属/证据依赖工具 | 17 passed | 5.8 s | `tools.xml`、`tools.log` |
| 公开性能 | 6 组通过，预先固定 100 配置 | 2571.3 s | `performance/summary.json`、`performance-summary.json` |
| 源接口及示例语法核对 | 通过 | — | `source-contract-audit.json` |
| 适用质量 hooks | 五项通过；最终暂存集合由收尾记录核对 | — | `quality-final.log` |

首轮新增测试误把 Q-used 之外的 O/LSE 槽当作有确定值，得到 3 passed / 1 failed；根据既有“不保证未用 query 输出”的契约修正比较范围后重跑，新集合及累计回归均通过。原失败保留在 `preflight/`，未修改实现或精度阈值来规避失败。

构建校验复用各 1.2 s；runtime 当次核对约 21 s。累计 E2E 的数学 reference 缓存 387 hit、ordinary 31 hit，待测实现全部实际执行。无设备数值/共享协议改动，本步未重跑历史组件单元、sanitizer 和 Trap；该基线证据仍按 `git show d42a7ff:docs/mla/status.md` 定位，不将其记为本步通过。

最终 8 个实机组的源码依赖与原始证据哈希均有效（`final-evidence-audit.json`）；496 个文档本地链接及锚点通过（`doc-links-final.json`）。

## 当次公开性能

各组沿用既有完整配置，没有按耗时筛选或截断。表内平均只计该组原口径的适用配置；单/显式 split 的原生自动 FD 及源非 MLA 模板对照保留单列，自动 split 组包含原生 Direct/FD。

| 组 | 实测/适用配置 | 平均耗时比 | 退化配置/分段 | 用时 |
| --- | --- | --- | --- | --- |
| `window-bsnd-profile` | 24/18 | 0.632044 | 0/0 | 658.0 s |
| `window-tnd-profile` | 24/18 | 0.447359 | 0/0 | 842.6 s |
| `auto-split-tnd-profile` | 14/12 | 0.799264 | 2/6 | 256.0 s |
| `auto-split-mixed-profile` | 14/12 | 0.637423 | 0/4 | 282.3 s |
| `auto-split-paged-bsnd-profile` | 12/12 | 0.864154 | 4/25 | 263.1 s |
| `split-paged-tnd-profile` | 12/4 | 0.531557 | 0/0 | 269.3 s |

合计 76 个适用配置等权平均比 **0.646915 ≤1**，6 个配置、35 个分段退化；另 24 个不适用源分派配置（原生 FD 或非 MLA 模板）依原规则单列。每组保留双 dtype 的全部单例、R 分段、wall/kernel 原始采样和固定源诊断，详情在对应组的 `public-profile/`。性能与固定 CANN 源比较，不作为相对整理前代码的加速声明。

## 环境与复现

Python `/usr/local/python3.11.15/bin/python3` 3.11.15，torch 2.7.1+cpu、torch_npu 2.7.1；CANN `/usr/local/Ascend/cann-9.0.0-beta.2`，CATLASS `769cd40a8716b28650b6bebb08db4834eea4462f`。宿主卡 6 为 Ascend910B3，实机检查在宿主执行。

实际加载本仓 `flash_attn_npu_3/flash_attn_npu_3.cpython-311-aarch64-linux-gnu.so`，SHA-256 `2bbbdaeac0b18209728a1a50da139a138482a6ed470f726d36e2a87adc76e1a6`；构建产物与数学缓存为 `/tmp/flash-attention-npu-mla-46928a793876/mla-validation-artifacts/`。native 必须显式用 `/tmp/mla-m10c3b/native`，固定源为 `ca18aee610729ca5270a44e2d84c751bc687fbf0`；驱动核验包内容与配方后使用。临时证据后续继承前须确认仍存在且依赖有效。

```bash
python3 -m tests.mla --e2e interface --device 6 --output /tmp/mla-interface-consolidation-20261015/repro-interface
python3 -m tests.mla --suite e2e ordinary --device 6 --output /tmp/mla-interface-consolidation-20261015/repro
MLA_NATIVE_ROOT=/tmp/mla-m10c3b/native python3 -m tests.mla --suite window-bsnd-profile window-tnd-profile auto-split-tnd-profile auto-split-mixed-profile auto-split-paged-bsnd-profile split-paged-tnd-profile --device 6 --output /tmp/mla-interface-consolidation-20261015/repro-profile
```

先 dry-run/collect-only。公开性能保持各组原始输入、精度、5 次预热、5×20 wall、10 次 profiler、R=1/4 和隔离条件；单配置的入口/R 先等权平均，所有单例和原生 FD 分类均保留。

## 当前功能与交接边界

512+64、Hkv=1、FP16/BF16 的五种布局支持 none/causal/window；非 PA 的 BSND、TND、混合及 BSND PA 支持自动/单/显式 split，TND PA 支持显式 1..256。TND PA 自动 0、192/128 prefill、softcap 与非默认调度仍明确拒绝，完整 graph replay 和 910C 实机未验收；详细规则见 [design 2.1–2.2](design.md#21-入口与记录选择)。本步未扩大开放能力。

接手已有六处独立循环修复完整保留，备份为 `preexisting/`、`preexisting.patch`；原始备注随既有工作区改动保留，均排除本步提交，保留核对为 `preservation-working.json`、`preservation-final.json`。本轮不启动无头循环；原状态 development=80/reviewed=78、pending=null 未变。
