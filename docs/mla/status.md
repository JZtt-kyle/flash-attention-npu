# MLA 当前状态

**代码、文档、测试和无头开发流程整理完成，必要检查通过。** 分支 `feat/mla-cann-migration`；功能基线仍为 M10c9b，入口基线为 R1（`006bbc9`）。当前闭环提交用 `git log -1 -- docs/mla/status.md` 定位。唯一下一步为 [M10c10：TND PA 原生自动 split 完整接入](plan.md#唯一下一步-m10c10tnd-pa-原生自动-split-完整接入)，不再单开风格整理轮；本次提交后停止。

## 整理结果

- 共享组件、固定源调度、输入、reference 和调用 helper 分别归入 `support/components/` 与 `support/public_cases/`；公共层对测试模块的反向导入从 **156 处降为 0**，公共层静态导入图无环。保留原来源与测试身份，PA reference 缓存依赖指向实际共享输入适配文件。
- 对照固定 CANN Kernel 和 CATLASS 分层，保留现有 Kernel/Block/Tile、模板实例与流水；位置布尔参数补名称，修正过期协议注释。五个 C++ 文件去除注释后的 token 完全相同；未改 CANN/CATLASS、公开接口或数值算法。
- Python 证据按实际导入闭包失效，未知动态加载继续保守处理；生产构建的 Python 输入从 **28 个降为 11 个**，保留实际构建器、原生源码、编译器、头文件和产物内容核验。独立 CPU/profile 构建器不再无故触发生产重建。
- `tools` 补入清单和归属检查，`--unit tools` 明确为 CPU 工具模式；修正历史生产注册、merge 消费者归属和搬迁后的 AST 检查路径。269 个既有组的选择参数保持一致，`tools` 增加两个通用测试文件；未缩减数值用例、精度阈值或性能采样。
- 接手已有六处循环修复已纳入本步：每五次成功开发后专门修正文档，保持失败停止、同会话恢复及只计数一次；新增“从干净工作区开始，必须干净结束”检查，防止漏提交文件流入下一轮。旧状态和接手已有工作保持可恢复。

## 当次证据

证据根 `/tmp/mla-maintenance-20261015/`。实现通过质量检查后固定，再进行最终 dry-run/collect-only 与完整验收；本步未使用 `--resume`，继承通过结果 **0**。

| 检查 | 结果 | 用时 |
| --- | --- | --- |
| 通用工具 | 307 passed，0 failed/skipped | 126.0 s |
| 完整工具单元 | 861 passed，0 failed/skipped | 611.1 s |
| 累计 MLA E2E | 300 passed，0 failed/skipped | 56.1 s |
| 普通 FA3 回归 | 516 passed，0 failed/skipped | 27.0 s |

工具组有重叠，不叠加为独立用例总数。完整工具单元含真实 AICPU 重编译和 CPU sanitizer 检查，约十分钟；仅在工具整体整理或相应改动时运行，不作为每轮功能开发门槛。日常 MLA E2E 本次约一分钟。

- `acceptance/summary.json`：最终四组结果、实际加载与原始证据；`final-evidence-audit.json`：源码依赖、原始证据及实现冻结核对全部有效。
- `collection-comparison.json`：受影响 122 个测试文件的 **13,163 个收集节点及顺序完全一致**；这是收集检查，不是执行这些历史专项。`final-equivalence.json` 核对 531 个搬出定义/常量、1028 个原数值测试函数及 C++ token；`constants-before.json`/`constants-after.json` 的 161 份配置一致。
- 预检中修复了过期断言、搬迁后的检查路径及 fixture 导入问题；完整工具诊断曾有 31 项失败，定点修复复核 31 passed 后重新完整验收。原失败保留在 `tools-preflight*`、`all-tools/`、`all-tools-diagnostic/` 和 `tool-repairs/`，未删除失败用例。独立审计的环境/pytest 根目录设置失误也保留原日志，成功复核以 `*-2` 和最终比较文件为准。
- 首次当前源码构建实际完成，但期间修正测试使准备组按规则失效（`device-acceptance/summary.json`，548.3 s）；最终准备重新核验并复用完整产物 **1.2 s**，runtime **21.0 s**。源码固定后全部相关实现实际执行，未把该次失效准备记为验收通过。
- 六项适用质量 hooks 通过，文档本地链接及锚点核对通过。公开调用、数值语句、性能输入和计时边界不变，本步未重跑 100 配置性能或历史设备组件矩阵；R1 的性能历史用 `git show 006bbc9:docs/mla/status.md` 查询，不记为本步实测或继承。

## 环境与使用

Python 3.11.15、torch 2.7.1+cpu、torch_npu 2.7.1；CANN `/usr/local/Ascend/cann-9.0.0-beta.2`，CATLASS `769cd40a8716b28650b6bebb08db4834eea4462f`。宿主卡 6 为 Ascend910B3。当前实际加载本仓 `flash_attn_npu_3/flash_attn_npu_3.cpython-311-aarch64-linux-gnu.so`，SHA256 `aad267e6b538b240530a768c8ed58eae71179296cf28fc4dd25cd1d1c049b6e9`。

产物与数学缓存为 `/tmp/flash-attention-npu-mla-46928a793876/mla-validation-artifacts/`；后续性能使用的 native 路径仍为 `/tmp/mla-m10c3b/native`，使用前核验配方和内容。`build/` 只含 `mla-codex-loop/`。临时证据被清理后不可继承；长期保留需配置持久 `MLA_WORK_DIR`。

原兼容启动脚本仍指向版本管理脚本，`--check` 与五轮预览已实际检查（CLI 0.162.1、已登录）；development=80 / reviewed=78 / pending=null 未改，本次未启动模型循环。可先检查，再运行一轮：

```bash
python3 /home/zhangqijia/repo/mla_codex_loop.py --check
python3 /home/zhangqijia/repo/mla_codex_loop.py --max-dev-runs 1
python3 -m tests.mla --suite e2e ordinary --device 6 --output /tmp/mla-maintenance-20261015/repro
```

验收复现先 dry-run/collect-only；整体工具复现用 `--suite tools unit-tools`，日常仅按改动范围选工具。无头连续开发省略 `--max-dev-runs`。循环成功计数与手动提交分开，本次未篡改旧会话状态；接手备份见 `preexisting/`、`preexisting.patch`。

512+64、Hkv=1、FP16/BF16 的五布局支持 none/causal/window；非 PA 三布局与 BSND PA 支持自动/单/显式 split，TND PA 支持显式 1..256。TND PA 自动 0、192/128 prefill、softcap、非默认调度仍拒绝，完整 graph replay 和 910C 实机未验收；本步不扩大功能范围，详见 [design](design.md#21-入口与记录选择)。
