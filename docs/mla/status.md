# MLA 当前交接

## 状态与下一步

- 当前步骤：M00，需求分析与文档约束完成；**方案待用户确认，未开始实现**。
- 当前分支：`feat/mla-cann-migration`。
- 本轮产物：[根目录约定](../../AGENTS.md)、[适配设计](design.md)、[候选迭代](plan.md) 和本交接文件。
- 本次闭环提交由 `git log -1 -- docs/mla/status.md` 定位，无需为补写自身 hash 再提交。
- 唯一下一步：收到方案确认后，检查本次文档提交和环境，执行 M01（reference 与源能力用例清单），完成一个 commit 后停止。

需要确认的是设计中的迁移交集与接口处理：包含 prefill 和压缩 MLA；使用 FA3 原入口；共享压缩 KV 语义；排除官方不可表达的 CANN 独立 512 维 K/V 等特性；metadata 对齐调用语义并使用 NPU 自己的调度内容。该确认来自用户本次“确认方案之后……开发”的要求，不是额外审批流程。

## 最近闭环的证据

M00 为文档步骤，不代表 MLA 功能、精度或性能已经通过。

| 检查 | 结果 |
| --- | --- |
| 现仓入口/C++/metadata/CATLASS 阅读 | 找到 qv 未使用、O/V 维度和 metadata 数值耦合问题，已记录到 design |
| CANN arch22 host/kernel/test 阅读 | 找到 512+64 专用路径、通用模板中的 128+64 prefill、PA stride 和独立 K/V 的差异 |
| 官方 FA3 源码 | 已在线核实并固定 SHA，官方链接见 design；临时副本在 `/tmp/mla-fa3-reference/d15f1531a460ba456f41b01a774f33ab2db8febf/`，不依赖该临时目录长期存在 |
| 文档检查 | PASS：四个文件的本地 Markdown 链接、冲突标记、行尾空白、文件末尾换行和代码围栏检查通过；`git diff --cached --check` 通过，暂存范围仅四个文档文件 |
| 构建 / NPU 精度 / 性能 | 本步未运行；后续分别由 M01/M02 建立参考和基线 |

## 已核实的环境快照

检查日期：2026-09-23。此表用于识别环境变化，不作为下次设备可用性的保证。

| 项目 | 本次结果 |
| --- | --- |
| 生态仓起点 | `cab322eda3112fdbfefd99c8632b6393273562f3` |
| CANN 源仓 | `ca18aee610729ca5270a44e2d84c751bc687fbf0`；FIA 和 attention/common 的 scoped status 无改动 |
| CATLASS | `769cd40a8716b28650b6bebb08db4834eea4462f` |
| Python | 可执行文件 `/usr/local/python3.11.15/bin/python` |
| Python 包元数据 | torch 2.7.1，torch-npu 2.7.1，pytest 9.0.3，numpy 1.26.4；未以这些版本号代替实际扩展运行检查 |
| CANN | `ASCEND_HOME_PATH` / `ASCEND_TOOLKIT_HOME` 指向 `/usr/local/Ascend/cann-9.0.0-beta.2`；安装信息为 9.0.0-beta.2 |
| 设备 | 经沙箱外只读 `npu-smi info` 查询为 8 张 910B3；当时 3、7 号健康且无进程，5、6 号告警；每次运行前重新查询 |
| 设备访问 | 沙箱内查询失败 `dcmi ... -8005`，沙箱外查询成功；不能据前者断言机器无 NPU |
| 构建状态 | 仓内存在被忽略的预编译 `.so`，尚未证明与源码起点一致；M02 需重建并核对导入路径 |
| 仓库初始未跟踪内容 | `atk_test/` 为既有内容，本轮不改、不提交、不清理 |
| shell | 工具的 login shell 初始化曾长时间无输出，`login=false` 可正常执行且已带 CANN 环境变量 |
| 提交质量工具 | 本次检查时未安装 pre-commit，且无 `.git/hooks/pre-commit`；文档检查可直接执行，下次仍需检查是否新增钩子 |

## 接手复核

```bash
git status --short
git log -3 --oneline
git show --stat --oneline HEAD
git diff --check
```

先核实最近一步是否确为 M00；如果期间有新提交或用户修改，依据实际差异更新判断。不要因为本表写了“下一步 M01”就跳过已有修改。

本轮未修改 kernel、Python API、构建脚本或测试；未创建新运行性能数据。
尚待实测的问题：安装的 CANN 算子与源码版本是否一致、两种 MLA 的实际精度/性能基线、固定 FA3 边界行为对应的完整用例。这些已安排在计划中，不算验证通过。
