# MLA 当前交接

## 状态与下一步

- 当前步骤：M01 已完成；独立 CPU reference、逻辑输入映射、源能力清单与本步质量检查通过。
- 当前分支：`feat/mla-cann-migration`；接手基点为 M00 `0c22614`，与原交接一致。
- 用户通过“继续MLA迁移开发吧”及后续“继续”授权按现有方案推进；已确认范围内不重复请求审批。
- 本轮产物：[reference](../../tests/common/mla_ref.py)、[CPU 验收](../../tests/test_flash_attn_npu_v3_mla_reference.py)、[设计中的源能力/用例清单](design.md) 4.1/4.2，以及计划和交接更新。
- 本次闭环提交由 `git log -1 -- docs/mla/status.md` 定位。
- 唯一下一步：M02（910B CANN 基线执行器、版本确认、精度/计时及普通 FA3 重建回归）；本轮提交 M01 后停止，不提前开始。

## 最近闭环的证据

本步只建立验收基础，**没有接通生产 MLA kernel/API，也不代表 NPU 精度或性能通过**。

| 检查 | 结果 |
| --- | --- |
| M00 接手复核 | PASS：分支、最近提交和四个文档差异一致；重跑本地 Markdown 链接、围栏、冲突标记、空白与末尾换行检查通过 |
| MLA CPU reference | PASS：`71 passed in 20.43s`，无 skip；FP64 标量 oracle 独立核对 512+64 与 192/128 的两项点积、GQA、默认/0/自定义 scale、softcap、O/LSE、外层 stride 和 storage offset |
| 输入映射与边界 | PASS：压缩内容 K/V 必须为同一 storage view；数值相等的独立 buffer 也拒绝；prefill 独立 V、split/combined RoPE、192 行 stride 和显式 CANN scale 往返正确 |
| mask / 空行 | PASS：手写可见列矩阵覆盖 Sq<Sk、Sq>Sk、单 query、双侧/单侧窗口及 causal 叠加；按固定 FA3 分支区分全局空 K 的 `+inf` 与全遮罩行的 `-inf` |
| 初轮失败处理 | FP16 `scale=0` 两例出现最终舍入差异；高精度参考改为先累加未归一化加权 V 再除以权重和，全部严格断言通过，未放宽阈值；初轮失败日志保留 |
| 源能力盘点 | 已按固定 arch22 host/kernel/test 与官方 FA3 补充来源、纳入/排除结论和 18 组代表用例；源码存在分支不记为设备验证通过 |
| 仓库质量 | PASS：`ci/code_quality.sh --staged` 检查本步五个文件；空白、末尾换行、冲突标记、Ruff 0.16.5 check/format 均通过，其他语言钩子无适用文件；文档链接/围栏与 18 个用例 ID 检查通过 |
| 构建 / NPU 精度 / 性能 | 本步未运行；公共入口、metadata、kernel、构建脚本及现有通用 reference 未修改，无普通 FA3 设备回归触发；M02 仍须重建 |

最小闭环复现（从仓库根目录执行）：

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m pytest -q tests/test_flash_attn_npu_v3_mla_reference.py
```

`TORCH_DEVICE_BACKEND_AUTOLOAD=0` 使该 CPU 测试不自动导入 torch_npu 或查询设备；无需 FA3 扩展或源仓副本。高精度累加使用 FP64，输出按现仓双 reference 比较约定返回输入 dtype 的 O 和 FP32 LSE。低精度 CPU 测试内的舍入断言不是 NPU 验收阈值，设备阈值留给 M02 固定。

质量入口（本次工具安装在临时目录；干净环境按 `ci/quality-requirements.txt` 安装）：

```bash
PYTHONPATH=/tmp/mla-m01-precommit PRE_COMMIT_HOME=/tmp/mla-m01-precommit-cache \
  ci/code_quality.sh --staged
git diff --check
git diff --cached --check
```

本轮临时证据在 `/tmp/mla-m01-validation/`：`pytest.log`、`pytest-initial-failure.log`、`npu-smi.log`、`precommit-install.log`、`quality.log`；版权注释及本交接结果更新后的检查为 `quality-final.log`。这些临时日志不替代仓内测试和复现命令，不作为最终实机性能的持久证据。

## 已核实的环境快照

环境核对跨 2026-09-23/24；9 月 24 日重新检查版本、源仓状态、已有扩展哈希及设备占用。

| 项目 | 本次结果 |
| --- | --- |
| CANN 源仓 | `ca18aee610729ca5270a44e2d84c751bc687fbf0`；FIA 和 attention/common scoped status 无改动，保持只读 |
| CATLASS | `769cd40a8716b28650b6bebb08db4834eea4462f`；子模块无改动 |
| Python / 包 | `/usr/local/python3.11.15/bin/python`；Python 3.11.15，实际 torch `2.7.1+cpu`，torch_npu 2.7.1，pytest 9.0.3，numpy 1.26.4；torch/torch_npu 均曾实际导入 |
| CANN / OPP | `/usr/local/Ascend/cann-9.0.0-beta.2`；安装信息 `9.0.0-beta.2` / `V100R001C10SPC001B214`；`ASCEND_OPP_PATH` 为其 `opp`，未设 `ASCEND_CUSTOM_OPP_PATH`；具体 FIA 算子与源码对应性尚待 M02 核实 |
| 设备 | 9 月 24 日沙箱外只读查询：8 张 910B3；3、7 号健康且无进程，5、6 号 Alarm；使用前仍需重查 |
| 实际扩展导入 | 9 月 23 日在健康空闲的 7 号卡环境中确认包和扩展均来自本仓；扩展为 `flash_attn_npu_3/flash_attn_npu_3.cpython-311-aarch64-linux-gnu.so`，伴有重复 tiling 注册警告；仅导入，不运行 attention |
| 已有扩展哈希 | 9 月 24 日 SHA256 `489ed4bee8aa617521b1d42527024c71fb88b093244d60ad3605ac59139b2c94`；**未重建，未证明来自当前源码**，不能用它验收后续实现 |
| 沙箱差异 | 沙箱内 `npu-smi` 返回 `-8005`，FA3 包设备检测也失败；沙箱外查询/导入成功。CPU 测试关闭 backend autoload 后正常，不据沙箱错误断言机器无 NPU |
| 质量工具 | 仓内无活动 pre-commit hook，系统未安装 pre-commit/Ruff；固定版本分别安装到 `/tmp/mla-m01-precommit`、`/tmp/mla-m01-quality`，未修改项目依赖 |
| 既有内容 / shell | `atk_test/` 为接手时既有未跟踪目录，不改、不提交；工具使用 `login=false` 避免 login 初始化长时间阻塞 |

## 下轮约束

先检查当前提交、`git status --short` 与本步 CPU 命令；源码/环境变动时重跑受影响检查。M02 从 design 的 D01/D02/P01 两种 dtype 建立可运行基线，再扩展该步必要代表输入；记录实际 FIA 库路径/版本或哈希，核实 split/combined prefill 布局和 scale，并重建普通 FA3 后确认加载路径。

TND/PA 解析、实际长度、split accumulator、metadata 与图仍未由本步 reference 接通；现仓生产入口原有 qv 未参与计算的问题也尚未修复，M03 前不能把调用成功当作 MLA 支持。
