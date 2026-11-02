# 新电脑使用指南

本项目是本地旋转框自动标注实验工作台，默认使用 YOLO26s-OBB 和 Ultralytics 8.4.91。网页负责抽帧、训练、自动标注和结果对比；人工画框、修正预测使用独立的 X-AnyLabeling。

## 安装与启动

1. 安装 64 位 Python 3.11 或 3.12，并在安装时勾选加入 PATH。
2. 从 [仓库](https://github.com/TBeiMu/autolabel-lab) 克隆或下载 ZIP，解压到自己的工作目录。
3. 在项目目录打开 PowerShell，运行：

   ```powershell
   powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\setup.ps1 -Training
   ```

4. 从 [官方发布页](https://github.com/CVHub520/X-AnyLabeling/releases/tag/v4.3.0) 下载并安装 Windows CPU 版 X-AnyLabeling。
5. 双击 `启动实验工作台.bat`，浏览器打开 `http://127.0.0.1:8765`。
6. 在「标注工具」选择安装后的 X-AnyLabeling 应用程序 `.exe`。安装包 `Setup.exe` 不是标注程序。

安装脚本创建项目 `.venv`，不会覆盖已有 Python 环境。只安装界面、抽帧和测试工具时省略 `-Training`；实际训练、自动标注和评价需要完整训练依赖。安装脚本被执行策略阻止时使用上面的 `-ExecutionPolicy Bypass` 命令，它只对该次启动生效。

## GPU 与已有训练环境

默认环境安装完成后检查：

```powershell
& .\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

返回 `True` 可在工作台选择 GPU 0；返回 `False` 可先选择 CPU。需要 GPU 时，按 [PyTorch 官方说明](https://pytorch.org/get-started/locally/) 在本项目 `.venv` 中安装适合显卡的 PyTorch，然后重新检查。

也可以复用已安装训练依赖的 Python，先在那个环境安装本项目依赖，再从同一个 PowerShell 启动工作台：

```powershell
$env:ALAB_PYTHON = 'D:\your-env\Scripts\python.exe'
& $env:ALAB_PYTHON -m pip install -e '.[train,test]'
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\start-web.ps1
```

需要以后双击也使用该环境时，在项目根目录创建 `runtime.local.json`：

```json
{"python": "D:/your-env/Scripts/python.exe"}
```

选择优先级为 `ALAB_PYTHON` → `runtime.local.json` → 项目 `.venv`。该配置留在本机，不上传 GitHub。

## 完成第一次实验

1. **新建实验**：填写名称和真实目标类别；类别顺序创建后固定。
2. **视频与数据**：选择视频，预览后抽帧，再固定种子、验证、测试划分。先准备足够的不同画面，默认种子 15 张。
3. **人工标注**：在 X-AnyLabeling 分别标注 seed、val、test，保存图片旁边的 JSON；无目标图片也要保存空标注。回工作台导入三组标注。
4. **模型训练**：检查数据，以 M0 为名先跑少量 epochs。显存不足时调小 batch 或尺寸；M0/M1 正式对比使用相同设置。
5. **自动标注与审核**：用 M0 生成 r1，打开审核目录逐图修正并保存。记录真实人工工时，确认全批审核后导入。
6. **再次训练**：训练 M1 并选择 r1，默认再次从随机权重开始。
7. **结果对比**：用固定测试集评价 M0/M1，更新报告。要比较省时比例，还要填写同批图片的纯人工标注实测工时。

实验保存在 `experiments/<实验目录标识>`。每个实验保存图片、标注、模型和报告；网页的「打开实验目录」可以定位它。原始预测 `raw` 和归档评价标注需要保持固定，修正只在 `editable` 目录操作。

## 验证与常见问题

检查代码和生成合成示例，不执行真实训练：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\verify.ps1
& .\.venv\Scripts\python.exe scripts/smoke.py --out _smoke/my_first_check
```

生成结果仅验证流程。需要在网页查看这个示例时，通过「导入已有实验」选择 `_smoke/my_first_check/experiment`；它没有真实训练好的模型。再次运行 smoke 时使用新的输出目录。

| 情况 | 处理方法 |
|---|---|
| 找不到 Python 环境 | 先运行安装命令，或检查 `ALAB_PYTHON`、`runtime.local.json` 中的路径 |
| 当前 Python 没有训练依赖 | 在实际运行环境安装 `.[train,test]`，然后重启工作台 |
| 端口 8765 被占用 | 运行 `powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\start-web.ps1 -Port 8766` |
| JSON 导入失败 | 检查类别名称、旋转框四角、边界、图片尺寸；缺少 JSON 的图片表示尚未标注 |
| 没有 GPU | 训练、预测、评价选择 CPU；执行速度会慢一些 |
| 模型或轮次名称已使用 | 用新名称重试，失败记录和原始数据会保留 |
| 训练时关闭浏览器 | 可重新双击打开；后台训练继续，网页会读取保存状态 |

更新程序可在项目目录运行 `git pull`，再执行安装脚本更新依赖。自己的实验目录和环境不在仓库提交中。
