# 在 X-AnyLabeling 加载自己的模型

本工作台可以直接用 `.pt` 自动标注，再将可编辑的 JSON 交给 X-AnyLabeling 修正。希望在 X-AnyLabeling 内直接运行模型时，使用以下 ONNX 方案。它针对 YOLO26s-OBB 和 X-AnyLabeling v4.3.0。

## 导出 ONNX

在项目目录的 PowerShell 中执行，将 `demo` 替换为实际实验目录，`imgsz` 保持与训练尺寸一致：

```powershell
& .\.venv\Scripts\yolo.exe export `
  model="experiments/demo/models/M0/best.pt" `
  format=onnx imgsz=640 batch=1 `
  dynamic=False half=False end2end=True nms=False `
  opset=17 simplify=False device=cpu
```

如果使用已有训练环境，将命令中的 `yolo.exe` 改为该环境 `Scripts` 目录里的程序。首次导出可能需要联网安装 ONNX 依赖；成功后，权重同目录生成 `best.onnx`。

## 创建 YAML

在 `best.onnx` 旁边创建 `my_yolo26_obb.yaml`：

```yaml
type: yolo26_obb
name: yolo26s-obb-r20260128
provider: Ultralytics
display_name: 我的 M0 YOLO26s-OBB
model_path: best.onnx
conf_threshold: 0.25
max_det: 300
classes:
  - 类别一
  - 类别二
```

将类别替换为实验 `project.json` 中 `classes` 的真实名称，顺序也必须一致。保留 `type`，修改 `display_name` 区分 M0/M1。相对路径相对于 YAML 目录，绝对 Windows 路径建议使用 `/`。

## 在界面使用

打开图片目录，点击左侧 AI 按钮或按 `Ctrl+A`，在模型下拉框选择「加载自定义模型」，选择 **YAML**。加载完成后先对一张图片运行，检查旋转框和类别，再继续标注并人工修正保存。

YOLO26 默认适配器用于 end-to-end、nms-free 输出，以上导出参数与它匹配。不要随意改成其他 YOLO 版本的 `type`。用于本实验审核时打开 `rounds/<轮次>/editable`，保留 `raw` 原始预测和已经归档的测试标注。

依据：[官方自定义模型说明](https://github.com/CVHub520/X-AnyLabeling/blob/v4.3.0/docs/zh_cn/custom_model.md)、[YOLO26s-OBB 模板](https://github.com/CVHub520/X-AnyLabeling/blob/v4.3.0/anylabeling/configs/auto_labeling/yolo26s_obb.yaml)。
