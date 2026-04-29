[README.md](https://github.com/user-attachments/files/27213335/README.md)
# HW1：NumPy 手写 MLP 实现 EuroSAT 图像分类

本项目使用 NumPy 从零实现三层 MLP 分类器，在 EuroSAT_RGB 遥感图像数据集上完成 10 类土地覆盖分类。代码不使用 PyTorch、TensorFlow、JAX 或自动微分框架，前向传播、反向传播、SGD 更新、学习率衰减、交叉熵损失和 L2 正则化均在 `mlp_numpy.py` 中手写实现。

## 项目文件

- `mlp_numpy.py`：数据读取、预处理、模型定义、手写反向传播、训练、超参数搜索、测试评估与可视化。
- `requirements.txt`：运行所需 Python 依赖。

## 环境依赖

```bash
pip install -r requirements.txt
```

主要依赖包括：numpy、pillow、matplotlib、scikit-learn。

## 运行训练与测试

执行下面命令会完成数据加载、训练、验证集选择最佳权重、测试集评估，并生成报告所需图表和指标文件：

```bash
python mlp_numpy.py --data_dir EuroSAT_RGB --out_dir outputs --image_size 32 --epochs 18
```

快速检查代码能否运行：

```bash
python mlp_numpy.py --data_dir EuroSAT_RGB --out_dir outputs_fast --image_size 32 --epochs 4 --fast
```

## 实验结果

- 最优配置：ReLU，隐藏层维度 256 和 128，初始学习率 0.05，Weight Decay 0.0005。
- 最佳验证集准确率：0.6516
- 测试集准确率：0.6546

## 提交链接

代码仓库：https://github.com/yilinliao520-eng/hw1-mlp-eurosat

模型权重：https://drive.google.com/file/d/1iBXzB2lM4hS3vxHtOJhTCDIGEhzopAHL/view?usp=sharing
