# AIMET (https://github.com/quic/aimet)

## 基本简介
---
来自Qualcomm Innovation Center (QuIC，高通创新中心？)

最初目标是为训练好的模型提供量化和模型压缩能力(本文章后续不关注模型压缩部分)，以及可视化工作
![AIMET_index](https://quic.github.io/aimet-pages/releases/1.19.1/_images/AIMET_index.PNG "AIMET_index")

其支持Pytorch和TensorFlow的模型

支持了比较高阶的训练后量化策略，以及量化感知训练

提供了包括分类，检测，分割，姿态，GAN，语音识别等模型。包含了原始模型以及量化后的模型，每个模型都提供了详细的修改指导以完成浮点到定点的量化
## PyTorch Models
### Model Zoo  <a name="pytorch-model-zoo"></a>
<table style="width:50%">
<tr>
    <th>Network</th>
    <th>Model Source <sup>[1]</sup></th>
    <th>Floating Pt (FP32) Model <sup>[2]</sup></th>
    <th>Quantized Model <sup>[3]</sup></th>
    <th>Results <sup>[4]</sup></th>
    <th>Documentation</th>
  </tr>
  <tr>
    <td>MobileNetV2</td>
    <td><a href="https://github.com/tonylins/pytorch-mobilenet-v2">GitHub Repo</a></td>
    <td><a href="https://drive.google.com/file/d/1jlto6HRVD3ipNkAl1lNhDbkBp7HylaqR/view">Pretrained Model</a></td>
    <td><a href="/../../releases/download/mobilenetv2-pytorch/mv2qat_modeldef.tar.gz">Quantized Model</a></td>
    <td>(ImageNet) Top-1 Accuracy <br>FP32: 71.67%<br> INT8: 71.14%</td>
    <td><a href="zoo_torch/Docs/MobilenetV2.md">MobileNetV2.md</a></td>
  </tr>
  <tr>
    <td>EfficientNet-lite0</td>
    <td><a href="https://github.com/rwightman/gen-efficientnet-pytorch">GitHub Repo</a></td>
    <td><a href="https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/efficientnet_lite0_ra-37913777.pth">Pretrained Model</a></td>
    <td><a href="/../../releases/download/pt-effnet-checkpoint/adaround_efficient_lite.pth">Quantized Model</a></td>
    <td>(ImageNet) Top-1 Accuracy <br> FP32: 75.42%<br> INT8: 74.44%</td>
    <td><a href="zoo_torch/Docs/EfficientNet-lite0.md">EfficientNet-lite0.md</a></td>
  </tr>
  <tr>
    <td>DeepLabV3+</td>
    <td><a href="https://github.com/jfzhang95/pytorch-deeplab-xception">GitHub Repo</a></td>
    <td><a href="https://drive.google.com/file/d/1G9mWafUAj09P4KvGSRVzIsV_U5OqFLdt/view">Pretrained Model</a></td>
    <td><a href="/../../releases/download/DeepLabV3-Torch/dlv3+_qat_renamed.tar.gz">Quantized Model</a></td>
    <td>(PascalVOC) mIOU <br>FP32: 72.62%<br> INT8: 72.22%</a></td>
    <td><a href="zoo_torch/Docs/DeepLabV3.md">DeepLabV3.md</a></td>
  </tr>
  <tr>
    <td>MobileNetV2-SSD-Lite</td>
    <td><a href="https://github.com/qfgaohao/pytorch-ssd">GitHub Repo</a></td>
    <td><a href="https://storage.googleapis.com/models-hao/mb2-ssd-lite-mp-0_686.pth">Pretrained Model</a></td>
    <td><a href="/../../releases/download/MV2SSD-Lite-Torch/adaround_mv2ssd_model_new.tar.gz">Quantized Model</a></td>
    <td>(PascalVOC) mAP<br> FP32: 68.7%<br> INT8: 68.6%</td>
    <td><a href="zoo_torch/Docs/MobileNetV2-SSD-lite.md">MobileNetV2-SSD-lite.md</a></td>
  </tr>
  <tr>
    <td>Pose Estimation</td>
    <td><a href="https://github.com/CMU-Perceptual-Computing-Lab/openpose">Based on Ref.</a></td>
    <td><a href="https://github.com/CMU-Perceptual-Computing-Lab/openpose">Based on Ref.</a></td>
    <td><a href="/../../releases/download/pose_estimation_pytorch/pose_estimation_pytorch_weights.tgz">Quantized Model</a></td>
    <td>(COCO) mAP<br>FP32: 0.364<br>INT8: 0.359<br> mAR <br> FP32: 0.436<br> INT8: 0.432</td>
    <td><a href="zoo_torch/Docs/PoseEstimation.md">PoseEstimation.md</a></td>
  </tr>
  <tr> 
    <td>SRGAN</td>
    <td><a href="https://github.com/andreas128/mmsr">GitHub Repo</a></td>
    <td><a href="/../../releases/download/srgan_mmsr_model/srgan_mmsr_MSRGANx4.gz">Pretrained Model</a> (older version from <a href="https://github.com/open-mmlab/mmediting/tree/master/configs/restorers/srresnet_srgan">here</a>)</td>    
    <td><a href="zoo_torch/examples/srgan_quanteval.py">See Example</a></td>
    <td>(BSD100) PSNR/SSIM <br> FP32: 25.51/0.653<br> INT8: 25.5/0.648<br><a href="#srgan-pytorch"> Detailed Results</a></td>
    <td><a href="zoo_torch/Docs/SRGAN.md">SRGAN.md</a></td>
  </tr>
  <tr>
    <td>DeepSpeech2</td>
    <td><a href="https://github.com/SeanNaren/deepspeech.pytorch">GitHub Repo</a></td>
    <td><a href="https://github.com/SeanNaren/deepspeech.pytorch/releases/download/v2.0/librispeech_pretrained_v2.pth">Pretrained Model</a></td>
    <td><a href="zoo_torch/examples/deepspeech2_quanteval.py">See Example</a></td>
    <td>(Librispeech Test Clean) WER <br> FP32<br> 9.92%<br> INT8: 10.22%</td>
    <td><a href="zoo_torch/Docs/DeepSpeech2.md">DeepSpeech2.md</a></td>
  </tr>
</table>

在高通量化白皮书中提到的几个PTQ策略都支持
<ul>
<li>Cross-Layer Equalization: Equalize weight tensors to reduce amplitude variation across channels
<li>Bias Correction: Corrects shift in layer outputs introduced due to quantization
<li>Adaptive Rounding: Learn the optimal rounding given unlabelled data
</ul>

以及还支持本文重点关注的QAT
<ul>
<li>Quantization Simulation: Simulate on-target quantized inference accuracy
<li>Quantization-aware Training: Use quantization simulation to train the model further to improve accuracy
</ul>
对于QAT部分还支持了RNN类的网络例如RNNs, LSTMs，GRUs

量化使用流程如下
![flow_feature_guidebook](https://quic.github.io/aimet-pages/releases/1.19.1/_images/flow_feature_guidebook.PNG "flow_feature_guidebook")
在进行QAT时，单个节点的前向模拟
![quant_4](https://quic.github.io/aimet-pages/releases/1.19.1/_images/quant_4.png "quant_4")
在进行QAT是，单个节点反向传播操作
![quant_5](https://quic.github.io/aimet-pages/releases/1.19.1/_images/quant_5.png "quant_5")

量化导出包含onnx和json文件，json每个节点会包含
<ul>
<li>Min: Numbers below these are clamped
<li>Max: Numbers above these are clamped
<li>Delta: Granularity of the fixed point numbers (is a function of the bit-width selected)
<li>Offset: Offset from zero
</ul>

$ delta=\frac {min-max} {{2}^{bitwidth}-1} $,
$ offset=\frac {-min} {delta} $

在使用Pytorch中使用需要让模型满足一些基本要求
<ul>
<li>要支持onnx导出</li>

```python
model = Model()
torch.onnx.export(model, <dummy_input>, <onnx_file_name>):
```

<li>要支持jit追踪</li>

```python
model = Model()
torch.jit.trace(model, <dummy_input>):
```

<li>层要定义为module，不能用torch.nn.functional等等替换（常见的就比如激活，+等），这一步可以通过 model-preparer 来进行部分替换</li>

```python
before
def forward(...):
    ...
    x = torch.nn.functional.relu(x)
    ...

after
def __init__(self,...):
    ...
    self.relu = torch.nn.ReLU()
    ...
def forward(...):
    ...
    x = self.relu(x)
    ...
```

<li>避免在类中重复使用定义的模块，如果需要多次使用就要定义多个），这一步可以通过model-preparer来进行部分替换</li>

```python
before
def __init__(self,...):
    ...
    self.relu = torch.nn.ReLU()
    ...

def forward(...):
    ...
    x = self.relu(x)
    ...
    x2 = self.relu(x2)
    ...

after
def __init__(self,...):
    ...
    self.relu = torch.nn.ReLU()
    self.relu2 = torch.nn.ReLU()
    ...

def forward(...):
    ...
    x = self.relu(x)
    ...
    x2 = self.relu2(x2)
    ...
```

<li>输入输出应该是torch.Tensors或者torch.Tensors的tuples</li>

```python
before
def __init__(self,...):
...
def forward(self, inputs: Dict[str, torch.Tensor]):
    ...
    x = self.conv1(inputs['image_rgb'])
    rgb_output = self.relu1(x)
    ...
    x = self.conv2(inputs['image_bw'])
    bw_output = self.relu2(x)
    ...
    return { 'rgb': rgb_output, 'bw': bw_output }

after
def __init__(self,...):
...
def forward(self, image_rgb, image_bw):
    ...
    x = self.conv1(image_rgb)
    rgb_output = self.relu1(x)
    ...
    x = self.conv2(image_bw)
    bw_output = self.relu2(x)
    ...
    return rgb_output, bw_output
```

<li>如果要使用model-preparer，需要使用Pytorch1.9+，因为是借助其fx tracing来完成的替换</li>

[model-preparer](https://quic.github.io/aimet-pages/releases/1.19.1/api_docs/torch_model_preparer.html#api-torch-model-preparer )

</ul>

对于torch.FX symbolic tracing API实现两个功能需求

1、用 torch.nn.Module 替换 torch.nn.functional

2、为重复，重用的module创建独立的 torch.nn.Module

基本流程为：追踪得到流程，然后通过类型进行判断替换

```Python
import torch
import torch.fx

# Sample module
class M(torch.nn.Module):
    def forward(self, x, y):
        return torch.add(x, y)

def transform(m: torch.nn.Module,
              tracer_class : type = fx.Tracer) -> torch.nn.Module:
    graph : fx.Graph = tracer_class().trace(m)
    # 对于graph中的node，FX会以顺序的形式来表示这个网络
    # 所以我们可以直接for循环来遍历：
    for node in graph.nodes:
        # 检测该node的IR类型是否是call_function
        if node.op == 'call_function':
            # 修改node.target为torch.mul，网络也因此变了
            if node.target == torch.add:
                node.target = torch.mul

    graph.lint() # Does some checks to make sure the
                 # Graph is well-formed.

    return fx.GraphModule(m, graph)
```

## 使用和代码解析：
---

**[QuantizationSimModel](https://github.com/quic/aimet/blob/986ece886c431f37ad992b3eb7b403f8806bf8a7/TrainingExtensions/torch/src/python/aimet_torch/quantsim.py#L117)** 添加模拟量化节点，模拟推理或者用于微调抵消量化影响

离线位置：TrainingExtensions->torch->src->python->aimet_torch->quantsim.py #line 117

1、校验量化参数是否合理

2、ConnectedGraph 构建图

3、记录每个module的输入输出个数

4、递归插入量化节点 -> 非rnn类默认使用 StaticGridQuantWrapper

5、取消bias的量化

6、调整transformer层的量化策略->修改为16bit？

7、根据第二步得到的图和配置的特殊op进行二次配置修改
