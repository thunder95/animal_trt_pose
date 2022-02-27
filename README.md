## 基于PaddlePaddle复现TrtPose

本文推荐英伟达开源的一个自底向上姿态模型，无需先进行人物检测，直接对姿态关键点进行估计，再进行多人匹配， 运行效率非常高。

TrtPose是一款轻量级，推理速度极快的姿态估计模型，作者在本地基于C++、Cuda和Tensorrt实现的TrtPose,单帧推理不足2ms， 在JetsonNano上也运行得非常快。

原代码基于PyTorch实现: https://github.com/NVIDIA-AI-IOT/trt_pose


## AnimalPose5数据集

本数据集来源： https://github.com/noahcao/animal-pose-dataset

包含有5种类别(cow, sheep, horse, cat, dog), 数据标注按照COCO格式，对于每个实例标注有边界框[xmin, ymin, xmax, ymax]， 以及关键点的二维坐标[x, y, visible]

20 关键点: Two eyes, Throat, Nose, Withers, Two Earbases, Tailbase, Four Elbows, Four Knees, Four Paws.


## Aistudio链接: https://aistudio.baidu.com/aistudio/projectdetail/3516206
