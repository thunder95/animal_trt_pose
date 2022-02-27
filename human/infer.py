from trt_pose_model import get_model
import cv2
import paddle.vision.transforms as transforms
import PIL.Image
from draw_objects import DrawObjects
from parse_objects import ParseObjects
import json
import sys
import paddle
from coco import coco_category_to_topology

# 获取图片路径
if len(sys.argv) < 2:        
    sys.exit("parameter error")    
img_path = sys.argv[1] #第一个参数指的是脚本名称

#读取姿态配置
with open("/home/aistudio/tmp/human_pose.json", "r") as f:
    pose_config = json.load(f)
num_parts = len(pose_config["keypoints"])
num_links = len(pose_config["skeleton"])
topology = coco_category_to_topology(pose_config)

#数据预处理
trans_func = transforms.Compose([
    paddle.vision.transforms.ToTensor(),
    paddle.vision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

ori_img = cv2.imread(img_path)
image = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
image = trans_func(image)

#加载模型
model = get_model(num_parts, 2 * num_links)
model.eval()
wgts = paddle.load("trt_pose.pdparams")
model.set_state_dict(wgts)

#模型推理
input = paddle.unsqueeze(image, axis=0)
cmap, paf = model(input)
print(cmap.shape, paf.shape)

#模型后处理
parse_objects = ParseObjects(topology)
draw_objects = DrawObjects(topology)
counts, objects, peaks = parse_objects(cmap.numpy(), paf.numpy())#, cmap_threshold=0.15, link_threshold=0.15)

#推理结果可视化保存图片
draw_objects(ori_img, counts, objects, peaks)
cv2.imwrite("/home/aistudio/tmp/result.jpg", ori_img)
