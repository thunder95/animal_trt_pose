import paddle
import tqdm
import numpy as np
import PIL.Image

# 根据变换参数获取变换矩阵
def get_quad(angle, translation, scale, aspect_ratio=1.0):
    if aspect_ratio > 1.0:
        # width > height =>
        # increase height region
        quad = np.array([
            [0.0, 0.5 - 0.5 * aspect_ratio],
            [0.0, 0.5 + 0.5 * aspect_ratio],
            [1.0, 0.5 + 0.5 * aspect_ratio],
            [1.0, 0.5 - 0.5 * aspect_ratio],
            
        ])
    elif aspect_ratio < 1.0:
        # width < height
        quad = np.array([
            [0.5 - 0.5 / aspect_ratio, 0.0],
            [0.5 - 0.5 / aspect_ratio, 1.0],
            [0.5 + 0.5 / aspect_ratio, 1.0],
            [0.5 + 0.5 / aspect_ratio, 0.0],
            
        ])
    else:
        quad = np.array([
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [1.0, 0.0],
        ])
        
    quad -= 0.5

    R = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]
    ])

    quad = np.dot(quad, R)
    quad -= np.array(translation)
    quad /= scale
    quad += 0.5
    return 

#PIL图像变换
def transform_image(image, size, quad):
    new_quad = np.zeros_like(quad)
    new_quad[:, 0] = quad[:, 0] * image.size[0]
    new_quad[:, 1] = quad[:, 1] * image.size[1]
    
    new_quad = (new_quad[0][0], new_quad[0][1],
            new_quad[1][0], new_quad[1][1],
            new_quad[2][0], new_quad[2][1],
            new_quad[3][0], new_quad[3][1])
    
    return image.transform(size, PIL.Image.QUAD, new_quad)

# 坐标系变换
def transform_points_xy(points, quad):
    p00 = quad[0]
    p01 = quad[1] - p00
    p10 = quad[3] - p00
    p01 /= np.sum(p01**2)
    p10 /= np.sum(p10**2)
    
    A = np.array([
        p10,
        p01,
    ]).transpose()
    
    return np.dot(points - p00, A)

# 获取新坐标系下的peaks
def transform_peaks(counts, peaks, quad):
    newpeaks = peaks.clone().numpy()
    C = counts.shape[0]
    for c in range(C):
        count = int(counts[c])
        newpeaks[c][0:count] = transform_points_xy(newpeaks[c][0:count][:, ::-1], quad)[:, ::-1]
    return paddle.to_tensor(newpeaks)

# 关键点名称
def coco_category_to_parts(coco_category):
    """Gets list of parts name from a COCO category
    """
    return coco_category['keypoints']

# 骨骼关键点的拓扑图
def coco_category_to_topology(coco_category):
    """Gets topology tensor from a COCO category
    """
    skeleton = coco_category['skeleton']
    K = len(skeleton)
    topology = []
    for k in range(K):
        topology.append([2 * k, 2 * k + 1, skeleton[k][0] - 1, skeleton[k][1] - 1])
    return np.asarray(topology)

# coco标注数据转为张量
def coco_annotations_to_tensors(coco_annotations,
                                image_shape,
                                parts,
                                topology,
                                max_count=100):
    """Gets tensors corresponding to peak counts, peak coordinates, and peak to peak connections
    """
    annotations = coco_annotations
    C = len(parts)
    K = topology.shape[0]
    M = max_count

    IH = image_shape[0]
    IW = image_shape[1]
    counts = paddle.zeros((C), dtype="int32")
    peaks = paddle.zeros((C, M, 2), dtype="float32")
    visibles = paddle.zeros((len(annotations), C), dtype="int32")
    connections = -paddle.ones((K, 2, M), dtype="int32")

    for ann_idx, ann in enumerate(annotations):
        kps = ann['keypoints']
        # add visible peaks
        for c in range(C): 
            x = kps[c * 3]
            y = kps[c * 3 + 1]
            visible = kps[c * 3 + 2]
            if visible: #只有关键点可见时
                peaks[c][counts[c]][0] = (float(y) + 0.5) / (IH + 1.0) # 图像宽高归一化, peak实际是归一化的坐标数据,不可见时为0
                peaks[c][counts[c]][1] = (float(x) + 0.5) / (IW + 1.0)
                counts[c] = counts[c] + 1 # 对每个关键点进行统计visible数量
                visibles[ann_idx][c] = 1 # visibles列表
        for k in range(K):
            c_a = topology[k][2]
            c_b = topology[k][3]
            if visibles[ann_idx][c_a] and visibles[ann_idx][c_b]:
                connections[k][0][counts[c_a] - 1] = counts[c_b] - 1
                connections[k][1][counts[c_b] - 1] = counts[c_a] - 1 # 构建关联节点 0：a->b, 1: b->a
    return counts, peaks, connections


def coco_annotations_to_mask_bbox(coco_annotations, image_shape):
    mask = np.ones(image_shape, dtype=np.uint8)
    for ann in coco_annotations:
        if 'num_keypoints' not in ann or ann['num_keypoints'] == 0:
            bbox = ann['bbox']
            x0 = round(bbox[0])
            y0 = round(bbox[1])
            x1 = round(x0 + bbox[2])
            y1 = round(y0 + bbox[3])
            mask[y0:y1, x0:x1] = 0
    return mask
            

class AnimalPoseDataset(paddle.io.Dataset):
    def __init__(self,
                 data_path = "/home/aistudio/data",
                 mode="train"):
        super(AnimalPoseDataset, self).__init__()

        # self.mode = mode
        # self.image_path = os.path.join(data_path, "images")
        # kp_path = os.path.join(data_path, "keypoints.json")
        # anno_dict = json.load(open(kp_path))
        # self.annotation_list = anno_dict["annotations"]
        # self.image_dict = anno_dict["images"]
        # self.num_samples = len(self.annotation_list)
        
        # 图像转换
        self.train_transform = T.Compose([
            T.Resize(size=(224, 224)),
            T.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.01),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        self.test_transform = T.Compose([
            T.Resize(size=(224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        # 预配置一些参数
        max_part_count = 100
        self.images_dir = "/home/aistudio/animalpose5/images"
        self.keep_aspect_ratio = False
        if self.mode == "train":
            self.random_angle = [-0.2, 0.2]
            self.random_scale = [0.5, 2.0]
            self.random_translate = [-0.2, 0.2]
        else:
            self.random_angle = [-0.0, 0.0]
            self.random_scale = [1.0, 1.0]
            self.random_translate = [-0.0, 0.0]

        #加载缓存文件
        annotations_file = "keypoints.json"
        paddle_cache_file = annotations_file + '.cache'
        
        if paddle_cache_file is not None and os.path.exists(paddle_cache_file):
            print('Cachefile found.  Loading from cache file...')
            cache = paddle.load(paddle_cache_file)
            self.counts = cache['counts']
            self.peaks = cache['peaks']
            self.connections = cache['connections']
            self.topology = cache['topology']
            self.parts = cache['parts']
            self.filenames = cache['filenames']
            self.samples = cache['samples']
            return
            

        #加载json标注文件
        anno_dict = json.load(open(annotations_file))
        coco_category = json.load(open("/home/aistudio/work/animal_pose.json"))

        #获取topology和parts
        self.topology = coco_category_to_topology(coco_category)
        self.parts = coco_category_to_parts(coco_category)
        
        # 构建samples, 1个img_id可以有多个ann
        img_map = anno_dict['images']
        samples = {}
        for ann in anno_dict['annotations']:
            img_id = str(ann['image_id'])
            img = img_map[img_id]

            if img_id not in samples:
                sample = {}
                sample['img'] = os.path.join(images_dir, img) #filename
                image = cv2.imread(sample['img'])
                sample['img_shape'] = image.shape
                sample['anns'] = [ann]
                samples[img_id] = sample
            else:
                samples[img_id]['anns'] += [ann]
        
        N = len(samples)
        C = len(parts)
        K = tp.shape[0]
        M = max_part_count

        # print(N, C, K, M)
        print('Generating intermediate tensors...')
        self.counts = paddle.zeros((N, C), dtype='int32')
        self.peaks = paddle.zeros((N, C, M, 2), dtype='float32')
        self.connections = paddle.zeros((N, K, 2, M), dtype='int32')
        self.filenames = []
        self.samples = []
        
        for i, sample in tqdm.tqdm(enumerate(samples.values())):
            filename = sample['img']
            self.filenames.append(filename)
            counts_i, peaks_i, connections_i = coco_annotations_to_tensors(
                sample['anns'], sample["img_shape"], self.parts, self.topology)
            self.counts[i] = counts_i
            self.peaks[i] = peaks_i
            self.connections[i] = connections_i
            self.samples += [sample]

        if paddle_cache_file is not None:
            print('Saving to intermediate tensors to cache file...')
            paddle.save({
                'counts': self.counts,
                'peaks': self.peaks,
                'connections': self.connections,
                'topology': self.topology,
                'parts': self.parts,
                'filenames': self.filenames,
                'samples': self.samples
            }, paddle_cache_file)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        image = PIL.Image.open(filename)
        
        im = self.samples[idx]['img']
        im_shape = self.samples[idx]["img_shape"]
        
        mask = coco_annotations_to_mask_bbox(self.samples[idx]['anns'], (im_shape[0], im_shape[1]))
        mask = PIL.Image.fromarray(mask)
        
        counts = self.counts[idx]
        peaks = self.peaks[idx]
        
        # affine transformation
        shiftx = float(paddle.rand(1)) * (self.random_translate[1] - self.random_translate[0]) + self.random_translate[0]
        shifty = float(paddle.rand(1)) * (self.random_translate[1] - self.random_translate[0]) + self.random_translate[0]
        scale = float(paddle.rand(1)) * (self.random_scale[1] - self.random_scale[0]) + self.random_scale[0]
        angle = float(paddle.rand(1)) * (self.random_angle[1] - self.random_angle[0]) + self.random_angle[0]
        
        if self.keep_aspect_ratio:
            ar = float(image.width) / float(image.height)
            quad = get_quad(angle, (shiftx, shifty), scale, aspect_ratio=ar)
        else:
            quad = get_quad(angle, (shiftx, shifty), scale, aspect_ratio=1.0)
        
        image = transform_image(image, (self.image_shape[1], self.image_shape[0]), quad)
        mask = transform_image(mask, (self.target_shape[1], self.target_shape[0]), quad)
        peaks = transform_peaks(counts, peaks, quad)
        
        counts = paddle.unsqueeze(count, axis=0)
        peaks = paddle.unsqueeze(peaks, axis=0)

        stdev = float(self.stdev * self.target_shape[0])

        #todo
        cmap = trt_pose.plugins.generate_cmap(counts, peaks,
            self.target_shape[0], self.target_shape[1], stdev, int(stdev * 5))

        #todo 
        paf = trt_pose.plugins.generate_paf(
            self.connections[idx][None, ...], self.topology,
            counts, peaks,
            self.target_shape[0], self.target_shape[1], stdev)

        image = image.convert('RGB')
        if self.transforms is not None:
            image = self.transforms(image)
            
        return image, cmap[0], paf[0], torch.from_numpy(np.array(mask))[None, ...]

    def __len__(self):
        return len(self.filenames)





