import paddle
import tqdm
import numpy as np
import PIL.Image

import json
import paddle.vision.transforms as T
import os
import cv2
import math
import random

paddle.disable_static()

def generate_cmap(counts, peaks,  height,  width,  stdev,  window):
    N = peaks.shape[0]
    C = peaks.shape[1]
    M = peaks.shape[2]
    H = height
    W = width
    w = int(window / 2)

    cmap = np.zeros((N, C, H, W), dtype=np.float32)
    var = stdev * stdev
    for n in range(N):
        for c in range(C):
            count = counts[n][c];
            for p in range(count):
                i_mean = peaks[n][c][p][0] * H
                j_mean = peaks[n][c][p][1] * W
                i_min = int(i_mean - w)
                i_max = int(i_mean + w + 1)
                j_min = int(j_mean - w)
                j_max = int(j_mean + w + 1)
                if i_min < 0: 
                    i_min = 0
                if (i_max >= H):
                    i_max = H
                if (j_min < 0):
                    j_min = 0
                if (j_max >= W):
                    j_max = W

                # print("debug...", n, c, p, i_min, i_max, i_mean, )
                for i in range(i_min, i_max):
                    d_i = float(i_mean - (float(i) + 0.5))
                    val_i = float(- (d_i * d_i))
                    for j in range(j_min, j_max):
                        d_j = float(j_mean - (float(j) + 0.5))
                        val_j = float(- (d_j * d_j))
                        val_ij = float(val_i + val_j)
                        val = math.exp(val_ij / var)
                        if val > cmap[n][c][i][j]:
                            cmap[n][c][i][j] = val

    return cmap

# 在默认的allclose无法通过，在atol=1e-5能通过
def generate_paf(connections, topology, counts, peaks, height, width, stdev):
    N = connections.shape[0]
    K = topology.shape[0]
    H = height
    W = width
    
    paf = np.zeros((N, 2 * K, H, W),dtype=np.float32)
    var = stdev * stdev

    for n in range(N):
        for k in range(K):
            k_i = int(topology[k][0])
            k_j = int(topology[k][1])
            c_a = int(topology[k][2])
            c_b = int(topology[k][3])
            count = int(counts[n][c_a])
            
            for i in range(H):
                for j in range(W):
                    p_c_i = i + 0.5
                    p_c_j = j + 0.5
                    
                    for i_a in range(count):
                        i_b = int(connections[n][k][0][i_a])
                        if i_b < 0:
                            continue # connection doesn't exist

                        p_a = peaks[n][c_a][i_a]
                        p_b = peaks[n][c_b][i_b]

                        p_a_i = p_a[0] * H
                        p_a_j = p_a[1] * W
                        p_b_i = p_b[0] * H
                        p_b_j = p_b[1] * W
                        p_ab_i = p_b_i - p_a_i
                        p_ab_j = p_b_j - p_a_j
                        p_ab_mag = math.sqrt(p_ab_i * p_ab_i + p_ab_j * p_ab_j) + 1e-5
                        u_ab_i = p_ab_i / p_ab_mag
                        u_ab_j = p_ab_j / p_ab_mag
                
                        p_ac_i = p_c_i - p_a_i
                        p_ac_j = p_c_j - p_a_j
                        
                        # dot product to find tangent bounds
                        dot = p_ac_i * u_ab_i + p_ac_j * u_ab_j
                        tandist = 0.0
                        if dot < 0.0 :
                            tandist = dot
                        elif dot > p_ab_mag:
                            tandist = dot - p_ab_mag
                        
                        # cross product to find perpendicular bounds
                        cross = p_ac_i * u_ab_j - p_ac_j * u_ab_i
                        
                        # scale exponentially RBF by 2D distance from nearest point on line segment
                        scale = math.exp(-(tandist*tandist + cross*cross) / var)
                        paf[n][k_i][i][j] += scale * u_ab_i
                        paf[n][k_j][i][j] += scale * u_ab_j
    return paf;



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
    return  quad

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
    newpeaks = peaks.copy()
    C = counts.shape[0]
    for c in range(C):
        count = int(counts[c])
        newpeaks[c][0:count] = transform_points_xy(newpeaks[c][0:count][:, ::-1], quad)[:, ::-1]
    return newpeaks

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
    # print("len skeleton", K)
    # exit()
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
    counts = np.zeros([C], dtype=np.int32)
    peaks = np.zeros((C, M, 2), dtype=np.float32)
    visibles = np.zeros((len(annotations), C), dtype=np.int32)
    connections = -np.ones((K, 2, M), dtype=np.int32)

    for ann_idx, ann in enumerate(annotations):
        kps = ann['keypoints']
        # print(C,  len(kps))
        # add visible peaks
        for c in range(C): 
            x = kps[c][0]
            y = kps[c][1]
            visible = kps[c][2]
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
    mask = np.ones(image_shape, dtype=np.uint32)
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

        self.mode = mode
        self.image_shape = [224, 224]
        self.target_shape = [56, 56]
        self.stdev=0.025
        self.test_ratio = 0.2

        # 图像转换
        self.transform = None
        if mode == "train":
            self.transforms = T.Compose([
                T.Resize(size=(224, 224)),
                T.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.01),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        else:
            self.transforms = T.Compose([
                T.Resize(size=(224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        # 预配置一些参数
        self.max_part_count = 100
        self.images_dir = os.path.join(data_path, "images")
        self.keep_aspect_ratio = False
        if self.mode == "train":
            self.random_angle = [-0.2, 0.2]
            self.random_scale = [0.5, 2.0]
            self.random_translate = [-0.2, 0.2]
        else:
            self.random_angle = [-0.0, 0.0]
            self.random_scale = [1.0, 1.0]
            self.random_translate = [-0.0, 0.0]
        self.annotations_file = os.path.join(data_path, "keypoints.json")

        #加载缓存文件
        self.train_cache_file = self.annotations_file + '.train_cache'
        self.test_cache_file = self.annotations_file + '.test_cache'

        if not os.path.exists(self.train_cache_file) or not os.path.exists(self.train_cache_file):
            self.create_cache_file()
        
        cache_file =  self.train_cache_file if mode == "train" else self.test_cache_file
        cache = paddle.load(cache_file)
        self.counts = cache['counts']
        self.peaks = cache['peaks'].numpy()
        self.connections = cache['connections'].numpy()
        self.topology = cache['topology'].numpy()
        self.parts = cache['parts']
        self.filenames = cache['filenames']
        self.samples = cache['samples']
        print("Loaded cache file... ... ", cache_file, len(self.filenames), len(self.samples))

    def create_cache_file(self):
        anno_dict = json.load(open(self.annotations_file))
        coco_category = json.load(open("animal_pose.json"))

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
                sample['img'] = os.path.join(self.images_dir, img) #filename
                image = cv2.imread(sample['img'])
                sample['img_shape'] = image.shape
                sample['anns'] = [ann]
                samples[img_id] = sample
            else:
                samples[img_id]['anns'] += [ann]

        # 分割数据集
        slist = list(samples.keys())
        random.shuffle(slist)
        split_num = int(len(slist) * self.test_ratio)
        test_idxs = slist[:split_num]
        self.save_cache_file(samples, test_idxs, split_num, True)
        self.save_cache_file(samples, test_idxs, split_num, False)

    def save_cache_file(self, ori_samples, test_idxs, split_num, is_train): 
        C = len(self.parts)
        K = self.topology.shape[0]
        M = self.max_part_count
        if is_train:
            N = len(ori_samples) - split_num
        else:
            N = split_num

        counts = np.zeros((N, C), dtype=np.int32)
        peaks = np.zeros((N, C, M, 2), dtype=np.float32)
        connections = np.zeros((N, K, 2, M), dtype=np.int32)
        filenames = []
        samples = []
        
        i = 0
        for (ix, idx) in tqdm.tqdm(enumerate(ori_samples.keys())):
            if is_train and idx in test_idxs:
                continue
            if not is_train and idx not in test_idxs:
                continue

            sample = ori_samples[idx]
            filename = sample['img']
            filenames.append(filename)
            counts_i, peaks_i, connections_i = coco_annotations_to_tensors(
                sample['anns'], sample["img_shape"], self.parts, self.topology)
            counts[i] = counts_i
            peaks[i] = peaks_i
            connections[i] = connections_i
            samples += [sample]
            i += 1
            
        cache_file = self.train_cache_file if is_train else self.test_cache_file
        print('Saving to intermediate tensors to train_cache file...')
        paddle.save({
            'counts': counts,
            'peaks': peaks,
            'connections': connections,
            'topology': self.topology,
            'parts': self.parts,
            'filenames': filenames,
            'samples': samples
        }, cache_file)

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
        shiftx = np.random.rand() * (self.random_translate[1] - self.random_translate[0]) + self.random_translate[0]
        shifty = np.random.rand() * (self.random_translate[1] - self.random_translate[0]) + self.random_translate[0]
        scale = np.random.rand() * (self.random_scale[1] - self.random_scale[0]) + self.random_scale[0]
        angle = np.random.rand() * (self.random_angle[1] - self.random_angle[0]) + self.random_angle[0]

        if self.keep_aspect_ratio:
            ar = float(image.width) / float(image.height)
            quad = get_quad(angle, (shiftx, shifty), scale, aspect_ratio=ar)
        else:
            quad = get_quad(angle, (shiftx, shifty), scale, aspect_ratio=1.0)
        
        image = transform_image(image, (self.image_shape[1], self.image_shape[0]), quad)
        mask = transform_image(mask, (self.target_shape[1], self.target_shape[0]), quad)

        peaks = transform_peaks(counts, peaks, quad)
        counts = np.expand_dims(counts, axis=0)
        peaks = np.expand_dims(peaks, axis=0)
        stdev = float(self.stdev * self.target_shape[0])

        cmap = generate_cmap(counts, peaks,
            self.target_shape[0], self.target_shape[1], stdev, int(stdev * 5))
  
        paf = generate_paf(
            self.connections[idx][None, ...], self.topology,
            counts, peaks,
            self.target_shape[0], self.target_shape[1], stdev)

        image = image.convert('RGB')
        if self.transforms is not None:
            image = self.transforms(image)
        return image, paddle.to_tensor(cmap[0]),  paddle.to_tensor(paf[0]),  paddle.to_tensor(np.array(mask)[None, ...], dtype="float32")

    def __len__(self):
        return len(self.filenames)

class CocoHumanPoseEval(object):
    
    def __init__(self, images_dir, annotation_file, image_shape, keep_aspect_ratio=False):
        
        self.images_dir = images_dir
        self.annotation_file = annotation_file
        self.image_shape = tuple(image_shape)
        self.keep_aspect_ratio = keep_aspect_ratio
        
        self.cocoGt = pycocotools.coco.COCO('annotations/person_keypoints_val2017.json')
        self.catIds = self.cocoGt.getCatIds('person')
        self.imgIds = self.cocoGt.getImgIds(catIds=self.catIds)
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
    def evaluate(self, model, topology):
        self.parse_objects = ParseObjects(topology, cmap_threshold=0.1, link_threshold=0.1, cmap_window=5, line_integral_samples=7, max_num_parts=100, max_num_objects=100)
        
        results = []

        for n, imgId in enumerate(self.imgIds[1:]):

            # read image
            img = self.cocoGt.imgs[imgId]
            img_path = os.path.join(self.images_dir, img['file_name'])

            image = PIL.Image.open(img_path).convert('RGB')#.resize(IMAGE_SHAPE)
            
            if self.keep_aspect_ratio:
                ar = float(image.width) / float(image.height)
            else:
                ar = 1.0
                
            quad = get_quad(0.0, (0, 0), 1.0, aspect_ratio=ar)
            image = transform_image(image, self.image_shape, quad)

            data = self.transform(image).cuda()[None, ...]

            cmap, paf = model(data)
            cmap, paf = cmap.cpu(), paf.cpu()

        #     object_counts, objects, peaks, int_peaks = postprocess(cmap, paf, cmap_threshold=0.05, link_threshold=0.01, window=5)
        #     object_counts, objects, peaks = int(object_counts[0]), objects[0], peaks[0]

            object_counts, objects, peaks = self.parse_objects(cmap, paf)
            object_counts, objects, peaks = int(object_counts[0]), objects[0], peaks[0]

            for i in range(object_counts):
                object = objects[i]
                score = 0.0
                kps = [0]*(17*3)
                x_mean = 0
                y_mean = 0
                cnt = 0
                for j in range(17):
                    k = object[j]
                    if k >= 0:
                        peak = peaks[j][k]
                        if ar > 1.0: # w > h w/h
                            x = peak[1]
                            y = (peak[0] - 0.5) * ar + 0.5
                        else:
                            x = (peak[1] - 0.5) / ar + 0.5
                            y = peak[0]

                        x = round(float(img['width'] * x))
                        y = round(float(img['height'] * y))

                        score += 1.0
                        kps[j * 3 + 0] = x
                        kps[j * 3 + 1] = y
                        kps[j * 3 + 2] = 2
                        x_mean += x
                        y_mean += y
                        cnt += 1

                ann = {
                    'image_id': imgId,
                    'category_id': 1,
                    'keypoints': kps,
                    'score': score / 17.0
                }
                results.append(ann)
            if n % 100 == 0:
                print('%d / %d' % (n, len(self.imgIds)))


        if len(results) == 0:
            return
        
        with open('trt_pose_results.json', 'w') as f:
            json.dump(results, f)
            
        cocoDt = self.cocoGt.loadRes('trt_pose_results.json')
        cocoEval = pycocotools.cocoeval.COCOeval(self.cocoGt, cocoDt, 'keypoints')
        cocoEval.params.imgIds = self.imgIds
        cocoEval.params.catIds = [1]
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
        
if __name__=='__main__':
    data = AnimalPoseDataset()
    print(111)
    for d in data:
        print(d)
        break




