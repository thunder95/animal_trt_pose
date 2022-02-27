from coco import coco_category_to_topology
from find_peaks import *
import json

class ParseObjects(object):
    
    def __init__(self, topology, cmap_threshold=0.1, link_threshold=0.1, cmap_window=5, line_integral_samples=7, max_num_parts=100, max_num_objects=100):
        self.topology = topology
        self.cmap_threshold = cmap_threshold
        self.link_threshold = link_threshold
        self.cmap_window = cmap_window
        self.line_integral_samples = line_integral_samples
        self.max_num_parts = max_num_parts
        self.max_num_objects = max_num_objects
    
    def __call__(self, cmap, paf):
        peak_counts, peaks = find_peaks(cmap, self.cmap_threshold, self.cmap_window, self.max_num_parts)
        print(peak_counts.shape, peaks.shape)

        ''' test ok!
        print(peak_counts)
        print(peaks)
        print(np.allclose(peaks, np.load("/home/aistudio/tmp/eval_peaks.npy")))
        '''

        normalized_peaks = refine_peaks(peak_counts, peaks, cmap, self.cmap_window)
        print(normalized_peaks.shape)

        ''' test ok!
        print(normalized_peaks)
        print(np.allclose(normalized_peaks, np.load("/home/aistudio/tmp/eval_np.npy")))
        '''

        score_graph = paf_score_graph(paf, self.topology, peak_counts, normalized_peaks, self.line_integral_samples)
        print(score_graph.shape)

        ''' test ok!
        print(score_graph)
        print(np.allclose(score_graph, np.load("/home/aistudio/tmp/eval_sg.npy")))
        '''

        connections = assignment(score_graph, self.topology, peak_counts, self.link_threshold)
        print(connections.shape)

        ''' test ok!
        print(np.allclose(connections, np.load("/home/aistudio/tmp/eval_conn.npy")))
        '''
        
        object_counts, objects = connect_parts(connections, self.topology, peak_counts, self.max_num_objects)
        print(objects.shape)

        ''' test ok!
        print(np.allclose(objects, np.load("/home/aistudio/tmp/eval_objs.npy")))
        '''

        return object_counts, objects, normalized_peaks

# 测试对齐pytorch代码
if __name__ == '__main__':
    with open("/home/aistudio/tmp/human_pose.json", "r") as f:
        human_pose = json.load(f)

    topology = coco_category_to_topology(human_pose)
    parse_objects = ParseObjects(topology)

    cmap = np.load("/home/aistudio/tmp/eval_emap.npy")
    paf =  np.load("/home/aistudio/tmp/eval_paf.npy")
    print(cmap.shape, paf.shape)
    parse_objects(cmap, paf)















    
    