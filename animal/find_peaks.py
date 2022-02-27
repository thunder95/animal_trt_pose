import numpy as np
import math
import munkres
from queue import Queue

def reflect(idx, _min, _max):
    if (idx < _min):
        return -idx
    elif (idx >= _max):
        return _max - (idx - _max) - 2
    else:
        return idx

def find_peaks(input, threshold, window_size, max_count):
    N, C, H, W = input.shape
    M = max_count
    win = int(window_size / 2)

    counts = np.zeros((N, C), dtype=np.int32)
    peaks = np.zeros((N, C, M, 2), dtype=np.int32)

    for n in range(N):
        for c in range(C):
            count = 0
            for i in range(H):
                if count >= M:
                    break

                for j in range(W):
                    if count >= M:
                        break

                    val = input[n][c][i][j]
                    if val < threshold:
                        continue

                    # H*W的win窗口范围内判断该点val是否是最大值
                    ii_min = i - win
                    if ii_min < 0:
                        ii_min = 0

                    jj_min = j - win
                    if jj_min < 0:
                        jj_min = 0

                    ii_max = i + win + 1
                    if ii_max > H:
                        ii_max = H

                    jj_max = j + win + 1
                    if jj_max > W:
                        jj_max = W

                    is_peak = True
                    for ii in range(ii_min, ii_max):
                        if not is_peak:
                            break
                        for jj in range(jj_min, jj_max):
                            if input[n, c, ii, jj] > val:
                                is_peak = False
                                break

                    if is_peak:
                        peaks[n, c, count, 0] = i
                        peaks[n, c, count, 1] = j
                        count+=1

            counts[n][c] = count

    return counts, peaks

def refine_peaks(counts, peaks, cmap, window_size):
    refined_peaks = np.zeros_like(peaks, dtype=np.float32)
    N, C, H, W = cmap.shape
    win = int(window_size / 2)

    for n in range(N):
        for c in range(C):
            count = counts[n][c]

            for m in range(count):
                refined_peaks[n, c, m, 0] = 0
                refined_peaks[n, c, m, 1] = 0

                i = peaks[n, c, m, 0]
                j = peaks[n, c, m, 1]
                weight_sum = 0.

                for ii in range(i - win, i + win + 1):
                    ii_idx = reflect(ii, 0, H)
                    for jj in range(j - win, j + win + 1):
                        jj_idx = reflect(jj, 0, W)
                        weight = cmap[n, c, ii_idx, jj_idx]
                        refined_peaks[n, c, m, 0] += weight * ii
                        refined_peaks[n, c, m, 1] += weight * jj
                        weight_sum += weight

                #像素加权平均归一化
                refined_peaks[n, c, m, 0] /= weight_sum;
                refined_peaks[n, c, m, 1] /= weight_sum;
                refined_peaks[n, c, m, 0] += 0.5 # center pixel
                refined_peaks[n, c, m, 1] += 0.5 # center pixel
                refined_peaks[n, c, m, 0] /= H   # normalize coordinates
                refined_peaks[n, c, m, 1] /= W   # normalize coordinates

    return refined_peaks

def paf_score_graph(paf, topology, counts, peaks, num_integral_samples):
    N, _, H, W = paf.shape
    K = topology.shape[0]
    _, C, M, _ = peaks.shape
    score_graph = np.zeros((N, K, M, M), dtype=np.float32)
    # print(peaks.shape, score_graph.shape)
    # PAF 将识别到的身体部位与图像中的每个人相关联。 参考：https://zhuanlan.zhihu.com/p/411412223
    # 通过对 u 的均匀间隔值进行采样和求和来近似积分
    # 一个关键点到另一个关键点的可能是同一个身体的置信图

    for n in range(N):
        for k in range(K):
            paf_i_idx, paf_j_idx, cmap_a_idx, cmap_b_idx = list(topology[k])
            # print(k, topology[k])
            counts_a = counts[n, cmap_a_idx]
            counts_b = counts[n, cmap_b_idx]

            for a in range(counts_a):
                pa_i = peaks[n, cmap_a_idx, a, 0] * H
                pa_j = peaks[n, cmap_a_idx, a, 1] * W
                for b in range(counts_b):
                    pb_i = peaks[n, cmap_b_idx, b, 0] * H
                    pb_j = peaks[n, cmap_b_idx, b, 1] * W

                    pab_i = pb_i - pa_i
                    pab_j = pb_j - pa_j
                    pab_norm = math.sqrt(pab_i * pab_i + pab_j * pab_j) + 1e-5
                    uab_i = pab_i / pab_norm
                    uab_j = pab_j / pab_norm

                    integral = 0.;
                    increment = 1.0 / num_integral_samples

                    for t in range(num_integral_samples):
                        """
                        compute integral point T
                        convert to int
                        note: we do not need to subtract 0.5 when indexing, because
                        round(x - 0.5) = int(x)
                        """

                        progress = float(t) / (float(num_integral_samples) - 1)
                        pt_i_int = int(pa_i + progress * pab_i)
                        if pt_i_int < 0:
                            continue
                        if pt_i_int >= H:
                            continue

                        pt_j_int = int(pa_j + progress * pab_j) # skip point if out of bounds (will weaken integral)
                        if pt_j_int < 0:
                            continue
                        if pt_j_int >= W:
                            continue

                        # get vector at integral point from PAF
                        pt_paf_i = float(paf[n, paf_i_idx, pt_i_int, pt_j_int])
                        pt_paf_j = float(paf[n, paf_j_idx, pt_i_int, pt_j_int])

                        # compute dot product of normalized A->B with PAF vector at integral point
                        dot = float(pt_paf_i * uab_i + pt_paf_j * uab_j)
                        integral += dot

                    integral /= num_integral_samples
                    score_graph[n, k, a, b] = integral
    return score_graph

def assignment(score_graph, topology, counts, score_threshold):
    N, C = counts.shape
    K = topology.shape[0]
    M = score_graph.shape[2]

    cost_graph = np.zeros((M, M), dtype=np.float32)
    connections = -np.ones((N, K, 2, M), dtype=np.int32)

    for n in range(N):
        for k in range(K):
            cmap_idx_a = topology[k, 2]
            cmap_idx_b = topology[k, 3]
            count_a = counts[n, cmap_idx_a]
            count_b = counts[n, cmap_idx_b]

            cost_graph[:count_a, :count_b] = -score_graph[n, k, :count_a, :count_b]
            # print(cost_graph)
            # exit()
            star_graph = munkres.PairGraph(count_a, count_b)
            munkres._munkres(cost_graph, M, star_graph, count_a, count_b)

            #fill output connections
            for i in range(count_a):
                for j in range(count_b):
                    if star_graph.isPair(i, j) and score_graph[n, k, i, j] > score_threshold:
                        connections[n, k, 0, i] = j
                        connections[n, k, 1, j] = i

    return connections

def connect_parts(connections, topology, peak_counts, max_num_objects):
    print(connections.shape)
    N, C = peak_counts.shape
    K = topology.shape[0]
    M = connections.shape[3]
    P = max_num_objects
    objects = -np.ones((N, max_num_objects, C), dtype=np.int32)
    object_counts = np.zeros((N, ), dtype=np.int32)
    for n in range(N):
        objects[n, :M, :C] = -1
        num_objects = 0
        visited = np.zeros((C, M))
        for c in range(C):
            if num_objects >= P:
                break

            for i in range(peak_counts[n, c]):
                if num_objects >= P:
                    break

                q = Queue()
                new_object = False
                q.put((c, i))
                while not q.empty():
                    tq = q.get()
                    c_n, i_n = tq[0], tq[1]

                    if visited[c_n, i_n]:
                       continue

                    visited[c_n, i_n] = 1
                    new_object = True
                    objects[n, num_objects, c_n] = i_n

                    for k in range(K):
                        c_a = topology[k, 2]
                        c_b = topology[k, 3]

                        if c_a == c_n:
                            i_b = connections[n, k, 0, i_n]
                            if i_b >= 0:
                                q.put((c_b, i_b))

                        if c_b == c_n:
                            i_a = connections[n, k, 1, i_n]
                            if i_a >= 0:
                                q.put((c_a, i_a))

                if new_object:
                    num_objects+=1
        object_counts[n] = num_objects
    
    return object_counts, objects







