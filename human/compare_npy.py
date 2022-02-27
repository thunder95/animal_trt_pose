import numpy as np

a = np.load("/home/aistudio/tmp/eval_conn_2.npy")
b = np.load("pd_conn.npy")
print(np.allclose(a, b, atol=1e-5))

a = np.load("/home/aistudio/tmp/eval_sg2.npy")
b = np.load("score_graph.npy")
print(np.allclose(a, b, atol=1e-5))

# print(a)
# print(b)