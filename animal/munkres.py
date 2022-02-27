import numpy as np

class CoverTable:
    def __init__(self, nrows, ncols):
        self.nrows = nrows
        self.ncols = ncols
        self.rows = np.zeros((nrows, )) 
        self.cols = np.zeros((ncols, )) 

    def coverRow(self, row):
        self.rows[row] = 1

    def coverCol(self, col):
        self.cols[col] = 1

    def uncoverRow(self, row):
        self.rows[row] = 0

    def uncoverCol(self, col):
        self.cols[col] = 0

    def isCovered(self, row, col):
        return self.rows[row] or self.cols[col]
    
    def isRowCovered(self, row):
        return self.rows[row]

    def isColCovered(self, col):
        return self.cols[col]

    def clear(self):
        self.rows = np.zeros((self.nrows, )) 
        self.cols = np.zeros((self.ncols, )) 

class PairGraph:
    def __init__(self, nrows, ncols):
        self.nrows = nrows
        self.ncols = ncols
        self.rows = -np.ones((nrows, )) 
        self.cols = -np.ones((ncols, )) 

    def colForRow(self, row):
        return self.rows[row]

    def rowForCol(self, col):
        return self.cols[col]

    def clear(self):
        self.rows = -np.ones((self.nrows, )) 
        self.cols = -np.ones((self.ncols, )) 

    def numPairs(self):
        return self.rows[self.rows >= 0].shape[0]
  
    def isRowSet(self, row):
        return self.rows[row] >= 0
    
    def isColSet(self, col):
        return self.cols[col] >= 0

    def pairs(self, ):
        a1 = np.where(self.rows >= 0)[0]
        a2 = self.rows[self.rows >= 0]
        return np.hstack((np.expand_dims(a1, axis=1), np.expand_dims(a2, axis=1)))

    def set(self, row, col):
        self.rows[row] = col
        self.cols[col] = row

    def isPair(self, row, col):
        return self.rows[row] == col
    
    def reset(self, row, col):
        self.rows[row] = -1
        self.cols[col] = -1

def subMinRow(cost_graph, M, nrows, ncols):
    for i in range(nrows):
        _min = cost_graph[i, 0]
        for j in range(ncols):
            if cost_graph[i, j] < _min:
                _min = cost_graph[i, j]
        cost_graph[i, :ncols] -= _min

def subMinCol(cost_graph, M, nrows, ncols):
    for j in range(ncols):
        _min = cost_graph[0, j]
        for i in range(nrows):
            if cost_graph[i, j] < _min:
                _min = cost_graph[i, j]
        cost_graph[:nrows, j] -= _min

def munkresStep1(cost_graph, M, star_graph, nrows, ncols):
    for i in range(nrows):
        for j in range(ncols):
            # print(i, j, star_graph.colForRow(i) , star_graph.rowForCol(j), cost_graph[i, j] )
            if not star_graph.isRowSet(i) and not star_graph.isColSet(j) and cost_graph[i, j] == 0:
                star_graph.set(i, j) # 若没有设定值 cost_graph(score_graph)=0
                # print("star_graph", i, j)
    return star_graph

def munkresStep2(star_graph, cover_table):
    k = star_graph.nrows if star_graph.nrows < star_graph.ncols else star_graph.ncols
    count = 0
    for j in range(star_graph.ncols):
        if (star_graph.isColSet(j)):
            cover_table.coverCol(j)
            count+=1
    return count >= k

def munkresStep3(cost_graph, M, star_graph, prime_graph, cover_table, p, nrows, ncols):
    for i in range(nrows):
        for j in range(ncols):
            # print("step3", i, j, cost_graph[i, j], star_graph.colForRow(i))
            if cost_graph[i, j] == 0 and not cover_table.isCovered(i, j):
                prime_graph.set(i, j)

                if star_graph.isRowSet(i):
                    cover_table.coverRow(i)
                    cover_table.uncoverCol(int(star_graph.colForRow(i)))
                else:
                    p[0] = i
                    p[1] = j
                    return 1, p
    return 0, p

def munkresStep4(star_graph, prime_graph, cover_table, p):
    while(star_graph.isColSet(int(p[1]))):
        s = [int(star_graph.rowForCol(p[1])), p[1]]
        star_graph.reset(s[0], s[1])
        p = [s[0], int(prime_graph.colForRow(s[0]))]

    star_graph.set(p[0], p[1])
    cover_table.clear()
    prime_graph.clear()

    return p

def munkresStep5(cost_graph, M,  cover_table, p, nrows, ncols):
    valid = False
    _min = -1
    for i in range(nrows):
        for j in range(ncols):
            if not cover_table.isCovered(i, j):
                if not valid:
                    _min = cost_graph[i, j]
                    valid = True
                elif cost_graph[i, j] < _min:
                    _min = cost_graph[i, j]

    for i in range(nrows):
        if cover_table.isRowCovered(i):
            for j in range(ncols):
                cost_graph[i, j] += _min

    for j in range(ncols):
        if not cover_table.isColCovered(j):
            for i in range(nrows):
                # print(i, j, _min, cost_graph[i, j])
                cost_graph[i, j] -= _min
                # print(i, j, _min, cost_graph[i, j])

#Munkres 算法步骤, 参考: https://blog.csdn.net/yiran103/article/details/103826367
def _munkres(cost_graph, M, star_graph, nrows, ncols):
    prime_graph = PairGraph(nrows, ncols)
    cover_table = CoverTable(nrows, ncols)
    prime_graph.clear()
    cover_table.clear()
    star_graph.clear()

    step = 0
    if ncols >= nrows:
        subMinRow(cost_graph, M, nrows, ncols) # 行去除最小值

    if ncols > nrows:
        step = 1;

    done = False
    p = [0, 0]

    tmp_cnt = 0
    while not done:
        tmp_cnt += 1
        if step == 0:
            # test_ok
            subMinCol(cost_graph, M, nrows, ncols) # 列去除最小值

        if step <= 1:
            # test_ok
            munkresStep1(cost_graph, M, star_graph, nrows, ncols)

        if step <= 2: 
            # test_ok
            # 在结果矩阵中找到零（ z z z）。如果其所在行和列中没有加星号的零，则将 z z z 标星。对矩阵中的每个元素重复此操作。转到步骤3。
            if munkresStep2(star_graph, cover_table):
                done = True
                continue

        if step <= 3:
            #覆盖包含加星零的列 test_ok
            res, p = munkresStep3(cost_graph, M, star_graph, prime_graph, cover_table, p, nrows, ncols)
            if not res:
                step = 5
                continue

        if step <= 4:
            p = munkresStep4(star_graph, prime_graph, cover_table, p)
            step = 2
            continue

        if step == 5:
            munkresStep5(cost_graph, M, cover_table, p, nrows, ncols)
            step = 3
            continue
     