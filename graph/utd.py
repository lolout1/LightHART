import numpy as np
import sys
import os

sys.path.extend(['../'])
from graph import tools
import networkx as nx

# skeleton_joints = [
#     "head",  # 0
#     "shoulder_center",1
#     "spine",2
#     "hip_center",3
#     "left_shoulder",4
#     "left_elbow",5  # 5
#     "left_wrist",6
#     "left_hand",7
#     "right_shoulder",8
#     "right_elbow",9
#     "right_wrist",10  # 10
#     "right_hand",11
#     "left_hip",12
#     "left_knee",13
#     "left_ankle",14
#     "left_foot",15  # 15
#     "right_hip",16
#     "right_knee",17
#     "right_ankle",18
#     "right_foot"19
# ]
outward = [(0, 1), (1,2),(1,4), (4,5),(5,6),(6,7), (2,3),(3,12),
          (1,8), (8,9),(9,10),(10,11) ,(12,13), (13,14), (14,15), 
          (3,16), (16,17), (17, 18), (18,19)]
# Edge format: (origin, neighbor)
num_node = 20
self_link = [(i, i) for i in range(num_node)]


inward = [(j, i) for (i, j) in outward]
neighbor = inward + outward


class Graph:
    def __init__(self, labeling_mode='spatial'):
        self.A = self.get_adjacency_matrix(labeling_mode)
        self.num_node = num_node
        self.self_link = self_link
        self.inward = inward
        self.outward = outward
        self.neighbor = neighbor

    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            A = tools.get_spatial_graph(num_node, self_link, inward, outward)
        else:
            raise ValueError()
        return A


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import os

    # os.environ['DISPLAY'] = 'localhost:11.0'
    A = Graph('spatial').get_adjacency_matrix()
    for i in A:
        plt.imshow(i, cmap='gray')
        plt.savefig('utdgraph')
    print(A)