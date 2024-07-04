import numpy as np
# from env_components.gpu import Gpu
# from networkx.drawing.nx_pydot import graphviz_layout
# import matplotlib
import networkx as nx
from operator import itemgetter

from env_components.gpu import Gpu
# import torch
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt

__all__ = ['NetworkTopology']


class NetworkTopology:
    """
        The class to hold the topology  of the environment.

        This class must be created once in beginning, it constructs the cluster topology,
        the adjacency matrix and then initialise all gpus.

        Attributes
        ----------
        G : networkx
            it is the created graph
        node_index : int
            used to index each node in the graph, it is increased by 1 when a node is added
        dic_nam_idx : dictionary
            this dictionary contains each node in the graph as a key and its index as value
        edges_array: ndarray
            contain all edges in the graph
        nvLink: int 40
            the weight of the edge between gpus sharing same host
            NVLink between two nodes within the same host
        infiniBand: int 4
            the weight of the edge between gpus sharing same n
            InfiniBand between GPUs across hosts
        peer_to_peer: int 2
            Peer-to-peer direct link within racks
        resources: ndarray
            ndarray containing the shape of all resource in the cluster as host rack gpu
        gpu_number: int
            total number of gpu in the cluster
        cluster_gpus: ndarray
            ndarray containing all gpus object
        adjacency_matrix: 2d-array
            matrix representing the adjacency between gpus, each cell contain the weigh of the path.
    """

    def __init__(self, args):

        self.num_gpus_per_machine = args.num_gpus_per_machine
        self.num_machines_per_rack = args.num_machines_per_rack
        self.num_racks_per_cluster = args.num_racks_per_cluster


        self.G = None
        self.node_index = 0
        self.dic_nam_idx = {}
        self.edges_array = []
        self.nvLink = 40
        self.infiniBand = 4
        self.peer_to_peer = 2
        # rack Host class, contains gpu
        self.resources = -np.ones((self.num_racks_per_cluster,
                                   self.num_machines_per_rack,
                                   self.num_gpus_per_machine), dtype=np.int32)
        self.gpu_number = self.num_racks_per_cluster * \
                          self.num_machines_per_rack * \
                          self.num_gpus_per_machine
        self.cluster_gpus = [Gpu(gpu,rack, machine,0,0) for rack in range(self.num_racks_per_cluster) for machine in range(self.num_machines_per_rack) for gpu in range(self.num_gpus_per_machine)]
        self.adjacency_matrix = np.ones((self.gpu_number, self.gpu_number))
        # self.node_index_build()
        # self.G = self.create_graph()

        self.create_adjacency_matrix()


        ### Yiming part
        self.G_clutser, self.G_cluster_only_gpus, self.demand_mask = self.cluster_G_creater()
        self.aj_matrix = nx.adjacency_matrix(self.G_cluster_only_gpus).todense()
        self.nodes_lst = np.array(list(self.G_clutser))
        self.demand_mask = None
        resources = nx.adjacency_matrix(self.G_cluster_only_gpus).todense()
        self.static = np.zeros((1, resources.shape[0], resources.shape[1]), dtype=np.float32)
        self.static[0:] = resources
        # self.root = torch.from_numpy(self.static[0][0].reshape(-1, 1))
        self.sp = dict(nx.all_pairs_shortest_path(self.G_clutser))



    ######## Yiming Functions VVV #########
    def cluster_G_creater(self):
        G_cluster = nx.Graph()
        G_cluster_pure_gpus = nx.Graph()
        # G_cluster_pure_gpus.add_node('center')
        cur = 0
        G_cluster.add_node('center')
        demand_mask = [True]
        for i in range(self.num_racks_per_cluster):
            node_r = f'r{i}'
            G_cluster.add_node(node_r)
            G_cluster.add_edge(node_r, 'center', weight=1 / self.peer_to_peer)
            demand_mask.append(True)
            for j in range(self.num_machines_per_rack):
                node_m = f'r{i}m{j}'
                G_cluster.add_node(node_m)
                G_cluster.add_edge(node_m, node_r, weight=1 / self.infiniBand)
                demand_mask.append(True)
                for k in range(self.num_gpus_per_machine):
                    node_g = f'r{i}m{j}g{k}'
                    G_cluster.add_node(node_g)
                    G_cluster.add_edge(node_g, node_m, weight=1 / self.nvLink)
                    demand_mask.append(True)
                    for l in range(k):
                        node_neighbor = f'r{i}m{j}g{k - l - 1}'
                        G_cluster.add_edge(node_g, node_neighbor, weight=1 / self.nvLink)
                    cur += 1

        for node_1 in G_cluster.nodes():
            if 'g' in node_1:
                G_cluster_pure_gpus.add_node(node_1)
        for node_1 in G_cluster_pure_gpus.nodes():
            for node_2 in G_cluster_pure_gpus.nodes():
                if node_1 != node_2:
                    G_cluster_pure_gpus.add_edge(node_1, node_2, weight=nx.dijkstra_path_length(G_cluster, source=node_1, target=node_2))
        return G_cluster, G_cluster_pure_gpus, demand_mask



    # def topology_render(self, G, nodes):
    #     nodes.append('center')
    #     routes, path = self.find_routes(nodes)
    #     plt.close('all')

    #     pos = graphviz_layout(G, prog="twopi")

    #     # nx.draw_networkx_nodes(self.G, pos, nodelist=[], node_color="g")

    #     plt.rcParams["figure.figsize"] = [12, 10]
    #     plt.rcParams["figure.dpi"] = 60
    #     plt.rcParams["figure.autolayout"] = False
    #     nx.draw_networkx(G, pos, with_labels=True)
    #     # for ctr, edgelist in enumerate(path):
    #     #     nx.draw_networkx_edges(self.G, pos=pos, edgelist=edgelist, edge_color='r', width=5)
    #     nx.draw_networkx_edges(G, pos=pos, edgelist=path, edge_color='r', width=1)

    #     plt.tight_layout()
    #     plt.savefig(f'{nodes}.png', bbox_inches='tight', dpi=200, format='png')


    def gpus_string_converter(self, gpus):
        result = ['center']

        for i in range(len(gpus[0])):
            result.append(f'r{gpus[0][i]}m{gpus[1][i]}g{gpus[2][i]}')
        return result

    def string_gpus_converter(self, gpu_str):

        r = np.array([int(x[1]) for x in gpu_str])
        m = np.array([int(x[3]) for x in gpu_str])
        g = np.array([int(x[5]) for x in gpu_str])
        return (r, m, g)

    def knn(self, graph, node, n):
        return list(map(itemgetter(1),sorted([(e[2]['weight'], e[1])
                                              for e in graph.edges(node, data=True)])[:n]))


    def find_routes(self, nodes):
        path = []
        routes = [(a, b) for idx, a in enumerate(nodes) for b in nodes[idx + 1:]]
        for pair in routes:
            path.extend(self.construct_edges(self.sp[pair[0]][pair[1]]))
        path = self.removeDuplicates(path)
        return routes, path

    def construct_edges(self, nodes):
        lst = []
        prev = nodes[0]
        for node in nodes:
            if prev != node:
                lst.append((prev, node))
            prev = node
        return lst

    def removeDuplicates(self, lst):
        result = []
        for item in lst:
            if item not in result and ((item[1], item[0])) not in result:
                result.append(item)
        return result


    # def distance_compute(self, tour_indices):
    #     """
    #     tour distance of selected GPUs
    #     """
    #     rewards = []
    #     for tour in tour_indices:
    #         nodes_i = np.array([self.nodes_lst[tour.cpu()]])
    #         nodes_i = np.append(nodes_i, 'center')
    #         routes, path = self.find_routes(nodes_i)
    #         length = 0
    #         for pair in path:
    #             length += self.G_clutser[pair[0]][pair[1]]["weight"]
    #         rewards.append(-length)

    #     return torch.tensor(rewards)


    ######## Yiming Functions ^^^ #########

    # def node_index_build(self):
    #     """
    #
    #     :return:
    #     """
    #     for i, racks in enumerate(self.resources):
    #         # r==> racks
    #         namer = 'r' + str(i)
    #         self.dic_nam_idx[namer] = self.node_index
    #         self.node_index += 1
    #         for l, _ in enumerate(self.resources):
    #             if i > l:
    #                 namer1 = 'r' + str(l)
    #                 self.edges_array.append((namer, namer1))
    #         for j, node in enumerate(racks):
    #             # h==>host
    #             nameh = namer + 'h' + str(j)
    #             self.dic_nam_idx[nameh] = self.node_index
    #             self.node_index += 1
    #             self.edges_array.append((namer, nameh))
    #             for k, gpu in enumerate(node):
    #                 #  g ==> gpu
    #                 nameg = nameh + 'g' + str(k)
    #                 self.dic_nam_idx[nameg] = self.node_index
    #                 self.node_index += 1
    #                 self.edges_array.append((nameh, nameg))
    #                 self.cluster_gpus.append(Gpu(gpu_id=self.dic_nam_idx[nameg],
    #                                              host_id=self.dic_nam_idx[nameh],
    #                                              rack_id=self.dic_nam_idx[namer],
    #                                              remaining_time=0,
    #                                              status=0
    #                                              ))
    #
    #     # dic_nam_idx: {'r0': 0, 'r0h0': 1, 'r0h0g0': 2, 'r0h0g1': 3, 'r0h1g0': 4, 'r0h1g1': 5, ...}
    #     # edges_array= [('r0', 'r0h0'), ('r0h0', 'r0h0g0'), ('r0h0', 'r0h0g1'), ('r0h1', 'r0h1g0'),
    #     # ('r0', 'r1'), ('r1', 'r1h0'), ...]
    #     #  cluster= [Gpu0, Gpu1, Gpu2, Gpu3, ...]
    #     #  Gpu0={'gpu_id' = 2, 'node_id' = 1, 'rack_id' = 0, 'remaining_time'=0}
    #
    # def create_graph(self):
    #     G = nx.Graph()
    #     for nid in self.dic_nam_idx.values():
    #         G.add_node(nid)
    #
    #     for edg in self.edges_array:
    #         G.add_edge(self.dic_nam_idx[edg[0]], self.dic_nam_idx[edg[1]])
    #     return G
    #
    def create_adjacency_matrix(self):
        for gpua in range(len(self.cluster_gpus)):
            for gpub in range(len(self.cluster_gpus)):
                if self.cluster_gpus[gpua].rack_id == self.cluster_gpus[gpub].rack_id:
                    if self.cluster_gpus[gpua].host_id == self.cluster_gpus[gpub].host_id:
                        if self.cluster_gpus[gpua].gpu_id == self.cluster_gpus[gpub].gpu_id:
                            self.adjacency_matrix[gpua, gpub] = 0
                        else:
                            self.adjacency_matrix[gpua, gpub] = 1 / self.nvLink
                    else:
                        self.adjacency_matrix[gpua, gpub] = 1 / self.infiniBand
                else:
                    # print(f'{gpua.gpu_id} - {gpub.gpu_id}')
                    self.adjacency_matrix[gpua, gpub] = 1 / self.peer_to_peer
    #
    # def get_gpu_distance_from_all(self, Gpu_ID):
    #     # Gpu_ID int
    #     return self.adjacency_matrix[Gpu_ID]
    #
    # def get_gpu_distance_gpu(self, Gpu_ID1, Gpu_ID2):
    #     # Gpu_ID int
    #     list = self.adjacency_matrix[Gpu_ID1]
    #     return list[Gpu_ID2]
    #
    def get_gpu_sort_nearest(self, Gpu_ID):
        # Gpu_ID int
        # returns
        return self.adjacency_matrix[Gpu_ID].argsort()  # first one is the node itself
    
    # def return_adj_matrx(self):
    #     return self.adjacency_matrix
    #
    # def compute_GPU_distances(self, job):
    #     switchtime = 0
    #     jobGpusIDlist = []
    #     WIDTH = 4  # (machine per rack)
    #     DEPTH = 4  # (gpu per machine)
    #     HEIGHT = 4
    #     distancefromothers = 0
    #     # print('job.gpus[0]): ',job.gpus)
    #     if job.gpus != []:
    #         for cordin in range(len(job.gpus[0])):  # find the selected GPUs' IDs
    #             x_ = job.gpus[0][cordin - 1]
    #             y_ = job.gpus[1][cordin - 1]
    #             z_ = job.gpus[2][cordin - 1]
    #             gpu_Id = x_ * HEIGHT * DEPTH + y_ * DEPTH + z_  # get the first gpu number dims found randomly in(x,y,z)
    #             # print('gpu_Id:',gpu_Id,x_,y_,z_)
    #             jobGpusIDlist.append(gpu_Id)
    #
    #         if (len(jobGpusIDlist) > 1):
    #             for gpu in jobGpusIDlist:
    #                 distancefromothers += self.get_gpu_distance_gpu(gpu, jobGpusIDlist[0])
    #
    #         return distancefromothers
