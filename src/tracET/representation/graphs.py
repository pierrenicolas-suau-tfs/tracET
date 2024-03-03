import os
import numpy as np
import sklearn.neighbors
from sklearn.neighbors import NearestNeighbors
import torch.nn.functional as F
import torch
from skimage.morphology import skeletonize_3d
import networkx as nx

#Create subsample and skeletonize a graph
def make_graph(T:np.ndarray,r:float)->tuple:
    """
    From an array with a point cloud, create a graph, conecting points if they are in a distance less than r
    :param T: numpy 3D array with point cloud (1 where is a detection in that position, 0 if not)
    :param r: radium (in pixel) maximum distance between points to have a conection
    :return: coords: numpy array (N,3) with N number of points. It contains the coordinates of the points
    :return: graph_array: sparse matrix with the connections
    """
    coords=np.argwhere(T)
    graph_array = sklearn.neighbors.radius_neighbors_graph(coords, r)
    return coords,graph_array

def subsample_pcloud(points: list, dist: float) -> list:
    """
    Subsample a point cloud by fixing a minimum distance
    :param points: list of point (coordinates as 1d array)
    :param dist: minimum distance
    :return:  subsampled points list
    """
    dist_2=dist*dist
    points_out=list()
    points_queue= points.copy()
    while len(points_queue):
        point = np.asarray(points_queue.pop(0))
        points_queue_arr=np.asarray(points_queue)
        if len(points_queue_arr)>0:
            points_queue= list(points_queue_arr[((point - np.asarray(points_queue))**2).sum(axis=1)>=dist_2])
            points_out.append(point)
    return points_out


def make_skeleton_graph(T:np.ndarray,r:float,subsample=0)->tuple:
    """
    From an array with a point cloud, create a graph, conecting points if they are in a distance less than r, applying
    first an skeletonization and a subsampling of points closer than another giving distance
    :param T: numpy 3D array with point cloud (1 where is a detection in that position, 0 if not)
    :param r: radium (in pixel) maximum distance between points to have a conection
    :param subsample (optional): If is not 0 (default), eliminate the points closer than the distance you put
    :return: coords: numpy array (N,3) with N number of points. It contains the coordinates of the points
    :return: graph_array: sparse matrix with the connections
    """
    skeleton =skeletonize_3d(T)
    coords=np.argwhere(skeleton)
    if subsample!=0:
        coords=np.array(subsample_pcloud(coords.tolist(),subsample))
    graph_array = sklearn.neighbors.radius_neighbors_graph(coords, r)
    return coords, graph_array

##Split in components
def split_into_components(sparse_matrix,node_coordinates):
    """
    Divide a graph (gived by a sparse matrix) by its connect components
    :param sparse_matrix: a graph as a sparse matrix
    :param node_coordinates: np array with the coordinates of the nodes
    :return: component matrices: list of sparse matrix with connect component
    :return: component coordinates: list of numpy arrays with coordinates of nodes of connect components
    """
    graph = nx.from_scipy_sparse_array(sparse_matrix)
    components = list(nx.connected_components(graph))

    component_matrices = []
    component_coordinates = []
    for component in components:
        subgraph = graph.subgraph(component)
        component_matrix = nx.to_scipy_sparse_array(subgraph)
        component_matrices.append(component_matrix)
        component_nodes = list(component)
        component_coords = node_coordinates[component_nodes]
        component_coordinates.append(component_coords)

    return component_matrices,component_coordinates

#Remove cycles
def spannig_tree_apply(sparse_graph):
    """
    Compute the minimum spanning tree:, the smallest acyclic subgraph that conect all the nodes
    :param sparse_graph: Sparse matrix with the graph information
    :return: Sparse matrix with the acyclic graph information
    """
    graph = nx.from_scipy_sparse_array(sparse_graph)
    graph_no_cycles=nx.minimum_spanning_tree(graph)
    return nx.to_scipy_sparse_array(graph_no_cycles)

#remove branches
def terminal_nodes(graph):
    """

    :param graph: NetworkX graph
    :return: list of the terminal nodes (Nodes with only one adyacent edge)
    """
    return list(filter(lambda node: graph.degree(node) == 1, graph.nodes()))


def remove_branches(sparse_graph,node_coordinates):
    """

    :param sparse_graph: Sparse matrix with an acyclic graph
    :param node_coordinates: numpy array with the coordinates of the nodes.
    :return: sparse matrix with the graph without smaller subbranches
    :return: numpy array with the coordinates of the nodes of the new graph.
    """
    graph = nx.from_scipy_sparse_array(sparse_graph)
    clean_graph=graph
    endpoints = terminal_nodes(clean_graph)
    path_coords = node_coordinates
    if len(endpoints)>2:
        clean_graph.remove_nodes_from(endpoints[1:len(endpoints) - 1])
        path_coords=np.delete(path_coords,endpoints[1:len(endpoints)-1],axis=0)
    sparse_clean_graph=(nx.to_scipy_sparse_array(clean_graph))
    return sparse_clean_graph,path_coords
