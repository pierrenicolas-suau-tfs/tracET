import os
import numpy as np
import sklearn.neighbors
from sklearn.neighbors import NearestNeighbors
import torch.nn.functional as F
import torch
from skimage.morphology import skeletonize_3d
import networkx as nx
import pandas as pd

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
def add_coords_to_graph(G,coords):
    for i,node in enumerate(G.nodes()):
        G.nodes[node]['Coords'] = coords[i]
def cal_edges_weights(G,coords):
    add_coords_to_graph(G,coords)
    for edge in enumerate(G.edges()):
        G[edge[1][0]][edge[1][1]]['weight']= np.linalg.norm(G.nodes[edge[1][0]]['Coords']-G.nodes[edge[1][1]]['Coords'])
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
    add_coords_to_graph(graph,node_coordinates)
    components = list(nx.connected_components(graph))

    component_matrices = []
    component_coordinates = []
    for component in components:
        subgraph = graph.subgraph(component)
        component_matrix = nx.to_scipy_sparse_array(subgraph)
        component_matrices.append(component_matrix)
        #component_nodes = list(component)
        component_coords = [subgraph.nodes[node]['Coords'] for node in subgraph]
        #component_coords = node_coordinates[component_nodes]
        component_coordinates.append(component_coords)

    return component_matrices,component_coordinates

#Remove cycles
def spannig_tree_apply(sparse_graph,coords):
    """
    Compute the minimum spanning tree:, the smallest acyclic subgraph that conect all the nodes
    :param sparse_graph: Sparse matrix with the graph information
    :return: Sparse matrix with the acyclic graph information
    """
    graph = nx.from_scipy_sparse_array(sparse_graph)
    cal_edges_weights(graph,coords)
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

def angle_3points(Point1,Point2,Point3):
    """

    :param Point1:
    :param Point2:
    :param Point3:
    :return:
    """
    vec12=Point2-Point1
    vec32=Point2-Point3
    mod_12=np.linalg.norm(vec12)
    mod_32=np.linalg.norm(vec32)
    scalar=np.dot(vec12,vec32)
    angle = np.arccos(scalar/mod_12/mod_32)
    if np.isnan(angle) == False:
        return angle
    else:
        return 0


def add_duplicate_node(G, index):
    atriv=G.nodes[index]
    new_id=max(G.nodes)+1
    G.add_node(new_id, **atriv)
    return new_id
def count_elements(array,subsets):
    return [next((i for i, subset in enumerate(subsets) if num in subset), -1) for num in array]
def label_branches(sparse_graph,coords):
    graph = nx.from_scipy_sparse_array(sparse_graph)
    add_coords_to_graph(graph,coords)
    nodes=graph.nodes()
    id_nodes=list(nodes)
    for i in range(len(id_nodes)):
        nodes_neig=list(graph.neighbors(id_nodes[i]))
        if len(nodes_neig)>2:
            if i == 0:
                continue
            elif i== len(id_nodes):
                continue
            else:
                node1=nodes[i-1]['Coords']
                node2=nodes[i]['Coords']
                angles=[]
                #del nodes_neig[id_nodes[i-1]]
                for j in range(len(nodes_neig)):
                    angles.append(angle_3points(node1,node2,nodes[nodes_neig[j]]['Coords']))
                angles=np.array(angles)
                cont_index=np.where(angles==max(angles))[0][0]

                del nodes_neig[cont_index]
                for j in range(len(nodes_neig)):
                    graph.remove_edge(i,nodes_neig[j])
                    new_node=add_duplicate_node(graph,i)
                    graph.add_edge(nodes_neig[j],new_node)

    branches = nx.connected_components(graph)
    branches_matrices = []

    for branch in branches:
        nodes_mat=[]
        for node in branch:
            nodes_mat.append(node)
        branches_matrices.append(nodes_mat)
        #component_nodes = list(component)
    L_branches=count_elements(nodes,branches_matrices)
    new_coords = [graph.nodes[node]['Coords'] for node in graph]
    new_sparse_graph = nx.to_scipy_sparse_array(graph)
    return new_sparse_graph, np.array(new_coords), np.array(L_branches)

def label_branches2(sparse_graph,coords):
    graph = nx.from_scipy_sparse_array(sparse_graph)
    add_coords_to_graph(graph,coords)
    nodes=graph.nodes()
    id_nodes=list(nodes)
    for i in range(len(id_nodes)):
        nodes_neig=list(graph.neighbors(id_nodes[i]))
        if len(nodes_neig)>2:
            if i == 0:
                continue
            elif i== len(id_nodes):
                continue
            else:

                node=nodes[i]['Coords']
                angles=np.zeros((len(nodes_neig),len(nodes_neig)))
                #del nodes_neig[id_nodes[i-1]]
                for j in range(len(nodes_neig)):
                    for k in range(len(nodes_neig)):
                        angles[j,k]=angle_3points(nodes[nodes_neig[j]]['Coords'],node,nodes[nodes_neig[k]]['Coords'])
                angles=np.triu(np.array(angles))
                cont_index=np.unravel_index(np.argmax(angles), angles.shape)

                del nodes_neig[cont_index[0]]
                del nodes_neig[cont_index[1]-1]

                for j in range(len(nodes_neig)):
                    graph.remove_edge(i,nodes_neig[j])
                    new_node=add_duplicate_node(graph,i)
                    graph.add_edge(nodes_neig[j],new_node)

    branches = nx.connected_components(graph)
    branches_matrices = []

    for branch in branches:
        nodes_mat=[]
        for node in branch:
            nodes_mat.append(node)
        branches_matrices.append(nodes_mat)
        #component_nodes = list(component)
    L_branches=count_elements(nodes,branches_matrices)
    new_coords = [graph.nodes[node]['Coords'] for node in graph]
    new_sparse_graph = nx.to_scipy_sparse_array(graph)
    return new_sparse_graph, np.array(new_coords), np.array(L_branches)


def sort_branches(branch_graph, branch_coord):
    graph = nx.from_scipy_sparse_array(branch_graph)
    add_coords_to_graph(graph,branch_coord)
    extrems = terminal_nodes(graph)
    order= nx.shortest_path(graph,source=extrems[0],target=extrems[1])

    sorted_coords=[graph.nodes[node]['Coords']for node in order]
    return np.array(sorted_coords)

def only_long_path(graph, coords):
    graph = nx.from_scipy_sparse_array(graph)
    cal_edges_weights(graph, coords)
    length = pd.DataFrame(dict(nx.all_pairs_dijkstra_path_length(graph)))
    extremes = [min(length.idxmax(axis=0)),max(length.idxmax(axis=0))]
    path = nx.shortest_path(graph,source = extremes[0], target = extremes[1], weight = 'weight')
    #clean_graph = graph.subgraph(path)
    clean_graph = nx.Graph()
    for i in path:
        clean_graph.add_node(i)
    for i in range(len(path)-1):
        clean_graph.add_edge(path[i],path[i+1])
    clean_coords = [graph.nodes[node]['Coords']for node in path]
    claen_sparse_graph = nx.to_scipy_sparse_array(clean_graph)
    return claen_sparse_graph, clean_coords
