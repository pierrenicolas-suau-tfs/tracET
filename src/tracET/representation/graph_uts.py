import os
import numpy as np
import sklearn.neighbors
from sklearn.neighbors import NearestNeighbors
import torch.nn.functional as F
import torch
from skimage.morphology import skeletonize_3d
import networkx as nx



#Create subsample and skeletonize a graph
def make_graph(T,r):
    """
    From an array with a point cloud, create a graph, conecting points if they are in a distance less than r
    :param T: numpy 3D array with point cloud (1 where is a detection in that position, 0 if not)
    :param r: radium (in pixel) maximum distance between points to have a conection
    :return: coords: numpy array (N,3) with N number of points. It contains the coordinates of the points
    :return: graph_array: sparse matrix with the connections  
    """
    coords=np.argwhere(T)
    graph_array = sklearn.neighbors.radius_neighbors_graph(coords, r)
    return([coords,graph_array])

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


def make_skeleton_graph(T,r,subsample=0):
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
    return ([coords, graph_array])

# Make a soft skeleton from a graph
def soft_erode(img):
    p1 = -F.max_pool3d(-img, (3, 1, 1), (1, 1, 1), (1, 0, 0))
    p2 = -F.max_pool3d(-img, (1, 3, 1), (1, 1, 1), (0 , 1, 0))
    p3 = -F.max_pool3d(-img, (1, 1, 3), (1, 1, 1), (0 , 0, 1))
    return (torch.min(torch.min(p1, p2), p3))

def soft_dilate(img):
    return (F.max_pool3d(img, (3, 3, 3), (1, 1, 1), (1, 1, 1)))

def soft_open(img):
    return (soft_dilate (soft_erode(img)))

def soft_skel(img, iter):
    img1 = soft_open(img)
    skel = F.relu(img-img1)
    print('bucle')
    for j in range (iter):
        img = soft_erode(img)
        img1 = soft_open(img)
        delta = F.relu (img-img1)
        #funciona=((delta-skel * delta).squeeze(0).numpy()>0).any()

        #skel = skel + F.relu(delta-skel * delta)
        skel = skel +(1-skel)*delta

        print(j)
    return (skel)

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

    return [component_matrices,component_coordinates]

#Eliminate cycles
def calculate_list_paths(G):
    """
    From a networkx graph, create a list with the shortest path between every pair of nodes.
    :param G: network x graph
    :return: paths: list with the sortest path between every pair of nodes
    """
    paths_dics = list(dict(nx.all_pairs_shortest_path(G)).values())
    paths=[]
    for paths_dic in paths_dics:
        paths_lists=list(paths_dic.values())
        for path in paths_lists:
            paths.append(path)
    return(paths)


def maximal_lists(lists):
    """
    from a list of list, eliminate the lists that are in others
    :param lists: list of list
    :return: max list: reduced list of list
    """
    sorted_lists=sorted(lists,key=len,reverse=True)
    max_lists=[]
    for list in sorted_lists:
        if not any(all(index in max_list for index in list) for max_list in max_lists):
            max_lists.append(list)
    return(max_lists)

def found_edges(cycle):
    """
    from a cycle (nx simple cycle format) found it edges
    :param cycle: nx simple cycle
    :return:edges: list of edges
    """
    edges=list(filter(lambda x: len(x) == 2, cycle))
    return(edges)

def count_paths_per_edges(paths,edge):
    """
    given a list of paths, counts how many of them contains a given edge
    :param paths: list of paths
    :param edge: a network x edge
    :return: count: number of paths that cross through edge
    """
    count=0
    for path in paths:
        for i in range(len(path)-1):
            if (path[i] ==edge[0] and path [i+1] == edge[1]) or (path[i] ==edge[1] and path [i+1] == edge[0]):
                count +=1
                break
    return (count)

def remove_cycles(sparse_graph):
    """
    Given a connected array, remove the cycles
    :param sparse_graph: connected array as a sparse matrix
    :return: L_sparse: sparse matrix with the array without cycles.
    """
    graph=nx.from_scipy_sparse_array(sparse_graph)
    cycles=list(nx.simple_cycles(graph))
    print('cycles calculated')



    paths=calculate_list_paths(graph)
    print('paths calculated')



    k=0
    while cycles !=[]:

        print('Enter in the bucle')
        for cycle in cycles:
            edges=found_edges(cycle)

            trips=[]
            c_edges=[]
            for edge in edges:
                trips.append(count_paths_per_edges(paths,edge))
                c_edges.append(edge)
            edge_index=np.where(np.array(trips)==min(trips))[0][0]

            edge_to_remove = c_edges[int(edge_index)]

            try:
                graph.remove_edge(edge_to_remove[0],edge_to_remove[1])
            except nx.exception.NetworkXError:
                continue
            #cycle=cycle.remove(edge_to_remove[1])
        cycles = list(nx.simple_cycles(graph))
        k = k + 1
        print(k)
        if k > 10000:
            print("It has to much cycles")

            break



    L_sparse=nx.to_scipy_sparse_array(graph)
    return(L_sparse)

def spannig_tree_apply(sparse_graph):
    graph = nx.from_scipy_sparse_array(sparse_graph)
    graph_no_cycles=nx.minimum_spanning_tree(graph)
    return(nx.to_scipy_sparse_array(graph_no_cycles))

def terminal_nodes(graph):
    return(list(filter(lambda node: graph.degree(node) == 1, graph.nodes())))

def mapping_change(graph):
    mapping={node: i+1 for i, node in enumerate(sorted(graph.nodes()))}
    return(mapping)
def remove_branches(sparse_graph,node_coordinates):
    graph = nx.from_scipy_sparse_array(sparse_graph)
    clean_graph=graph

    endpoints = terminal_nodes(clean_graph)
    path_coords = node_coordinates


    if len(endpoints)>2:
    #    if i>0:
    #        mapping = mapping_change(clean_graph)
    #        list_points = [mapping[endpoint] for endpoint in endpoints]
    #        clean_graph.remove_nodes_from(endpoints[1:len(endpoints) - 1])

    #        path_coords = np.delete(path_coords, list_points[1:len(list_points) - 1], axis=0)
     #   else:
        clean_graph.remove_nodes_from(endpoints[1:len(endpoints) - 1])
        path_coords=np.delete(path_coords,endpoints[1:len(endpoints)-1],axis=0)

        #endpoints=terminal_nodes(clean_graph)
        #i=i+1
    #long_path=nx.shortest_path(graph,endpoints[0],endpoints[len(endpoints)-1])
    #clean_graph=graph.subgraph(long_path)


    sparse_clean_graph=(nx.to_scipy_sparse_array(clean_graph))
    #path_nodes = list(sparse_clean_graph)
    #path_coords = node_coordinates[long_path]
    return(sparse_clean_graph,path_coords)
