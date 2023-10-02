import mrcfile
import vtk
import os
from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkCommonDataModel import vtkMutableUndirectedGraph
from vtkmodules.vtkFiltersSources import vtkGraphToPolyData
import numpy as np
import matplotlib.pyplot as ptl
import sklearn.neighbors
from sklearn.neighbors import NearestNeighbors
import torch.nn.functional as F
import torch
from skimage.morphology import skeletonize_3d
import networkx as nx



def load_mrc(fname, mmap=False, swapxz=True):
    """
    Load an input MRC tomogram as ndarray
    :param fname: the input MRC
    :param mmap: if True (default) load a numpy memory map instead of an ndarray
    :param swapxz: if True then X and Z axis are swapped for compatibility with IMOD tomograms.
                   Don't use it with mmaps
    :return: a ndarray or a memmap if mmap is True
    """
    if mmap:
        mrc = mrcfile.mmap(fname, permissive=True)
    else:
        mrc = mrcfile.open(fname, permissive=True)
    if swapxz:
        return np.swapaxes(mrc.data, 0, 2)
    else:
        return np.swapaxes(mrc.data, 0, 2)


def write_mrc(tomo, fname, v_size=1, dtype=None, swapxz=True):
    """
    Saves a tomo (3D dataset) as MRC file
    :param tomo: tomo to save as ndarray
    :param fname: output file path
    :param v_size: voxel size (default 1)
    :param dtype: data type (default None, then the dtype of tomo is considered)
    :param swapxz: if True then X and Z axis are swapped for compatibility with IMOD tomograms.
                   Don't use it with mmaps
    :return: a numpy memory map is mmap is True, otherwise None
    """
    with mrcfile.new(fname, overwrite=True) as mrc:
        if swapxz:
            tomo = np.swapaxes(tomo, 0, 2)
        if dtype is None:
            mrc.set_data(tomo)
        else:
            mrc.set_data(tomo.astype(dtype))
        mrc.voxel_size.flags.writeable = True
        mrc.voxel_size = (v_size, v_size, v_size)
        mrc.set_volume()
        # mrc.header.ispg = 401
def make_graph(T,r):
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
    skeleton =skeletonize_3d(T)
    coords=np.argwhere(skeleton)
    if subsample!=0:
        coords=np.array(subsample_pcloud(coords.tolist(),subsample))
    graph_array = sklearn.neighbors.radius_neighbors_graph(coords, r)
    return ([coords, graph_array])


# Converts an iterable of points into a poly
# points: iterable with 3D points coordinates
# normals: iterable with coordinates for the normals
# n_name: name for the normal (default 'normal')
def points_to_poly(points, normals=None, n_name='n_normal'):
    poly = vtk.vtkPolyData()
    p_points = vtk.vtkPoints()
    p_cells = vtk.vtkCellArray()

    if normals is not None:
        p_norm = vtk.vtkFloatArray()
        p_norm.SetName(n_name)
        p_norm.SetNumberOfComponents(3)
        for i, point, normal in zip(list(range(len(points))), points, normals):
            p_points.InsertNextPoint(point)
            p_cells.InsertNextCell(1)
            p_cells.InsertCellPoint(i)
            p_norm.InsertTuple(i, normal)
    else:
        for i, point in enumerate(points):
            p_points.InsertNextPoint(point)
            p_cells.InsertNextCell(1)
            p_cells.InsertCellPoint(i)
    poly.SetPoints(p_points)
    poly.SetVerts(p_cells)
    if normals is not None:
        poly.GetPointData().AddArray(p_norm)

    return poly

def make_graph_polydata(coords,source,target):
    g = vtkMutableUndirectedGraph()
    points = vtkPoints()
    num_points=np.shape(coords)[0]
    indices=np.arange(num_points)
    num_edges=len(source)
    for i in range(num_points):
        indices[i]=g.AddVertex()
        points.InsertNextPoint(coords[i,0],coords[i,1],coords[i,2])
    g.SetPoints(points)
    for i in range(num_edges):
        g.AddEdge(source[i],target[i])
    polyGraph=vtkGraphToPolyData()
    polyGraph.SetInputData(g)
    polyGraph.Update()
    output = polyGraph.GetOutputPort()
    producer = output.GetProducer()
    poly= producer.GetOutput()
    return(poly)

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

def save_vtp(poly, fname):
    """
    Store data vtkPolyData as a .vtp file
    :param poly: input vtkPolyData to store
    :param fname: output path file
    :return:
    """

    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(fname)
    writer.SetInputData(poly)
    if writer.Write() != 1:
        raise IOError


def save_vti(image, fname):
    """
    Store data vtkPolyData as a .vti file
    :param image: input image as numpy array
    :param fname: output path file
    :return:
    """

    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName(fname)
    writer.SetInputData(image)
    if writer.Write() != 1:
        raise IOError


def load_poly(fname):
    """
    Load data vtkPolyData object from a file
    :param fname: input .vtp file
    :return: the vtkPolyData object loaded
    """

    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(fname)
    reader.Update()

    return reader.GetOutput()


def split_into_components(sparse_matrix,node_coordinates):
    """

    :param sparse_matrix:
    :param node_coordinates:
    :return:
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

def calculate_list_paths(G):
    """

    :param G:
    :return:
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

    :param lists:
    :return:
    """
    sorted_lists=sorted(lists,key=len,reverse=True)
    max_lists=[]
    for list in sorted_lists:
        if not any(all(index in max_list for index in list) for max_list in max_lists):
            max_lists.append(list)
    return(max_lists)

def found_edges(cycle):
    """

    :param cycle:
    :return:
    """
    edges=[]
    for i in range(len(cycle)-1):
        edges.append(cycle[i:i+2])
    return(edges)

def count_paths_per_edges(paths,edge):
    """

    :param paths:
    :param edge:
    :return:
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

    :param sparse_graph:
    :return:
    """
    graph=nx.from_scipy_sparse_array(sparse_graph)
    cycles=list(nx.simple_cycles(graph))
    max_cycles=maximal_lists(cycles)


    paths=calculate_list_paths(graph)


    for cycle in max_cycles:

        while (cycle in cycles) == True:
            edges=found_edges(cycle)
            trips=[]
            c_edges=[]
            for edge in edges:
                trips.append(count_paths_per_edges(paths,edge))
                c_edges.append(edge)
            edge_index=np.where(np.array(trips)==min(trips))[0][0]

            edge_to_remove = c_edges[int(edge_index)]
            graph.remove_edge(edge_to_remove[0],edge_to_remove[1])
            cycle=cycle.remove(edge_to_remove[1])



    L_sparse=nx.to_scipy_sparse_array(graph)
    return(L_sparse)