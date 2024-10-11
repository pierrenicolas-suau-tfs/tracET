import vtk
import numpy as np
# Converts an iterable of points into a poly
# points: iterable with 3D points coordinates
# normals: iterable with coordinates for the normals
# n_name: name for the normal (default 'normal')
def points_to_poly(points, normals=None, n_name='n_normal'):
    """
    Transform a cloud point into a polydata
    Args:
        points: iterable with 3D points coordinates
        normals: iterable with coordinates for the normals
        n_name: name for the normal (default 'normal')

    Returns:
        Polydata with the points.

    """
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
    """
    Giving coordinates and conexions of the points of a graph, it creates a polydata
    Args:
        coords: numpy array with coordinates
        source: numpy array with the indices of the vertex that are origin of an edge
        target: numpy array with the indices of the vertex that are target of an edge

    Returns: polydata with the graph connections

    """
    g = vtk.vtkMutableUndirectedGraph()
    points = vtk.vtkPoints()
    num_points=np.shape(coords)[0]
    indices=np.arange(num_points)
    num_edges=len(source)
    for i in range(num_points):
        indices[i]=g.AddVertex()
        points.InsertNextPoint(coords[i,0],coords[i,1],coords[i,2])
    g.SetPoints(points)
    for i in range(num_edges):
        g.AddEdge(source[i],target[i])
    polyGraph=vtk.vtkGraphToPolyData()
    polyGraph.SetInputData(g)
    polyGraph.Update()
    output = polyGraph.GetOutputPort()
    producer = output.GetProducer()
    poly= producer.GetOutput()
    return(poly)

def add_atributte_to_poly(poly, array, att_name):
    vtk_scalar_values = vtk.vtkFloatArray()
    vtk_scalar_values.SetNumberOfValues(len(array))
    vtk_scalar_values.SetArray(array, len(array), 1)

    poly.GetPointData().AddArray(vtk_scalar_values)
    poly.GetPointData().GetArray(poly.GetPointData().GetNumberOfArrays() - 1).SetName(att_name)

def merge_polys(poly_1, poly_2):
    """
    Merges two input poly_data in single one
    :param poly_1: input poly_data 1
    :param poly_2: input poly_data 2
    :return: an poly_data that merges the two inputs
    """
    assert isinstance(poly_1, vtk.vtkPolyData) and isinstance(poly_2, vtk.vtkPolyData)
    app_flt = vtk.vtkAppendPolyData()
    app_flt.AddInputData(poly_1)
    app_flt.AddInputData(poly_2)
    app_flt.Update()
    return app_flt.GetOutput()


def add_label_to_poly(poly, lbl, p_name):
    """
    Add a label to all cells in a poly_data
    :param poly: input poly_data
    :param lbl: label (integer) value
    :p_name: property name used for labels, if not exist in poly_dota is created
    """
    assert isinstance(poly, vtk.vtkPolyData)
    lbl, p_name = int(lbl), str(p_name)
    arr = vtk.vtkIntArray()
    n_cells = poly.GetNumberOfCells()
    arr.SetName(p_name)
    arr.SetNumberOfComponents(1)
    arr.SetNumberOfValues(n_cells)
    for i in range(n_cells):
        arr.SetValue(i, lbl)
    poly.GetCellData().AddArray(arr)
def add_labels_to_poly(poly, lbl, p_name):
    """
    Add a label to all cells in a poly_data
    :param poly: input poly_data
    :param lbl: array of label (integer) values
    :p_name: property name used for labels, if not exist in poly_dota is created
    """
    assert isinstance(poly, vtk.vtkPolyData)
    assert isinstance(lbl,np.ndarray)
    lbl, p_name = lbl, str(p_name)
    arr = vtk.vtkIntArray()
    n_points = poly.GetNumberOfPoints()
    arr.SetName(p_name)
    arr.SetNumberOfComponents(1)
    arr.SetNumberOfValues(n_points)
    for i in range(n_points):
        arr.SetValue(i, int(lbl[i]))
    poly.GetPointData().AddArray(arr)

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