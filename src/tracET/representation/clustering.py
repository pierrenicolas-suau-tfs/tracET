import numpy as np
from sklearn.cluster import MeanShift, AffinityPropagation
from src.tracET.core.vtk_uts import *

def get_coords_from_pc(T):
    """

    :param T:
    :return:
    """
    coords_ids = np.where(T == 1)
    coords = np.zeros((len(coords_ids[0]), 3))
    coords[:, 0] = coords_ids[0]
    coords[:, 1] = coords_ids[1]
    coords[:, 2] = coords_ids[2]
    return coords

def get_MS_cluster(T,blob_size,n_jobs):
    """
    Cluster and label the blobs and calculate the centroid.

    :param T: Input tomogram (A 3D numpy array)
    :param blob_size: Size of the blob in pixel (int)
    :param n_jobs: Number of jobs to calculate the clustering (int)
    :return:
    labels : Labels of which cluster is each positive point
    centers : Coordinates of the centers of the blobs
    out_T: tomogram with the points and it labels
    """

    coords = get_coords_from_pc(T)

    clusters = MeanShift(bandwidth=blob_size,n_jobs=n_jobs)
    clusters.fit(coords)
    labels = clusters.labels_
    labels = np.array(labels) + 1
    centers = clusters.cluster_centers_
    out_T = np.zeros(np.shape(T))
    out_T[(coords[:, 0]).astype(np.int32), (coords[:, 1]).astype(np.int32), (coords[:, 2]).astype(
        np.int32)] = labels.astype(np.int32)
    out_poly = points_to_poly(coords)
    add_labels_to_poly(out_poly,labels,'blob')
    return labels, centers, out_T, out_poly

def get_AF_cluster(T):
    """

    :param T:
    :return:
    """
    coords = get_coords_from_pc(T)
    clusters= AffinityPropagation(damping = 0.9)
    clusters.fit(coords)
    labels = clusters.labels_
    labels = np.array(labels) + 1
    centers = clusters.cluster_centers_
    out_T = np.zeros(np.shape(T))
    out_T[(coords[:, 0]).astype(np.int32), (coords[:, 1]).astype(np.int32), (coords[:, 2]).astype(
        np.int32)] = labels.astype(np.int32)
    out_poly = points_to_poly(coords)
    add_labels_to_poly(out_poly, labels, 'blob')
    return labels, centers, out_T, out_poly