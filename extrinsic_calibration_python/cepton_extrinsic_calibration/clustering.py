###################################################################
##                                                               ##
##  Copyright(C) 2021 Cepton Technologies. All Rights Reserved.  ##
##  Contact: https://www.cepton.com                              ##
##                                                               ##
###################################################################

import numpy as np
from sklearn.cluster import KMeans
from scipy import spatial

def clustering_kmeans(point_cloud: np.ndarray, num_cluster: int, 
               initial_centers: np.ndarray):
  """
  Global clustering. Given the number of target, the function will assign a 
    target label for each point.
  Input:
    point_cloud [ndarray]: input point cloud, after data preprocessing. Array
      of size (n, 4). The last column, intensity, won't be useful in this 
      function.
    num_cluster [int]: number of target in the field of view.
    initial_centers [ndarray]: a good initial cluster center can save some time
      Array of size (num_cluster, 3).
  Return:
    labels [ndarray]: point-wise label, array of size (n, ).
    cluster_centers [ndarray]: cluster centers. May not be useful.
  """

  kmeans = KMeans(n_clusters=num_cluster, random_state=0)
  kmeans.cluster_centers_ = initial_centers
  kmeans.fit(point_cloud[:, :3])
  return kmeans.labels_, kmeans.cluster_centers_

def clustering_connectivity(point_cloud: np.ndarray, num_clusters, radius = 0.1, 
                            min_sample = 30, downsample=25000):
  """
  Use BFS and KDTree to expand the cluster based on connectivity (density).
  This method will also filter some outliers.
  Input:
    point_cloud [ndarray]: Input point cloud.
    num_clusters [int]: Number of clusters to be detected, the largest n clusters 
      will be returned.
    radius [float]: Neighbor searching radius.
    min_sample [int]: Minimum number of neighbors to be considered as connected.
    downsample [int]: Choose a subset of points to do the clustering.
  Return:
    labels [ndarray]: Point-wise label, array of size (n, ).
    cluster_centers [ndarray]: Cluster centers. May not be useful.
    valid_clusters [ndarray]: An array of cluster index, sorted by cluster size
      in descending order.
  """

  rand_index = np.random.choice(len(point_cloud), downsample, replace=False)

  point_cloud_sample = point_cloud[rand_index]

  # Use x, y, z in the KDTree
  tree = spatial.KDTree(point_cloud_sample[:, :3])
  # labels can be -1, 0, 1, ...
  # -1 means outlier, 0 means not visited, >=1 means the point is in cluster 
  labels = np.zeros(len(point_cloud_sample))

  curr_label = 0

  for i in range(len(labels)):
    if not labels[i] == 0:
      # if visited
      continue
    else:
      neighbors = np.array([i])
      curr_label += 1
      while not len(neighbors) == 0:
        # pop from the queue
        curr_idx = neighbors[0]
        neighbors = np.delete(neighbors, 0)
        if not labels[curr_idx] == 0:
          # if visited
          continue
        # query neighbors
        expand_neighbors = tree.query_ball_point(point_cloud_sample[curr_idx, :3], radius, workers=-1)
        if len(expand_neighbors) < min_sample:
          # too sparse to be considered as connected
          labels[curr_idx] = -1
          continue 
        else:
          # if the point is connected, label it and push all its neighbors to the queue
          labels[curr_idx] = curr_label
          neighbors = np.concatenate([neighbors, expand_neighbors])
          neighbors = np.unique(neighbors)
          neighbors = neighbors[labels[neighbors] == 0]
        

  # TODO: alert when the number of points in the cluster is less than a threshold
  count = []
  # count each points in each cluster
  for cluster_idx in range(0, int(labels.max())+1):
    count.append((labels==cluster_idx).sum())
  
  # clusters with the most points are considered as the true clusters
  valid_clusters = np.argsort(count)[::-1][:num_clusters]

  # outliers will be discarded
  labeled_point = np.concatenate([
                                    point_cloud_sample[np.isin(labels, valid_clusters), :], 
                                    labels[np.isin(labels, valid_clusters), None]
                                  ], 1)

  # build KDTree with labeled point cloud
  tree = spatial.KDTree(labeled_point[:, :3])

  # label all the points based on nearest neighbor
  dist, match_idx = tree.query(point_cloud[:, :3], workers=-1)
  labels_point_cloud = np.where(dist > 0.1, -1, labeled_point[match_idx, 4])

  cluster_centers = []
  for cluster_idx in valid_clusters:
    cluster_centers.append(np.mean(point_cloud[labels_point_cloud==cluster_idx, :3], axis=0))
  cluster_centers = np.array(cluster_centers)

  return labels_point_cloud, cluster_centers, valid_clusters