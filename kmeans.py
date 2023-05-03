import ipdb
import torch
from torch import nn
from torch.nn import functional as F
import torch_scatter
import os
import numpy as np


def calculate_prototypes_from_labels(embeddings,
                                     labels,
                                     num_protos=None):
    """Calculates prototypes from labels.
    This function calculates prototypes (mean direction) from embedding
    features for each label. This function is also used as the m-step in
    k-means clustering.
    Args:
        embeddings: A 2-D or 4-D float tensor with feature embedding in the
        last dimension (embedding_dim).
        labels: An N-D long label map for each embedding pixel.
        max_label: The maximum value of the label map. Calculated on-the-fly
        if not specified.
    Returns:
        A 2-D float tensor with shape `[num_prototypes, embedding_dim]`.
    """
    if num_protos is None:
        num_protos = labels.max() + 1
    # prototypes = torch.zeros((num_protos, embeddings.shape[-1]),
    #                         dtype=embeddings.dtype,
    #                         device=embeddings.device)

    prototypes = torch_scatter.scatter(embeddings, labels, dim=0, reduce='mean')
    # prototypes.scatter_(0, labels, embeddings, reduce='mean')
    return prototypes

def find_nearest_prototypes(embeddings, prototypes):
    """Finds the nearest prototype for each embedding.
    Args:
        embeddings: An N-D float tensor with embedding features in the last
        dimension (N, embedding_dim).
        prototypes: A 2-D float tensor with shape
        `[num_prototypes, embedding_dim]`.
    Returns:
        A 1-D long tensor with length `[N]` containing the index
        of the nearest prototype for each pixel.
    """
    
    dist = (embeddings[:, None] - prototypes[None]).pow(2).sum(-1)
    labels = dist.argmin(1)
    return labels

def kmeans_with_initial_prototypes(embeddings, prototypes, iterations=10, num_protos=3, return_proto=False):
    """Performs the k-means clustering with initial
    labels.
    Args:
        embeddings: A 2-D float tensor with shape
        `[num_pixels, embedding_dim]`.
        initial_labels: A 1-D long tensor with length [num_pixels].
        K-means clustering will start with this cluster labels if
        provided.
        max_label: An integer for the maximum of labels.
        iterations: Number of iterations for the k-means clustering.
    Returns:
        A 1-D long tensor of the cluster label for each pixel.
    """

    for i in range(iterations):
        # E-step of the k-means clustering.
        labels = find_nearest_prototypes(embeddings, prototypes)

        # M-step of the k-means clustering.
        prototypes = calculate_prototypes_from_labels(embeddings, labels, num_protos)
        
        if False and i % 5 == 0:
            print("Finished kMeans iter", i)
            print(labels)

    if return_proto:
        return labels, prototypes
    return labels

def kmeans(embeddings, iterations=10, num_protos=3, return_proto=False):
    """Performs the k-means clustering with initial
    labels.
    Args:
        embeddings: A 2-D float tensor with shape
        `[num_pixels, embedding_dim]`.
        initial_labels: A 1-D long tensor with length [num_pixels].
        K-means clustering will start with this cluster labels if
        provided.
        max_label: An integer for the maximum of labels.
        iterations: Number of iterations for the k-means clustering.
    Returns:
        A 1-D long tensor of the cluster label for each pixel.
    """

    labels = torch.randint(num_protos, (embeddings.shape[0], )).to(embeddings.device)
    for i in range(iterations):

        # M-step of the k-means clustering.
        prototypes = calculate_prototypes_from_labels(embeddings, labels, num_protos)
        
        # E-step of the k-means clustering.
        labels = find_nearest_prototypes(embeddings, prototypes)
        if False and i % 5 == 0:
            print("Finished kMeans iter", i)
            print(labels)

    if return_proto:
        return labels, prototypes
    return labels
