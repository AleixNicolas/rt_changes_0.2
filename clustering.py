### This Python script contains the functions used for clustering

import numpy as np

def agglomerative_clustering(y, method='ward', alpha=1, K=None, verbose=0, algorithm="generic"):

    """
    Perform hierarchical, agglomerative clustering on given
    condensed distance matrix y. Adapted (as well as the
    auxiliary functions, excepting compute_polarisation)
    from Scipy's 'linkage' method [1] and the main reference
    therein [2].

    Parameters
    ----------
    y : ndarray
        A condensed matrix containing the pairwise distances of
        the observations.
    method : str, optional
        The distance update scheme: "ward" (the default), "centroid"
        or "poldist".
    alpha : double, optional
        Value of the `polarisation sensitivity` parameter [3], only
        used for computing polarisation and, if method="poldist",
        distance update. Default is alpha=1.
    K : double, optional
        Normalisation factor used for computing the polarisation [3].
        If none is specified, K = (2/n)**(2+alpha)/2 (the inverse of
        the maximum polarisation for a given population
        with max(y) = 1).
    verbose : int, optional
        Level of information display upon execution: 0 for no
        information, 1 for some on the last stages of the algorithm.
        Default is 0.
    algorithm : str, optional
        Algorithm to use for clustering, either "nn_chain" (faster,
        but only works with method "ward") or
        "generic" (slower, but works with any distance update scheme).
        Default is "generic".

    Returns
    -------
    Z : ndarray, shape (n - 1, 4)
        Computed linkage matrix: a stepwise dendogram with a fourth
        column containing the size of the newly formed cluster at
        each step.
    pol : ndarray, shape (n - 1,)
        Polarisation at each step, from the first (point clusters)
         to the next-to-last (2 clusters) configuration.

    References
    ----------
    .. [1] https://github.com/scipy/scipy/blob/v1.7.0/scipy/cluster/hierarchy.py

    .. [2] Daniel Mullner, "Modern hierarchical, agglomerative
            clustering algorithms", :arXiv:`1109.2378v1`.

    .. [3]  Esteban, J., & Ray, D. (1994). "On the Measurement of
            Polarization". Econometrica, 62(4), 819- 851. doi:10.2307/2951734
    """
    #y = _convert_to_double(np.asarray(y, order='c')) # to normalise input, I guess

    if y.ndim != 1:
        raise ValueError("`y` must be 1-dimensional.")

    if not np.all(np.isfinite(y)):
        raise ValueError("The condensed distance matrix must contain only "
                         "finite values.")
    L = y.shape[0]
    n = int(round((1+np.sqrt(1+8*L))/2))

    if algorithm == "nn_chain":
        if method in ["centroid", "poldist"]:
            raise ValueError("The nn_chain algorithm cannot be "
                             "used with the centroid nor poldist "
                             "methods. Use algorithm='generic' instead.")
        Z, pol = nn_chain(y, n, method, alpha, K, verbose)

    elif algorithm == "generic":
        Z, pol = generic_clustering(y, n, method, alpha, K, verbose)

    return Z, pol

def generic_clustering(dists, n, method, alpha, K, verbose):
    """
    Adapted from Scipy's 'fast_linkage' method [1] and the
    main reference therein [2].

    Performs hierarchical, agglomerative clustering with the
    "Generic Clustering Algorithm" from [2].

    The worst case time complexity is O(n^3). The original algorithm
    has a best case time complexity of O(n^2) and it usually works
    quite close to that, but adding the computation of the polarisation at 
    every step makes the complexity O(n^3) in any case.

    Parameters
    ----------
    dists : ndarray
        A condensed matrix containing the pairwise distances of the
        observations.
    n : int
        The number of observations.
    method : str
        The linkage method: "centroid", "ward" or "poldist".
    alpha : double
        Value of the `polarisation sensitivity` parameter [3], only
        used for computing polarisation and, if method="poldist",
        distance update.
    K : double
        Normalisation factor used for computing the polarisation [3].
        If K=None, K is set to (2/n)**(2+alpha)/2 (the inverse of
        the maximum polarisation for the given population
        with max(dists) = 1).
    verbose : int
        Level of information display upon execution: 0 for no
        information, 1 for some on the last stages of the algorithm.

    Returns
    -------
    Z : ndarray, shape (n - 1, 4)
        Computed linkage matrix.
    pol : ndarray, shape (n - 1,)
        Polarisation at each step, from the first (point clusters)
         to the next-to-last (2 clusters) configuration.

    References
    ----------
    .. [1] https://github.com/scipy/scipy/blob/v1.7.1/scipy/cluster/_hierarchy.pyx

    .. [2] Daniel Mullner, "Modern hierarchical, agglomerative
            clustering algorithms", :arXiv:`1109.2378v1`.

    .. [3]  Esteban, J., & Ray, D. (1994). "On the Measurement of
            Polarization". Econometrica, 62(4), 819- 851. doi:10.2307/2951734
    """
    Z = np.empty((n - 1, 4))

    D = dists.copy()  # Distances between clusters.
    size = np.ones(n, dtype=np.intc)  # Sizes of clusters.
    # ID of a cluster to put into linkage matrix.
    cluster_id = np.arange(n, dtype=np.intc)
    pol = np.empty(n-1) # polarisation at each level
    # by default, set K to 1/max(pol) (for given population and with max dist = 1)
    if K == None:
        K = (2/n)**(2+alpha)/2

    #if poldist method, add normalisation constant and size factor 
    #to initial centroid distances
    if method == 'poldist':
        D *= K*2
    
    # Nearest neighbor candidate and lower bound of the distance to the
    # true nearest neighbor for each cluster among clusters with higher
    # indices (thus size is n - 1).
    neighbor = np.empty(n - 1, dtype=np.intc)
    min_dist = np.empty(n - 1)

    for x in range(n - 1):
        pair = find_min_dist(n, D, size, x)
        neighbor[x] = pair[0]#pair.key
        min_dist[x] = pair[1]#pair.value
    min_dist_heap = Heap(min_dist)

    for k in range(n - 1):
        # Theoretically speaking, this can be implemented as "while True", but
        # having a fixed size loop when floating point computations involved
        # looks more reliable. The idea that we should find the two closest
        # clusters in no more that n - k (1 for the last iteration) distance
        # updates.
        if (k < n-3) | ((verbose == 0) & (k < n-2)):
            print(f"Iteration {k}/{n-2}...", end='\r')
        else:
            print(f"Iteration {k}/{n-2}...")
        #compute polarisation
        pol[k] = compute_polarisation(D, size, method, alpha, K, n, k, verbose)

        for i in range(n - k):
            x, dist = min_dist_heap.get_min()
            y = neighbor[x]

            if dist == D[condensed_index(n, x, y)]:
                break
            #J-> recompute NN and update heap
            y, dist = find_min_dist(n, D, size, x)
            neighbor[x] = y
            min_dist[x] = dist
            min_dist_heap.change_value(x, dist)
        min_dist_heap.remove_min()

        id_x = cluster_id[x]
        id_y = cluster_id[y]
        nx = size[x]
        ny = size[y]

        if id_x > id_y:
            id_x, id_y = id_y, id_x

        Z[k, 0] = id_x
        Z[k, 1] = id_y
        Z[k, 2] = dist
        Z[k, 3] = nx + ny

        size[x] = 0  # Cluster x will be dropped.
        size[y] = nx + ny  # Cluster y will be replaced with the new cluster.
        cluster_id[y] = n + k  # Update ID of y.

        # Update the distance matrix.
        for z in range(n):
            nz = size[z]
            if nz == 0 or z == y:
                continue

            D[condensed_index(n, z, y)] = distance_update(#new_dist
                D[condensed_index(n, z, x)], D[condensed_index(n, z, y)],
                dist, nx, ny, nz, method, alpha)

        # Reassign neighbor candidates from x to y.
        # This reassignment is just a (logical) guess.
        for z in range(x):
            if size[z] > 0 and neighbor[z] == x:
                neighbor[z] = y

        # Update lower bounds of distance.
        for z in range(y):
            if size[z] == 0:
                continue

            dist = D[condensed_index(n, z, y)]
            if dist < min_dist[z]:
                neighbor[z] = y
                min_dist[z] = dist
                min_dist_heap.change_value(z, dist)

        # Find nearest neighbor for y.
        if y < n - 1:
            z, dist = find_min_dist(n, D, size, y)
            if z != -1:
                neighbor[y] = z
                min_dist[y] = dist
                min_dist_heap.change_value(y, dist)

    return Z, pol
#J-> Z#.base: Base object if memory is from some other object-> had to remove it
#to get the actual Z

def nn_chain(dists, n, method, alpha, K, verbose):
    """
    Adapted from Scipy's 'nn_chain' method [1] and the
    main reference therein [2].

    Performs hierarchy clustering using nearest-neighbor chain
    algorithm reviewed in [2].

    The worst case time complexity would be O(n^2) for the original 
    nn_chain, but adding the computation of the polarisation at every step 
    makes it O(n^3).

    Parameters
    ----------
    dists : ndarray
        A condensed matrix containing the pairwise distances of the
        observations.
    n : int
        The number of observations.
    method : str
        The linkage method: "centroid", "ward" or "poldist".
    alpha : double
        Value of the `polarisation sensitivity` parameter [3], only
        used for computing polarisation and, if method="poldist",
        distance update.
    K : double
        Normalisation factor used for computing the polarisation [3].
        If K=None, K is set to (2/n)**(2+alpha)/2 (the inverse of
        the maximum polarisation for the given population
        with max(dists) = 1).
    verbose : int
        Level of information display upon execution: 0 for no
        information, 1 for some on the last stages of the algorithm.

    Returns
    -------
    Z : ndarray, shape (n - 1, 4)
        Computed linkage matrix.
    pol : ndarray, shape (n - 1,)
        Polarisation at each step, from the first (point clusters)
         to the next-to-last (2 clusters) configuration.

    References
    ----------
    .. [1] https://github.com/scipy/scipy/blob/v1.7.1/scipy/cluster/_hierarchy.pyx

    .. [2] Daniel Mullner, "Modern hierarchical, agglomerative
            clustering algorithms", :arXiv:`1109.2378v1`.

    .. [3]  Esteban, J., & Ray, D. (1994). "On the Measurement of
            Polarization". Econometrica, 62(4), 819- 851. doi:10.2307/2951734
    """
    ### DEFINE QUANTITIES
    Z_arr = np.empty((n - 1, 4))
    Z = Z_arr # this is just to rename the Z_arr: they are synchronised

    D = dists.copy()  # Distances between clusters.

    # by default, set K to 1/max(pol) (for given population and with max dist = 1)
    if K == None:
        K = (2/n)**(2+alpha)/2
    
    #if poldist method, add normalisation constant and size factor 
    #to initial centroid distances
    if method == 'poldist':
        D *= K*2
        
    size = np.ones(n, dtype=np.intc)  # Sizes of clusters.
    pol = np.empty(n-1) # polarisation at each level

    # Variables to store neighbors chain.
    cluster_chain = np.ndarray(n, dtype=np.intc)
    chain_length = 0

    ### ALGORITHM
    for k in range(n - 1):
        if (k < n-3) | ((verbose == 0) & (k < n-2)):
            print(f"Iteration {k}/{n-2}...", end='\r')
        else:
            print(f"Iteration {k}/{n-2}...") 
        #compute polarisation
        pol[k] = compute_polarisation(D, size, method, alpha, K, n, k, verbose)

        #If chain emplty, pick one existing cluster (the first in size list)
        if chain_length == 0:
            chain_length = 1
            for i in range(n):
                if size[i] > 0:
                    cluster_chain[0] = i
                    break

        # Go through chain of neighbors until two mutual neighbors are found.
        while True:  #THIS IS EXITTED WITH THE BREAK BELOW
            x = cluster_chain[chain_length - 1]

            #CHECK FIRST DIST TO PREVIOUS CLUSTER IN CHAIN
            # We want to prefer the previous element in the chain as the
            # minimum, to avoid potentially going in cycles.
            if chain_length > 1:
                y = cluster_chain[chain_length - 2]
                current_min = D[condensed_index(n, x, y)]
            else:
                current_min = np.inf

            #FOR EVERY OTHER CLUSTER; CHECK DIST AND KEEP THE MIN
            for i in range(n):
                #SKIP DIST TO ITSELF AND TO ALREADY MERGED CLUSTERS
                if size[i] == 0 or x == i:
                    continue

                dist = D[condensed_index(n, x, i)]
                if dist < current_min:
                    current_min = dist
                    y = i

            #IF THE MIN DIST CLUSTER IS ALREADY IN CHAIN, IT IS THE PREVIOUS ELEMENT
            #AND WE HAVE MUTUAL MIN DIST, SO BREAK FROM WHILE AND MERGE CLUSTERS
            if chain_length > 1 and y == cluster_chain[chain_length - 2]:
                break
                
            #OTHERWISE; JUST CONTINUE STACKING CLUSTERS IN CHAIN
            cluster_chain[chain_length] = y
            chain_length += 1

        # Merge clusters x and y and pop them from stack.
        chain_length -= 2

        # This is a convention used in fastcluster.
        if x > y:
            x, y = y, x

        # get the original numbers of points in clusters x and y
        nx = size[x]
        ny = size[y]

        # Record the new node. Updating Z updates Z_arr as well
        Z[k, 0] = x
        Z[k, 1] = y
        Z[k, 2] = current_min
        Z[k, 3] = nx + ny
        size[x] = 0  # Cluster x will be dropped.
        size[y] = nx + ny  # Cluster y will be replaced with the new cluster

        # Update the distance matrix.
        for i in range(n):
            ni = size[i]
            if ni == 0 or i == y:
                continue
            D[condensed_index(n, i, y)] = distance_update(
                    D[condensed_index(n, i, x)],
                    D[condensed_index(n, i, y)],
                    current_min, nx, ny, ni, method, alpha)

    # Sort Z and pol by cluster distances (the nn_chain algorithm does not produce that
    #order in general)
    order = np.argsort(Z_arr[:, 2], kind='mergesort')#an (n-1)x1 index array
    Z_arr = Z_arr[order]# orders rows of Z according to "order"
    pol = pol[order] #same for polarisation vector

    # Find correct cluster labels inplace (up to now, the new clusters had no label):
    # we label them acording to their order of appearance in the newly sorted Z
    label(Z_arr, n)

    return Z_arr, pol


def compute_polarisation(D, size, method, alpha, K, n, k, verbose):
    """
    Compute system polarisation [1] at step k of clustering.

    Parameters
    ----------
    D : ndarray
        A condensed matrix containing the pairwise distances of
        the observations.
    size : ndarray
        Array containing the current (step k) cluster sizes
    method : str
        The linkage method: "centroid", "ward" or "poldist".
    alpha : double
        Value of the `polarisation sensitivity` parameter [1].
    K : double
        Normalisation factor used for computing the polarisation [1].
    n : int
        The number of observations.
    k : int
        The clustering step.
    verbose : int
        Level of information display: 0 for no information,
        1 for some on the last stages of the clustering.

    Returns
    -------
    p : double
        Polarisation at step k.

    References
    ----------
    .. [1]  Esteban, J., & Ray, D. (1994). "On the Measurement of
            Polarization". Econometrica, 62(4), 819- 851. doi:10.2307/2951734
    """
    p = 0
    for i in range(n):
        ni = size[i]
        if ni == 0:
            continue
        for j in range(i+1, n):
            nj = size[j]
            if nj == 0: #or j == i:
                continue
            if method == "centroid":
                p += K * (ni ** (1 + alpha) * nj
                          + nj ** (1 + alpha) * ni) * D[condensed_index(n, i, j)]
            elif method == "ward":
                coef = np.sqrt((ni + nj)/(2*ni*nj))
                p += K * (ni ** (1 + alpha) * nj
                          + nj ** (1 + alpha) * ni) * coef * D[condensed_index(n, i, j)]
            elif method == "poldist":
                p += D[condensed_index(n, i, j)]
            if (k >= n - 3) & (verbose > 0):
                print( f"Cluster {i} (size {ni}) and cluster {j} (size {nj}) "
                    f"have dist {round(D[condensed_index(n, i, j)], 2)}")
    if (k >= n - 3) & (verbose > 0):
        print(f"and total polarisation is {(round(p, 2))}")
    return p


def distance_update(d_xi, d_yi, d_xy, size_x, size_y, size_i, method, alpha):
    """
    Adapted from Scipy function of the same name in [1].
    Calculates the distance from cluster i to the new
    cluster xy after merging cluster x and cluster y.
    
    Parameters
    ----------
    d_xi : double
        Distance from cluster x to cluster i
    d_yi : double
        Distance from cluster y to cluster i
    d_xy : double
        Distance from cluster x to cluster y
    size_x : int
        Size of cluster x
    size_y : int
        Size of cluster y
    size_i : int
        Size of cluster i
    method : str
        The linkage method: "centroid", "ward" or "poldist".
    alpha : double
        Value of the `polarisation sensitivity` parameter [2],
        only used if method="poldist".

    Returns
    -------
    d_xyi : double
        Distance from the new cluster xy to cluster i

    References
    ----------
    .. [1]  https://github.com/scipy/scipy/blob/v1.7.0/scipy/cluster/_hierarchy_distance_update.pxi
    .. [2]  Esteban, J., & Ray, D. (1994). "On the Measurement of
            Polarization". Econometrica, 62(4), 819- 851. doi:10.2307/2951734
    """
    if method=="centroid":
        return np.sqrt((((size_x * d_xi * d_xi) + (size_y * d_yi * d_yi)) -
                     (size_x * size_y * d_xy * d_xy) / (size_x + size_y)) /
                    (size_x + size_y))

    elif method=="ward":
        t = 1.0 / (size_x + size_y + size_i)
        return np.sqrt((size_i + size_x) * t * d_xi * d_xi +
                    (size_i + size_y) * t * d_yi * d_yi -
                    size_i * t * d_xy * d_xy)

    elif method=="poldist":
        #centroid dist from new cluster xy to i
        d_xi /= (size_i*size_x**(1+alpha)+size_x*size_i**(1+alpha))
        d_yi /= (size_i*size_y**(1+alpha)+size_y*size_i**(1+alpha))
        d_xy /= (size_y*size_x**(1+alpha)+size_x*size_y**(1+alpha))

        d_c_xyi = np.sqrt((((size_x * d_xi * d_xi) + (size_y * d_yi * d_yi)) -
                     (size_x * size_y * d_xy * d_xy) / (size_x + size_y)) /
                    (size_x + size_y))
        #pol dist from new cluster xy to i
        return d_c_xyi*(size_i*(size_x+size_y)**(1+alpha)+
                       (size_x+size_y)*size_i**(1+alpha))

def condensed_index(n, i, j):
    """
    Adapted from Scipy function of the same name in [1].

    Calculate the condensed index of element (i, j) in an n x n condensed
    matrix.

    References
    ----------
    .. [1]  https://github.com/scipy/scipy/blob/v1.7.0/scipy/cluster/_hierarchy.pyx

    """
    if i < j:
        return int(round(n * i - (i * (i + 1) / 2) + (j - i - 1)))
    elif i > j:
        return int(round(n * j - (j * (j + 1) / 2) + (i - j - 1)))
    
### The following class and function are just for relabeling
# clusters in the linkage matrix Z produced by the nn_chain
# algorithm
class LinkageUnionFind:
    """
    Adapted from Scipy class of the same name in [1].

    Structure for fast cluster labeling in unsorted dendrogram
    (used in nn_chain algorithm).

    References
    ----------
    .. [1]  https://github.com/scipy/scipy/blob/v1.7.0/scipy/cluster/_hierarchy.pyx

    """

    def __init__(self, n):
        self.parent = np.arange(2 * n - 1, dtype=np.intc)
        self.next_label = n
        self.size = np.ones(2 * n - 1, dtype=np.intc)

    def merge(self, x, y):
        self.parent[x] = self.next_label
        self.parent[y] = self.next_label
        size = self.size[x] + self.size[y]
        self.size[self.next_label] = size
        self.next_label += 1
        return size

    def find(self, x):
        p = x

        while self.parent[x] != x:
            x = self.parent[x]

        while self.parent[p] != x:
            p, self.parent[p] = self.parent[p], x

        return x
    
def label(Z, n):
    """
    Adapted from Scipy function of the same name in [1].

    Correctly label clusters in unsorted dendrogram
    (used in nn_chain algorithm).

    References
    ----------
    .. [1]  https://github.com/scipy/scipy/blob/v1.7.0/scipy/cluster/_hierarchy.pyx

    """
    uf = LinkageUnionFind(n)
    
    for i in range(n - 1):
        x, y = int(Z[i, 0]), int(Z[i, 1])
        x_root, y_root = uf.find(x), uf.find(y)
        if x_root < y_root:
            Z[i, 0], Z[i, 1] = x_root, y_root
        else:
            Z[i, 0], Z[i, 1] = y_root, x_root
        # weird: without this, new clusters are not correctly labelled
        #even if sizes are already assigned in nn__chain
        Z[i, 3] = uf.merge(x_root, y_root)

#only for generic_clustering
def find_min_dist(n, D, size, x):
    """
    Adapted from Scipy function of the same name in [1].

    Find the minimal distance and the corresponding nearest
    neighbor candidate y to cluster x among the clusters with
    higher  index (i > x), D being the condensed distance matrix
    of the set of n clusters, whose sizes are stored in "size".
    Used in generic_clustering algorithm.

    References
    ----------
    .. [1]  https://github.com/scipy/scipy/blob/v1.7.0/scipy/cluster/_hierarchy.pyx

    """
    current_min = np.Inf
    y = -1
    for i in range(x + 1, n):
        if size[i] == 0:
            continue
        dist = D[condensed_index(n, x, i)]
        if dist < current_min:
            current_min = dist
            y = i
    return y, current_min


#only for generic_clustering
class Heap:
    """
    Adapted from Scipy class of the same name in [1].

    Binary heap.
    Heap stores values and keys. Values are passed explicitly, whereas keys
    are assigned implicitly to natural numbers (from 0 to n - 1).
    The supported operations (all have O(log n) time complexity):
        * Return the current minimum value and the corresponding key.
        * Remove the current minimum value.
        * Change the value of the given key. Note that the key must be still
          in the heap.
    The heap is stored as an array, where children of parent i have indices
    2 * i + 1 and 2 * i + 2. All public methods are based on  `sift_down` and
    `sift_up` methods, which restore the heap property by moving an element
    down or up in the heap.

    References
    ----------
    .. [1]  https://github.com/scipy/scipy/blob/v1.7.1/scipy/cluster/_structures.pxi

    """
    
    def __init__(self, values):
        self.size = values.shape[0]
        self.index_by_key = np.arange(self.size, dtype=np.intc)
        self.key_by_index = np.arange(self.size, dtype=np.intc)
        self.values = values.copy()

        # Create the heap in a linear time. The algorithm sequentially sifts
        # down items starting from lower levels.
        #J-> introduced round coz float gave error
        for i in reversed(range(round(self.size / 2))):
            self.sift_down(i)

    def get_min(self):
        return self.key_by_index[0], self.values[0]

    def remove_min(self):
        self.swap(0, self.size - 1)
        self.size -= 1
        self.sift_down(0)

    def change_value(self, key, value):
        index = self.index_by_key[key]
        old_value = self.values[index]
        self.values[index] = value
        if value < old_value:
            self.sift_up(index)
        else:
            self.sift_down(index)

    def sift_up(self, index):
        parent = Heap.parent(index)
        while index > 0 and self.values[parent] > self.values[index]:
            self.swap(index, parent)
            index = parent
            parent = Heap.parent(index)

    def sift_down(self, index):
        child = Heap.left_child(index)
        while child < self.size:
            if (child + 1 < self.size and
                    self.values[child + 1] < self.values[child]):
                child += 1

            if self.values[index] > self.values[child]:
                self.swap(index, child)
                index = child
                child = Heap.left_child(index)
            else:
                break

    @staticmethod
    def left_child(parent):
        return (parent << 1) + 1

    @staticmethod
    def parent(child):
        return (child - 1) >> 1

    def swap(self, i, j):
        self.values[i], self.values[j] = self.values[j], self.values[i]
        key_i = self.key_by_index[i]
        key_j = self.key_by_index[j]
        self.key_by_index[i] = key_j
        self.key_by_index[j] = key_i
        self.index_by_key[key_i] = j
        self.index_by_key[key_j] = i