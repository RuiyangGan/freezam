# A function library for the construction of a falconn table
# that does approximate nearest neighbor search by multi-probe
# locality sensitive hashing. The following code is heavily borrowed
# from the GloVe example on the github page of FALCONN-LIB. The link
# is attached below:
# https://github.com/FALCONN-LIB/FALCONN/blob/master/src/examples/glove/glove.py

import falconn
import numpy as np
import copy


def falconn_table(sig_mat):
    ''' Construct a falconn table with given signature. Return
    a falconn table and the random seed used (for random rotation)
    to construct the falconn table.

    Keyword Argument:

    sig_mat -- A numpy ndarray, where each row is signature at a time
    window center
    '''

    # pre-processing the signature matrix
    # coerce the ndarray into 32-bit floating number

    if sig_mat.dtype != np.float32:
        sig_mat = sig_mat.astype(np.float32)

    # Normalize and center the signature matrix so that
    # the observations are on a unit hypersphere
    sig_mat /= max(1e-6, max(np.linalg.norm(sig_mat, axis=1).reshape(-1, 1)))
    center = np.mean(sig_mat, axis=0)
    sig_mat -= center

    # Instantiate the parameters for the falconn table
    params_cp = falconn.LSHConstructionParameters()
    params_cp.dimension = len(sig_mat[0])
    # Set the LSH family to be Cross Polytope
    params_cp.lsh_family = falconn.LSHFamily.CrossPolytope
    # Set the distance function to be the L2_norm
    # which is the cosine distance
    params_cp.distance_function = falconn.DistanceFunction.EuclideanSquared
    # # Set the randomly-picked seed for table construction
    # params_cp.seed = cp_seed
    # Set the number of random rotation, since the signature is very likely
    # a large sparse matrix
    params_cp.num_rotations = 2
    # select the number of hash tables
    params_cp.l = 50
    params_cp.seed = 5721840
    # Set the thread usage (0 for using all) and storage formats of the
    # falconn table
    params_cp.num_setup_threads = 0
    params_cp.storage_hash_table = (falconn.StorageHashTable.
                                    BitPackedFlatHashTable)

    # select the number of hash functions according the size
    # of the signature matrix
    num_obs = sig_mat.shape[0]
    bit_num = int(np.log2(num_obs))
    falconn.compute_number_of_hash_functions(bit_num, params_cp)

    # Construct falconn table with configured parameters
    falconn_tab = falconn.LSHIndex(params_cp)
    falconn_tab.setup(sig_mat)

    return falconn_tab


def falconn_que(falconn_tab, sig_mat, acy_level=.95,
                auto_probe=True):
    ''' Construct a falconn queryable object with the given falconn table
    and desired accuracy level

    Keyword Argument:

    falconn_tab -- A given falconn table to search for the nearest neighbor
    of the query points

    sig_mat -- The dataset where each row specifies a signature
    at a time window

    acy_level -- The desired accruacy level of the nearest neighbor search

    auto_probe -- A boolean variable indicating whether to search for
    the number of probes in multiprobe hashing
    '''
    # Construct the queryable from the falconn table just created
    falconn_queryable = falconn_tab.construct_query_object()
    # Set the number of probes of the queryable according to the desired
    # level of accuracy
    if auto_probe and sig_mat.shape[0] >= 200:
        num_of_probes = probe_num_search(sig_mat, acy_level)
        falconn_queryable.set_num_probes(num_of_probes)

    # Return the falconn queryable object
    return [falconn_queryable, acy_level]


def probe_num_search(sig_mat, acy_level):
    '''This function searches the number of probe
    given the desired accraucy level

    Keyword Argument:

    acy_level -- Minimum accuracy level of the approximate
    nearest neighbor search

    sig_mat -- The signature matrix that is used to construct the table
    '''
    # randomly select a portion of observations of the points to be
    # sample queries
    dataset = copy.deepcopy(sig_mat)
    if dataset.dtype != np.float32:
        dataset = dataset.astype(np.float32)
    dataset /= max(1e-6, max(np.linalg.norm(dataset, axis=1).reshape(-1, 1)))
    center = np.mean(dataset, axis=0)
    dataset -= center
    number_of_queries = int(dataset.shape[0]*(1-acy_level))
    np.random.shuffle(dataset)
    queries = dataset[len(dataset) - number_of_queries:]
    dataset = dataset[:len(dataset) - number_of_queries]

    # Use linear scan to search for the nearest neighbor of the sample
    # queries
    answers = []
    for query in queries:
        answers.append(np.dot(dataset, query).argmax())

    # Construc a falconn table based on the randomly splitted
    # dataset and a corresponding queryable
    table = falconn_table(dataset)
    queryable = table.construct_query_object()

    # Find the number of probes used to ensure the desired accuracy level
    # Set the default number of probes, which equals to the numbers of
    # hash table used in the
    number_of_tables = table._params.l
    number_of_probes = number_of_tables
    while True:
        accuracy = evaluate_number_of_probes(answers, queries,
                                             queryable,
                                             number_of_probes)
        # If desired level of accuracy is achieved, then break
        # out the while loop
        if accuracy >= acy_level:
            break
        # If the desired accuracy level is not achieved, double
        # the size of the number of probes used in searching
        number_of_probes = number_of_probes * 2

    if number_of_probes > number_of_tables:
        # Search for the smallest number of probes that ensure
        # the desired accuracy level
        left = number_of_probes // 2
        right = number_of_probes
        while right - left > 1:
            number_of_probes = (left + right) // 2
            accuracy = evaluate_number_of_probes(answers, queries,
                                                 queryable,
                                                 number_of_probes)
            if accuracy >= acy_level:
                right = number_of_probes
            else:
                left = number_of_probes
        number_of_probes = right
    return number_of_probes


def evaluate_number_of_probes(answers, queries, queryable, number_of_probes):
    '''This function evaluates accuracy of queries in the given queryables
    given the number of probes used

    Keyword Argument:

    answers -- The answer of the sample queryable

    queries -- The sample queries

    queryable -- The queryable object

    number_of_probes -- The number of probes used in search
    '''
    queryable.set_num_probes(number_of_probes)
    score = 0
    for (i, query) in enumerate(queries):
        if answers[i] in queryable.get_candidates_with_duplicates(query):
            score += 1
    return float(score) / len(queries)


def falconn_search(query_pts, falconn_queryable, k=10):
    ''' Find the k approximate nearest neighbor to the nearest point
    in the data set by using the falconn table. The return will be a list
    of index for the n.

    Keyword Argument:

    query_pts -- The given point(s) for nearest neighbor searching.

    falconn_queryable -- The given falconn_queryable

    k -- The number of nearest neighbor to be found in the data set
    '''
    # Convert the query points to float 32 if they are not already
    if query_pts.dtype != np.float32:
        query_pts = query_pts.astype(np.float32)

    # Normalize and Center the query points so that they have zero mean
    # and unit varaince
    query_pts /= max(1e-6,
                     max(np.linalg.norm(query_pts, axis=1).reshape(-1, 1)))
    center = np.mean(query_pts, axis=0)
    query_pts -= center

    num_obs = query_pts.shape[0]
    knn_idx = []
    # Perform knn search for each observations
    for query_pt in query_pts:
        knn_idx.append(falconn_queryable.find_k_nearest_neighbors(query_pt, k))
    # return the k nearest neighbors of given query points
    return knn_idx
