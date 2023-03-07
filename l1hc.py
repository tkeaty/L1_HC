from networkx import DiGraph, is_directed_acyclic_graph, topological_sort, shortest_path_length
from sklearn.preprocessing import StandardScaler
from mmpc import *
from l1mb import *
from math import fabs
from time import time

# ALL FUNCTIONS ASSUME DATA IS STANDARD SCALED
# PLEASE PERFORM STANDARD SCALING BEFORE USING.


def add_edge(edge_list, p, c, mode=1):
    """
    Adds an edge to the given list
    :param edge_list: a dictionary of format {p: (c, mode)}
    :param p: identifier of the parent (source) node of the new edge
    :param c: identifier of the child (sink) node of the new edge
    :param mode: 1 = add, 0 = delete, -1 = reverse
    :return: the new edge list
    """
    if p in edge_list:
        edge_list[p].append((c, mode))
    else:
        edge_list[p] = [(c, mode)]

    return edge_list


def compute_single_gaussian(x, mean):
    """
    This function assumes var=1
    :param x:
    :param mean:
    :return:
    """
    # Compute the constant
    const = 1/np.sqrt(2.0*np.pi)

    # Compute the exponential
    x_term = -0.5*((x - mean)**2)

    # Multiply the two terms together
    prob = const*np.exp(x_term)

    return prob


def compute_log_gaussian_prob(mean, var, data):
    """
    Compute the log likelihood of the given univariate Gaussian data
    :param mean: sample mean for the feature
    :param var: sample variance for the feature
    :param data: Nx1 column vector of the observations for one feature
    :return: the log likelihood of the data
    """
    log_prob = 0
    warnings.filterwarnings('error')
    for d in data.values:
        try:
            log_prob += np.log(compute_single_gaussian(d, mean))
        except Warning:
            print('x: %f, mean: %f' % (d, mean))
            log_prob += np.log(10 ** -30)

    return log_prob


def compute_log_gaussian_prob2(y_hat, y_true):
    """
    Compute the log likelihood of the observations given mean=y_hat, obs=y_true (used with product of linear regression)
    :param y_hat: Nx1 column vector corresponding to beta.T * X linear regression output
    :param y_true: Nx1 column vector corresponding to the observations of a given feature
    :return: the log likelihood of the data
    """
    log_prob = 0
    std = np.std(y_hat)

    warnings.filterwarnings('error')
    for h, t in zip(y_hat, y_true):
        try:
            log_prob += np.log(compute_single_gaussian(t, h))
        except Warning:
            print('x: %f, mean: %f' % (t, h))
            log_prob += np.log(10**-30)

    return log_prob


def compute_log_linear_prob2(x_vec, y, data):
    """
    Compute the least-squares estimate of beta, for y=X*beta
    :param x_vec: indices of the covariate features for y
    :param y: index of feature y
    :param data: pandas DataFrame
    :return: the log likelihood of the regression model P(y|X) = N(beta*x, var)
    """
    x_mat = data.iloc[:, x_vec].values
    y_vals = data.iloc[:, y].values

    if len(x_mat.shape) > 1:
        x_mat = np.hstack((np.ones(x_mat.shape[0]).reshape(-1, 1), x_mat))
    else:
        x_mat = np.hstack((np.ones(x_mat.shape[0]).reshape(-1, 1), x_mat.reshape(-1, 1)))

    try:
        w_vec = np.linalg.inv(x_mat.T @ x_mat) @ x_mat.T @ y_vals
    except np.linalg.LinAlgError:
        w_vec = np.linalg.pinv(x_mat.T @ x_mat) @ x_mat.T @ y_vals

    y_hat = x_mat @ w_vec
    log_p = compute_log_gaussian_prob2(y_hat.flatten(), y_vals.flatten())

    return log_p


def compute_mdl2(data, cand_g, c):
    """
    Compute the change in minimum description length
    :param data: pandas DataFrame
    :param cand_g: the new, augmented candidate graph
    :param c: the newly added child (sink) node
    :return: the change in log likelihood given the new graph structure
    """
    prev_prob = cand_g.nodes[c]['ll']
    u_vec = []

    for u in cand_g.predecessors(c):
        u_vec.append(u)

    if len(u_vec) == 0:
        mean = np.mean(data.iloc[:, c].values)
        var = np.std(data.iloc[:, c].values)
        new_prob = compute_log_gaussian_prob(mean, var, data.iloc[:, c])
    else:
        new_prob = compute_log_linear_prob2(u_vec, c, data)

    return new_prob - (2+len(u_vec)) * np.log2(data.shape[0])/2 - prev_prob


def augment_graph2(g, p, child):
    """
    Augment graph g given the parent, child, and mode of augmentation
    :param g: a DAG object
    :param p: index of the parent (source) node
    :param child: a 2-tuple of (child node index, mode of augmentation)
    :return: the augmented graph
    """
    c, mode = child
    new_g = g.copy()

    if mode == 1:
        new_g.add_edge(p, c)
    elif mode == 0:
        new_g.remove_edge(p, c)
    else:
        new_g.remove_edge(p, c)
        new_g.add_edge(c, p)

    return new_g


def augment_data(data, data_hat, g, c):
    """
    Update the model estimates to reflect the current graph structure
    :param data: The original DataFrame
    :param data_hat: DataFrame containing the model estimates for each node
    :param g: DiGraph object
    :param c: The node which has been modified
    :return: The updated data_hat
    """
    u_vec = []

    for u in g.predecessors(c):
        u_vec.append(u)

    if len(u_vec) == 0:
        data_hat.iloc[:, c] = data.iloc[:, c]
    else:
        x_mat = data.iloc[:, u_vec].values
        y_vals = data.iloc[:, c].values

        if len(x_mat.shape) > 1:
            x_mat = np.hstack((np.ones(x_mat.shape[0]).reshape(-1, 1), x_mat))
        else:
            x_mat = np.hstack((np.ones(x_mat.shape[0]).reshape(-1, 1), x_mat.reshape(-1, 1)))

        try:
            w_vec = np.linalg.inv(x_mat.T @ x_mat) @ x_mat.T @ y_vals
        except np.linalg.LinAlgError:
            w_vec = np.linalg.pinv(x_mat.T @ x_mat) @ x_mat.T @ y_vals

        y_hat = x_mat @ w_vec

        data_hat.iloc[:, c] = y_hat

    return data_hat


def select_edge2(data, g, candidate_edges):
    """
    Selects the candidate edge resulting in the greatest increase in log likelihood
    :param data: pandas DataFrame
    :param g: a DAG object
    :param candidate_edges: a dictionary of format {p: (c, mode)}
    :return:
        new_p: parent node
        new_c: child node
        new_m: mode of augmentation
        best_ll: the most positive change in log likelihood over all candidate edges
        prune_set: a list of edges decreasing the log likelihood
    """
    log_likes = []
    ll_map = []
    prune_set = {}

    for p, children in candidate_edges.items():
        # Add cand edge to list, then compute LL
        for child_mode in children:
            try:
                c, mode = child_mode
            except TypeError:
                print('TypeError')
                print(p)
                print(child_mode)
            cand_g = augment_graph2(g, p, (c, mode))

            if is_directed_acyclic_graph(cand_g):
                ll = compute_mdl2(data, cand_g, c)
                if ll < 0:
                    prune_set = add_edge(prune_set, p, c, mode)

                log_likes.append(ll)
                ll_map.append((p, c, mode))

            else:
                prune_set = add_edge(prune_set, p, c, mode)

    if len(log_likes) > 0:
        best_ll = max(log_likes)
        ind = np.argmax(log_likes)
        new_p, new_c, new_m = ll_map[ind]

    else:
        best_ll = 0
        new_p, new_c, new_m = (-1, -1, -1)

    return new_p, new_c, new_m, best_ll, prune_set


def preprocess_probs(g, data):
    """
    Preprocess the log likelihood of each node in the DAG
    :param g: a DAG object
    :param data: pandas DataFrame
    :return: g
    """
    for v in topological_sort(g):
        u_vec = []
        for u in g.predecessors(v):
            u_vec.append(u)

        if len(u_vec) == 0:
            mean = np.mean(data.iloc[:, v].values)
            var = np.std(data.iloc[:, v].values)
            g.nodes[v]['ll'] = compute_log_gaussian_prob(mean, var, data.iloc[:, v])

        else:
            g.nodes[v]['ll'] = compute_log_linear_prob2(u_vec, v, data)

    return g


def preprocess_single_prob(g, data, new_p, new_c, mode):
    """
    Preprocess the log likelihood of a single node
    :param g: a DAG object
    :param data: pandas DataFrame
    :param new_p: index of the new parent node
    :param new_c: index of the new child node
    :param mode: mode of graph augmentation
    :return: g
    """
    if mode == -1:
        g = preprocess_single(g, data, new_p)
        g = preprocess_single(g, data, new_c)

    else:
        g = preprocess_single(g, data, new_c)

    return g


def preprocess_single(g, data, v):
    """
    Helper function to process log likelihoods
    :param g: a DAG object
    :param data: pandas DataFrame
    :param v: the vertex to process
    :return: g
    """
    u_vec = []
    for u in g.predecessors(v):
        u_vec.append(u)

    if len(u_vec) == 0:
        mean = np.mean(data.iloc[:, v].values)
        var = np.std(data.iloc[:, v].values)
        g.nodes[v]['ll'] = compute_log_gaussian_prob(mean, var, data.iloc[:, v])

    else:
        g.nodes[v]['ll'] = compute_log_linear_prob2(u_vec, v, data)

    return g


def initialize_cand_edges(pcs, cols):
    """
    Initializes the set of candidate edges from edge selection output
    :param pcs: a dictionary {parent: [children]} of the candidate edges
    :param cols: list of columns in the data
    :return: a dictionary of candidate edges
    """
    candidate_edges = {}
    for parent, children in pcs.items():
        p = cols.get_loc(parent)
        if len(children) > 0:
            for child in children:
                c = cols.get_loc(child)
                candidate_edges = add_edge(candidate_edges, p, c)

    return candidate_edges


def remove_edge(edge_list, p, c, mode):
    """
    Remove an edge from the edge list
    :param edge_list: a dictionary of format {p: (c, mode)}
    :param p: identifier of the parent (source) node of the new edge
    :param c: identifier of the child (sink) node of the new edge
    :param mode: 1 = add, 0 = delete, -1 = reverse
    :return: the new edge list
    """
    if len(edge_list[p]) <= 1:
        del edge_list[p]
    else:
        edge_list[p].remove((c, mode))

    return edge_list


def trim_prune_cands(prune, new_c):
    """
    Remove edges with the newly added child node from the pruned set
    :param prune: set of candidate edges for pruning
    :param new_c: newly added child node
    :return: prune
    """
    discard = {}
    for p, children in prune.items():
        for cm in children:
            c, mode = cm
            if c == new_c:
                discard = add_edge(discard, p, c, mode)

    for p, children in discard.items():
        for cm in children:
            prune = remove_edge(prune, p, cm[0], cm[1])

    return prune


def trim_cand_edges(prune, cand):
    """
    Prune the candidate edges using the given pruning set
    :param prune: edge list
    :param cand: edge list
    :return: cand
    """
    for p, children in prune.items():
        for cm in children:
            if cm in cand[p]:
                cand = remove_edge(cand, p, cm[0], cm[1])

    return cand


def unprune_new_cands(prune, cand, new_c):
    """
    Un-prune edges with the newly added child node
    :param prune: edge list
    :param cand: edge list
    :param new_c: newly added child node
    :return: prune, cand
    """
    discard = {}
    for p, children in prune.items():
        for cm in children:
            if cm[0] == new_c:
                cand = add_edge(cand, p, cm[0], cm[1])
                discard = add_edge(discard, p, cm[0], cm[1])

    for p, children in discard.items():
        for cm in children:
            prune = remove_edge(prune, p, cm[0], cm[1])

    return prune, cand


def condense_prune_set(old, new):
    """
    Combine the persistent and candidate pruning sets
    :param old: persistent pruning set
    :param new: candidate pruning set
    :return: new persistent pruning set
    """
    for p, children in new.items():
        for cm in children:
            old = add_edge(old, p, cm[0], cm[1])

    return old


def recompute_prune_cand(old_prune, new_prune, cand, new_c):
    """
    Wrapper to re-compute the persistent pruning set and candidate sets given new pruning candidates and new child node
    :param old_prune: persistent pruning set
    :param new_prune: candidate pruning set
    :param cand: candidate set
    :param new_c: newly added child node
    :return: cand, old_prune
    """
    new_prune = trim_prune_cands(new_prune, new_c)
    cand = trim_cand_edges(new_prune, cand)
    old_prune, cand = unprune_new_cands(old_prune, cand, new_c)
    old_prune = condense_prune_set(old_prune, new_prune)

    return cand, old_prune


def hill_climb2(data, pcs):
    """
    Perform a greedy hill-climbing search for the optimal DAG
    :param data: pandas DataFrame
    :param pcs: edge selection output from MMPC
    :return: a DAG object
    """
    g = DiGraph()
    g.add_nodes_from(range(data.shape[1]))
    candidate_edges = initialize_cand_edges(pcs, data.columns)
    done = False
    pruning_set = {}
    g = preprocess_probs(g, data)
    data_hat = data.copy()

    while not done:
        print('Edges added: %i' % len(g.edges))
        print('# of candidates: %i' % len(candidate_edges))

        new_p, new_c, mode, ll, prune_cands = select_edge2(data_hat, g, candidate_edges)
        if new_p == -1 or ll < 0:
            done = True
            break

        if mode == 1:
            candidate_edges = add_edge(candidate_edges, new_p, new_c, mode=-1)
            candidate_edges = add_edge(candidate_edges, new_p, new_c, mode=0)
        elif mode == -1:
            candidate_edges = add_edge(candidate_edges, new_c, new_p, mode=-1)
            candidate_edges = add_edge(candidate_edges, new_c, new_p, mode=0)
        else:
            candidate_edges = add_edge(candidate_edges, new_p, new_c, mode=1)
            candidate_edges = add_edge(candidate_edges, new_c, new_p, mode=1)

            # Handle the reverse edge in candidates if removing
            if (new_c, -1) in candidate_edges[new_p]:
                candidate_edges = remove_edge(candidate_edges, new_p, new_c, -1)
            elif (new_c, -1) in pruning_set[new_p]:
                pruning_set = remove_edge(pruning_set, new_p, new_c, -1)

        candidate_edges = remove_edge(candidate_edges, new_p, new_c, mode)
        candidate_edges, pruning_set = recompute_prune_cand(pruning_set, prune_cands, candidate_edges, new_c)

        g = augment_graph2(g, new_p, (new_c, mode))
        data_hat = augment_data(data, data_hat, g, new_c)
        g = preprocess_single_prob(g, data_hat, new_p, new_c, mode)

    return g


def weight_edges(g, data):
    edge_dict = {}
    for v in g.nodes:
        u_vec = []
        for u in g.predecessors(v):
            u_vec.append(u)

        x = data.iloc[:, u_vec]
        y = data.iloc[:, v]

        try:
            w_vec = np.linalg.inv(x.T @ x) @ x.T @ y
        except np.linalg.LinAlgError:
            w_vec = np.linalg.pinv(x.T @ x) @ x.T @ y

        for u, w in zip(u_vec, w_vec):
            if u not in edge_dict:
                edge_dict[u] = {v: fabs(1/w)}
            else:
                edge_dict[u][v] = fabs(1/w)

    e = [
        (u, v, {"weight": d})
        for u, nbrs in edge_dict.items()
        for v, d in nbrs.items()
    ]

    g.update(edges=e, nodes=edge_dict)

    return g


def mmhc(data, selection_method='l1'):
    """
    Run the MaxMin Hill-Climbing algorithm
    :param data: pandas DataFrame
    :return: a DAG object
    """
    parent_child = {}

    if selection_method == 'l1':
        parent_child = get_l1mb_pcs(data)

    else:
        cpc_dict = mmpc_bar(data)

        for c in data.columns:
            parent_child[c] = mmpc(c, cpc_dict)

    dag = hill_climb2(data, parent_child)
    dag = weight_edges(dag, data)

    return dag


def drop_zeros(data):
    means = data.mean(axis=0)
    data = data.loc[:, means > 0]

    return data


def filter_by_dispersion(data, n_keep=2000):
    vals = data.values

    mean = np.mean(vals, axis=0)
    std = np.var(vals, axis=0)
    disp = std/mean

    inds = np.argsort(disp)[-n_keep:]

    return data.iloc[:, inds]


def write_to_csv(g, cols, path):
    arr = []
    for e in g.edges.data('weight', default=1):
        arr.append([cols[e[0]], cols[e[1]], e[2]])

    df = pd.DataFrame(data=arr, columns=['Source', 'Sink', 'Weight'])
    df.to_csv(path)


def preprocess_data(df1, df2, df3, gene_list=None, n_keep=200):
    slices = [df1.shape[0], df2.shape[0]+df1.shape[0]]
    count_data = np.vstack((df1.values, df2.values, df3.values))
    raw_df = pd.DataFrame(data=count_data, columns=df1.columns)

    if gene_list is None:
        raw_df = drop_zeros(raw_df)
        raw_df = filter_by_dispersion(raw_df, n_keep=n_keep)

    else:
        raw_df = raw_df.loc[:, gene_list]

    df_scaled = pd.DataFrame(data=StandardScaler().fit_transform(raw_df), columns=raw_df.columns)
    return df_scaled.iloc[:slices[0], :], df_scaled.iloc[slices[0]:slices[1], :], df_scaled.iloc[slices[1]:, :]


def read_bens_genes(path):
    genes = []
    with open(path) as f:
        for line in f:
            genes.append(line.strip())

    return genes


def get_shortest_paths(dag):
    paths = {}
    for n in dag.nodes:
        paths[n] = shortest_path_length(dag, source=n, weight='weight')

    sp_mat = np.zeros((len(paths), len(paths)))

    for u, p in paths.items():
        for v, w in p.items():
            if u != v:
                sp_mat[u, v] = w
                sp_mat[v, u] = w

    max_val = np.max(sp_mat)

    for i in range(sp_mat.shape[0]):
        for j in range(sp_mat.shape[0]):
            if i != j and sp_mat[i, j] == 0:
                sp_mat[i, j] = max_val + 1

    return sp_mat


def grow_gene_set(genes, data):
    new_genes = []

    for g in genes:
        mb = get_l1_markov_blanket(g, data)
        new_genes = new_genes + mb

    return new_genes


def add_gene_set(genes, data, raw_data):
    for g in genes:
        data[g] = raw_data[g]

    return data


def compute_graph_ll(g):
    ll = 0
    for v in topological_sort(g):
        ll += g.nodes[v]['ll']

    # mdl = (2*g.number_of_nodes() + g.number_of_edges()) * np.log2(n) / 2

    return -1*ll


def permutation_test(all_df, aml_df, n=100):
    split = all_df.shape[0]
    data = np.vstack((all_df.values, aml_df.values))
    mdls = np.zeros((2, n))

    for i in range(n):
        print('Permutation %i' % i)
        rng = np.random.default_rng(i)
        rng.shuffle(data)

        all_perm = pd.DataFrame(data=data[:split, :], columns=all_df.columns)
        aml_perm = pd.DataFrame(data=data[split:, :], columns=aml_df.columns)

        all_dag = mmhc(all_perm)
        aml_dag = mmhc(aml_perm)

        mdls[0, i] = compute_graph_ll(all_dag)
        mdls[1, i] = compute_graph_ll(aml_dag)

    return mdls
