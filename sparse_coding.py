import spams
import torch
import numpy as np
import time


def dim_reduction_embed_old(X, method = 'all', param = None):
    # X : array-like, shape (n_dim, n_samples)
    n_dim, n_samples = X.shape
    # assert n_samples <= n_dim, 'n_samples must be smaller than n_dim'

    # move X onto GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    gram_matrix = X.T @ X

    assert isinstance(X, torch.Tensor), 'current implementation accepts only torch tensor for X'

    # do eigenvalue decomposition on the gram matrix
    L, Q = torch.linalg.eigh(gram_matrix)
    # Q @ torch.diag_embed(L) @ Q.mH == A
    L = torch.max(L, torch.zeros_like(L))
    # note that L is in ascending order
    assert torch.all(L[:-1] <= L[1:]), 'L is supposed to be in ascending order'


    if method is 'all':
        return torch.diag_embed(torch.sqrt(L)) @ Q.T
    elif method is 'top_d':
        # assert param is integer  
        assert isinstance(param, int), 'when method is top, param must be an integer meaning the dimension'
        return torch.diag_embed(torch.sqrt(L[-param:])) @ Q.T[-param:]
    elif method is 'top_energy':
        # assert param is float
        assert isinstance(param, float), 'when method is top_energy, param must be a float'
        L_flipped = torch.flip(L, [0])
        L_cumsum = torch.cumsum(L_flipped, 0)
        L_cumsum = L_cumsum / L_cumsum[-1]
        n_dim = torch.sum(L_cumsum < param)
        return torch.diag_embed(torch.sqrt(L[-n_dim:])) @ Q.T[-n_dim:]


def dim_reduction_embed(X, method = 'all', param = None):
    # X : array-like, shape (n_dim, n_samples)
    n_dim, n_samples = X.shape

    # Using torch.svd_lowrank for dimensionality reduction
    # U, S, V = torch.svd_lowrank(X, q=min(n_dim, n_samples))
    # print("running random svd")
    U, S, V = torch.svd_lowrank(X, q=500)

    # print("running original svd")
    # U, S, V = torch.svd(X)

    if method == 'all':
        # For 'all', we return V @ diag(S), which is the projection of all samples into the reduced space
        return (V * S).T  # Equivalent to V @ diag(S), but more efficient
    elif method == 'top_d':
        # assert param is an integer  
        assert isinstance(param, int), 'when method is top_d, param must be an integer meaning the dimension'
        # Return the projection on the top-d principal components
        return (V[:, :param] * S[:param]).T
    elif method == 'top_energy':
        # assert param is a float
        assert isinstance(param, float), 'when method is top_energy, param must be a float'
        S_cumsum = torch.cumsum(S**2, 0)
        S_total = S_cumsum[-1]
        energy_threshold = S_total * param
        n_dim_reduced = torch.sum(S_cumsum < energy_threshold) + 1  # +1 to include the dimension meeting the threshold
        return (V[:, :n_dim_reduced] * S[:n_dim_reduced]).T
    else:
        raise ValueError("Method not recognized. Use 'all', 'top_d', or 'top_energy'.")


def decompose(target, dl_dict, tau = 0.9, alpha = 0.1, print_level = 3, normalize = True, return_rectr_err = False, method = 'omp'):

    if print_level >= 2:
        print('shape of target is {}'.format(target.shape))
        print('length of dl_dict is {}'.format(len(dl_dict)))

    data = []
    data.append(target.view(-1))
    for atom in dl_dict:
        atom = atom.view(-1)
        data.append(atom)

    data = torch.stack(data, dim=0)

    if print_level >= 2:
        print(f"Successfully loaded {data.shape[0]} emotions, each in {data.shape[1]} dimension.")
        # print data dimension
        print("data dimension: ", data.shape)
        # print norm of each row of data
        print("norm of each row of data: ", torch.norm(data, dim=1))

    # print data @ data.T
    data_new = dim_reduction_embed(data.T).T
    # inner_1 = data @ data.T
    # inner_2 = data_new @ data_new.T
    # print("size of data_new: ", data_new.size())
    # # sanity check: inner_1 and inner_2 should be the same
    # # assert torch.allclose(inner_1, inner_2)
    # # print the difference between inner_1 and inner_2
    # error = inner_1 - inner_2
    # print("max inner product error: ", torch.max(torch.abs(error)))
    # print("mean inner product error: ", torch.mean(torch.abs(error)))
    # print("median inner product error: ", torch.median(torch.abs(error)))


    # data_new: n_samples x n_feature_dim
    data_new = data_new.numpy()

    # check for nan and inf
    if not np.all(np.isfinite(data_new)):
        print('data_new contains nan or inf')
        import pdb; pdb.set_trace()

    # which row of data_new corresponds to emotion "love"
    id_target = 0
    y = data_new[id_target]
    y = np.copy(y.reshape(1, -1))

    if normalize:
        norm_y = np.linalg.norm(y) / 1
        y = y / norm_y
        norm_data_new = np.linalg.norm(data_new, axis=1) / 1
        data_new = data_new / norm_data_new[:, None]

    data_new[id_target] = 0

    
    # benchmark the time of enet and omp

    # start = time.time()
    # # if method == 'enet':
    # c_spams = spams.lasso(np.asfortranarray(y.T), D=np.asfortranarray(data_new.T), 
    #                                     lambda1=tau * alpha, lambda2=(1.0-tau) * alpha, mode=2)
    # c_spams = np.asarray(c_spams.todense()).T[0]
    # end = time.time()
    # print(f"Time taken for enet: {end - start:.4f} seconds")
    # # if print_level >= 2:
    # #     print(c_spams)
    # # start = time.time()
    # # # elif method == 'omp':
    # # c_spams = orthogonal_matching_pursuit_single(data_new, y, n_nonzero=200, thr=1.0e-6)
    # # # convert c_spams shape properly
    # # c_spams = c_spams.squeeze()
    # # end = time.time()
    # # print(f"Time taken for omp: {end - start:.4f} seconds")

    
    # # make c_spam a row array
    # c_spams = c_spams.reshape(1, -1)
    c_spams = np.linalg.lstsq(data_new.T, y.T, rcond=None)[0].T

    def obj_elastic_net(X, y, c, alpha, tau):
        reconstruction_error = np.linalg.norm(y.T - X.T @ c.T) 
        l1_reg = np.sum(np.abs(c))
        l2_reg = np.sum(c ** 2)
        total_error = 0.5 * (reconstruction_error ** 2) + alpha * (tau * l1_reg + 0.5 * (1.0 - tau) * l2_reg)
        return reconstruction_error, l1_reg, l2_reg, total_error


    if print_level >= 1:
        reconstruction_error, l1_reg, l2_reg, total_error = obj_elastic_net(data_new, y, c_spams, alpha, tau)
        print('===elastic net===')
        print(f'recstr = {reconstruction_error:.4f}, l1_reg = {l1_reg:.4f}, l2_reg = {l2_reg:.4f}, total = {total_error:.4f}')

        c_lstsq = np.linalg.lstsq(data_new.T, y.T, rcond=None)[0].T

        reconstruction_error, l1_reg, l2_reg, total_error = obj_elastic_net(data_new, y, c_lstsq, alpha, tau)
        print('===least squares===')
        print(f'recstr = {reconstruction_error:.4f}, l1_reg = {l1_reg:.4f}, l2_reg = {l2_reg:.4f}, total = {total_error:.4f}')
    
    c_spams = c_spams[0]

    if normalize:
        c_spams = c_spams / norm_data_new * norm_y

    # set control_coefficient to be everything of c_spams except the first one
    control_coefficient = torch.tensor(c_spams[1:])
    
    if print_level >= 1:
        print('l1 norm of control_coefficient: ', torch.sum(torch.abs(control_coefficient)))

    if return_rectr_err:
        return control_coefficient, reconstruction_error
    
    return control_coefficient


def orthogonal_matching_pursuit_single(X, y, n_nonzero=100, thr=1.0e-6):
    """Sparse subspace clustering by orthogonal matching pursuit (SSC-OMP) for a single sample
    Compute self-representation vector c_i by solving the following optimization problem
    min_{c_i} ||y - c_i X ||_2^2 s.t. ||c_i||_0 <= n_nonzero
    via OMP, where c_i is the self-representation vector for y.

    Parameters
    -----------
    X : array-like, shape (n_samples, n_features)
        Input data to be clustered
    y : array-like, shape (n_features,)
        Target sample to process
    n_nonzero : int, default 100
        Termination condition for omp.
    thr : float, default 1.0e-6
        Termination condition for omp.	

    Returns
    -------
    c_i : array, shape: (n_samples,)
        The self-representation vector for the target y.
    """	
    n_samples = X.shape[0]
    residual = y.copy()  # initialize residual
    supp = np.empty(shape=(0), dtype=int)  # initialize support
    residual_norm_thr = np.linalg.norm(y) * thr

    for t in range(n_nonzero):  # for each iteration of OMP  
        # compute coherence between residuals and X     
        coherence = abs(np.matmul(residual, X.T))
        # update support
        supp = np.append(supp, np.argmax(coherence))
        # compute coefficients
        coeff = np.linalg.lstsq(X[supp, :].T, y.T, rcond=None)[0]
        # compute residual
        residual = y - np.matmul(coeff.T, X[supp, :])
        # check termination
        if np.sum(residual ** 2) < residual_norm_thr:
            break

    c_i = np.zeros(n_samples)
    c_i[supp] = coeff.squeeze()
    return c_i