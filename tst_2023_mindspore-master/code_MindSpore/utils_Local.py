import sys
sys.path.append('../')
from utils_sqrtm import sqrtm
import numpy as np
import torch
import torch.utils.data
from past.utils import old_div
import scipy.stats as stats
from sklearn.utils import check_random_state
from sklearn.neighbors import NearestNeighbors
import mindspore
from mindspore import Tensor as MSTensor
from mindspore import Tensor, nn, ops, context,Parameter
import mindspore.numpy as mnp
from mindspore.common import dtype as mstype
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.ops import composite as C
is_cuda = False

def density_estimate_test(S, N1):
    """KNN density estimator"""
    p_size = 20
    density1 = []
    density2 = []

    d = len(S[0])

    N2 = len(S) - N1
    nbrs1 = NearestNeighbors(n_neighbors=p_size + 1, algorithm='ball_tree').fit(S[:N1])
    distances1, indices1 = nbrs1.kneighbors(S)
    nbrs2 = NearestNeighbors(n_neighbors=p_size + 1, algorithm='ball_tree').fit(S[N1:])
    distances2, indices2 = nbrs2.kneighbors(S)

    for i in range(0, len(S)):
        density1.append(p_size / (N1 * distances1[i][p_size] ** d))
        density2.append(p_size / (N2 * distances2[i][p_size - 1] ** d))
    for i in range(N1, len(S)):
        density1.append(p_size / (N1 * distances1[i][p_size - 1] ** d))
        density2.append(p_size / (N2 * distances2[i][p_size] ** d))
    return np.array(density1), np.array(density2)

def sample_BLOB(N, rs, check):
    rs = check_random_state(rs)
    rows = 3
    cols = 3
    if check == 0:
        """Generate Blob-S for testing type-I error. X and Y are drawn from an identical distribution"""
        sep = 1
        correlation = 0
        # generate within-blob variation
        mu = np.zeros(2)
        sigma = np.eye(2)
        X = rs.multivariate_normal(mu, sigma, size=N)
        corr_sigma = np.array([[1, correlation], [correlation, 1]])
        Y = rs.multivariate_normal(mu, corr_sigma, size=N)
        # assign to blobs
        X[:, 0] += rs.randint(rows, size=N) * sep
        X[:, 1] += rs.randint(cols, size=N) * sep
        Y[:, 0] += rs.randint(rows, size=N) * sep
        Y[:, 1] += rs.randint(cols, size=N) * sep
    else:
        """Generate Blob-D for testing or test power. X and Y are drawn from two different distributions"""
        sigma_mx_2_standard = np.array([[0.03, 0], [0, 0.03]])
        sigma_mx_2 = np.zeros([9, 2, 2])
        for i in range(9):
            sigma_mx_2[i] = sigma_mx_2_standard
            if i < 4:
                sigma_mx_2[i][0, 1] = -0.02 - 0.002 * i
                sigma_mx_2[i][1, 0] = -0.02 - 0.002 * i
            if i == 4:
                sigma_mx_2[i][0, 1] = 0.00
                sigma_mx_2[i][1, 0] = 0.00
            if i > 4:
                sigma_mx_2[i][1, 0] = 0.02 + 0.002 * (i - 5)
                sigma_mx_2[i][0, 1] = 0.02 + 0.002 * (i - 5)

        mu = np.zeros(2)
        sigma = np.eye(2) * 0.03
        X = rs.multivariate_normal(mu, sigma, size=N)
        Y = rs.multivariate_normal(mu, np.eye(2), size=N)
        X[:, 0] += rs.randint(rows, size=N)
        X[:, 1] += rs.randint(cols, size=N)
        Y_row = rs.randint(rows, size=N)
        Y_col = rs.randint(cols, size=N)
        locs = [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2], [2, 0], [2, 1], [2, 2]]
        for i in range(9):
            corr_sigma = sigma_mx_2[i]
            L = np.linalg.cholesky(corr_sigma)
            ind = np.expand_dims((Y_row == locs[i][0]) & (Y_col == locs[i][1]), 1)
            ind2 = np.concatenate((ind, ind), 1)
            Y = np.where(ind2, np.matmul(Y, L) + locs[i], Y)
    return X, Y
def Local_OPT(S, N1, n_Anchors, N_epoch, learning_rate, percent, split_adjust, seed, device, dtype, batch_size=None):
    reg = torch.tensor(1e-5)
    """initialization for test locations"""
    Anchors = init_locs_2randn(S, N1, n_Anchors, seed + 5)
    Anchors = MatConvert(Anchors, device, dtype)
    med = meddistance(S, 1000)
    list_gwidth = np.hstack(((med ** 2) * (2.0 ** np.linspace(-3, 4, 30))))
    list_gwidth.sort()
    list_gwidth = MatConvert(list_gwidth, device, dtype)

    S = MatConvert(S, device, dtype)

    """initialization for parameter gamma of Mahalanobis kernels"""
    besti, powers = grid_search_gwidth(S, N1, Anchors, list_gwidth, 0.05, device, dtype)
    gwidth = list_gwidth[besti]
    gwidths = np.repeat(gwidth, n_Anchors)
    gwidths = MatConvert(gwidths, device, dtype)

    """initialization for Mahalanobis matrices of Mahalanobis kernels"""
    M_matrix = np.identity(S.shape[1])
    M_matrixs = np.tile(M_matrix, (n_Anchors, 1)).reshape((-1, S.shape[1], S.shape[1]))
    M_matrixs = MatConvert(M_matrixs, device, dtype)
    # M_matrixs = M_matrix_initial(S, Anchors, device, dtype)

    np.random.seed(seed=1102)
    torch.manual_seed(1102)
    torch.cuda.manual_seed(1102)
    Anchors=torch.from_numpy(Anchors.asnumpy())
    Anchors.requires_grad = True
    gwidths=torch.from_numpy(gwidths.asnumpy())
    gwidths.requires_grad = True
    M_matrixs=torch.from_numpy(M_matrixs.asnumpy())
    M_matrixs.requires_grad = True
    optimizer_u = torch.optim.Adam([Anchors] + [gwidths] + [M_matrixs], lr=learning_rate)

    for t in range(N_epoch):
        S1 = S[:N1, :]
        S1 =torch.from_numpy(S1.asnumpy())
        S2 = S[N1:, :]
        S2 = torch.from_numpy(S2.asnumpy())
        epoch = max(min(int(len(S1) / batch_size) * 2, int(len(S2) / batch_size) * 2), 1)
        for i in range(epoch):
            if int(len(S1) / batch_size) * 2 <= 1 or int(len(S2) / batch_size) * 2 <= 1:
                ind1 = np.random.choice(np.arange(len(S1)), min(len(S1), len(S2)), replace=False)
                ind2 = np.random.choice(np.arange(len(S2)), min(len(S1), len(S2)), replace=False)
            else:
                ind1 = np.random.choice(np.arange(len(S1)), int(batch_size / 2), replace=False)
                ind2 = np.random.choice(np.arange(len(S2)), int(batch_size / 2), replace=False)

            X = torch.cat([S1[ind1], S2[ind2]], 0)
            if is_cuda:
                S1 = torch.index_select(S1, 0, torch.tensor(np.delete(np.arange(len(S1)), ind1, 0), dtype=torch.long).cuda())
                S2 = torch.index_select(S2, 0, torch.tensor(np.delete(np.arange(len(S2)), ind2, 0), dtype=torch.long).cuda())
            else:
                S1 = np.delete(S1, ind1, 0)
                S2 = np.delete(S2, ind2, 0)

            loss, _ = HT_Statistics_Mkernels(X, int(len(X) / 2), Anchors, gwidths, M_matrixs, device, dtype)

            optimizer_u.zero_grad()
            loss.backward(retain_graph=True)
            # Update weights using gradient
            optimizer_u.step()

        # map Mahalanobis matrices to the positive-definite cone
        if (t + 1) % 5 == 0:
            with torch.no_grad():
                for j in range(len(M_matrixs)):
                    eigvalues, eigvectors = torch.eig(M_matrixs[j], eigenvectors=True)
                    eigvalues = torch.max(eigvalues[:, 0], reg)
                    eigvectors = eigvectors.t().reshape(eigvectors.shape[0], -1, eigvectors.shape[1])
                    M_matrixs[j] = eigvalues[0] * eigvectors[0].t() * eigvectors[0]
                    for i in range(1, len(eigvalues)):
                        M_matrixs[j] += eigvalues[i] * eigvectors[i] * eigvectors[i].t()
        if (t + 1) % 100 == 0 or t == 0:
            print("STAT_value: ", loss.item())

    gwidths = gwidths.detach()
    Anchors = Anchors.detach()
    M_matrixs = M_matrixs.detach()

    T_level = int(np.log2(1 / percent))
    Tree = gen_tree(S, N1, T_level, split_adjust, Anchors, gwidths, M_matrixs, device, dtype)

    return Anchors, gwidths, M_matrixs, Tree, T_level

def gen_tree(S, N1, T_level, split_adjust, Anchors, gwidths, M_matrixs, device, dtype):
    reg = torch.tensor(1e-5)
    assert N1 >= len(S) - N1

    M = len(S)
    J = Anchors.shape[0]

    gwidths = gwidths.detach()
    Anchors = Anchors.detach()
    M_matrixs = M_matrixs.detach()

    """Calculate the embedding samples"""
    gwidths = torch.max(gwidths, reg)
    Mats_A = torch.sum(torch.mul(M_matrixs, Anchors.reshape(Anchors.shape[0], -1, Anchors.shape[1])), 2)
    A_Mats = torch.sum(torch.mul(M_matrixs.transpose(1, 2), Anchors.reshape(Anchors.shape[0], -1, Anchors.shape[1])), 2)
    A_Mats_A = torch.sum(torch.mul(Mats_A, Anchors), 1)
    S = torch.from_numpy(S.asnumpy())
    S_Mats = torch.matmul(S, M_matrixs)
    S_Mats_S = torch.sum(torch.mul(S_Mats, S), 2).t()
    S_Mats_A = torch.sum(torch.mul(S_Mats.transpose(0, 1).reshape(-1, Anchors.shape[0], Anchors.shape[1]), Anchors), 2)
    A_Mats_S = torch.sum(torch.mul(A_Mats.reshape(Anchors.shape[0], -1, Anchors.shape[1]), S).transpose(0, 1).reshape(-1, Anchors.shape[0], Anchors.shape[1]), 2)
    D = S_Mats_S - A_Mats_S - S_Mats_A + A_Mats_A
    D = torch.max(D, torch.tensor(0.0))
    D = torch.exp(torch.div(-D, 2.0 * gwidths))


    """construct partition tree"""
    Tree = []
    split_INDEXS = [np.arange(M)]
    stat_chi, diff = HT_Statistics_cal(D, N1, device, dtype)
    split_F = [torch.sign(diff)]
    split_N1 = [N1]

    min_N1 = int(N1 / 2 ** T_level / split_adjust)
    for i in range(T_level):
        level_indicator = 2 ** i

        split_INDEXS_left = None
        split_INDEXS_right = None
        split_F_left = None
        split_F_right = None
        split_N1_left = None
        split_N1_right = None

        for l in range(level_indicator):
            INDEXS = split_INDEXS[0]
            del split_INDEXS[0]
            split_index = 0
            split_value = 0
            p_temp = 1

            N1_node = split_N1[0]
            del split_N1[0]

            for j in range(J):
                value = torch.median(D[INDEXS[:N1_node], j])  ### split based on the median of sample X

                left_idx = np.argwhere(D[INDEXS, j] <= value).reshape(-1)
                right_idx = np.argwhere(D[INDEXS, j] > value).reshape(-1)
                if len(right_idx) == 0:
                    continue
                INDEXS_right = INDEXS[right_idx]
                INDEXS_left = INDEXS[left_idx]
                N1_left = sum(np.array(INDEXS_left).reshape(-1) < N1)
                N1_right = sum(np.array(INDEXS_right).reshape(-1) < N1)

                if N1_left < min_N1 * 2 ** (T_level - i - 1) or N1_right < min_N1 * 2 ** (T_level - i - 1):
                    continue

                stat_chi_left, diff_left = HT_Statistics_cal(D[INDEXS_left], N1_left, device, dtype)
                stat_chi_left = stat_chi_left.cpu().detach().numpy()
                p_chi_left = stats.chi2.sf(stat_chi_left, J)
                if np.isnan(p_chi_left):   ### len(D[INDEXS_left]) == N1_left
                    p_chi_left = 0.0
                p_left = p_chi_left

                stat_chi_right, diff_right = HT_Statistics_cal(D[INDEXS_right], N1_right, device, dtype)
                stat_chi_right = stat_chi_right.cpu().detach().numpy()
                p_chi_right = stats.chi2.sf(stat_chi_right, J)
                if np.isnan(p_chi_right):  ### len(D[INDEXS_right]) == N1_right
                    p_chi_right = 0.0
                p_right = p_chi_right

                if p_left * p_right < p_temp:
                    p_temp = p_left * p_right

                    split_index = j
                    split_value = value
                    split_INDEXS_left = INDEXS_left
                    split_INDEXS_right = INDEXS_right
                    split_F_left = torch.sign(diff_left)
                    split_F_right = torch.sign(diff_right)
                    split_N1_left = N1_left
                    split_N1_right = N1_right

            Tree.append([split_index, split_value, split_F[0]])
            del split_F[0]

            split_INDEXS.append(split_INDEXS_left)
            split_INDEXS.append(split_INDEXS_right)
            split_F.append(split_F_left)
            split_F.append(split_F_right)
            split_N1.append(split_N1_left)
            split_N1.append(split_N1_right)

    assert len(split_INDEXS) == len(split_F)

    for i in range(len(split_INDEXS)):
        Tree.append([None, None, split_F[0]])
        del split_F[0]
    return Tree

"""test procedure for two-sample test"""
def Local_TEST(S, N1, Anchors, gwidths, M_matrixs, infer_dire, alpha, beta, device, dtype):
    S = MatConvert(S, device, dtype)
    stat, diff = HT_Statistics_Mkernels(S, N1, Anchors, gwidths, M_matrixs, device, dtype)
    test_flags = diff
    stat = stat.cpu().detach().numpy()
    J, d = Anchors.shape

    pvalue = stats.chi2.sf(-stat, J)
    if sum(test_flags * infer_dire) >= 0:
        h = int(pvalue <= beta * alpha)
    else:
        pvalue = stats.chi2.sf(-stat, J)
        h = int(pvalue <= (2 - beta) * alpha)
    return h ## return the test results h

"""explore the local significant difference"""
def explore_regions(S, N1, Anchors, gwidths, M_matrixs, T_level, Tree, alpha, beta, split_adjust, device, dtype, min_num):   ### min_num is the parameter v given in Eqn. (28)
    """Calculate the embedding samples"""
    S = MatConvert(S, device, dtype)
    gwidths = torch.max(gwidths, torch.tensor(10 ** -5))
    Mats_A = torch.sum(torch.mul(M_matrixs, Anchors.reshape(Anchors.shape[0], -1, Anchors.shape[1])), 2)
    A_Mats = torch.sum(torch.mul(M_matrixs.transpose(1, 2), Anchors.reshape(Anchors.shape[0], -1, Anchors.shape[1])), 2)
    A_Mats_A = torch.sum(torch.mul(Mats_A, Anchors), 1)
    S_Mats = torch.matmul(S, M_matrixs)
    S_Mats_S = torch.sum(torch.mul(S_Mats, S), 2).t()
    S_Mats_A = torch.sum(torch.mul(S_Mats.transpose(0, 1).reshape(-1, Anchors.shape[0], Anchors.shape[1]), Anchors), 2)
    A_Mats_S = torch.sum(torch.mul(A_Mats.reshape(Anchors.shape[0], -1, Anchors.shape[1]), S).transpose(0, 1).reshape(-1, Anchors.shape[0], Anchors.shape[1]), 2)
    D = S_Mats_S - A_Mats_S - S_Mats_A + A_Mats_A
    D = torch.max(D, torch.tensor(0.0))
    D = torch.exp(torch.div(-D, 2.0 * gwidths))

    """partition samples into rectangle regions based on tree"""
    M, J = D.shape
    min_N1 = int(N1 / 2 ** T_level / split_adjust)
    split_N1 = [N1]
    DATA_INDEXS = [np.arange(M)]
    for i in range(T_level):
        for k in range(2 ** i):
            split_index, split_value, split_F = Tree[0]
            del Tree[0]
            INDEXS = DATA_INDEXS[0]
            Node_N1 = split_N1[0]
            del DATA_INDEXS[0]
            del split_N1[0]
            left_idx = np.argwhere(D[INDEXS, split_index] <= split_value).reshape(-1)
            right_idx = np.argwhere(D[INDEXS, split_index] > split_value).reshape(-1)
            try:
                left_N1 = sum(np.array(INDEXS[left_idx] < N1))
                right_N1 = sum(np.array(INDEXS[right_idx] < N1))
            except:
                left_N1 = 0
                right_N1 = 0

            """avoid empty rectangle region"""
            if left_N1 < min_N1 * 2 ** (T_level - i - 1) or right_N1 < min_N1 * 2 ** (T_level - i - 1):
                split_value = np.median(D[INDEXS[:Node_N1], split_index])
                left_idx = np.argwhere(D[INDEXS, split_index] <= split_value).reshape(-1)
                right_idx = np.argwhere(D[INDEXS, split_index] > split_value).reshape(-1)
                left_N1 = sum(np.array(INDEXS[left_idx] < N1))
                right_N1 = sum(np.array(INDEXS[right_idx] < N1))

            split_N1.append(left_N1)
            split_N1.append(right_N1)
            DATA_INDEXS.append(INDEXS[left_idx])
            DATA_INDEXS.append(INDEXS[right_idx])
    assert len(Tree) >= len(DATA_INDEXS)
    assert len(DATA_INDEXS) == 2 ** T_level == len(split_N1)

    thres = min(min_num, max(1, int(len(DATA_INDEXS) / 2)))
    p_ = 1 - 2 ** (np.log2(1 - alpha) / thres)  ###  the parameter p_*

    H = []
    G = []
    for i in range(2 ** T_level):
        _, _, split_F = Tree[i]
        if len(DATA_INDEXS[0]) == split_N1[0]:
            p_chi = 0
            diff = torch.mean(D[DATA_INDEXS[0]], dim=0)
        else:
            stat_chi, diff = HT_Statistics_cal(D[DATA_INDEXS[0]], split_N1[0], device, dtype)
            stat_chi = stat_chi.cpu().detach().numpy()
            # p_chi = sf_chi2(J, stat_chi) ###  calculate the p-value by simulation
            p_chi = stats.chi2.sf(stat_chi, J)

        del DATA_INDEXS[0]
        del split_N1[0]

        if sum(diff * split_F) > 0:
            h = int(p_chi < beta * p_)
            g = G_fun(p_chi, p_, beta, 1)
        else:
            h = int(p_chi < (2 - beta) * p_)
            g = G_fun(p_chi, p_, beta, -1)

        H.append(h)
        G.append(g)

    H = np.array(H)
    # diff_results = np.argwhere(H == 1).reshape(-1)
    G = np.array(G)
    num_reject = sum(H == 0)
    num_accept = sum(H > 0)
    idx_sort = np.argsort(-G)
    H = H[idx_sort]
    G = G[idx_sort]

    for i in range(2 ** T_level):
        if num_reject <= thres:
            break

        if H[0] == 0:
            num_reject -= 1
            H = H[1:]
            G = G[1:]
            idx_sort = idx_sort[1:]
        else:
            num_accept -= 1
            H = H[1:]
            G = G[1:]
            idx_sort = idx_sort[1:]

    return idx_sort[np.argwhere(H == 1).reshape(-1)][::-1]   ### return the indexs set of rectangle regions of significant differences

"""measure the density difference in the identified rectangle regions"""
def TEST_density_diff(S, N1, Anchors, gwidths, M_matrixs, T_level, Tree, DIFF_idx, split_adjust, density1, density2, device, dtype):
    S = MatConvert(S, device, dtype)
    gwidths = torch.max(gwidths, torch.tensor(10 ** -5))
    Mats_A = torch.sum(torch.mul(M_matrixs, Anchors.reshape(Anchors.shape[0], -1, Anchors.shape[1])), 2)
    A_Mats = torch.sum(torch.mul(M_matrixs.transpose(1, 2), Anchors.reshape(Anchors.shape[0], -1, Anchors.shape[1])), 2)
    A_Mats_A = torch.sum(torch.mul(Mats_A, Anchors), 1)
    S_Mats = torch.matmul(S, M_matrixs)
    S_Mats_S = torch.sum(torch.mul(S_Mats, S), 2).t()
    S_Mats_A = torch.sum(torch.mul(S_Mats.transpose(0, 1).reshape(-1, Anchors.shape[0], Anchors.shape[1]), Anchors), 2)
    A_Mats_S = torch.sum(torch.mul(A_Mats.reshape(Anchors.shape[0], -1, Anchors.shape[1]), S).transpose(0, 1).reshape(-1, Anchors.shape[0], Anchors.shape[1]), 2)
    D = S_Mats_S - A_Mats_S - S_Mats_A + A_Mats_A
    D = torch.max(D, torch.tensor(0.0))
    D = torch.exp(torch.div(-D, 2.0 * gwidths))

    min_N1 = int(N1 / 2 ** T_level / split_adjust)

    M, J = D.shape
    split_N1 = [N1]
    DATA_INDEXS = [np.arange(M)]
    for i in range(T_level):
        for k in range(2 ** i):
            split_index, split_value, _ = Tree[0]
            del Tree[0]
            INDEXS = DATA_INDEXS[0]
            Node_N1 = split_N1[0]
            del DATA_INDEXS[0]
            del split_N1[0]

            left_idx = np.argwhere(D[INDEXS, split_index] <= split_value).reshape(-1)
            right_idx = np.argwhere(D[INDEXS, split_index] > split_value).reshape(-1)
            try:
                left_N1 = sum(np.array(INDEXS[left_idx] < N1))
                right_N1 = sum(np.array(INDEXS[right_idx] < N1))
            except:
                left_N1 = 0
                right_N1 = 0

            if left_N1 < min_N1 * (T_level - 1) or right_N1 < min_N1 * (T_level - 1):
                split_value = np.median(D[INDEXS[:Node_N1], split_index])
                left_idx = np.argwhere(D[INDEXS, split_index] <= split_value).reshape(-1)
                right_idx = np.argwhere(D[INDEXS, split_index] > split_value).reshape(-1)
                left_N1 = sum(np.array(INDEXS[left_idx] < N1))
                right_N1 = sum(np.array(INDEXS[right_idx] < N1))

            split_N1.append(left_N1)
            split_N1.append(right_N1)
            DATA_INDEXS.append(INDEXS[left_idx])
            DATA_INDEXS.append(INDEXS[right_idx])

    sum_diffs = 0
    for i in DIFF_idx:
        diff = sum(np.abs(density1[DATA_INDEXS[i]] - density2[DATA_INDEXS[i]])) / len(DATA_INDEXS[i])
        sum_diffs += diff
    return sum_diffs / len(DIFF_idx)   ### return the density difference

"""calculation of masked p-value"""
def G_fun(p, p_, b, flag):
    g = None
    if flag == 1 or b == 2:
        if p < b * p_:
            g = p / b
        else:
            g = p_ / (1 - b * p_) * (1 - p)
    else:
        if p < (2 - b) * p_:
            g = p / (2 - b)
        else:
            g = p_ / (1 - (2 - b) * p_) * (1 - p)
    return g

"""calculate the statistic and the vector for inference direction based on embedding samples"""
def HT_Statistics_cal(D, N1, device, dtype):
    diag = 1e-5 * torch.eye(D.shape[1])
    Cst = torch.div((len(D) - N1) * N1, len(D))
    Sig1, u1 = torch_cov(D[:N1])
    Sig2, u2 = torch_cov(D[N1:])
    Sig = torch.div(Sig1 + Sig2, max(len(D) - 2, 1))
    T = Cst * torch.mv(torch.inverse(Sig + diag).t(), u1 - u2).dot(u1 - u2)
    L_inv = torch.inverse(sqrtm(Sig + diag))
    D = torch.matmul(L_inv, D.transpose(0, 1)).transpose(0, 1)
    diff = torch.mean(D[:N1], dim=0) - torch.mean(D[N1:], dim=0)
    return T, diff   ### return the statistic and the vector for inference direction

"""calculate the statistic and the vector for inference direction based on original samples"""


def HT_Statistics_Mkernels(S, N1, Anchors, gwidths, Matrixs, device, dtype):
    gwidths = torch.max(gwidths, torch.tensor(10 ** -5))
    Mats_A = torch.sum(torch.mul(Matrixs, Anchors.reshape(Anchors.shape[0], -1, Anchors.shape[1])), 2)
    A_Mats = torch.sum(torch.mul(Matrixs.transpose(1, 2), Anchors.reshape(Anchors.shape[0], -1, Anchors.shape[1])), 2)
    A_Mats_A = torch.sum(torch.mul(Mats_A, Anchors), 1)
    if isinstance(S, MSTensor):
       S = torch.from_numpy(S.asnumpy())
    S1_Mats = torch.matmul(S[:N1], Matrixs)
    S1_Mats_S1 = torch.sum(torch.mul(S1_Mats, S[:N1]), 2).t()
    S1_Mats_A = torch.sum(torch.mul(S1_Mats.transpose(0, 1).reshape(-1, Anchors.shape[0], Anchors.shape[1]), Anchors), 2)
    A_Mats_S1 = torch.sum(torch.mul(A_Mats.reshape(Anchors.shape[0], -1, Anchors.shape[1]), S[:N1]).transpose(0, 1).reshape(-1, Anchors.shape[0], Anchors.shape[1]), 2)
    D1 = S1_Mats_S1 - A_Mats_S1 - S1_Mats_A + A_Mats_A
    S2_Mats = torch.matmul(S[N1:], Matrixs)
    S2_Mats_S2 = torch.sum(torch.mul(S2_Mats, S[N1:]), 2).t()
    S2_Mats_A = torch.sum(torch.mul(S2_Mats.transpose(0, 1).reshape(-1, Anchors.shape[0], Anchors.shape[1]), Anchors), 2)
    A_Mats_S2 = torch.sum(torch.mul(A_Mats.reshape(Anchors.shape[0], -1, Anchors.shape[1]), S[N1:]).transpose(0, 1).reshape(-1, Anchors.shape[0], Anchors.shape[1]), 2)
    D2 = S2_Mats_S2 - A_Mats_S2 - S2_Mats_A + A_Mats_A

    D1 = torch.max(D1, torch.tensor(0.0))
    D2 = torch.max(D2, torch.tensor(0.0))
    D1 = torch.exp(torch.div(-D1, 2.0 * gwidths))
    D2 = torch.exp(torch.div(-D2, 2.0 * gwidths))
    Sig1, u1 = torch_cov(D1)
    Sig2, u2 = torch_cov(D2)
    Sig = torch.div(Sig1 + Sig2, max(len(S) - 2, 1))
    diag = 1e-5 * torch.eye(Anchors.shape[0])
    L_inv = torch.inverse(sqrtm(Sig + diag))

    D1 = torch.matmul(L_inv, D1.transpose(0, 1)).transpose(0, 1)
    D2 = torch.matmul(L_inv, D2.transpose(0, 1)).transpose(0, 1)
    diff = torch.mean(D1, dim=0) - torch.mean(D2, dim=0)

    Cst = torch.div((len(S) - N1) * N1, len(S))
    T = Cst * torch.mv(torch.inverse(Sig + diag).t(), u1 - u2).dot(u1 - u2)
    return -T, diff.detach()   ### return the statistic and the vector for inference direction



"""calculate p-value based on simulation"""
def sf_chi2(J, x):
    mean = np.zeros(J)
    cov = np.eye(J)
    X = np.random.multivariate_normal(mean, cov, 10000)
    S = np.sum(np.abs(X ** 2), 1)
    n = np.shape(S)[0]
    m = np.shape(S[S > x])[0]
    res = m / n
    return res

"""calculate the statistic in initialization"""
def HT_Statistics_kernel(S, N1, Anchors, gwidth, device, dtype):
    # Convert the inputs to MindSpore Tensors if they're not already
    S = Tensor(S, dtype)
    Anchors = Tensor(Anchors, dtype)

    diag = 1e-5 * mnp.eye(Anchors.shape[0])
    Cst = Tensor((len(S) - N1) * N1 / len(S), dtype)
    N1 = int(N1)
    #print(S.shape,N1)
    D1 = mnp.sum(S[:N1] ** 2, 1).reshape((-1, 1)) - 2 * mnp.matmul(S[:N1], mnp.transpose(Anchors)) + mnp.sum(Anchors ** 2, 1).reshape((1, -1))
    D2 = mnp.sum(S[N1:] ** 2, 1).reshape((-1, 1)) - 2 * mnp.matmul(S[N1:], mnp.transpose(Anchors)) + mnp.sum(Anchors ** 2, 1).reshape((1, -1))
    
    D1 = mnp.exp(-D1 / (2.0 * gwidth))
    D2 = mnp.exp(-D2 / (2.0 * gwidth))
    #print(D1.shape)
    Sig1, u1 = torch_cov2(D1)  # Assuming this function is already converted to work with MindSpore
    Sig2, u2 = torch_cov2(D2)  # Same as above
    
    Sig = (Sig1 + Sig2) / max(len(S) - 2, 1)
    if isinstance(Sig, torch.Tensor):
        # Convert PyTorch tensor to numpy array
        Sig_numpy = Sig.cpu().numpy()  # Use `.cpu()` if tensor is on GPU
        # Convert numpy array to MindSpore tensor
        Sig = Tensor(Sig_numpy)
   
    Sig_numpy = Sig.asnumpy()

    # Assuming diag is a MindSpore tensor and converting it to numpy
    diag_numpy = diag.asnumpy()

    # Perform addition using NumPy
    Sig_plus_diag = np.add(Sig_numpy, diag_numpy)

    # Matrix inversion using NumPy
    k_numpy = np.linalg.inv(Sig_plus_diag)

    # Convert the result back to MindSpore Tensor
    k = Tensor(k_numpy)

    # Convert other variables to MindSpore tensors if they are not already

    # Now, you can continue with your operations using k
    # ... for example, if you need to perform a matmul operation:
    result = mnp.matmul(k, u1 - u2)  # Adjust as necessary for your application
    difference = mnp.subtract(u1, u2)

    # Perform element-wise multiplication
    elementwise_product = mnp.multiply(result, difference)

    # Sum the result to get the dot product
    dot_product = mnp.sum(elementwise_product)

    # Now compute your final result
    T = Cst * dot_product
    return T  # or whatever variable you need to return

def grid_search_gwidth(S, N1, Anchors, list_gwidth, alpha, device, dtype):
    """
    Linear search for the best Gaussian width in the list that maximizes
    the test power, fixing the test locations to T.
    The test power is given by the CDF of a non-central Chi-squared
    distribution.
    return: (best width index, list of test powers)
    """

    S = Tensor(S, dtype) if not isinstance(S, Tensor) else S
    N1 = Tensor(N1, dtype) if not isinstance(N1, Tensor) else N1
    Anchors = Tensor(Anchors, dtype) if not isinstance(Anchors, Tensor) else Anchors
    list_gwidth = Tensor(list_gwidth, dtype) if not isinstance(list_gwidth, Tensor) else list_gwidth

    num_widths = list_gwidth.shape[0]
    powers = mnp.zeros(num_widths)
    lambs = mnp.zeros(num_widths)
    thresh = stats.chi2.isf(alpha, df=Anchors.shape[0])

    for wi in range(num_widths):
        gwidth = list_gwidth[wi]
        try:
            lamb = HT_Statistics_kernel(S, N1, Anchors, gwidth, device, dtype)
            lamb_numpy = lamb.asnumpy()
            if lamb_numpy <= 0:
                raise np.linalg.LinAlgError
            if np.iscomplex(lamb_numpy):
                print('Lambda is complex. Truncate the imag part. lamb: %s' % (str(lamb_numpy)))
                lamb_numpy = np.real(lamb_numpy)

            power = stats.ncx2.sf(thresh, df=Anchors.shape[0], nc=lamb_numpy)
            powers_np = powers.asnumpy()
            lambs_np = lambs.asnumpy()
            powers_np[wi] = power
            lambs_np[wi] = lamb_numpy
            powers = Tensor(powers_np, dtype=dtype)
            lambs = Tensor(lambs_np, dtype=dtype)

        except np.linalg.LinAlgError:
            print('LinAlgError. skip width (%d, %.3g)' % (wi, gwidth))
            powers_np = powers.asnumpy()
            lambs_np = lambs.asnumpy()
            powers_np[wi] = np.NINF
            lambs_np[wi] = np.NINF
            powers = Tensor(powers_np, dtype=dtype)
            lambs = Tensor(lambs_np, dtype=dtype)

    # Convert powers to numpy for the argmax operation, then convert the result back to Tensor
    besti_numpy = np.argmax(np.around(powers.asnumpy(), 3))
    besti = Tensor(besti_numpy, dtype=mnp.int32)

    return besti, powers
def M_matrix_initial(S, Anchors, device, dtype):
    M_matrixs = torch.tensor([])
    for i in range(len(Anchors)):
        M_matrixs = torch.cat((M_matrixs, torch_cov(S - Anchors[i])[0] / torch.tensor(len(S) - 1)))
        # M_matrixs = torch.cat((M_matrixs,torch.inverse(torch_cov(S-Anchors[i])[0]/torch.tensor(len(S)-1).to(device, dtype))))
    return M_matrixs.reshape((-1, S.shape[1], S.shape[1]))

def init_locs_randn(S, N1, n_Anchors, seed=1):
    """Fit a Gaussian to the merged data of the two samples and draw
    n_test_locs points from the Gaussian"""
    # set the seed
    rand_state = np.random.get_state()
    np.random.seed(seed)

    X = S[:N1]
    Y = S[N1:]
    d = X.shape[1]
    # fit a Gaussian in the middle of X, Y and draw sample to initialize T
    xy = np.vstack((X, Y))
    mean_xy = np.mean(xy, 0)
    cov_xy = np.cov(xy.T)
    [Dxy, Vxy] = np.linalg.eig(cov_xy + 1e-3 * np.eye(d))
    Dxy = np.real(Dxy)
    Vxy = np.real(Vxy)
    Dxy[Dxy <= 0] = 1e-3
    eig_pow = 0.9  # 1.0 = not shrink
    reduced_cov_xy = Vxy.dot(np.diag(Dxy ** eig_pow)).dot(Vxy.T) + 1e-3 * np.eye(d)

    T0 = np.random.multivariate_normal(mean_xy, reduced_cov_xy, n_Anchors)
    # reset the seed back to the original
    np.random.set_state(rand_state)
    return T0

def init_locs_2randn(S, N1, n_Anchors, seed=1):
    """Fit a Gaussian to each dataset and draw half of n_test_locs from
    each. This way of initialization can be expensive if the input
    dimension is large.
    """
    np.random.seed(seed)
    if n_Anchors == 1:
        return init_locs_randn(S, N1, n_Anchors, seed)

    # Assuming S is a MindSpore Tensor here. If not, you will need to convert it.
    X = S[:N1]
    Y = S[N1:]
    d = X.shape[1]

    # fit a Gaussian to each of X, Y
    mean_x = mnp.mean(X, 0)
    mean_y = mnp.mean(Y, 0)
    cov_x = mnp.cov(X.T)
    
    # Eigen-decomposition, check if there's an equivalent in MindSpore
    cov_x_np = cov_x.asnumpy()  # Convert to numpy array for processing
    Dx, Vx = np.linalg.eig(cov_x_np + 1e-3 * np.eye(d))
    Dx = np.real(Dx)
    Vx = np.real(Vx)
    # Convert back to MindSpore tensors
    Dx = Tensor(Dx, dtype=mstype.float32)
    Vx = Tensor(Vx, dtype=mstype.float32)
    
    # Ensure real parts; MindSpore might handle this differently
   
    Dx[Dx <= 0] = 1e-3

    eig_pow = 0.9
    reduced_cov_x = Vx.dot(mnp.diag(Dx ** eig_pow)).dot(Vx.T) + 1e-3 * mnp.eye(d)
    cov_y = mnp.cov(Y.T)
    cov_y_np = cov_y.asnumpy()
    Dy, Vy = np.linalg.eig(cov_y_np + 1e-3 * np.eye(d))
    Vy = np.real(Vy)
    Dy = np.real(Dy)
    Dy = Tensor(Dy, dtype=mstype.float32)
    Vy = Tensor(Vy, dtype=mstype.float32)
    
    Dy[Dy <= 0] = 1e-3
    reduced_cov_y = Vy.dot(mnp.diag(Dy ** eig_pow).dot(Vy.T)) + 1e-3 * mnp.eye(d)
    
    Jx = old_div(n_Anchors, 2)  # Make sure old_div is replaced or defined for integer division
    Jy = n_Anchors - Jx

    assert Jx + Jy == n_Anchors, 'total test locations is not n_Anchors'
    
    # No direct equivalent in MindSpore for multivariate_normal, might need to implement manually or keep this part in NumPy
    Tx = np.random.multivariate_normal(mean_x.asnumpy(), reduced_cov_x.asnumpy(), Jx)
    Ty = np.random.multivariate_normal(mean_y.asnumpy(), reduced_cov_y.asnumpy(), Jy)
    T0 = np.vstack((Tx, Ty))

    # Convert final result back to Tensor for compatibility with the rest of the code
    return Tensor(T0)

def MatConvert(S, device, dtype):
    """convert the numpy to a torch tensor."""
    S = mindspore.Tensor(S, dtype=dtype)
    return S
def torch_cov(input_vec):
    u = torch.mean(input_vec, 0)
    x = input_vec - u
    cov_matrix = torch.matmul(x.t(), x)
    return cov_matrix, u

def torch_cov2(input_vec):
    # Set device target to CPU (or "GPU" if you're working on a GPU)
    

    # Check if input_vec is a MindSpore Tensor; if not, and it's a PyTorch tensor, convert it.
    if not isinstance(input_vec, Tensor) and isinstance(input_vec, torch.Tensor):
        input_vec = Tensor(input_vec.cpu().numpy())

    # Calculate the mean along the first axis (dim=0 in PyTorch)
    u = mnp.mean(input_vec, axis=0)

    # Subtract the mean from the data
    x = input_vec - u

    # Calculate the covariance matrix; note that this is not divided by (n-1) for an unbiased estimate
    cov_matrix = mnp.matmul(mnp.transpose(x), x)

    # Convert results back to PyTorch tensors before returning, if necessary
   

    return cov_matrix, u
def meddistance(X, subsample=None, mean_on_fail=True):
    """
    Compute the median of pairwise distances of points in the matrix.
    Parameters:
    X : n x d numpy array or MindSpore Tensor
    mean_on_fail: True/False. If True, use the mean when the median distance is 0.
    Return:
    median distance
    """
    if isinstance(X, Tensor):
        X = X.asnumpy()  # Convert MindSpore tensor to numpy array for processing

    if subsample is None:
        D = dist_matrix(X, X)  # Assume dist_matrix is compatible or define a compatible one
        Itri = np.tril_indices(D.shape[0], -1)
        Tri = D[Itri]
        med = np.median(Tri)
        if med <= 0:
            # use the mean
            return np.mean(Tri)
        return med
    else:
        assert subsample > 0
        rand_state = np.random.get_state()
        np.random.seed(9827)
        n = X.shape[0]
        ind = np.random.choice(n, min(subsample, n), replace=False)
        np.random.set_state(rand_state)
        # recursion just one level
        return meddistance(X[ind, :], None, mean_on_fail)

def dist_matrix(X, Y):
    """
    Construct a pairwise Euclidean distance matrix of size X.shape[0] x Y.shape[0]
    """
    sx = np.sum(X ** 2, 1)
    sy = np.sum(Y ** 2, 1)
    D2 = sx[:, np.newaxis] - 2.0 * np.dot(X, Y.T) + sy[np.newaxis, :]
    # to prevent numerical errors from taking sqrt of negative numbers
    D2[D2 < 0] = 0
    D = np.sqrt(D2)
    return D