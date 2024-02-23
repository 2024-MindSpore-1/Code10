import mindspore
from mindspore import context, Tensor, nn, ops
from mindspore.common import dtype as mstype
from mindspore.ops import operations as P
from utils_Local import Local_OPT, density_estimate_test, sample_BLOB, explore_regions, TEST_density_diff
import warnings

warnings.filterwarnings("ignore")

context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

check = 1
KK = 10
K = 10
alpha = 0.05
beta = 2.0

mindspore.ops.seed(1102)

Results = Tensor([0, 0], dtype=mstype.float32)
Middle_results = Tensor.zeros([KK, K], mstype.float32)

N = 10000
min_percent = 0.015625
check_percent = 0.0625
split_adjust = 1.2

N_epoch = 1000
learning_rate = 0.007

n_Anchors = 17
batch_size = 10000
NUM = int(check_percent / min_percent)

X, Y = sample_BLOB(N=500000, rs=123, check=check)
S = P.Concat(0)((X, Y))
density1, density2 = density_estimate_test(S, 500000)

for kk in range(KK):
    X_tr, Y_tr = sample_BLOB(N=int(N/2), rs=123 * kk + 3, check=check)

    # Train
    Anchors, gwidths, M_matrixs, Tree, T_level = Local_OPT(P.Concat(0)((X_tr, Y_tr)), X_tr.shape[0], n_Anchors, N_epoch, learning_rate, min_percent, split_adjust, 123 * kk + 3, None, mstype.float32, batch_size=batch_size)

    for k in range(K):
        X_te, Y_te = sample_BLOB(N=int(N/2), rs=321 * kk + 110, check=check)
        DIFF_idx = explore_regions(P.Concat(0)((X_te, Y_te)), X_te.shape[0], Anchors, gwidths, M_matrixs, T_level, Tree.copy(), alpha, beta, split_adjust, None, mstype.float32, 20)

        mean_diffs_ours = TEST_density_diff(S, 500000, Anchors, gwidths, M_matrixs, T_level, Tree.copy(), DIFF_idx[:NUM], split_adjust, density1, density2, None, mstype.float32)

        Middle_results = P.Assign()(Middle_results[kk], P.Insert()(Middle_results[kk], k, mean_diffs_ours))

Results = P.Assign()(Results, P.Insert()(Results, 0, P.ReduceSum()(P.Div()(P.ReduceSum()(Middle_results, 1), float(K))) / float(KK)))
Results = P.Assign()(Results, P.Insert()(Results, 1, P.Sqrt()(P.ReduceMean()(P.Square()(P.Sub()(Middle_results, P.ReduceMean()(Middle_results, 1)))))))

print("Mean and std of DIFFS: ", str(Results[0].asnumpy()), str(Results[1].asnumpy()))
