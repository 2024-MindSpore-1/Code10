import mindspore
import mindspore.numpy as mnp
from mindspore import context, Tensor, nn, ops
from mindspore.common import dtype as mstype
import torch
from utils_Local import Local_OPT, Local_TEST, sample_BLOB
import warnings

warnings.filterwarnings("ignore")

context.set_context(mode=context.GRAPH_MODE, device_target="CPU")  # Set to CPU mode

KK = 10  # Number of test rounds
K = 100  # Number of tests per round
device=torch.device("cpu")
mindspore.common.set_seed(1102)  # Set random seed

Results = mnp.zeros([2], dtype=mstype.float32)  # Initialize results tensor with zeros using MindSpore's numpy module
Middle_results = mnp.zeros([KK, K], dtype=mstype.float32)  # Initialize intermediate results

alpha = 0.05  # Significance level
n_Anchors = 17  # Number of test positions
beta = 2.0
N_epoch = 1000  # Optimization epochs
learning_rate = 0.007
batch_size = 900

check = 1  # check=0 for Type I error; check=1 for testing power
N = 900

for kk in range(KK):
    X_tr, Y_tr = sample_BLOB(N=int(N/2), rs=123 * (kk + 330), check=check)
    X_tr = Tensor(X_tr, dtype=mstype.float32)
    Y_tr = Tensor(Y_tr, dtype=mstype.float32)
    S_tr = ops.Concat(axis=0)((X_tr, Y_tr))

    # Training
    Anchors, gwidths, M_matrixs, Tree, T_level = Local_OPT(S=S_tr, N1=X_tr.shape[0], n_Anchors=n_Anchors, N_epoch=N_epoch, learning_rate=learning_rate, percent=1, split_adjust=1, seed=123 * (kk + 330), device=None, dtype=mstype.float32, batch_size=batch_size)

    for k in range(K):
        X_te, Y_te = sample_BLOB(N=int(N/2), rs=321 * (kk + k + 330), check=check)
        X_te = Tensor(X_te, dtype=mstype.float32)
        Y_te = Tensor(Y_te, dtype=mstype.float32)
        S_te = ops.Concat(axis=0)((X_te, Y_te))

        # Testing
        h = Local_TEST(S=S_te, N1=S_te.shape[0] // 2, Anchors=Anchors, gwidths=gwidths, M_matrixs=M_matrixs, infer_dire=Tree[0][-1], alpha=alpha, beta=beta, device=None, dtype=mstype.float32)
        Middle_results[kk][k] = h

# Calculate results
Results[0] = Middle_results.sum() / float(K * KK)
Results[1] = Middle_results.std()

if check == 0:
    print("Average and standard deviation of Type I error: ", str(Results[0]), str(Results[1]))
else:
    print("Average and standard deviation of test power: ", str(Results[0]), str(Results[1]))
