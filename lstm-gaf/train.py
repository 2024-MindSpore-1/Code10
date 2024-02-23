# -- coding: utf-8 --**
import argparse

from mindspore import Tensor

from mindnlp.modules import Glove, StaticLSTM, Fasttext
from itertools import islice
import mindspore as ms
import mindspore
from mindspore.dataset import GeneratorDataset, transforms
from mindspore import nn, context
from mindnlp.transforms import PadTransform
from mindnlp.transforms.tokenizers import BertTokenizer
from mindnlp.metrics import Accuracy
import os
from mindspore import ops, ms_function
from mindspore.amp import init_status, DynamicLossScaler
from mindspore.train.serialization import save_checkpoint
import pandas as pd
import json
import moxing as mox
from mindspore.communication.management import init
from tqdm.autonotebook import tqdm
import numpy as np
from mindnlp.metrics.utils import _check_onehot_data, _check_shape, _convert_data_type

os.system('pip install /cache/code/LSTM-GAF/mindnlp-0.1.1-py3-none-any.whl')
os.system('pip install regex')
os.system('pip install regex\n')
parser = argparse.ArgumentParser(description='train_bert')
parser.add_argument('--multi_data_url',
                    help='使用数据集时，需要定义的参数',
                    default='[{}]')
parser.add_argument('--pretrain_url',
                    help='使用预训练模型时，需要定义的参数',
                    default='[{}]')
parser.add_argument('--train_url',
                    help='回传结果到启智，需要定义的参数',
                    default='/cache/output')

parser.add_argument('--epoch_size',
                    type=int,
                    default=30,
                    help='Training epochs.')
parser.add_argument('--eval_count',
                    help='count',
                    type=int,
                    default=3)
args, unk = parser.parse_known_args()
res = {}

device_id = int(os.getenv('RANK_ID'))


ms.set_context(mode=ms.GRAPH_MODE, device_target="Ascend")
init()
data_dir = '/cache/data'
model_dir = '/cache/model'
save_dir = '/cache/output'
dir_list = [data_dir, model_dir, save_dir]


def openi_multidataset_to_env(multi_data_url, data_dir, key):
    """
    copy single or multi dataset to training image
    """
    multi_data_json = json.loads(multi_data_url)
    for i in range(len(multi_data_json)):
        path = data_dir + "/" + multi_data_json[i][key + "_name"]
        if os.path.exists(path):
            return
        #    os.makedirs(path)
        try:
            mox.file.copy_parallel(multi_data_json[i][key + "_url"], path)
            if path.endswith('zip'):
                import zipfile
                f = zipfile.ZipFile(path)
                f.extractall(data_dir)
                f.close()
            elif path.endswith('tar'):
                import tarfile
                f = tarfile.TarFile(path)
                f.extractall(data_dir)
                f.close()
            print((os.listdir(data_dir)))
            print("Successfully Download {} to {}".format(multi_data_json[i][key + "_url"], path))
        except Exception as e:
            print('moxing download {} to {} failed: '.format(
                multi_data_json[i][key + "_url"], path) + str(e))
    return


def check_env():
    for dir in dir_list:
        if not os.path.exists(dir):
            os.makedirs(dir, exist_ok=True)
    # download ckpt to model dir
    openi_multidataset_to_env(args.pretrain_url, model_dir, 'model')
    openi_multidataset_to_env(args.multi_data_url, data_dir, 'dataset')


check_env()



def from_pretrained(cls, name='1M', dims=300, special_first=True, **kwargs):
    embeddings = []
    fasttext_file_path = model_dir + "/wiki-news-300d-1M.vec"
    with open(fasttext_file_path, encoding='utf-8') as file:
        for line in islice(file, 1, None):
            _, embedding = line.split(maxsplit=1)
            embeddings.append(np.fromstring(embedding, dtype=np.float32, sep=' '))

    if special_first:
        embeddings.insert(0, np.random.rand(dims))
        embeddings.insert(1, np.zeros((dims,), np.float32))
    else:
        embeddings.append(np.random.rand(dims))
        embeddings.append(np.zeros((dims,), np.float32))

    embeddings = np.array(embeddings).astype(np.float32)

    requires_grad = kwargs.get('requires_grad', True)
    dropout = kwargs.get('dropout', 0.0)
    return cls(Tensor(embeddings), requires_grad, dropout)


embed = from_pretrained(Fasttext, '1M', 300, special_tokens=["<unk>", "<pad>"])
embed_size = 300
lstm_hidden_size = 200
num_layers = 2
dense_hidden_size = 200
num_classes = 8
bidirectional = True
batch_size = 32
from model import SentimentNet

model = SentimentNet(10000,
                     embed_size,
                     200,
                     2,
                     bidirectional,
                     num_classes,
                     embed,
                     batch_size)
from mindspore.ops import functional as F, composite as C, operations as P

_atan_opt = C.MultitypeFuncGraph("atan_opt")
map = C.Map()


# ----GAF-----
@_atan_opt.register("Tensor")
def atan_p(x):
    a = P.Atan()
    return 0.2 * a(x * 10)


class sgd_act(nn.SGD):

    def construct(self, g):
        g = map(_atan_opt, g)
        return super().construct(g)


# --------------



class R8Dataset:
    def __init__(self, path):
        self.path = path
        self._labels, self._text_a = [], []
        self._load()

    def _load(self):
        self.df = pd.read_csv(self.path)
        labels = {'ship': 0,
                  'money-fx': 1,
                  'grain': 2,
                  'acq': 3,
                  'trade': 4,
                  'earn': 5,
                  'crude': 6,
                  'interest': 7}
        self.df['intent'] = [labels[i] for i in self.df['intent']]

    def __getitem__(self, index):
        return self.df['text'][index], self.df['intent'][index]

    def __len__(self):
        return len(self.df)


def process_dataset(source, tokenizer, pad_value, batch_size=32, shuffle=True):
    column_names = ["text_a", "label"]
    rename_columns = ["input_ids", "label"]

    dataset = GeneratorDataset(source, column_names=column_names, shuffle=shuffle)
    # transforms
    pad_op = PadTransform(64, pad_value=pad_value)
    type_cast_op = transforms.TypeCast(mindspore.int32)

    # map dataset
    dataset = dataset.map(operations=[tokenizer, pad_op], input_columns="text_a")
    dataset = dataset.map(operations=[type_cast_op], input_columns="label")
    # rename dataset
    dataset = dataset.rename(input_columns=column_names, output_columns=rename_columns)
    # batch dataset
    dataset = dataset.batch(batch_size)

    return dataset


tokenizer = BertTokenizer.from_pretrained(data_dir)
pad_value = tokenizer.token_to_id('[PAD]')
dataset_train = process_dataset(R8Dataset(data_dir + "/r8/r8-train-stemmed.csv"),
                                tokenizer, pad_value)
dataset_val = process_dataset(R8Dataset(data_dir + "/r8/r8-dev-stemmed.csv"), tokenizer,
                              pad_value)
dataset_test = process_dataset(R8Dataset(data_dir + "/r8/r8-test-stemmed.csv"), tokenizer,
                               pad_value, shuffle=False)

# set bert config and define parameters for training
reslut_record = {'loss': [], 'acc': []}
total = dataset_train.get_dataset_size()
loss_list = []
acc_list = []
# model = BertForSequenceClassification.from_pretrained(model_dir)
# model = auto_mixed_precision(model, 'O1')

if device_id % 2 == 0:
    optimizer = nn.SGD(model.trainable_params(), learning_rate=1e-1, weight_decay=1e-8)  #
else:
    optimizer = sgd_act(model.trainable_params(),
                        learning_rate=1e-1, weight_decay=1e-8)  #

metric = Accuracy()

loss_scaler = DynamicLossScaler(scale_value=2 ** 10, scale_factor=2, scale_window=1000)



def eval_(model, dataset):
    model.set_train(False)
    _correct_num = 0
    _total_num = 0
    with tqdm(total=dataset.get_dataset_size()) as progress:
        progress.set_description('Evaluate')
        for batch_idx, (input_ids, labels) in enumerate(dataset.create_tuple_iterator()):
            model.set_train(False)
            outputs = model(input_ids)
            preds = outputs[0]
            y_pred = _convert_data_type(preds)  # np
            y_true = _convert_data_type(labels)
            if y_pred.shape[1] != 8:
                raise ValueError(f'For `Accuracy.update`, class numbers do not match. Last input '
                                 f'predicted data contain {8} classes, but current '
                                 f'predicted data contain {y_pred.shape[1]} classes. Please check '
                                 f'your predicted value (`preds`).')

            if y_pred.ndim == y_true.ndim and \
                    (_check_onehot_data(y_true) or y_true[0].shape == (1,)):
                y_true = y_true.argmax(axis=1)
            _check_shape(y_pred, y_true, 8)
            indices = y_pred.argmax(axis=1)
            res = (np.equal(indices, y_true) * 1).reshape(-1)
            _correct_num += res.sum()
            _total_num += res.shape[0]
            progress.update(1)
    progress.close()
    print(f'Evaluate Score: {_correct_num / _total_num}')
    return _correct_num / _total_num, _correct_num, _total_num,


def forward_fn(inputs, labels):
    logits = model(inputs, labels=labels)
    loss = logits[0]
    return loss


grad_fn = ops.value_and_grad(forward_fn, None, optimizer.parameters)


@ms_function
def train_step(data, label):
    status = init_status()
    data = ops.depend(data, status)
    loss, grads = grad_fn(data, label)
    loss = ops.depend(loss, optimizer(grads))
    return loss


for epoch in range(100):
    print(f'=========epoch:{epoch}============')
    loss_total = 0
    loss_100 = 0
    cur_step_nums = 0
    for batch_idx, (input_ids, labels) in enumerate(dataset_train.create_tuple_iterator()):
        cur_step_nums += 1
        loss = train_step(input_ids, labels)
        loss_total += loss
        loss_100 += float(loss)
        print(loss)
        if cur_step_nums % 25 == 0:
            loss_list.append(loss_100 / 25)
            loss_100 = 0
    if epoch % 9 == 0:
        save_checkpoint(model, model_dir + f'/mindspore_{epoch}.ckpt')
        re = eval_(model, dataset_val)
        print(f"Accuracy:{re}")
        acc_list.append(re[0])
reslut_record['loss'].append(loss_list)
reslut_record['acc'].append(acc_list)
with open(save_dir + f'/result_{device_id}.json', 'w+') as f:
    json.dump(reslut_record, fp=f, ensure_ascii=False, indent=True)
    f.close()