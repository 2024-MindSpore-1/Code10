import mindspore.nn as nn
import mindspore as ms
import mindspore.ops.composite as C
import mindspore.ops.operations as P
from mindspore import nn
from mindspore.ops import operations as P

class SentimentNet(nn.Cell):
    """Sentiment network structure."""

    def __init__(self,
                 vocab_size,
                 embed_size,
                 num_hiddens,
                 num_layers,
                 bidirectional,
                 num_classes,
                 embed,
                 batch_size):
        super(SentimentNet, self).__init__()
        # Mapp words to vectors
        self.embedding = embed  # nn.Embedding(vocab_size,
        #             embed_size,
        #             embedding_table=weight)
        self.embedding.requires_grad = False
        self.trans = P.Transpose()
        self.perm = (1, 0, 2)

        self.encoder = nn.LSTM(input_size=embed_size,
                               hidden_size=num_hiddens,
                               num_layers=num_layers,
                               has_bias=True,
                               bidirectional=bidirectional,
                               dropout=0.0)
        self.loss = nn.CrossEntropyLoss()
        self.concat = P.Concat(1)
        self.squeeze = P.Squeeze(axis=0)
        if bidirectional:
            self.decoder = nn.Dense(num_hiddens * 4, num_classes)
        else:
            self.decoder = nn.Dense(num_hiddens * 2, num_classes)
        self.num_classes = num_classes

    def construct(self, inputs, labels=None):
        # input：(32,64)
        embeddings = self.embedding(inputs)
        # (32,64,300)
        embeddings = self.trans(embeddings, self.perm)
        # (64,32,300)
        output, _ = self.encoder(embeddings)
        # (64,32,400)
        # states[i] size(64,200)  -> encoding.size(64,400)
        encoding = self.concat((self.squeeze(output[0:1:1]), self.squeeze(output[63:64:1])))
        outputs = self.decoder(encoding)
        if labels is not None:
            loss = self.loss(outputs.view(-1, self.num_classes), labels.view(-1))
            if isinstance(outputs, tuple):
                outputs = (loss,) + outputs
            else:
                outputs = (loss, outputs)
            return outputs
        return (outputs,)
