from mindspore.nn import Cell
from mindspore import ops, nn,Parameter,Tensor
import mindspore as ms
import step_lr
from models import model_factory
import sys

class SAM(nn.Optimizer):
    """定义优化器"""
    def __init__(self, params, learning_rate,base_optimizer, rho=0.05,adaptive=False,**kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        super(SAM, self).__init__(learning_rate, params)
        self.state = dict()
        self.adaptive = adaptive
        self.rho = Parameter(Tensor(rho, ms.float32), name="rho")
        self.base_optimizer = base_optimizer(params,learning_rate,**kwargs)

        self.parameters = self.base_optimizer.parameters

    def first_step(self,gradients):
        params = self.parameters
        grad_norm = self._grad_norm(gradients)
        scale = self.rho / (grad_norm + 1e-12)
        pow = ops.Pow()
        for i in range(len(params)):
            if gradients[i] is None:continue
            if params[i].name in self.state:
                self.state[params[i].name].update({"old_p": params[i].value().copy()})
            else:
                self.state.update({params[i].name: {"old_p": params[i].value().copy()}})
            self.state[params[i].name].update({"old_g": gradients[i].copy()})
            e_w = (pow(params[i], 2) if self.adaptive else 1.0)
            e_w = e_w * gradients[i]
            e_w = e_w * scale
            update = params[i]+e_w
            ops.assign(params[i], update)


    def construct(self, gradients):
        params = self.parameters
        gradients = list(gradients)
        for i in range(len(params)):
            if gradients[i] is None:continue
            gradients[i] = self.state[str(params[i].name)]["old_g"]+ gradients[i]
            params[i].set_data(self.state[str(params[i].name)]["old_p"])
            ##value = self.state[str(params[i].name)]["old_p"]
            #ops.assign(gradients[i],self.state[str(params[i].name)]["old_g"]+ gradients[i])
            #ops.assign(params[i],self.state[str(params[i].name)]["old_p"])
        gradients = tuple(gradients)
        success = self.base_optimizer(gradients)
        return success

    def _grad_norm(self, gradients):
        params = self.parameters
        stack = ops.Stack()
        test = []
        for i in range(len(params)):
            if gradients[i] is None: continue
            if self.adaptive:
                value = ops.abs(params[i])
            else:
                value = 1.0
            value = value * gradients[i]
            dims = [i for i in range(len(value.shape))]
            dims = tuple(dims)
            value = ops.norm(value)
            test.append(value)
        test = stack(test)
        test = ops.norm(test)
        return test


# net = model_factory.get_network("resnet18")(num_classes=10)
# # 设置优化器待优化的参数和学习率为0.01
# lr = step_lr.stepLR(learning_rate=0.003,steps_per_epoch=100,total_epochs=100)
# opt = SAM(net.trainable_params(), 0.01,base_optimizer=nn.SGD,momentum=0.9,weight_decay=5e-4)
# test =  opt.parameters
# print(0)