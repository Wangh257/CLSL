import torch
from torch import nn
from .alias_multinomial import AliasMethod
import math
import numpy as np

def swap_rows_with_target_tensor(batchSize, K, y):
    """
    为每行生成唯一的随机索引 (0-1699)，并将第0列与目标值所在位置的元素交换。
    最终结果转为 PyTorch Tensor 并放置在指定设备上。
    
    :param batchSize: 批量大小
    :param K: 每行的列数 - 1
    :param y: 每一行的目标值 (NumPy 数组或 Tensor)
    :param device: 目标设备 ("cuda" 或 "cpu")
    :return: 调整后的索引矩阵 (Tensor)
    """
    # 每行生成从 0 到 1699 的随机排列
    data = np.array([np.random.permutation(K + 1) for _ in range(batchSize)])
    for i in range(batchSize):
        # 找到目标值 y[i] 的位置
        # import pdb;pdb.set_trace()
        y_pos = np.where(data[i] == y[i])[0][0]
        # 交换第 0 列和目标值所在位置
        data[i, 0], data[i, y_pos] = data[i, y_pos], data[i, 0]
    return data

class Feature_Dict(nn.Module):
    def __init__(self, feature_dim, data_size, K, T=0.07, momentum=0.5, use_softmax=False):
        super(Feature_Dict, self).__init__()
        self.nLem = data_size
        self.unigrams = torch.ones(self.nLem)
        self.multinomial = AliasMethod(self.unigrams)
        self.multinomial.cuda()
        self.K = data_size - 1
        self.use_softmax = use_softmax

        self.register_buffer('params', torch.tensor([K, T, -1, -1, momentum]))
        stdv = 1. / math.sqrt(feature_dim / 3)
        self.register_buffer('memory_fringe', torch.rand(data_size, feature_dim).mul_(2 * stdv).add_(-stdv))
        self.register_buffer('memory_phase', torch.rand(data_size, feature_dim).mul_(2 * stdv).add_(-stdv))

    def forward(self, fea_f, fea_p, y, idx=None):
        K = int(self.params[0].item())
        T = self.params[1].item()

        Z_l = self.params[2].item()
        Z_ab = self.params[3].item()

        momentum = self.params[4].item()
        batchSize = fea_f.size(0)
        data_size = self.memory_fringe.size(0)
        feature_size = self.memory_fringe.size(1)
        # import pdb;pdb.set_trace()
        # score computation
        if idx is None:
            idx = swap_rows_with_target_tensor(batchSize, self.K, y.cpu().numpy())
            idx = torch.tensor(idx, dtype=torch.long).cuda()
            # idx = self.multinomial.draw(batchSize * (self.K + 1)).view(batchSize, -1)
            # idx.select(1, 0).copy_(y.data)

        # import pdb;pdb.set_trace()
        # sample
        weight_fringe = torch.index_select(self.memory_fringe, 0, idx.view(-1)).detach()
        weight_fringe = weight_fringe.view(batchSize, self.K + 1, feature_size)
        out_phase = torch.bmm(weight_fringe, fea_p.view(batchSize, feature_size, 1))

        # sample
        weight_phase = torch.index_select(self.memory_phase, 0, idx.view(-1)).detach()
        weight_phase = weight_phase.view(batchSize, self.K + 1, feature_size)
        # weight_phase_norm = weight_phase.norm(dim=2, keepdim=True)
        # weight_phase_normlized = weight_phase.div(weight_phase_norm)
        out_fringe = torch.bmm(weight_phase, fea_f.view(batchSize, feature_size, 1))

        if self.use_softmax:
            out_fringe = torch.div(out_fringe, T)
            out_phase = torch.div(out_phase, T)
            out_phase = out_phase.contiguous()
            out_fringe = out_fringe.contiguous()
        else:
            out_phase = torch.exp(torch.div(out_phase, T))
            out_fringe = torch.exp(torch.div(out_fringe, T))
            # set Z_0 if haven't been set yet,
            # Z_0 is used as a constant approximation of Z, to scale the probs
            if Z_l < 0:
                self.params[2] = out_fringe.mean() * data_size
                Z_l = self.params[2].clone().detach().item()
                print("normalization constant Z_l is set to {:.1f}".format(Z_l))
            if Z_ab < 0:
                self.params[3] = out_phase.mean() * data_size
                Z_ab = self.params[3].clone().detach().item()
                print("normalization constant Z_ab is set to {:.1f}".format(Z_ab))
            # compute out_l, out_ab
            out_fringe = torch.div(out_fringe, Z_l).contiguous()
            out_phase = torch.div(out_phase, Z_ab).contiguous()

        # # update memory
        with torch.no_grad():
            f_pos = torch.index_select(self.memory_fringe, 0, y.view(-1)) # Batch_size * feature_size
            f_pos.mul_(momentum)
            f_pos.add_(torch.mul(fea_f, 1 - momentum))
            f_norm = f_pos.norm(dim=1, keepdim=True)
            # l_norm = l_pos.pow(2).sum(1, keepdim=True).pow(0.5)
            updated_f = f_pos.div(f_norm)
            self.memory_fringe.index_copy_(0, y, updated_f)

            p_pos = torch.index_select(self.memory_phase, 0, y.view(-1))
            p_pos.mul_(momentum)
            p_pos.add_(torch.mul(fea_p, 1 - momentum))
            p_norm = p_pos.norm(dim=1, keepdim=True)
            # p_norm = p_pos.pow(2).sum(1, keepdim=True).pow(0.5)
            updated_p = p_pos.div(p_norm)
            self.memory_phase.index_copy_(0, y, updated_p)

        return out_fringe, out_phase

class Feature_Dict_Singel_Encoder(nn.Module):
    def __init__(self, feature_dim, data_size, K, T=0.07, momentum=0.5, use_softmax=False):
        super(Feature_Dict_Singel_Encoder, self).__init__()
        self.nLem = data_size
        self.unigrams = torch.ones(self.nLem)
        self.multinomial = AliasMethod(self.unigrams)
        self.multinomial.cuda()
        self.K = data_size - 1
        self.use_softmax = use_softmax
        
        self.register_buffer('params', torch.tensor([K, T, -1, -1, momentum]))
        stdv = 1. / math.sqrt(feature_dim / 3)
        self.register_buffer('memory_fringe', torch.rand(data_size, feature_dim).mul_(2 * stdv).add_(-stdv))
        self.register_buffer('memory_fenzi', torch.rand(data_size, feature_dim).mul_(2 * stdv).add_(-stdv))
        self.register_buffer('memory_fenmu', torch.rand(data_size, feature_dim).mul_(2 * stdv).add_(-stdv))

    def forward(self, fea_f, fea_fenzi, fea_fenmu, y, idx=None):
        K = int(self.params[0].item())
        T = self.params[1].item()

        Z_l = self.params[2].item()
        Z_ab = self.params[3].item()

        momentum = self.params[4].item()
        batchSize = fea_f.size(0)
        data_size = self.memory_fringe.size(0)
        feature_size = self.memory_fringe.size(1)
        # import pdb;pdb.set_trace()
        # score computation
        if idx is None:
            idx = swap_rows_with_target_tensor(batchSize, self.K, y.cpu().numpy())
            idx = torch.tensor(idx, dtype=torch.long).cuda()

        import pdb;pdb.set_trace()
        # sample  fringe
        weight_fenzi = torch.index_select(self.memory_fenzi, 0, idx.view(-1)).detach()
        weight_fenzi = weight_fenzi.view(batchSize, self.K + 1, feature_size)
        f_fenzi = torch.bmm(weight_fenzi, fea_f.view(batchSize, feature_size, 1))

        weight_fenmu = torch.index_select(self.memory_fenmu, 0, idx.view(-1)).detach()
        weight_fenmu = weight_fenmu.view(batchSize, self.K + 1, feature_size)
        f_fenmu = torch.bmm(weight_fenmu, fea_f.view(batchSize, feature_size, 1))

        # sample  fenzi
        weight_fringe = torch.index_select(self.memory_fringe, 0, idx.view(-1)).detach()
        weight_fringe = weight_fringe.view(batchSize, self.K + 1, feature_size)
        fenzi_f = torch.bmm(weight_fringe, fea_fenzi.view(batchSize, feature_size, 1))
        fenzi_fenmu = torch.bmm(weight_fenmu, fea_fenzi.view(batchSize, feature_size, 1))

        # sample fenmu
        fenmu_f = torch.bmm(weight_fringe, fea_fenmu.view(batchSize, feature_size, 1))
        fenmu_fenzi = torch.bmm(weight_fenzi, fea_fenmu.view(batchSize, feature_size, 1))
        
        if self.use_softmax:
            f_fenzi = torch.div(f_fenzi, T)
            f_fenmu = torch.div(f_fenmu, T)
            fenzi_f = torch.div(fenzi_f, T)
            fenzi_fenmu = torch.div(fenzi_fenmu, T)
            fenmu_f = torch.div(fenmu_f, T)
            fenmu_fenzi = torch.div(fenmu_fenzi, T)

            f_fenzi = f_fenzi.contiguous()
            f_fenmu = f_fenmu.contiguous()
            fenzi_f = fenzi_f.contiguous()
            fenzi_fenmu = fenzi_fenmu.contiguous()
            fenmu_f = fenmu_f.contiguous()
            fenmu_fenzi = fenmu_fenzi.contiguous()
        
        else:
            out_phase = torch.exp(torch.div(out_phase, T))
            out_fringe = torch.exp(torch.div(out_fringe, T))
            # set Z_0 if haven't been set yet,
            # Z_0 is used as a constant approximation of Z, to scale the probs
            if Z_l < 0:
                self.params[2] = out_fringe.mean() * data_size
                Z_l = self.params[2].clone().detach().item()
                print("normalization constant Z_l is set to {:.1f}".format(Z_l))
            if Z_ab < 0:
                self.params[3] = out_phase.mean() * data_size
                Z_ab = self.params[3].clone().detach().item()
                print("normalization constant Z_ab is set to {:.1f}".format(Z_ab))
            # compute out_l, out_ab
            out_fringe = torch.div(out_fringe, Z_l).contiguous()
            out_phase = torch.div(out_phase, Z_ab).contiguous()

        # # update memory
        with torch.no_grad():
            f_pos = torch.index_select(self.memory_fringe, 0, y.view(-1)) # Batch_size * feature_size
            f_pos.mul_(momentum)
            f_pos.add_(torch.mul(fea_f, 1 - momentum))
            f_norm = f_pos.norm(dim=1, keepdim=True)
            # l_norm = l_pos.pow(2).sum(1, keepdim=True).pow(0.5)
            updated_f = f_pos.div(f_norm)
            # import pdb;pdb.set_trace()
            self.memory_fringe.index_copy_(0, y, updated_f)

            fenzi_pos = torch.index_select(self.memory_fenzi, 0, y.view(-1))
            fenzi_pos.mul_(momentum)
            fenzi_pos.add_(torch.mul(fea_fenzi, 1 - momentum))
            fenzi_norm = fenzi_pos.norm(dim=1, keepdim=True)
            # p_norm = p_pos.pow(2).sum(1, keepdim=True).pow(0.5)
            updated_fenzi = fenzi_pos.div(fenzi_norm)
            self.memory_fenzi.index_copy_(0, y, updated_fenzi)

            fenmu_pos = torch.index_select(self.memory_fenmu, 0, y.view(-1))
            fenmu_pos.mul_(momentum)
            fenmu_pos.add_(torch.mul(fea_fenmu, 1 - momentum))
            fenmu_norm = fenmu_pos.norm(dim=1, keepdim=True)
            # p_norm = p_pos.pow(2).sum(1, keepdim=True).pow(0.5)
            updated_fenmu = fenmu_pos.div(fenmu_norm)
            self.memory_fenmu.index_copy_(0, y, updated_fenmu)

        return f_fenzi, f_fenmu, fenzi_f, fenzi_fenmu, fenmu_f, fenmu_fenzi

class NCEAverage(nn.Module):

    def __init__(self, inputSize, outputSize, K, T=0.07, momentum=0.5, use_softmax=False):
        super(NCEAverage, self).__init__()
        self.nLem = outputSize
        self.unigrams = torch.ones(self.nLem)
        self.multinomial = AliasMethod(self.unigrams)
        self.multinomial.cuda()
        self.K = K
        self.use_softmax = use_softmax

        self.register_buffer('params', torch.tensor([K, T, -1, -1, momentum]))
        stdv = 1. / math.sqrt(inputSize / 3)
        self.register_buffer('memory_l', torch.rand(outputSize, inputSize).mul_(2 * stdv).add_(-stdv))
        self.register_buffer('memory_ab', torch.rand(outputSize, inputSize).mul_(2 * stdv).add_(-stdv))

    def forward(self, l, ab, y, idx=None):
        K = int(self.params[0].item())
        T = self.params[1].item()
        Z_l = self.params[2].item()
        Z_ab = self.params[3].item()

        momentum = self.params[4].item()
        batchSize = l.size(0)
        outputSize = self.memory_l.size(0)
        inputSize = self.memory_l.size(1)

        # score computation
        if idx is None:
            idx = self.multinomial.draw(batchSize * (self.K + 1)).view(batchSize, -1)
            idx.select(1, 0).copy_(y.data)
        # sample
        weight_l = torch.index_select(self.memory_l, 0, idx.view(-1)).detach()
        weight_l = weight_l.view(batchSize, K + 1, inputSize)
        out_ab = torch.bmm(weight_l, ab.view(batchSize, inputSize, 1))
        # sample
        weight_ab = torch.index_select(self.memory_ab, 0, idx.view(-1)).detach()
        weight_ab = weight_ab.view(batchSize, K + 1, inputSize)
        out_l = torch.bmm(weight_ab, l.view(batchSize, inputSize, 1))

        if self.use_softmax:
            out_ab = torch.div(out_ab, T)
            out_l = torch.div(out_l, T)
            out_l = out_l.contiguous()
            out_ab = out_ab.contiguous()
        else:
            out_ab = torch.exp(torch.div(out_ab, T))
            out_l = torch.exp(torch.div(out_l, T))
            # set Z_0 if haven't been set yet,
            # Z_0 is used as a constant approximation of Z, to scale the probs
            if Z_l < 0:
                self.params[2] = out_l.mean() * outputSize
                Z_l = self.params[2].clone().detach().item()
                print("normalization constant Z_l is set to {:.1f}".format(Z_l))
            if Z_ab < 0:
                self.params[3] = out_ab.mean() * outputSize
                Z_ab = self.params[3].clone().detach().item()
                print("normalization constant Z_ab is set to {:.1f}".format(Z_ab))
            # compute out_l, out_ab
            out_l = torch.div(out_l, Z_l).contiguous()
            out_ab = torch.div(out_ab, Z_ab).contiguous()

        # # update memory
        with torch.no_grad():
            l_pos = torch.index_select(self.memory_l, 0, y.view(-1))
            l_pos.mul_(momentum)
            l_pos.add_(torch.mul(l, 1 - momentum))
            l_norm = l_pos.pow(2).sum(1, keepdim=True).pow(0.5)
            updated_l = l_pos.div(l_norm)
            self.memory_l.index_copy_(0, y, updated_l)

            ab_pos = torch.index_select(self.memory_ab, 0, y.view(-1))
            ab_pos.mul_(momentum)
            ab_pos.add_(torch.mul(ab, 1 - momentum))
            ab_norm = ab_pos.pow(2).sum(1, keepdim=True).pow(0.5)
            updated_ab = ab_pos.div(ab_norm)
            self.memory_ab.index_copy_(0, y, updated_ab)

        return out_l, out_ab


# =========================
# InsDis and MoCo
# =========================

class MemoryInsDis(nn.Module):
    """Memory bank with instance discrimination"""
    def __init__(self, inputSize, outputSize, K, T=0.07, momentum=0.5, use_softmax=False):
        super(MemoryInsDis, self).__init__()
        self.nLem = outputSize
        self.unigrams = torch.ones(self.nLem)
        self.multinomial = AliasMethod(self.unigrams)
        self.multinomial.cuda()
        self.K = K
        self.use_softmax = use_softmax

        self.register_buffer('params', torch.tensor([K, T, -1, momentum]))
        stdv = 1. / math.sqrt(inputSize / 3)
        self.register_buffer('memory', torch.rand(outputSize, inputSize).mul_(2 * stdv).add_(-stdv))

    def forward(self, x, y, idx=None):
        K = int(self.params[0].item())
        T = self.params[1].item()
        Z = self.params[2].item()
        momentum = self.params[3].item()

        batchSize = x.size(0)
        outputSize = self.memory.size(0)
        inputSize = self.memory.size(1)

        # score computation
        if idx is None:
            idx = self.multinomial.draw(batchSize * (self.K + 1)).view(batchSize, -1)
            idx.select(1, 0).copy_(y.data)

        # sample
        weight = torch.index_select(self.memory, 0, idx.view(-1))
        weight = weight.view(batchSize, K + 1, inputSize)
        out = torch.bmm(weight, x.view(batchSize, inputSize, 1))

        if self.use_softmax:
            out = torch.div(out, T)
            out = out.squeeze().contiguous()
        else:
            out = torch.exp(torch.div(out, T))
            if Z < 0:
                self.params[2] = out.mean() * outputSize
                Z = self.params[2].clone().detach().item()
                print("normalization constant Z is set to {:.1f}".format(Z))
            # compute the out
            out = torch.div(out, Z).squeeze().contiguous()

        # # update memory
        with torch.no_grad():
            weight_pos = torch.index_select(self.memory, 0, y.view(-1))
            weight_pos.mul_(momentum)
            weight_pos.add_(torch.mul(x, 1 - momentum))
            weight_norm = weight_pos.pow(2).sum(1, keepdim=True).pow(0.5)
            updated_weight = weight_pos.div(weight_norm)
            self.memory.index_copy_(0, y, updated_weight)

        return out


class MemoryMoCo(nn.Module):
    """Fixed-size queue with momentum encoder"""
    def __init__(self, inputSize, outputSize, K, T=0.07, use_softmax=False):
        super(MemoryMoCo, self).__init__()
        self.outputSize = outputSize
        self.inputSize = inputSize
        self.queueSize = K
        self.T = T
        self.index = 0
        self.use_softmax = use_softmax

        self.register_buffer('params', torch.tensor([-1]))
        stdv = 1. / math.sqrt(inputSize / 3)
        self.register_buffer('memory', torch.rand(self.queueSize, inputSize).mul_(2 * stdv).add_(-stdv))
        print('using queue shape: ({},{})'.format(self.queueSize, inputSize))

    def forward(self, q, k):
        batchSize = q.shape[0]
        k = k.detach()

        Z = self.params[0].item()

        # pos logit
        l_pos = torch.bmm(q.view(batchSize, 1, -1), k.view(batchSize, -1, 1))
        l_pos = l_pos.view(batchSize, 1)
        # neg logit
        queue = self.memory.clone()
        l_neg = torch.mm(queue.detach(), q.transpose(1, 0))
        l_neg = l_neg.transpose(0, 1)

        out = torch.cat((l_pos, l_neg), dim=1)

        if self.use_softmax:
            out = torch.div(out, self.T)
            out = out.squeeze().contiguous()
        else:
            out = torch.exp(torch.div(out, self.T))
            if Z < 0:
                self.params[0] = out.mean() * self.outputSize
                Z = self.params[0].clone().detach().item()
                print("normalization constant Z is set to {:.1f}".format(Z))
            # compute the out
            out = torch.div(out, Z).squeeze().contiguous()

        # # update memory
        with torch.no_grad():
            out_ids = torch.arange(batchSize).cuda()
            out_ids += self.index
            out_ids = torch.fmod(out_ids, self.queueSize)
            out_ids = out_ids.long()
            self.memory.index_copy_(0, out_ids, k)
            self.index = (self.index + batchSize) % self.queueSize

        return out
