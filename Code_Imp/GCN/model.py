import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import os
from torch.nn.parameter import Parameter
from torch import optim
import math

class GraphConvolution(nn.Module):
    def __init__(self,features_num,hidden_size):
        super(GraphConvolution,self).__init__()
        self.w = Parameter(torch.FloatTensor(features_num,hidden_size))
        self.b = Parameter(torch.FloatTensor(hidden_size))
        stdv = 1. / math.sqrt(self.w.size(1))
        self.w.data.uniform_(-stdv,stdv)
        self.b.data.uniform_(-stdv,stdv)
    
    def forward(self, x,adj):
        x = torch.mm(x,self.w)
        output = torch.spmm(adj,x)#spmm是稀疏矩阵乘法
        return output + self.b

class GCN(nn.Module):
    def __init__(self,input_size,hidden_size,output_size,dropout):
        super(GCN,self).__init__()
        self.gc1 = GraphConvolution(input_size,hidden_size)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.gc2 = GraphConvolution(hidden_size,output_size)
       
    def forward(self,x,adj):
        x = self.gc1(x,adj)
        x = self.relu(x)
        x = self.drop(x)
        x = self.gc2(x,adj)
        return x


def build_symmetric_adj(edges,node_num):
    adj = np.zeros((node_num,node_num))
    for i,j in edges:
        adj[i,j] = 1
        adj[j,i] = 1
    for i in range(node_num):
        adj[i,i]=1
    return adj

def load_cora_data(data_path):
    print("Loading Cora dataset...")
    content = np.genfromtxt(data_path+'/cora.content' ,dtype=np.dtype(str))
    idx = content[:,0].astype(np.int32)#
    features = content[:,1:-1].astype(np.float32)#2708*1433
    labels = encode_labels(content[:,-1])#2708
    node_num  = len(idx)
    cites = np.genfromtxt(data_path+'/cora.cites' ,dtype=np.int32)
    idx_map = {j:i for i,j in enumerate(idx)}
    edges = [(idx_map[i],idx_map[j]) for i,j in cites]#5429*2
    edges = np.array(edges,dtype=np.int32)
    adj = build_symmetric_adj(edges,node_num)
    print(f"adj;{adj.shape}")
    adj = normalize(adj)
    features = torch.FloatTensor(features)
    labels = torch.LongTensor(labels)
    adj = adj_to_sparse_tensor(adj)
    return features,labels,adj

 
# load_cora_data("D:\deep learning\Code_Imp\GCN\cora")

def encode_labels(labels):
    classes = sorted(set(labels))
    label2index = {label : idx for idx,label in enumerate(classes)}#标签到索引的映射
    indices = [label2index[label] for label in labels]
    indices = np.array(indices)
    return indices

def normalize(mx):
    degree = np.sum(mx,axis = 1)
    d_inv_sqrt = np.power(degree,-0.5).flatten()#展成一维数组
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_inv_sqrt_mat = np.diag(d_inv_sqrt)
    mx = d_inv_sqrt_mat @ mx @ d_inv_sqrt_mat
    return mx

def adj_to_sparse_tensor(adj):
    adj = torch.FloatTensor(adj)
    adj = adj.to_sparse()
    return adj

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    features,labels,adj = load_cora_data("D:\deep learning\Code_Imp\GCN\cora")
    print(features.shape,labels.shape,adj.shape)
    features = features.to(device)
    labels = labels.to(device)
    adj = adj.to(device)
    assert len(features) == len(labels)#断言语句 确保数据的一致性 监督学习或图神经网络的训练过程中，输入和标签的长度必须一致
    assert len(features) == len(adj)
    sample_num = len(features)
    train_num = int(sample_num*0.15)
    test_num = sample_num - train_num
    feature_num = features.shape[1]
    hidden_size = 16
    class_num = labels.max().item()+1
    dropout = 0.5
    model = GCN(feature_num,hidden_size,class_num,dropout).to(device)
    model.train()
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    n_epoch = 500
    for epoch in range(1,n_epoch+1):
        optimizer.zero_grad()
        outputs = model(features,adj)
        loss = criterion(outputs[:train_num],labels[:train_num])
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print(f"Epoch:{epoch},Loss:{loss.item():.3f}")
    model.eval()
    outputs = model(features,adj)
    predicted = torch.argmax(outputs[train_num:],dim=1)
    correct = (predicted == labels[train_num:]).sum().item()
    accuracy = 100*correct/test_num
    print(f"Accuracy:{correct/test_num:.3f}%")
