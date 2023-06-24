import argparse
import os.path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans

from evaluation import eva
from utils import load_data, normalize_weight, elbo_kl_loss
from torch.optim import Adam
from models import MultiGraphAutoEncoder


# ============================ 1.parameters ==========================
# from visulization import plot_loss, plot_tsne


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='acm', help='acm, dblp, amazon_photos, amazon_computers')
parser.add_argument('--train', type=bool, default=False, help='training mode')
parser.add_argument('--model_name', type=str, default='vgmgc_acm.pkl', help='model name')

parser.add_argument('--path', type=str, default='./data/', help='')
parser.add_argument('--order', type=int, default=8, help='aggregation orders')
parser.add_argument('--weight_soft', type=float, default=0.9, help='parameter of p')  # acm=0, dblp=[0.5]
parser.add_argument('--min_belief', type=float, default=0.7, help='the value of minist belief [0.5-1]')
parser.add_argument('--max_belief', type=float, default=0.99, help='the value of minist belief [0.5-1]')
parser.add_argument('--kl_step', type=float, default=5, help='lambda kl')
parser.add_argument('--lam_elbo_kl', type=float, default=1, help='lambda elbo_kl')
parser.add_argument('--temperature', type=float, default=5, help='')

parser.add_argument('--lam_emd', type=float, default=1, help='trade off between global self-attention and gnn')
parser.add_argument('--threshold', type=float, default=.8, help='threshold of Edge generation')
parser.add_argument('--hidden_dim', type=int, default=512, help='lambda consis')  # citeseer=[512] others=default
parser.add_argument('--latent_dim', type=int, default=512, help='lambda consis')  # citeseer=[16] others=default


parser.add_argument('--epoch', type=int, default=10000, help='')
parser.add_argument('--patience', type=int, default=100, help='')
parser.add_argument('--lr', type=float, default=1e-3, help='')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='')
parser.add_argument('--cuda_device', type=int, default=0, help='')
parser.add_argument('--use_cuda', type=bool, default=False, help='')
parser.add_argument('--update_interval', type=int, default=3, help='')
parser.add_argument('--random_seed', type=int, default=2023, help='')
parser.add_argument('--add_graph', type=bool, default=True, help='')


args = parser.parse_args()

train = args.train
dataset = args.dataset
path = args.path
order = args.order
weight_soft = args.weight_soft
min_belief = args.min_belief
max_belief = args.max_belief
kl_step = args.kl_step
kl_max = 1000
lam_emd = args.lam_emd
lam_elbo_kl = args.lam_elbo_kl
threshold = args.threshold

add_graph=args.add_graph
hidden_dim = args.hidden_dim
latent_dim = args.latent_dim
epoch = args.epoch
patience = args.patience
lr = args.lr
weight_decay = args.weight_decay
temprature = args.temperature
cuda_device = args.cuda_device
use_cuda = args.use_cuda
update_interval = args.update_interval
random_seed = args.random_seed

torch.manual_seed(random_seed)

EPS = 1e-10


# ============================ 2.dataset and model preparing ==========================
labels, adjs, features, adjs_labels, feature_labels, shared_feature, shared_feature_label, graph_num = load_data(dataset, path)


adjs = [a.to_sparse() for a in adjs]
adjs_labels = [a.to_sparse() for a in adjs_labels]

class_num = labels.max()+1
feat_dim = [d.shape[1] for d in features]
feat_dim.append(shared_feature.shape[1])
node_num = features[0].shape[0]



print(
    'dataset informations:\n',
    'class_num:{}\n'.format(class_num),
    'graph_num:{}\n'.format(graph_num),
    'feat_dim:{}\n'.format(feat_dim),
    'node_num:{}'.format(features[0].shape[0]),end='\n'
)
for i in range(graph_num):
    print('G^{} edge num:{}'.format(i+1, int(adjs_labels[i].values().sum())))

model = MultiGraphAutoEncoder(feat_dim, hidden_dim, latent_dim, class_num, lam_emd=lam_emd, order=order,
                              view_num=graph_num, temperature=temprature, threshold=threshold)

if use_cuda:
    torch.cuda.set_device(cuda_device)
    torch.cuda.manual_seed(random_seed)
    model = model.cuda()
    adjs = [a.cuda() for a in adjs]
    adjs_labels = [adj_labels.cuda() for adj_labels in adjs_labels]
    features = [f.cuda() for f in features]
    feature_labels = [fl.cuda() for fl in feature_labels]
    shared_feature = shared_feature.cuda()
    shared_feature_label = shared_feature_label.cuda()

device = shared_feature.device
# ------------------------------------------- optimizer -------------------------------
param_ge = []
param_ae = []
param_gg = []
for i in range(graph_num):
    param_ae.append({'params': model.FeatDec[i].parameters()})
    # param_ae.append({'params': model.LatentMap[i].parameters()})
    param_ge.append({'params': model.GraphEnc[i].parameters()})
    param_ae.append({'params': model.cluster_layer[i]})
param_ae.append({'params': model.cluster_layer[graph_num]})
param_gg.append({'params': model.GraphGen.parameters()})
# optimizer_gg = Adam(param_gg, lr=3e-3, weight_decay=1e-7)

optimizer = Adam(param_ge + param_ae + param_gg,
                 lr=lr, weight_decay=weight_decay)
optimizer_gg = Adam(param_gg,lr=lr, weight_decay=weight_decay)

# cluster parameter initiate
y = labels.cpu().numpy()


# ============================ 3.Training ==========================
if train:
    with torch.no_grad():
        model.eval()
        zs = []
        kmeans = KMeans(n_clusters=class_num, n_init=3)
        for i in range(graph_num):
            q, z = model(features[i], shared_feature, adjs[i], view=i)
            zs.append(z)
            y_pred = kmeans.fit_predict(z.data.cpu().numpy())
            y_pred_last = y_pred
            model.cluster_layer[i].data = torch.tensor(kmeans.cluster_centers_).to(device)
            eva(y, y_pred, 'K{}'.format(i))

        z = torch.cat(zs, dim=-1)
        y_pred = kmeans.fit_predict(z.data.cpu().numpy())
        y_pred_last = y_pred
        model.cluster_layer[-1].data = torch.tensor(kmeans.cluster_centers_).to(device)
        nmi, acc, ari, f1 = eva(y, y_pred, 'Kz')
        # print()
        model.train()

    bad_count = 0
    best_loss = 100
    best_acc = 1e-12
    best_nmi = 1e-12
    best_ari = 1e-12
    best_f1 = 1e-12
    best_epoch = 0
    l = 0.0
    best_a = [1e-12 for i in range(graph_num)]
    weights = normalize_weight(best_a)


    for epoch_num in range(epoch):
        model.train()

        zs = []
        x_preds = []
        qs = []
        re_loss = 0.
        consis_loss = 0.
        re_feat_loss = 0.
        kl_loss = 0.
        re_global_feat_loss = 0.
        re_global_adj_loss = 0.
        elbo = 0.
        # ----------------------------- compute reconstruct loss for each view---------------------
        believes = []
        target_probs = []
        attn_sum = 0.
        for v in range(graph_num):
            A_pred, z, q, x_pred, x_global_pred, attn = model(features[v], shared_feature, adjs[v], view=v)
            zs.append(z)
            qs.append(q)
            x_preds.append(x_pred.unsqueeze(0))

            re_loss += F.binary_cross_entropy(A_pred.view(-1), adjs_labels[v].to_dense().view(-1))
            re_feat_loss += F.binary_cross_entropy(x_pred.view(-1), feature_labels[v].view(-1))

            re_global_feat_loss += F.binary_cross_entropy(x_global_pred.view(-1), shared_feature_label.view(-1))
            re_feat_loss += re_global_feat_loss
            if weights[v] < min_belief:
                belief = min_belief
            elif weights[v] > max_belief:
                belief = max_belief
            else:
                belief = weights[v]
            believes.append(belief)
            target_prob = belief * adjs_labels[v].to_dense() + (1 - belief) * (
                        1 - adjs_labels[v].to_dense() + torch.eye(node_num, device=device))
            target_probs.append(target_prob)

            attn_sum += attn

        attn_avg = attn_sum / graph_num
        attn = model.GraphGen.reparameter(attn_avg, hard=False, training=True)
        KL = elbo_kl_loss(attn, believes, target_probs)
        elbo += (KL + re_loss)

        # ------------------------------- weight assignment with pseudo labels ---------------------------------------
        with torch.no_grad():
            h_prim = torch.cat([zs[i] * weights[i] for i in range(graph_num)], dim=-1).detach()
            kmeans = KMeans(n_clusters=class_num, n_init=3)
            y_prim = kmeans.fit_predict(h_prim.cpu().numpy())
            for v in range(graph_num):
                y_pred = kmeans.fit_predict(zs[v].detach().cpu().numpy())
                a = eva(y_prim, y_pred, visible=False, metrics='nmi')
                best_a[v] = a
            weights = normalize_weight(best_a, p=weight_soft)

        # ---------------------------------------- kl loss------------------------------------
        h = torch.cat([zs[i] * weights[i] for i in range(graph_num)], dim=-1)

        qh = model.predict_distribution(h, -1)
        p = model.target_distribution(qh)
        kl_loss += F.kl_div(qh.log(), p, reduction='batchmean')
        for i in range(graph_num):
            kl_loss += F.kl_div(qs[i].log(), p, reduction='batchmean')
        if l < kl_max:
            l = kl_step * epoch_num
        else:
            l = kl_max
        kl_loss *= l

        loss = re_feat_loss + lam_elbo_kl * elbo + kl_loss


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # ============================ 4.evaluation ==========================
        if epoch_num % update_interval == 0:
            model.eval()
            with torch.no_grad():
                zs = []
                qs = []
                q = 0.
                for v in range(graph_num):
                    tmp_q, z = model(features[v], shared_feature, adjs[v], view=v)
                    zs.append(z)
                    qs.append(tmp_q)

            z = torch.cat([zs[i] * weights[i] for i in range(len(zs))], dim=-1)
            kmeans = KMeans(n_clusters=class_num, n_init=10)
            res2 = kmeans.fit_predict(z.data.cpu().numpy())
            nmi, acc, ari, f1 = eva(y, res2, str(epoch_num) + 'Kz')

            print('acc:{},  nmi:{},  ari:{},  f1:{}, loss:{}, bestepoch:{}'.format(
                acc, nmi, ari, f1, loss, best_epoch))

            # for i in range(graph_num):
            #     res1 = kmeans.fit_predict(zs[i].data.cpu().numpy())
            #     n, a, _, _ = eva(y, res1, str(epoch_num) + 'K'+str(i))

            print(weights)
            model.train()


    # ======================================= 5. postprocess ======================================

        print('Epoch:{}'.format(epoch_num),
              'bad_count:{}'.format(bad_count),
              'kl:{:.4f}'.format(kl_loss),
              're_feat:{:.4f}'.format(re_feat_loss.item()),
              'elbo_kl:{:.4f}'.format(elbo.item()),
              're_global_feat:{:.4f}'.format(re_global_feat_loss.item()),
              end='\n')

        if acc > best_acc:
            if os.path.exists('./pkl/vgmgc_{}_acc{:.4f}.pkl'.format(dataset, best_acc)):
                os.remove('./pkl/vgmgc_{}_acc{:.4f}.pkl'.format(dataset, best_acc))
            best_acc = acc
            best_nmi = nmi
            best_ari = ari
            best_f1 = f1
            best_epoch = epoch_num
            if loss < best_loss:
                best_loss = loss
            torch.save({'state_dict':model.state_dict(),
                        'weights': weights}, './pkl/vgmgc_{}_acc{:.4f}.pkl'.format(dataset, best_acc,))
            bad_count = 0
        else:
            bad_count += 1

        print('best acc:{}, best nmi:{}, best ari:{}, best f1:{},best loss:{}, bestepoch:{}'.format(
             best_acc, best_nmi, best_ari, best_f1, best_loss, best_epoch))
        print()

        if bad_count >= patience:
            print('complete training, best acc:{}, best nmi:{}, best ari:{}, best f1:{},best loss:{}, bestepoch:{}'.format(
                best_acc, best_nmi, best_ari, best_f1, best_loss.item(), best_epoch))
            break


# ============================================== Test =====================================================
if not train:
    model_name = args.model_name
else:
    model_name = 'vgmgc_{}_acc{:.4f}.pkl'.format(dataset, best_acc)
# print('Loading model:{}...'.format(model_name))
best_model = torch.load('./pkl/'+model_name, map_location=features[0].device)
weights = best_model['weights']
print('weights,'+ args.dataset + str(weights))
state_dict = best_model['state_dict']
model.load_state_dict(state_dict)
print('Evaluating....')

model.eval()
with torch.no_grad():
    zs = []
    qs = []
    q = 0.
    for v in range(graph_num):
        tmp_q, z = model(features[v], shared_feature, adjs[v], view=v)
        zs.append(z)
        qs.append(tmp_q)

z = torch.cat([zs[i] * weights[i] for i in range(graph_num)], dim=-1)
kmeans = KMeans(n_clusters=class_num, n_init=100)
res2 = kmeans.fit_predict(z.data.cpu().numpy())
nmi, acc, ari, f1 = eva(y, res2, str('eva:') + 'Kz')

for i in range(graph_num):
    res1 = kmeans.fit_predict(zs[i].data.cpu().numpy())
    eva(y, res1, str('eva:') + 'K' + str(i), visible=True)

print('Results: acc:{},  nmi:{},  ari:{},  f1:{}, '.format(
            acc, nmi, ari, f1))

