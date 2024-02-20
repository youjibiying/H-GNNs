import os
import time
import copy
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import pprint as pp
from models.model import GCNModel
from datasets import load_feature_construct_H, generate_H_from_dist
# from datasets import source_select
from parsing import train_args
from torch.nn.utils import clip_grad_norm_
from utils import hypergraph_utils as hgut
import scipy.sparse as sp
from datasets import data


# CUDA_LAUNCH_BLOCKING=1

args = train_args()
device = torch.device('cuda', args.gpu) if torch.cuda.is_available() else torch.device('cpu')
# initialize visual object classification data
data_dir = args.modelnet40_ft if args.on_dataset == 'ModelNet40' \
    else args.ntu2012_ft
adj = None

if args.activate_dataset.startswith('coauthorship') or args.activate_dataset.startswith('cocitation'):
    dataset, idx_train, idx_test = data.load(args)
    idx_val = idx_test
    hypergraph, fts, lbls = dataset['hypergraph'], dataset['features'], dataset['labels']
    lbls = np.argmax(lbls, axis=1)
    H = np.zeros((dataset['n'], len(hypergraph)))
    for i, (a, p) in enumerate(hypergraph.items()):
        H[list(p), i] = 1

else:
    fts, lbls, idx_train, idx_test, mvcnn_dist, gvcnn_dist = \
        load_feature_construct_H(data_dir,
                                 gamma=args.gamma,
                                 K_neigs=args.K_neigs,
                                 is_probH=args.is_probH,
                                 use_mvcnn_feature=args.use_mvcnn_feature,
                                 use_gvcnn_feature=args.use_gvcnn_feature,
                                 use_mvcnn_feature_for_structure=args.mvcnn_feature_structure,
                                 use_gvcnn_feature_for_structure=args.gvcnn_feature_structure)
    idx_val = idx_test



n_class = int(lbls.max()) + 1

# transform data to device
fts = torch.Tensor(fts).to(device)  # features -> fts
lbls = torch.Tensor(lbls).squeeze().long().to(device)
idx_train = torch.Tensor(idx_train).long().to(device)
idx_test = torch.Tensor(idx_test).long().to(device)
idx_val = torch.Tensor(idx_val).long().to(device)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

    torch.cuda.current_device()
    torch.cuda._initialized = True

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    # sparse_mx = sp.coo_matrix(sparse_mx)
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape).to(device)

def get_data(args, device):

    if args.activate_dataset.startswith('coauthorship') or args.activate_dataset.startswith('cocitation'):
        dataset, idx_train1, idx_test1 = data.load(args)  # reloading due to different splits.
        idx_val1 = idx_test1
        hypergraph, fts1, lbls1 = dataset['hypergraph'], dataset['features'], dataset['labels']
        lbls1 = np.argmax(lbls1, axis=1)

        H = np.zeros((dataset['n'], len(hypergraph)))
        for i, (a, p) in enumerate(hypergraph.items()):
            H[list(p), i] = 1

        else:
            if args.type.lower() == 'mlp':
                fts1 = hgut.normalise(fts1)
                G = sp.coo_matrix([1])
            else:
                G = hgut._generate_G_from_H_sparse(H, args=args)
        fts1 = torch.Tensor(fts1).to(device)  # features -> fts
        lbls1 = torch.Tensor(lbls1).squeeze().long().to(device)
        idx_train1 = torch.Tensor(idx_train1).long().to(device)
        idx_test1 = torch.Tensor(idx_test1).long().to(device)
        idx_val1 = torch.Tensor(idx_val1).long().to(device)

    else:  # object classification
        H = generate_H_from_dist(mvcnn_dist=mvcnn_dist,
                                gvcnn_dist=gvcnn_dist,
                                split_diff_scale=False,
                                gamma=args.gamma,
                                K_neigs=args.K_neigs,
                                is_probH=args.is_probH,
                                use_mvcnn_feature_for_structure=args.mvcnn_feature_structure,
                                use_gvcnn_feature_for_structure=args.gvcnn_feature_structure)

        G = hgut.generate_G_from_H(H, args=args)  # D_v^1/2 H W D_e^-1 H.T D_v^-1/2 :
        fts1 = fts
        lbls1 = lbls
        idx_train1 = idx_train
        idx_test1 = idx_test
        idx_val1 = idx_val


    if args.activate_dataset.startswith('coauthorship') or args.activate_dataset.startswith(
            'cocitation'):  # args.type == 'sgc' or args.activate_dataset.endswith('dblp'):
        H = sparse_mx_to_torch_sparse_tensor(sp.lil_matrix(H))

        G = sparse_mx_to_torch_sparse_tensor(G)
    else:
        H = sparse_mx_to_torch_sparse_tensor(sp.csr_matrix(H))
        G = sparse_mx_to_torch_sparse_tensor(G)  # torch.Tensor(G).to(device)

    return G, H, fts1, lbls1, idx_train1, idx_test1, idx_val1




def train_model(model, criterion, optimizer, scheduler, num_epochs=25, print_freq=500, args=None):
    time_start = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    if args.tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(
            os.path.join(args.save_dir, f'{time_start}'))
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc, best_epoch, best_val_loss = -1, 0, -1.0
    G, H, fts1, lbls1, idx_train1, idx_test1, idx_val1 = get_data(args=args,device=device)

    for epoch in range(num_epochs):

        if epoch % print_freq == 0:
            # print('-' * 10)
            print(f'Epoch {epoch}/{num_epochs - 1}')

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode
            running_loss = 0.0
            running_corrects = 0

            idx = idx_train1 if phase == 'train' else idx_val1  # idx_test

            # Iterate over data.
            optimizer.zero_grad()
            with torch.set_grad_enabled(phase == 'train'):

                outputs = model(fts1, G=G, adj=H)

                loss = criterion(outputs[idx], lbls1[idx])

                _, preds = torch.max(outputs, 1)
                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    clip_grad_norm_(model.parameters(), 40)
                    optimizer.step()
                    scheduler.step(loss)

            # statistics
            running_loss += loss.item() * fts.size(0)
            running_corrects += torch.sum(preds[idx] == lbls.data[idx])

            avg_loss = running_loss / len(idx)
            epoch_acc = running_corrects.double() / len(idx)

            if epoch % print_freq == 0:
                print(
                    f'{phase} avgLoss: {avg_loss:.4f} || per_loss: {loss:.4f} || type:{args.type}  ||'
                    f' {f"dataset:{args.activate_dataset}" if args.activate_dataset!="none" else f"dataset:{args.on_dataset}"}' \
                    f'|| lr: {optimizer.param_groups[0]["lr"]} || Acc: {epoch_acc:.4f} || dropout: {args.dropout} ')
            # deep copy the model
            if phase == 'val':
                if epoch_acc > best_acc:
                    early_stop = 0
                    best_acc = epoch_acc
                    best_epoch = epoch
                    best_val_loss = loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                else:
                    early_stop += 1

            if args.tensorboard:
                if phase == 'train':
                    writer.add_scalar('loss/train', loss, epoch)
                    writer.add_scalar('acc/train', epoch_acc, epoch)
                else:
                    writer.add_scalar('loss/val', loss, epoch)
                    writer.add_scalar('acc/val', epoch_acc, epoch)
                    writer.add_scalar('best_acc/val', best_acc, epoch)


        if epoch % print_freq == 0:
            print(f'Best val Acc: {best_acc:4f} at epoch: {best_epoch}, n_layers: {args.nbaseblocklayer} ')
            print('-' * 20)

        if early_stop > args.early_stopping:
            print(f'early stop at epoch {epoch}, n_layers: {args.nbaseblocklayer}' )
            break

    time_elapsed = time.time() - since
    print(f'\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    model.eval()
    # test
    outputs = model(fts, adj=H, G=G)

    _, preds = torch.max(outputs, 1)
    test_acc = torch.sum(preds[idx_test1] == lbls.data[idx_test1]).double() / len(idx_test1)
    # print(args)
    print(f"test_acc={test_acc}\n"
          f"best_val_acc={best_acc}\n"
          f"best_val_epoch={best_epoch}\n"
          f"best_val_loss={best_val_loss}")

    if args.tensorboard:
        writer.add_histogram('best_acc', test_acc)
    return best_epoch, float(test_acc),time_elapsed




def param_count(model):
    print(model)
    for n, p in model.named_parameters():
        print(n, p.shape)
    param_count = sum(param.numel() for param in model.parameters() if param.requires_grad)
    print(f'Number of parameters = {param_count:,}')

def _main():
    print(args)
    model = GCNModel(nfeat=fts.shape[1],
                     nhid=args.hidden,
                     nclass=n_class,
                     nhidlayer=args.nhiddenlayer,
                     dropout=args.dropout,
                     baseblock=args.type,
                     nbaselayer=args.nbaseblocklayer,
                     activation=F.relu,
                     withbn=args.withbn,
                     args=args)

    param_count(model)
    model = model.to(device)
    if args.type == 'gcnii':
        optimizer = optim.Adam([
            {'params': model.params1, 'weight_decay': args.wd1},
            {'params': model.params2, 'weight_decay': args.wd2},
        ], lr=args.lr)
    else:

        optimizer = optim.Adam(model.parameters(), lr=args.lr,
                               weight_decay=args.weight_decay)

    schedular = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=100,
                                                           verbose=False,
                                                           threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0,
                                                           eps=1e-08)
    criterion = torch.nn.CrossEntropyLoss()

    best_epoch, test_acc,time_elapsed = train_model(model, criterion, optimizer, schedular, args.epochs, print_freq=args.print_freq,
                                        args=args)

    return test_acc, best_epoch,time_elapsed


if __name__ == '__main__':
    elapsed_times=[]
    # citation network
    if args.activate_dataset.startswith('coauthor') or args.activate_dataset.startswith('cocitation'):
        setup_seed(args.seed)
        if args.debug:
            splits = [args.split]
        else:
            splits = [args.split + i for i in range(10)]
        results = []
        for split in splits:
            print(f"split: {split}/{splits}")
            args.split = split
            test_acc, best_epoch,time_elapsed = _main()
            results.append(test_acc)
            elapsed_times.append(time_elapsed)
            print('Acc array: ', results)
    else: # visual object
        if args.debug:
            seed_nums = [args.seed]  # 1000
        else:
            seed_nums = [args.seed + i for i in range(10)]  # 1000
        results = []
        for seed_num in seed_nums:
            print(f"seed:{seed_num}/{seed_nums}")
            setup_seed(seed_num)
            test_acc, best_epoch, time_elapsed = _main()
            results.append(test_acc)
            elapsed_times.append(time_elapsed)
            print('Acc array: ', results)


    results = np.array(results)
    elapsed_times = np.array(elapsed_times)
    print(f"\nAvg_test_acc={results.mean():.5f} \n"
          f"std={results.std():.5f}\n"
          f"elapsed_times:{elapsed_times.mean():.4f}+-{elapsed_times.std():.4f}s.")

