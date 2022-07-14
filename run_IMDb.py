# _*_ coding: utf-8 _*_
# @Time   : 2020/9/16 9:58 下午
# @Author : Zezhi Shao
# @File   : run_IMDb.py
# @Desc   : This model is greatly affected by the random seed, so we set the seed to a fixed number: 0.
import  argparse
import  torch    as th
import  time
import  optuna

from config                 import  *
from dataset.dataset_utils  import  load_IMDB_data
from train_utils            import  *
from dataloader             import  ModelDataset
from model.imdb_model       import  IMDBModel

def run_imdb(args):
    # set_config()
    # train recorder
    svm_macro_f1_lists  = []
    svm_micro_f1_lists  = []
    nmi_mean_list       = []
    nmi_std_list        = []
    ari_mean_list       = []
    ari_std_list        = []

    # other parameters
    use_node_features   = [True, False, False]  # 除了A节点都用One Hot表示
    dataset             = 'IMDb'
    save_path           = home_path + 'checkpoint/Uni_HetG_checkpoint_{}.pt'.format(dataset)
    if args.cuda:
        device          = th.device('cuda:0')
    else:
        device          = th.device('cpu')

    # get features
    _, _, features, _, type_mask, _, _  = load_IMDB_data(home_path, only_load_feature=True)
    features, input_feature_dims        = set_node_feature_type(features, use_node_features, device)

    # datasets
    train_dataset       = ModelDataset(name=dataset, mode='train')
    validate_dataset    = ModelDataset(name=dataset, mode='validate')
    test_dataset        = ModelDataset(name=dataset, mode='test')
    
    # repeat
    for _ in range(args.repeat):
        imdb_model  = IMDBModel(type_mask, input_feature_dims, args).to(device)
        optimizer   = torch.optim.Adam(imdb_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        early_stopping = EarlyStopping(patience=args.patience, verbose=True, save_path=save_path)

        dur1 = []  # training avg time
        dur2 = []  # opt avg time

        print("------------------------------ Training Network ------------------------------")
        for epoch in range(num_epochs):
            t_start = time.time()
            imdb_model.train()
            for iteration, (input_data, labels, target_index) in enumerate(train_dataset):
                # print(iteration, end='\r')
                t0                              = time.time()
                input_data                      = set_inputs_to_device(input_data, device, dataset)
                logits, embeddings, trick_loss  = imdb_model(input_data, features, target_index=target_index)
                log_p                           = F.log_softmax(logits, 1)
                ce_loss                         = F.nll_loss(log_p, torch.LongTensor(labels).to(device))
                train_loss                      = ce_loss + trick_loss
                t1                              = time.time()
                dur1.append(t1 - t0)    # train time
                
                # opt
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()

                t2          = time.time()
                dur2.append(t2 - t1)    # opt time
                if iteration % 50 == 0:
                    pass 
                    # print('Epoch {:05d} | Iteration {:05d} | ce_Loss {:.4f} | trick_loss {:.4f} | Time1(s) {:.4f} | Time2(s) {:.4f}'.format(
                            # epoch, iteration, ce_loss.item(), trick_loss.item(), np.mean(dur1), np.mean(dur2)))

            # ============================== Validating Network ==============================
            imdb_model.eval()
            val_log_p               = []
            validate_labels_list    = []
            with torch.no_grad():
                # ignore target index(_) here because it is useless in training procedure
                for iteration, (input_data, labels, target_index) in enumerate(validate_dataset):
                    # print(iteration, end='\r')

                    input_data                      = set_inputs_to_device(input_data, device, dataset)
                    logits, embeddings, trick_loss  = imdb_model(input_data, features, target_index=target_index)
                    log_p                           = F.log_softmax(logits, 1)

                    val_log_p.append(log_p)
                    validate_labels_list.extend(list(labels))
                val_ce_loss = F.nll_loss(torch.cat(val_log_p, 0), torch.LongTensor(validate_labels_list).to(device))
            t_end = time.time()
            # print('Epoch {:05d} | Val_ce_Loss {:.4f} | Val_trick_Loss {:.4f} | Time(s) {:.4f}'.format(epoch, val_ce_loss.item(), trick_loss.item(), t_end - t_start))
            # early stopping
            early_stopping(val_ce_loss, imdb_model)
            if early_stopping.early_stop:
                print('Early stopping!')
                break

        print("------------------------------ Testing Network ------------------------------")
        imdb_model.load_state_dict(torch.load(save_path))
        imdb_model.eval()
        test_embeddings = []
        test_labels_list = []
        with torch.no_grad():
            # ignore target index(_) here because it is useless in training procedure
            for iteration, (input_data, labels, target_index) in enumerate(test_dataset):

                input_data = set_inputs_to_device(input_data, device, dataset)
                logits, embeddings, trick_loss  = imdb_model(input_data, features, target_index=target_index)

                test_embeddings.append(embeddings)
                test_labels_list.extend(list(labels))

            test_embeddings = torch.cat(test_embeddings, 0)
            save_embeddings(test_embeddings, test_labels_list, dataset)

            svm_macro_f1_list, svm_micro_f1_list, nmi_mean, nmi_std, ari_mean, ari_std = evaluate_results_nc(test_embeddings.cpu().numpy(), torch.LongTensor(test_labels_list).cpu().numpy(), num_classes=args.output_dim)
            svm_macro_f1_lists.append(svm_macro_f1_list)
            svm_micro_f1_lists.append(svm_micro_f1_list)
            nmi_mean_list.append(nmi_mean)
            nmi_std_list.append(nmi_std)
            ari_mean_list.append(ari_mean)
            ari_std_list.append(ari_std)

    print_test_result(svm_macro_f1_lists, svm_micro_f1_lists, nmi_mean_list, nmi_std_list, ari_mean_list, ari_std_list)

if __name__ == "__main__":
    parser  = argparse.ArgumentParser(description='MVHetGNN')
    # model args
    parser.add_argument('-hidden_dim',    default=64,               type=int,   help='dimension of output hidden vector of each view') # d'
    parser.add_argument('-intra_dim',     default=64,               type=int,   help='dimension of fused feature')                      # d^{M/2}
    parser.add_argument('-inter_dim',     default=64,               type=int,   help='dimension of fused feature')                      # d
    # parser.add_argument('-fusion_dim',     default=48,               type=int,   help='dimension of fused feature')
    
    parser.add_argument('-batch_size',    default=IMDb_batch_size,  type=int,   help='batch size')
    # parser.add_argument('-dropout_rate',  default=0.3,            type=float, help='dropout rate of message passing matrix in Ego Graph Encoding module') TODO
    parser.add_argument('-dropout_rate',  default=0.5,              type=float, help='dropout rate of transform matrix')
    parser.add_argument('-gain',          default=1.414,                type=float, help='gain parameter while initializing the FC')
    parser.add_argument('-output_dim',    default=3,                type=int,   help='class number')
    parser.add_argument('-mse',           default=0.1,             type=float, help='weight decay')
    
    # training process control
    parser.add_argument('-patience',      default=10,               type=int,   help='patience of early stopping')
    parser.add_argument('-repeat',        default=1,                type=int,   help='repeat times of model training')
    parser.add_argument('-cuda',          default=True,             type=float, help='weight decay')

    # backpropagation
    parser.add_argument('-lr',            default=0.001,               type=float, help='learning rate')
    parser.add_argument('-weight_decay',  default=0.001,     type=float, help='weight decay')
    args = parser.parse_args()

    run_imdb(args)
