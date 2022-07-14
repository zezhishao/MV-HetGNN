import  argparse
import  torch    as th
import  time

from config                 import  *
from dataset.dataset_utils  import  load_LastFM_data
from train_utils            import  *
from dataloader             import  ModelDataset
from model.lastfm_model     import  LastFMModel
from sklearn.metrics        import  roc_auc_score, average_precision_score

def run_lastfm(args):
    # train recoder
    auc_list = []
    ap_list = []

    # other parameters
    use_node_features = [False, False, False]  # 除了A节点都用One Hot表示
    dataset             = 'LastFM'
    save_path           = home_path + 'checkpoint/MVHetGNN_checkpoint_{}.pt'.format(dataset)
    if args.cuda:
        device          = th.device('cuda:0')
    else:
        device          = th.device('cpu')
    
    # get feat
    _, _, _, type_mask, train_val_test_pos_user_artist, train_val_test_neg_user_artist = load_LastFM_data(home_path)
    features, input_feature_dims                                                        = get_lastfm_node_feat(type_mask, use_node_features, device)

    # unsupervised labels
    test_pos_user_artist    = train_val_test_pos_user_artist['test_pos_user_artist']
    test_neg_user_artist    = train_val_test_neg_user_artist['test_neg_user_artist']
    y_true_test             = np.array([1] * len(test_pos_user_artist) + [0] * len(test_neg_user_artist))
    
    # datasets
    train_dataset   = ModelDataset(name='LastFM', mode='train')
    val_dataset     = ModelDataset(name='LastFM', mode='validate')
    test_dataset    = ModelDataset(name='LastFM', mode='test')

    # repeat
    for _ in range(args.repeat):
        # initilize model
        lastfm_model    = LastFMModel(type_mask, input_feature_dims, args).to(device)
        
        optimizer       = torch.optim.Adam(lastfm_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        
        early_stopping = EarlyStopping(patience=args.patience, verbose=True, save_path=save_path)
        
        dur1            = []  # training avg time
        dur2            = []  # opt avg time

        print("------------------------------ Training Network ------------------------------")
        for epoch in range(num_epochs):
            t_start     = time.time()
            lastfm_model.train()
            # iterations
            for iteration, [positive_data, negative_data] in enumerate(train_dataset):
                print(iteration, end='\r')
                t0      = time.time()
                positive_data, negative_data = set_inputs_to_device([positive_data, negative_data], device, dataset)

                train_pos_g_lists, train_pos_indices_lists, train_pos_idx_batch_mapped_lists, pos_hierarchical_graph_all = positive_data
                train_neg_g_lists, train_neg_indices_lists, train_neg_idx_batch_mapped_lists, neg_hierarchical_graph_all = negative_data
                
                # use hidden state rather output of fc
                _, [pos_embedding_user, pos_embedding_artist], trick_loss1 = lastfm_model(pos_hierarchical_graph_all, features)
                _, [neg_embedding_user, neg_embedding_artist], trick_loss2 = lastfm_model(neg_hierarchical_graph_all, features)  # print(random.randint(0,9))

                pos_embedding_user      = pos_embedding_user.view(-1, 1, pos_embedding_user.shape[1])
                pos_embedding_artist    = pos_embedding_artist.view(-1, pos_embedding_artist.shape[1], 1)
                neg_embedding_user      = neg_embedding_user.view(-1, 1, neg_embedding_user.shape[1])
                neg_embedding_artist    = neg_embedding_artist.view(-1, neg_embedding_artist.shape[1], 1)

                pos_out     = torch.bmm(pos_embedding_user, pos_embedding_artist)
                neg_out     = -torch.bmm(neg_embedding_user, neg_embedding_artist)
                unsup_loss  = -torch.mean(F.logsigmoid(pos_out) + F.logsigmoid(neg_out))
                trick_loss  = trick_loss1 + trick_loss2
                train_loss  = unsup_loss + trick_loss
                
                # print(train_loss)
                t1 = time.time()
                dur1.append(t1 - t0)  # train time

                # opt  
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()

                t2  = time.time()
                dur2.append(t2 - t1)     # opt time
                if iteration % 10 == 0:
                    print('Epoch {:05d} | Iteration {:05d} | unsupervised_Loss {:.4f} | trick_Loss {:.4f} | Time1(s) {:.4f} | Time2(s) {:.4f}'.format(
                        epoch, iteration, unsup_loss.item(), trick_loss.item(), np.mean(dur1), np.mean(dur2)))

            # ============================== Validating Network ==============================
            lastfm_model.eval()
            val_loss = []
            with torch.no_grad():
                for iteration, [positive_data, negative_data] in enumerate(val_dataset):
                    print(iteration, end='\r')
                    # forward
                    positive_data, negative_data = set_inputs_to_device([positive_data, negative_data], device, dataset)

                    val_pos_g_lists, val_pos_indices_lists, val_pos_idx_batch_mapped_lists, pos_hierarchical_graph_all = positive_data
                    val_neg_g_lists, val_neg_indices_lists, val_neg_idx_batch_mapped_lists, neg_hierarchical_graph_all = negative_data

                    # use hidden state rather output of fc
                    _, [pos_embedding_user, pos_embedding_artist], trick_loss1 = lastfm_model(pos_hierarchical_graph_all, features)
                    _, [neg_embedding_user, neg_embedding_artist], trick_loss2 = lastfm_model(neg_hierarchical_graph_all, features)  # print(random.randint(0,9))

                    pos_embedding_user      = pos_embedding_user.view(-1, 1, pos_embedding_user.shape[1])
                    pos_embedding_artist    = pos_embedding_artist.view(-1, pos_embedding_artist.shape[1], 1)
                    neg_embedding_user      = neg_embedding_user.view(-1, 1, neg_embedding_user.shape[1])
                    neg_embedding_artist    = neg_embedding_artist.view(-1, neg_embedding_artist.shape[1], 1)

                    pos_out = torch.bmm(pos_embedding_user, pos_embedding_artist)
                    neg_out = -torch.bmm(neg_embedding_user, neg_embedding_artist)
                    val_loss.append(-torch.mean(F.logsigmoid(pos_out) + F.logsigmoid(neg_out)))
    
                val_loss = torch.mean(torch.tensor(val_loss))
            t_end = time.time()
            print('Epoch {:05d} | Val_Loss {:.4f} | Time(s) {:.4f}'.format(epoch, val_loss.item(), t_end - t_start))

            early_stopping(val_loss, lastfm_model)
            if early_stopping.early_stop:
                print('Early stopping!')
                break

        print("------------------------------ Testing Network ------------------------------")
        lastfm_model.load_state_dict(torch.load(save_path))
        lastfm_model.eval()

        pos_proba_list = []
        neg_proba_list = []
        with torch.no_grad():
            for iteration, [positive_data, negative_data] in enumerate(test_dataset):
                print(iteration, end='\r')
                positive_data, negative_data = set_inputs_to_device([positive_data, negative_data], device, dataset)
                
                test_pos_g_lists, test_pos_indices_lists, test_pos_idx_batch_mapped_lists, pos_hierarchical_graph_all = positive_data
                test_neg_g_lists, test_neg_indices_lists, test_neg_idx_batch_mapped_lists, neg_hierarchical_graph_all = negative_data

                _, [pos_embedding_user, pos_embedding_artist], trick_loss1 = lastfm_model(pos_hierarchical_graph_all, features)
                _, [neg_embedding_user, neg_embedding_artist], trick_loss2 = lastfm_model(neg_hierarchical_graph_all, features)  # print(random.randint(0,9))

                pos_embedding_user      = pos_embedding_user.view(-1, 1, pos_embedding_user.shape[1])
                pos_embedding_artist    = pos_embedding_artist.view(-1, pos_embedding_artist.shape[1], 1)
                neg_embedding_user      = neg_embedding_user.view(-1, 1, neg_embedding_user.shape[1])
                neg_embedding_artist    = neg_embedding_artist.view(-1, neg_embedding_artist.shape[1], 1)

                pos_out = torch.bmm(pos_embedding_user, pos_embedding_artist).flatten()
                neg_out = torch.bmm(neg_embedding_user, neg_embedding_artist).flatten()
                pos_proba_list.append(torch.sigmoid(pos_out))
                neg_proba_list.append(torch.sigmoid(neg_out))

            y_proba_test = torch.cat(pos_proba_list + neg_proba_list)
            y_proba_test = y_proba_test.cpu().numpy()
        auc = roc_auc_score(y_true_test, y_proba_test)
        ap  = average_precision_score(y_true_test, y_proba_test)

        print('Link Prediction Test')
        print('AUC = {}'.format(auc))
        print('AP = {}'.format(ap))
        auc_list.append(auc)
        ap_list.append(ap)
            
        print('----------------------------------------------------------------')
        print('Link Prediction Tests Summary')
        print('AUC_mean = {}, AUC_std = {}'.format(np.mean(auc_list), np.std(auc_list)))
        print('AP_mean = {}, AP_std = {}'.format(np.mean(ap_list), np.std(ap_list)))


if __name__ == "__main__":
    set_config(0)
    
    parser  = argparse.ArgumentParser(description='MVHetGNN')
    # model args
    parser.add_argument('-hidden_dim',    default=64,              type=int,   help='dimension of output hidden vector of each view')
    parser.add_argument('-intra_dim',     default=64,               type=int,   help='dimension of fused feature')
    parser.add_argument('-inter_dim',     default=64,               type=int,   help='dimension of fused feature')

    parser.add_argument('-batch_size',    default=LastFM_batch_size,    type=int,   help='batch size')
    # parser.add_argument('-dropout_rate',  default=0.3,                 type=float, help='dropout rate of message passing matrix in Ego Graph Encoding module') TODO
    parser.add_argument('-dropout_rate',  default=0.3,                  type=float, help='dropout rate of transform matrix')
    parser.add_argument('-gain',          default=1,                    type=float, help='gain parameter while initializing the FC')
    parser.add_argument('-output_dim',    default=4,                    type=int,   help='class number')
    
    # training process control
    parser.add_argument('-patience',      default=5,                    type=int,   help='patience of early stopping')
    parser.add_argument('-repeat',        default=50,                   type=int,   help='repeat times of model training')
    parser.add_argument('-cuda',          default=True,                 type=float, help='weight decay')

    # backpropagation
    parser.add_argument('-lr',            default=0.001,                type=float, help='learning rate')
    parser.add_argument('-weight_decay',  default=0.001,                type=float, help='weight decay')
    args = parser.parse_args()

    run_lastfm(args)
