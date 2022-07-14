import  argparse
from    platform import java_ver
import  torch    as th
import  time

from config                 import  *
from dataset.dataset_utils  import  load_DBLP_data
from train_utils            import  *
from dataloader             import  ModelDataset
from model.dblp_model       import  DBLPModel
from easydict import EasyDict
from set_env import set_env
import setproctitle
setproctitle.setproctitle("MV-HetGNN")

def run_dblp(args):
    # set_config(0)
    ENV = EasyDict()
    ENV.SEED    = args.seed
    ENV.CUDNN_ENABLED = False
    set_env(ENV)
    # train recorder
    svm_macro_f1_lists  = []
    svm_micro_f1_lists  = []
    nmi_mean_list       = []
    nmi_std_list        = []
    ari_mean_list       = []
    ari_std_list        = []

    # other parameters
    use_node_features   = [True, True, False, False]
    dataset             = 'DBLP'
    save_path           = home_path + 'checkpoint/MVHetGNN_checkpoint_{}.pt'.format(dataset)
    if args.cuda:
        device          = th.device('cuda:0')
    else:
        device          = th.device('cpu')
    
    # get features
    _, _, features, _, type_mask, _, _  = load_DBLP_data(home_path)
    features, input_feature_dims        = set_node_feature_type(features, use_node_features, device)

    # datasets
    train_dataset       = ModelDataset(name=dataset, mode='train')
    validate_dataset    = ModelDataset(name=dataset, mode='validate')
    test_dataset        = ModelDataset(name=dataset, mode='test')

    # repeat
    for _ in range(args.repeat):
        # initilize model
        dblp_model      = DBLPModel(type_mask, input_feature_dims, args).to(device)

        # print(dblp_model)
        optimizer       = torch.optim.Adam(dblp_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        lr_sche_steps   = [25, 50, 75]
        lr_decay_ratio  = 0.5
        lr_scheduler    = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_sche_steps, gamma=lr_decay_ratio)
        
        early_stopping  = EarlyStopping(patience=args.patience, verbose=True, save_path=save_path)

        dur1            = []  # training avg time
        dur2            = []  # opt avg time

        print("------------------------------ Training Network ------------------------------")
        for epoch in range(num_epochs):
            t_start     = time.time()
            dblp_model.train()
            # iterations
            for iteration, (input_data, labels) in enumerate(train_dataset):
                print(iteration, end='\r')
                t0                              = time.time()
                input_data                      = set_inputs_to_device(input_data, device, dataset)
                logits, embeddings, trick_loss  = dblp_model(input_data, features=features)
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
                    print('Epoch {:05d} | Iteration {:05d} | ce_Loss {:.4f} | trick_loss {:.4f} | Time1(s) {:.4f} | Time2(s) {:.4f}'.format(
                            epoch, iteration, ce_loss.item(), trick_loss.item(), np.mean(dur1), np.mean(dur2)))
            
            # ============================== Validating Network ==============================
            dblp_model.eval()
            val_log_p               = []
            validate_labels_list    = []
            with torch.no_grad():
                for iteration, (input_data, labels) in enumerate(validate_dataset):
                    print(iteration, end='\r')

                    input_data                      = set_inputs_to_device(input_data, device, dataset)
                    logits, embeddings, trick_loss  = dblp_model(input_data, features=features)
                    log_p                           = F.log_softmax(logits, 1)

                    val_log_p.append(log_p)
                    validate_labels_list.extend(list(labels))

                val_ce_loss     = F.nll_loss(torch.cat(val_log_p, 0), torch.LongTensor(validate_labels_list).to(device))
            t_end   = time.time()
            current_lr = optimizer.param_groups[0]['lr']
            print('Epoch {:05d} | current_lr {:.6f} | Val_ce_Loss {:.4f} | Val_trick_Loss {:.4f} | Time(s) {:.4f}'.format(epoch, current_lr, val_ce_loss.item(), trick_loss.item(), t_end - t_start))
            # early stopping
            early_stopping(val_ce_loss, dblp_model)
            if early_stopping.early_stop:
                print('Early stopping!')
                break
            lr_scheduler.step()

        print("------------------------------ Testing Network ------------------------------")
        dblp_model.load_state_dict(torch.load(save_path))
        dblp_model.eval()
        test_embeddings     = []
        test_labels_list    = []
        with torch.no_grad():
            for iteration, (input_data, labels) in enumerate(test_dataset):

                input_data                      = set_inputs_to_device(input_data, device, dataset)
                logits, embeddings, trick_loss  = dblp_model(input_data, features=features)

                if len(embeddings.shape) == 1:
                    embeddings = embeddings.unsqueeze(0)
                
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
    parser.add_argument('-hidden_dim',    default=64,              type=int,   help='dimension of output hidden vector of each view')
    parser.add_argument('-intra_dim',     default=64,               type=int,   help='dimension of fused feature')
    parser.add_argument('-inter_dim',     default=64,               type=int,   help='dimension of fused feature')
    
    parser.add_argument('-batch_size',    default=DBLP_batch_size,  type=int,   help='batch size')
    parser.add_argument('-dropout_rate',  default=0.3,              type=float, help='dropout rate of transform matrix')
    parser.add_argument('-gain',          default=1,                type=float, help='gain parameter while initializing the FC')
    parser.add_argument('-output_dim',    default=4,                type=int,   help='class number')
    parser.add_argument('-seed',          default=40,            type=float, help='weight decay')
    
    # training process control
    parser.add_argument('-patience',      default=50,               type=int,   help='patience of early stopping')
    parser.add_argument('-repeat',        default=1,               type=int,   help='repeat times of model training')
    parser.add_argument('-cuda',          default=True,             type=float, help='weight decay')

    # backpropagation
    parser.add_argument('-lr',            default=0.005,            type=float, help='learning rate')
    parser.add_argument('-weight_decay',  default=0.005,            type=float, help='weight decay')  

    parser.add_argument('-mse',           default=0.1,            type=float, help='weight decay')

    args = parser.parse_args()

    run_dblp(args)
