import argparse, os, time, torch
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split

from sdsmm.mdn.helpers import negative_log_likelihood, negative_log_likelihood_loss, \
    measurement_bias
from sdsmm.mdn.model import DistanceBearingMDN
from sdsmm.mdn.trainable_model import TrainableDistanceBearingMDN

import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset

from utils import read_real_learning_data, MrclamLearningDataset, read_data_with_data_ids, \
    read_artificial_learning_data, read_bootstraped_learning_data

from collections import OrderedDict

def main():
    DT_ARTL = 'artificial'
    DT_BSTP = 'bootstrap'
    DT_REAL = 'all_real'
    DATA_TYPE_LIST = [ DT_ARTL, DT_BSTP, DT_REAL ]

    parser = argparse.ArgumentParser()
    parser.add_argument('-b', "--robot_id", help="The robot ID.", type=int,
        choices=[1,2,3,4,5], default=0)

    parser.add_argument('-e', "--n_epochs", type=int, default=35,
        help="The max number of epochs to train for.")
    parser.add_argument('-r', "--lrate", type=float, default=0.0001,
        help="The learning rate.")
    parser.add_argument("--batch_size", type=int, default=32,
        help="The batch size.")
    parser.add_argument('-l', '--l2reg', type=float, default=0.,
        help='L2 regularization amount.')
    parser.add_argument('--save_best_model', action='store_true',
        help='Saves the model with the lowest validation loss.')

    parser.add_argument('--reduce_lr', action='store_true',
        help='Use Reduce LR on Plateau.')
    parser.add_argument('-p', '--lrpatience', type=int,
        help='The number of epochs to wait before reducing the learning rate.')
    parser.add_argument('-f', '--lrfactor', type=float,
        help='The factor to reduce the learning rate by.')
    parser.add_argument('-t', '--lrthreshold', type=float, default=0.01,
        help='The threshold for the learning rate scheduler.')

    parser.add_argument('-d', "--datatype", choices=DATA_TYPE_LIST,
        help="The type of data to use during training.")

    parser.add_argument('--pretraindir', help='Directory of the pretrained model.')
    parser.add_argument('-z', '--bs_size',
        choices=['all', '10k', '5k', '2.5k', '0.5k', '0.1k'],
        help='Size of the bootstrapped dataset.')
    parser.add_argument('-x', '--bs_index', type=int,
        help='The bootstrap index (1 - 100).')

    parser.add_argument('--cuda', type=int, default=-1,
        help='Use a GPU if one is available.')

    args = parser.parse_args()

    # Sanity checks.
    assert args.n_epochs > 0
    assert args.lrate > 0
    assert args.batch_size > 0
    assert args.l2reg >= 0.
    assert args.bs_index is None or 1 <= args.bs_index <= 100
    if args.reduce_lr:
        assert 0.05 <= args.lrfactor <= 0.5
        assert args.lrpatience >= 0
        assert args.lrthreshold > 0

    SAVE_IMG_DPI = 800

    # Select the compute device (cpu or cuda?).
    print('Selecting the compute device.')
    cuda_id = args.cuda
    device = torch.device(f'cuda:{cuda_id}') \
        if args.cuda >= 0 and torch.cuda.is_available() \
            else torch.device('cpu')
    print('Compute device will be: ' + str(device) + '\n')

    # Create the write directory if necessary.
    if args.datatype == DT_ARTL:
        run_id = 'pretrained'

    elif args.datatype == DT_BSTP:
        if args.pretraindir is None:
            run_id = 'lrdo'
        else:
            run_id = 'flrd'

    elif args.datatype == DT_REAL:
        run_id = 'rdo'
    #end if

    if args.datatype == DT_BSTP:
        assert args.bs_size is not None
        assert args.bs_index is not None
        run_id = os.path.join(run_id, 'ss-' + args.bs_size,
            'si-{:05}'.format(args.bs_index))
    #end if
    run_id = os.path.join(run_id, 'robot-{}'.format(args.robot_id))
    writedir = os.path.join(os.environ['IROS21_SDSMM'], 'exps/models', run_id)

    print('Run ID: ' + run_id)
    print('Write directory: ' + writedir)
    if not os.path.exists(writedir):
        os.makedirs(writedir)

    # Load all the data from disk.
    if args.datatype == DT_ARTL:
        print('\nLoading training data for the artificial dataset.')
        X, y = read_artificial_learning_data(set_type='train')

        print('\nLoading testing data for artificial dataset.')
        X_test, y_test = read_artificial_learning_data(set_type='test')

    elif args.datatype == DT_BSTP:
        print('\nLoading training data.')
        X_train, y_train = read_bootstraped_learning_data(args.robot_id, args.bs_size,
            args.bs_index, 'train')

        print('\nLoading validation data.')
        X_valid, y_valid = read_bootstraped_learning_data(args.robot_id, args.bs_size,
            args.bs_index, 'valid')

        print('\nLoading testing data.')
        X_test, y_test = read_bootstraped_learning_data(args.robot_id, args.bs_size,
            args.bs_index, 'test')

    elif args.datatype == DT_REAL:
        train_valid_data_ids = [ 2, 3, 5, 6, 8, 9 ]
        print('\nLoading training and validation data. Course IDs: ' + str(train_valid_data_ids))
        X, y = read_data_with_data_ids(args.robot_id, train_valid_data_ids)

        test_data_ids = [ 1, 4, 7 ]
        print('\nLoading testing data. Course IDs: ' + str(test_data_ids))
        X_test, y_test = read_data_with_data_ids(args.robot_id, test_data_ids)

    else:
        raise Exception('Unknown option: ' + args.datatype)
    #end if

    if args.datatype != DT_BSTP:
        print('\nDividing the training dataset into training and validation.')
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.1, shuffle=True)

    trainloader = MrclamLearningDataset(X=X_train, y=y_train, batch_size=args.batch_size,
        shuffle=True, device=device)

    validloader = MrclamLearningDataset(X=X_valid, y=y_valid, batch_size=X_valid.shape[0],
        shuffle=False, device=device)

    testloader = MrclamLearningDataset(X=X_test, y=y_test, batch_size=X_test.shape[0],
        shuffle=False, device=device)

    print('\n')
    print(' - Train size: {:,}'.format(trainloader.nobs))
    print(' - Valid size: {:,}'.format(validloader.nobs))
    print(' - Test size:  {:,}'.format(testloader.nobs))
    print('')

    # Build the MDN.
    print('Building the mixture density network.')
    net = TrainableDistanceBearingMDN().to(device)

    if args.pretraindir is not None:
        print('Loading pretrained model parameters.')
        net.load_model(os.path.join(args.pretraindir, 'model.pt'), device)

    print('Creating optimizer.')
    optimizer = optim.Adam(net.parameters(), lr=args.lrate, weight_decay=args.l2reg)

    # Create the learning rate scheduler.
    lr_scheduler = None
    if args.reduce_lr:
        print('Creating `Reduce LR on Plateau`.')
        lr_scheduler = ReduceLROnPlateau(optimizer=optimizer, mode='min',
            factor=args.lrfactor, patience=args.lrpatience, threshold=args.lrthreshold,
            threshold_mode='rel', verbose=True)
    #end if

    # Train the model.
    print('Training the model.')
    train_hist, valid_hist = net.fit(trainloader=trainloader, validloader=validloader,
        n_epochs=args.n_epochs, lossfnc=negative_log_likelihood_loss('mean'),
        biasfnc=measurement_bias, optimizer=optimizer, lr_scheduler=lr_scheduler,
        writedir=writedir, save_best=args.save_best_model)

    # Evaluate the model.
    print('Evaluating the model.')
    test_hist = net.evaluate(dataloader=testloader,
        lossfnc=negative_log_likelihood_loss('mean'), biasfnc=measurement_bias)

    # Load the last saved model from disk.
    print('Loading the saved trained model.')
    del net
    loaded_net = DistanceBearingMDN().to(device)
    loaded_net.load_model(modelpath=os.path.join(writedir, 'model.pt'), load_to_device=device)

    # Graph and save the model losses.
    print('Graphing model losses.')
    n_epochs_kept = len(train_hist.losses)
    epoch_array = np.arange(1, n_epochs_kept + 1)
    plt.title('Training Losses')
    plt.plot(epoch_array, train_hist.losses, label='Train loss')
    plt.plot(epoch_array, valid_hist.losses, label='Valid loss')
    plt.plot(epoch_array, [ test_hist.final_loss() ] * n_epochs_kept, label='Test loss')
    plt.xlabel('Epochs')
    plt.ylabel('NLL')
    plt.grid()
    plt.legend()
    plt.savefig(os.path.join(writedir, 'training-loss.png'), dpi=SAVE_IMG_DPI)
    plt.close()

    print('Computing the measurement biases for datasets.')
    plot_indices = list(range(1, 7, 2))
    dataloaders = [trainloader, validloader, testloader]
    datatypes = ['Train', 'Valid', 'Test']
    for p_i, dloader, dttype in zip(plot_indices, dataloaders, datatypes):
        # Compute the bias for this dataset.
        print(f'Computing the point biases for {dttype} set.')
        biases = measurement_bias(net_out=loaded_net(dloader.X), meas_true=dloader.y)
        biases = biases.detach().clone().t().cpu()

        plt.subplot(3, 2, p_i)
        plt.hist(biases[0], density=True,
            bins=list(range(int(min(biases[0]))-1, int(max(biases[0]))+1)))
        plt.ylabel(dttype + ' Freq.')
        if p_i + 1 == 6:
            plt.xlabel('Distance bias [cm]')

        plt.subplot(3, 2, p_i + 1)
        plt.hist(biases[1], density=True,
            bins=list(range(int(min(biases[1]))-1, int(max(biases[1]))+1)))
        if p_i + 1 == 6:
            plt.xlabel('Bearing bias [degs]')
    #end for
    plt.savefig(os.path.join(writedir, 'training-biases.png'), dpi=SAVE_IMG_DPI)
    plt.close()

    # Print training results.
    results_str = """
        Run ID:               {}
        Data type:            {}
        Pretrain model path:  {}
        Bootstrap data:       {} -- index {}
        Robot ID:             {}
        Epochs:               {}
        Learning rate:        {:.8f}
        Batch size:           {}
        L2 regularization:    {:.8f}

        Save best model?      {}
        Epochs trained for:   {}

        CUDA ID?:             {}

        Reduce LR on Plateau? {}
        LR Patience:          {}
        LR Reduce Factor:     {}
        LR Threshold:         {}

        Train size:           {}
        Test size:            {}

        Train NLL:            {:.6f}
        Valid NLL:            {:.6f}
         Test NLL:            {:.6f}

        Train distance MSE:   {:05.6f} [cm]
        Valid distance MSE:   {:05.6f} [cm]
         Test distance MSE:   {:05.6f} [cm]

        Train bearing MSE:    {:05.6f} [deg]
        Valid bearing MSE:    {:05.6f} [deg]
         Test bearing MSE:    {:05.6f} [deg]
    """.format(
        run_id, args.datatype,
        'n/a' if args.pretraindir is None else args.pretraindir,
        args.bs_size, args.bs_index,
        args.robot_id, args.n_epochs, args.lrate, args.batch_size, args.l2reg,

        'Yes' if args.save_best_model else 'No', n_epochs_kept,

        args.cuda,

        'Yes' if args.reduce_lr else 'No',
        args.lrpatience, args.lrfactor, args.lrthreshold,

        X_train.size()[0] + X_valid.size()[0], X_test.size()[0],

        # NLL
        train_hist.final_loss(),
        valid_hist.final_loss(),
        test_hist.final_loss(),

        # Distance
        train_hist.final_biases()[0],
        valid_hist.final_biases()[0],
        test_hist.final_biases()[0],

        # Bearing
        train_hist.final_biases()[1],
        valid_hist.final_biases()[1],
        test_hist.final_biases()[1])

    print(results_str)
    with open(os.path.join(writedir, 'log.txt'), 'w') as f:
        f.write(results_str)

    print('\nDone! Write dir: ' + writedir)
#end def

if __name__ == '__main__':
    main()
