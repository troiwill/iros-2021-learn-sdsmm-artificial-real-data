import os
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from sdsmm.mdn.model import DistanceBearingMDN

class History(object):

    def __init__(self):
        self.losses = list()
        self.biases = list()

    def final_loss(self):
        return self.losses[-1]

    def final_biases(self):
        return self.biases[-1]

    def retain_first_n_histories(self, n_epochs):
        self.losses = self.losses[:n_epochs]
        self.biases = self.biases[:n_epochs]
    #end def
#end class

class TrainableDistanceBearingMDN(DistanceBearingMDN):

    def __init__(self):
        super(TrainableDistanceBearingMDN, self).__init__()

        self.train_status = \
            'TRAINING --- Epoch: ({:03} / {:03}) [{:05} / {:05} ({:07.3f}%)]    ' \
                + 'Loss: {:09.4f}   MSE: {:07.3f} cm, {:07.3f} degs          \r'
        self.valid_status = \
            'VALIDATION                      [{:05} / {:05} ({:07.3f}%)]    ' \
                + 'Loss: {:09.4f}   MSE: {:07.3f} cm, {:07.3f} degs          \r'
        self.eval_status = \
            'EVALUATION --- [{:05} / {:05} ({:07.3f}%)]    ' \
                + 'Loss: {:09.4f}   MSE: {:07.3f} cm, {:07.3f} degs          \r'
        print('Architecture:\n' + str(self))
    #end def

    def __train_model__(self, dataloader, lossfnc, biasfnc, optimizer, status_str, epoch_i,
        n_epochs):
        """
        Internal function that trains the model for one epoch.
        """
        # Set up for training.
        self.train()
        losses = list()
        biases = list()
        nb_batches = dataloader.nbatches

        # Perform the training loop for this epoch.
        for batch_i, (X, y) in enumerate(dataloader):
            optimizer.zero_grad()
            net_out = self.forward(X)
            loss = lossfnc(net_out, y)

            bias = biasfnc(net_out=net_out, meas_true=y).detach().clone()
            biases += bias.cpu().numpy().tolist()
            bias = torch.mean(torch.square(bias), 0)

            loss.backward()
            optimizer.step()

            losses.append(loss.item())

            print(status_str.format(epoch_i + 1, n_epochs, batch_i + 1, nb_batches,
                100. * (batch_i + 1) / nb_batches, losses[-1], bias[0], bias[1]), end='')
        #end for
        return losses, biases
    #end def

    def __eval_model__(self, dataloader, lossfnc, biasfnc, status_str):
        """
        Internal function that evaluates the model using the provided data.
        """
        # Set up for evaluating.
        self.eval()
        losses = list()
        biases = list()
        nb_batches = dataloader.nbatches
        
        with torch.no_grad():
            for batch_i, (X, y) in enumerate(dataloader):
                net_out = self.forward(X)
                loss = lossfnc(net_out, y)
                losses.append(loss.item())

                bias = biasfnc(net_out=net_out, meas_true=y).detach().clone()
                biases += bias.cpu().numpy().tolist()
                bias = torch.mean(torch.square(bias), 0)

                print(status_str.format(batch_i + 1, nb_batches, 100. * (batch_i + 1) / nb_batches,
                    losses[-1], bias[0], bias[1]), end='')
            #end for
        #end with
        return losses, biases
    #end def

    def fit(self, trainloader, validloader, n_epochs, lossfnc, biasfnc, optimizer, lr_scheduler,
        writedir, save_best):
        """
        Trains the model using the data from the training set `trainloader` and evaluates using
        the validation set `validloader`. Trains for `n_epochs` using the loss function `lossfnc`.
        """
        # Stores the training and validation loss histories.
        train_hist = History()
        valid_hist = History()

        # Records the "best validation loss" thus far.
        best_valid_loss = 1e8
        last_best_epoch = n_epochs

        for epoch_i in range(n_epochs):
            # Train the model.
            losses, biases = self.__train_model__(dataloader=trainloader, lossfnc=lossfnc, 
                biasfnc=biasfnc, optimizer=optimizer, status_str=self.train_status,
                epoch_i=epoch_i, n_epochs=n_epochs)

            train_hist.losses.append(np.mean(losses))
            train_hist.biases.append(np.mean(np.square(biases), axis=0).tolist())

            nb_batches = trainloader.nbatches
            print(self.train_status.format(epoch_i + 1, n_epochs, nb_batches, nb_batches, 100.,
                train_hist.final_loss(), train_hist.final_biases()[0], train_hist.final_biases()[1]))

            # Evaluate the model.
            losses, biases = self.__eval_model__(dataloader=validloader, lossfnc=lossfnc, 
                biasfnc=biasfnc, status_str=self.valid_status)

            valid_hist.losses.append(np.mean(losses))
            valid_hist.biases.append(np.mean(np.square(biases), axis=0).tolist())

            # Adjust the learning rate if possible.
            if lr_scheduler:
                lr_scheduler.step(float(valid_hist.final_loss()))

            nb_batches = validloader.nbatches
            print(self.valid_status.format(nb_batches, nb_batches, 100.,
                valid_hist.final_loss(), valid_hist.final_biases()[0], valid_hist.final_biases()[1]))

            # Save the best model thus far if requested.
            if save_best is True and valid_hist.final_loss() < best_valid_loss:
                print('* Saving \'best\' model (old valid loss = {:.5f}, new valid loss = {:.5f})'.format(
                    best_valid_loss, valid_hist.final_loss()))

                best_valid_loss = valid_hist.final_loss()
                last_best_epoch = epoch_i + 1

                self.save_model(writedir)
            #end if
            print('')
        #end for

        # Save the current model if 'save_best' is False.
        if save_best is True:
            print('Clipping histories up to epoch ' + str(last_best_epoch))
            train_hist.retain_first_n_histories(last_best_epoch)
            valid_hist.retain_first_n_histories(last_best_epoch)

        else:
            print('Saving the final model with validation loss = {:.5f}\n'.format(
                valid_hist.final_loss()))
            self.save_model(writedir)
        #end if

        return train_hist, valid_hist
    #end def

    def evaluate(self, dataloader, lossfnc, biasfnc):
        """
        Evaluates the model using the dataset in `dataloader`, loss function `lossfnc` and bias
        function `biasfnc`.
        """
        # Evaluate the model.
        losses, biases = self.__eval_model__(dataloader=dataloader, lossfnc=lossfnc, 
            biasfnc=biasfnc, status_str=self.eval_status)

        eval_hist = History()
        eval_hist.losses.append(np.mean(losses))
        eval_hist.biases.append(np.mean(np.square(biases), axis=0).tolist())

        nb_batches = dataloader.nbatches
        print(self.eval_status.format(nb_batches, nb_batches, 100.,
            eval_hist.losses[-1], eval_hist.final_biases()[0], eval_hist.final_biases()[1]))
        print('')

        return eval_hist
    #end def

    def save_model(self, writedir):
        torch.save(self.state_dict(), os.path.join(writedir, 'model.pt'))
    #end def
#end class
