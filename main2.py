from __future__ import print_function

import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import utils
from config import args
from model import LBL, CondCopy
from torch import Tensor
import torch.cuda


def train(model, optimizer, data_iter, text_field, args):
    model.train()
    loss_function_tot = nn.NLLLoss(size_average=False)
    loss_function_avg = nn.NLLLoss(size_average=True)
    total_loss = 0
    data_size = 0

    iter_len = len(data_iter)
    batch_idx = 0
    for batch in data_iter:
        context = torch.transpose(batch.text, 0, 1)
        target = (batch.target[-1, :]).cuda()
        batch_size = context.size(0)
        # zero out gradients
        optimizer.zero_grad()
        # get output
        output = model(context).cuda()
        # calculate loss
        loss = loss_function_avg(output, target)
        total_loss += loss_function_tot(output, target).data.cpu().numpy()[0]
        data_size += batch_size
        # calculate gradients
        loss.backward()
        # update parameters
        optimizer.step()
        # enforce the max_norm constraint
        #model.max_norm_embedding()
        # skip the last batch
        if batch_idx >= iter_len - 2:
            break

        batch_idx += 1

    avg_loss = total_loss / data_size
    return model, optimizer, np.exp(avg_loss)


def evaluate(model, data_iter, text_field, args):
    model.eval()
    loss_function_tot = nn.NLLLoss(size_average=False)
    total_loss = 0
    data_size = 0
    iter_len = len(data_iter)
    batch_idx = 0
    for batch in data_iter:
        context = torch.transpose(batch.text, 0, 1)
        target = (batch.target[-1, :]).cuda()
        batch_size = context.size(0)
        # get model output
        output = model(context).cuda()
        # calculate total loss
        loss = loss_function_tot(output, target)  # loss is already averaged
        total_loss += loss.data.cpu().numpy()[0]
        data_size += batch_size

        # skip last batch
        if batch_idx >= iter_len - 2:
            break

        batch_idx += 1

    avg_loss = total_loss / data_size
    perplexity = np.exp(avg_loss)  # use exp here because the loss uses ln
    return perplexity


def main():
    train_iter, val_iter, test_iter, text_field = utils.load_ptb(
        ptb_path='data3.zip',
        ptb_dir='data3',
        bptt_len=args.context_size,
        batch_size=args.batch_size,
        gpu=args.GPU,
        reuse=False, repeat=False,
        shuffle=True
    )

    lr = args.initial_lr
    model = CondCopy(text_field.vocab.vectors, args.context_size, args.dropout) 

    # Specify embedding weights
    embedding_dim = (model.vocab_size, model.hidden_size)
    if args.init_weights == 'rand_norm':
        model.embedding_layer.weight.data = \
            Tensor(np.random.normal(size=embedding_dim))
        print('Initializing random normal weights for embedding')
    elif args.init_weights == 'rand_unif':
        model.embedding_layer.weight.data = \
            Tensor(np.random.uniform(size=embedding_dim))
        print('Initializing random uniform weights for embedding')
    elif args.init_weights == 'ones':
        model.embedding_layer.weight.data = \
            Tensor(np.ones(shape=embedding_dim))
        print('Initializing all ones as weights for embedding')
    elif args.init_weights == 'zeroes':
        model.embedding_layer.weight.data = \
            Tensor(np.zeros(shape=embedding_dim))
        print('Initializing all zeroes as weights for embedding')
    else:
        raise ValueError('{} is not a valid embedding weight \
                          initializer'.format(args.init_weights))
    model.output_shortlist.weight.data = model.embedding_layer.weight.data
    
    location_dim = (model.hidden_size, model.hidden_size)
    model.output_location.weight.data = \
        Tensor(np.random.normal(size=location_dim))

    switch_dim = (1, model.hidden_size)
    model.switch.weight.data = \
        Tensor(np.random.normal(size=switch_dim))

    # Specify optimizer
    if args.optimizer == "Adamax":
        print("Optimizer: Adamax")
        optimizer = optim.Adamax(model.get_train_parameters(),
                                 lr=lr, weight_decay=args.l2)
    elif args.optimizer == "Adam":
        print("Optimizer: Adam")
        optimizer = optim.Adam(model.get_train_parameters(),
                               lr=lr, weight_decay=args.l2)
    elif args.optimizer == "SGD":
        print("Optimizer: SGD")
        optimizer = optim.SGD(model.get_train_parameters(),
                              lr=lr, weight_decay=args.l2)
    else:
        raise ValueError('{} is not a valid optimizer'.format(args.optimizer))

    # load model from file
    if args.resume != "NONE":
        filename = os.path.join(args.model_dir, args.resume)
        if os.path.isfile(filename):
            print("=> loading checkpoint %s" % filename)
            checkpoint = torch.load(filename)
            args.start_epoch = checkpoint["start_epoch"]
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            print("=> loaded checkpoint %s (start at epoch %d)"
                  % (filename, args.start_epoch))
        else:
            print("=> no checkpoint found at %s" % filename)
        # just test and return if mode is test
        if args.mode == "test":
            test_perp = evaluate(model, test_iter, text_field, args)
            print("TEST PERPLEXITY %.5lf" % test_perp)
            return

    # train and evaluate
    print("Model: %s" % model)
    val_perps = []
    for epoch in range(args.start_epoch, args.epochs):
        model, optimizer, train_perp = train(model, optimizer, train_iter,
                                             text_field, args)
        print("TRAIN [EPOCH %d]: PERPLEXITY %.5lf" % (epoch, train_perp))
        val_perp = evaluate(model, val_iter, text_field, args)
        print("VALIDATE [EPOCH %d]: PERPLEXITY %.5lf" % (epoch, val_perp))
        val_perps.append(val_perp)

        # adjust learning rate
        if len(val_perps) > args.adapt_lr_epoch and \
           np.min(val_perps[-args.adapt_lr_epoch:]) > \
           np.min(val_perps[:-args.adapt_lr_epoch]):
            #if lr <= 0.00003:
            lr *= 0.5
            #else:
                #lr = 0.00003
            print("=> changing learning rate to %.8lf" % lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            # test model every 5 epochs
        if epoch % 5 == 0:
            test_perp = evaluate(model, test_iter, text_field, args)
            print("TEST [EPOCH %d]: PERPLEXITY %.5lf" % (epoch, test_perp))

        # saving model
        def save_checkpoint(state, filename):
            print("=> saving current model to checkpoint %s" % filename)
            torch.save(state, filename)

        checkpoint_name = os.path.join(args.model_dir, "%s-epoch%d"
                                       % (args.model_suffix, epoch))
        save_checkpoint({
            'start_epoch': epoch + 1,
            'args': args,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, checkpoint_name)

    # test trained model
    test_perp = evaluate(model, test_iter, text_field, args)
    print("TEST [EPOCH %d]: PERPLEXITY %.5lf" % (epoch, test_perp))


if __name__ == "__main__":
    main()
