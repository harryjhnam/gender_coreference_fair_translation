import os
import argparse

from tqdm import trange

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import finetune_layer
from utils import utils, dataloader


parser = argparse.ArgumentParser()

parser.add_argument('--GPU_ID', type=int)

parser.add_argument('--data_dir', type=str, default='./WinoBias_Dataset/processed')
parser.add_argument('--model_dir', type=str, default='./experiments/ckpts')
parser.add_argument('--tensorboard_dir', type=str, default='./experiments/runs')

parser.add_argument('--layer_hidden_dims', type=int, nargs='+', default=[1024, 256, 128, 16, 3])
parser.add_argument('--bert_hidden_dim', type=int, default=1024)
parser.add_argument('--max_len', type=int, default=22)

parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--train_batch_size', type=int, default=64)
parser.add_argument('--eval_batch_size', type=int, default=1024)
parser.add_argument('--learning_rate', type=float, default=5e-3)
parser.add_argument('--weight_decay', type=float, default=0)

parser.add_argument('--random_seed', type=int, default=1234)



def train(model, data_iterator, optimizer, loss_fn, writer, epoch, num_steps):

    # set model to training mode
    model.train()

    # set running average accuracy object
    avg_accuracy = utils.running_avg_accuracy()

    t = trange(num_steps)
    for i in t:
        # fetch the next training batch
        input_batch, target_batch = next(data_iterator)

        # input_batch = [batch_size, max_len, bert_hidden_dim]
        # target_batch = [batch_size, max_len]

        pred_batch = model(input_batch)

        # pred_batch = [batch_size, max_len, num_tags]

        num_tags = pred_batch.size(2)

        loss = loss_fn(pred_batch.view(-1, num_tags), target_batch.view(-1))

        # clear the previous gradients and update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update the average accuracy
        avg_accuracy.update(pred_batch, target_batch)

        # update the postfix of the progress bar
        t.set_postfix(loss='{:05.3f}'.format(loss.item()))

    # update writer once in an epoch
    writer.add_scalar('Train/loss', loss.item(), epoch+1)
    writer.add_scalar('Train/accuracy', avg_accuracy(), epoch+1)
    for n, p in model.named_parameters():
        writer.add_histogram(f'Train/parameters/{n}', p.data, epoch+1)
        writer.add_histogram(f'Train/gradients/{n}', p.grad, epoch+1)

    return avg_accuracy()


def evaluate(model, data_iterator, loss_fn, writer, epoch, num_steps):

    # set model to evaluating mode
    model.eval()

    # set running average accuracy object
    avg_accuracy = utils.running_avg_accuracy()

    for i in range(num_steps):
        # fetch the next evaluating batch
        input_batch, target_batch = next(data_iterator)

        # input_batch = [batch_size, max_len, bert_hidden_dim]
        # target_batch = [batch_size, max_len]

        pred_batch = model(input_batch)

        # pred_batch = [batch_size, max_len, num_tags]

        num_tags = pred_batch.size(2)

        loss = loss_fn(pred_batch.view(-1, num_tags), target_batch.view(-1))

        # update the average accuracy
        avg_accuracy.update(pred_batch, target_batch)

    # update writer once in an epoch
    writer.add_scalar('Eval/loss', loss.item(), epoch+1)
    writer.add_scalar('Eval/accuracy', avg_accuracy(), epoch+1)

    return avg_accuracy()


def train_and_evaluate(model, train_data, valid_data, optimizer, loss_fn, writer, model_dir, comment):
    
    # best valid accuracy
    best_train_accuracy = 0.0
    best_valid_accuracy = 0.0

    for epoch in range(args.epochs):

        print(f"\n\n* Running Epoch {epoch}/{args.epochs} *")

        # Train the model
        train_data_iterator = data_loader.data_iterator(train_data, args.train_batch_size, device, cuda, shuffle=True)
        train_num_steps = train_data['size'] // args.train_batch_size + 1
        train_accuracy = train(model, train_data_iterator, optimizer, loss_fn, writer, epoch, train_num_steps)
        best_train_accuracy = max(best_train_accuracy, train_accuracy)

        # evaluate the model
        eval_data_iterator = data_loader.data_iterator(valid_data, args.eval_batch_size, device, cuda, shuffle=False)
        eval_num_steps = valid_data['size'] // args.eval_batch_size + 1
        valid_accuracy = evaluate(model, eval_data_iterator, loss_fn, writer, epoch, eval_num_steps)

        # update the best accuracy and best model
        if valid_accuracy > best_valid_accuracy:

            print('- found new best accuracy!')

            best_valid_accuracy = valid_accuracy

            # save the best model
            best_model_path = os.path.join(model_dir, f'{comment}/model.best.ckpt')

            utils.save_checkpoint(state = {'epoch': epoch+1,
                                           'comment' : comment,
                                           'accuracy' : best_valid_accuracy,
                                           'state_dict': model.state_dict(),
                                           'optim_dict': optimizer.state_dict()},
                                  model_dir = model_dir, 
                                  model_path = best_model_path)

            
    # save the last model
    last_model_path = os.path.join(model_dir, f'{comment}/epoch{epoch}.last.ckpt')
    utils.save_checkpoint(state = {'epoch': epoch+1,
                                   'comment' : comment,
                                   'accuracy' : valid_accuracy,
                                   'state_dict': model.state_dict(),
                                   'optim_dict': optimizer.state_dict()},
                          model_dir = model_dir, 
                          model_path = last_model_path)


    return best_train_accuracy, best_valid_accuracy

if __name__ == '__main__':
    
    args = parser.parse_args()

    # set device
    cuda = torch.cuda.is_available()
    if cuda:
        device = torch.device(f"cuda:{args.GPU_ID}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    
    # set random seed for reproducible experiments
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args.random_seed)
    if cuda:
        torch.cuda.manual_seed(args.random_seed)
        torch.cuda.manual_seed_all(args.random_seed)

    # set tensorboard writer
    comment = f'BATCH{args.train_batch_size}_HIDDENS_{"_".join([str(i) for i in args.layer_hidden_dims])}_LR{args.learning_rate}_WD{args.weight_decay}'
    writer = SummaryWriter(os.path.join(args.tensorboard_dir, comment))

    print('- Loading data...')

    # Loading the train and validation datasets
    data_loader = dataloader.DataLoader(args.data_dir, args.max_len, args.random_seed)
    data = data_loader.data_split() # data is dict with 'train', 'valid', 'test' as keys
    train_data = data['train']    
    valid_data = data['valid']

    print('- Defining and initializing model')

    # Define model and initialize parameters
    model = finetune_layer.finetune_layer(args.layer_hidden_dims, args.bert_hidden_dim, data_loader.num_tags)
    model.init_weights()
    model.to(device)

    # Define optimizer
    optimizer = optim.Adam(model.parameters(), 
                           lr = args.learning_rate,
                           weight_decay = args.weight_decay)

    # Define loss
    loss_fn = nn.CrossEntropyLoss(ignore_index=-1)

    print(f'\n- Start Training for {args.epochs} epochs')

    best_train_acc, best_valid_acc = train_and_evaluate(model, train_data, valid_data, optimizer, loss_fn, writer, args.model_dir, comment)