import os
import argparse
from timeit import default_timer as timer

import torch
import torch.nn as nn

import finetune_layer
from utils import utils, dataloader


parser = argparse.ArgumentParser()

parser.add_argument('--GPU_ID', type=int)

parser.add_argument('--data_dir', type=str, default='./WinoBias_Dataset/processed')
parser.add_argument('--model_dir', type=str, default='./experiments/ckpts')

parser.add_argument('--layer_hidden_dims', type=int, nargs='+', default=[1024, 256, 128, 16, 3])
parser.add_argument('--bert_hidden_dim', type=int, default=1024)
parser.add_argument('--max_len', type=int, default=22)

parser.add_argument('--train_batch_size', type=int, default=64)
parser.add_argument('--test_batch_size', type=int, default=1024)
parser.add_argument('--learning_rate', type=float, default=5e-3)
parser.add_argument('--weight_decay', type=float, default=0)


def test(model, test_data_iterator, loss_fn, comment):
    
    # set model to evaluating mode
    model.eval()

    # fetch test data
    # batch size should be bigger or equal than the test size

    input_batch, target_batch = next(test_data_iterator)

    # input_batch = [test_size, max_len, bert_hidden_dim]
    # target_batch = [test_size, max_len]

    pred_batch = model(input_batch)

    # pred_batch = [batch_size, max_len, num_tags]

    # print examples
    print("")
    print("* Test samples:")
    for i in range(10):
        target = ' '.join([str(int(t)) for t in target_batch[i] if t != -1])
        pred = ' '.join([str(int(p)) for p in torch.argmax(pred_batch[i], dim=1)])
        print(f"- target {i}    : {target}")
        print(f"- prediction {i}: {pred}")

    # calculate average
    running_avg_acc = utils.running_avg_accuracy()
    running_avg_acc.update(pred_batch, target_batch)
    accuracy = running_avg_acc()

    num_tags = pred_batch.size(2)

    loss = loss_fn(pred_batch.view(-1, num_tags), target_batch.view(-1)).item()
    
    return accuracy, loss


if __name__ == '__main__':

    args = parser.parse_args()

    # set device
    cuda = torch.cuda.is_available()
    if cuda:
        device = torch.device(f"cuda:{args.GPU_ID}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    # set model path
    comment = f'BATCH{args.train_batch_size}_HIDDENS_{"_".join([str(i) for i in args.layer_hidden_dims])}_LR{args.learning_rate}_WD{args.weight_decay}'
    ckpt_path = os.path.join(args.model_dir, f'{comment}/model.best.ckpt')

    # Loading the train and validation datasets
    data_loader = dataloader.DataLoader(args.data_dir, args.max_len)
    data = data_loader.data_split() # data is dict with 'train', 'valid', 'test' as keys
    test_data = data['test']
    test_data_iterator = data_loader.data_iterator(test_data, args.test_batch_size, device, cuda, shuffle=False)

    # Define model and load the trained parameters
    print("- Load Model ...")
    start = timer()
    model = finetune_layer.finetune_layer(args.layer_hidden_dims, args.bert_hidden_dim, data_loader.num_tags)
    checkpoint = utils.load_checkpoint(ckpt_path, model)
    model.to(device)
    end = timer()
    print("- Loading model completed ({:04.3f}s elapsed)".format(end-start))

    # Define loss
    loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
    
    test_acc, test_loss = test(model, test_data_iterator, loss_fn, comment)

    print("")
    print(f"* Test accuracy : {test_acc}")
    print(f"* Test loss : {test_loss}")
