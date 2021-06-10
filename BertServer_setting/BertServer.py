"""
Start or Stop the `BertServer` from python

Arguments:
    --start (flag) : to start the BertServer
    --stop  (flat) : to stop the BertServer

Reference:
    https://github.com/hanxiao/bert-as-service#q-are-you-suggesting-using-bert-without-fine-tuning

Download the pretrained BERT model listed in the page below:
    https://github.com/hanxiao/bert-as-service

"""

from bert_serving.server.helper import get_args_parser
from bert_serving.server import BertServer

import os, argparse

parser = argparse.ArgumentParser()

# Start/Stop the server
parser.add_argument('--start', default=False, action='store_true')
parser.add_argument('--stop', default=False, action='store_true')

# Arguments for server settings
parser.add_argument('--bert_model_dir', type=str, default='../temp/uncased_L-24_H-1024_A-16/',
                    help='path to the pretrained model')
parser.add_argument('--port', type=str, default='5555', 
                    help='port for pushing data from client to server')
parser.add_argument('--port_out', type=str, default='5556',
                    help='port for publishing results from server to client')
parser.add_argument('--max_seq_len', type=str, default='24')



def start_BertServer(args):
    server_args = get_args_parser().parse_args(['-model_dir', args.bert_model_dir,
                                                '-port', args.port,
                                                '-port_out', args.port_out,
                                                '-max_seq_len', args.max_seq_len,
                                                '-pooling_strategy', 'NONE',
                                                '-mask_cls_sep'])

    server = BertServer(server_args)
    server.start()



if __name__ == '__main__':
    
    args = parser.parse_args()

    if args.start and args.stop:
        raise ValueError('Only one of --start or --stop flag should be given; both flags are given.')

    if not (args.start or args.stop):
        raise ValueError('Either of --start or --stop flag is required; No flag is given.')

    
    # Start BertServer
    if args.start:
        print(args.bert_model_dir, args.port, args.port_out, args.max_seq_len)
        start_BertServer(args)

    if args.stop:
        BertServer.shutdown(int(args.port))