import torch.nn as nn

class finetune_layer(nn.Module):
    def __init__(self, layer_hidden_dims, bert_hidden_dim, num_tags):
        """
        layer_hidden_dims (list[int])   : list of ints which are the input dims of all layers and final output dim
                                          len(layer_hidden_dims) is the (n_layers + 1)
        bert_hidden_dim (int)           : hidden dim of bert word embeddings
        num_tags (int)                  : the number of tags in labels
        """
        super(finetune_layer, self).__init__()
        
        assert layer_hidden_dims[0] == bert_hidden_dim, f"The first hidden dimension ({layer_hidden_dims[0]}) should be the same as the hidden dimension of BERT ({bert_hidden_dim})"
        assert layer_hidden_dims[-1] == num_tags, f"The last hidden dimension ({layer_hidden_dims[-1]}) should be same as the number of tags ({num_tags})"

        n_layers = len(layer_hidden_dims) - 1
        
        self.layers = nn.ModuleList( [ nn.Linear(layer_hidden_dims[i], layer_hidden_dims[i+1]) for i in range(n_layers) ] )     
        self.ReLU = nn.ReLU()
        self.softmax = nn.Softmax(dim=2)

    def init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                nn.init.zeros_(p.data)

    def forward(self, inputs):

        # inputs = [batch_size, seq_len, bert_hidden_dim]

        x = inputs
        for layer in self.layers:
            x = self.ReLU(layer(x))

        # x = [batch_size, seq_len, num_tags]

        return self.softmax(x)      
