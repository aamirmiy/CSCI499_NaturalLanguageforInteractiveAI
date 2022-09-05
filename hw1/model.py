# IMPLEMENT YOUR MODEL CLASS HERE
import torch
import torch.nn as nn


class semanticNet(nn.Module):
    def __init__(self, device,vocab_size, output_size1, output_size2, embedding_dim, hidden_dim, n_layers): #try with dropout afterwards
        super(semanticNet, self).__init__()
        self.device = device
        self.output_size1 = output_size1 #check this afterwards
        self.output_size2 = output_size2
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, output_size1)
        self.fc2 = nn.Linear(hidden_dim, output_size2)
        #self.softmax = nn.Softmax()

    def forward(self, x, hidden):
        batch_size = x.size(0)
        x = x.long()
        embeds = self.embedding(x)
        lstm_out, hidden = self.lstm(embeds, hidden)
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)

        #out = self.dropout(lstm_out)
        out1 = self.fc1(lstm_out)
        out2 = self.fc2(lstm_out)
        #output1 = self.softmax(out1)
        #output2 = self.softmax(out2)
        out1 = out1.view(batch_size, -1)
        out1 = out1[:,-1]
        out2 = out2.view(batch_size, -1)
        out2 = out2[:,-1]
        # output1 = output1.view(batch_size, -1)
        # output1 = output1[:,-1]
        # output2 = output2.view(batch_size, -1)
        # output2 = output2[:,-1]
        return out1, out2
    
    # def init_hidden(self,batch_size):
    #     weight = next(self.parameters()).data
    #     hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device),
    #                   weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device))
    #     return hidden

        

