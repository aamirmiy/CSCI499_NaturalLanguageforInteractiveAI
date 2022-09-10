import torch.nn as nn
import torch

class semanticNet(nn.Module):
    def __init__(self, device, output_size1, output_size2, embedding_dim, embedding_matrix, hidden_dim, n_layers): #try with dropout afterwards
        super(semanticNet, self).__init__()
        self.device = device
        self.output_size1 = output_size1 #number of classes in actions
        self.output_size2 = output_size2 #number of classes in targets
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        #model architecture
        self.embedding = nn.Embedding(embedding_matrix.shape[0], embedding_dim)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix,dtype=torch.float32))
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, output_size1)
        self.fc2 = nn.Linear(hidden_dim, output_size2)


    def forward(self, x):
        #x is input having shape (batch_size,sequence_length)
        
        #embedding input = (batch_size, sequence_length)
        embeds = self.embedding(x)
        #embedding output =(batch_size,sequence_length,embedding_dim)
        
        #lstm_input = (batch_size, sequence_length,embedding_dim)
        lstm_out, (h0,c0) = self.lstm(embeds)
        #lstm_out = (batch_size, sequence_length, hidden_dim)

        #h0 = (1, batch_size, hidden_dim)
 
        #fc_input = (batch_size,hidden_dim)     
        out1 = self.fc1(h0.squeeze(0))
        out2 = self.fc2(h0.squeeze(0))
       
        return out1, out2
