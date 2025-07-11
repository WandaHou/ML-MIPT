from transformers import LlamaConfig, LlamaForCausalLM
import torch
from math import ceil

class NN(torch.nn.Module):
    "Linear Neural network"
    def __init__(self, input_size, output_size, dims, dropout_prob=0.):
        super().__init__()
        self.input_size, self.output_size = input_size, output_size
        self.dims = dims
        layers = []
        for d in dims:
            layers.append(torch.nn.Linear(input_size, d))
            layers.append(torch.nn.ReLU())
            #layers.append(torch.nn.SiLU())
            #layers.append(torch.nn.Tanh())
            layers.append(torch.nn.Dropout(dropout_prob))
            input_size = d
        layers.append(torch.nn.Linear(dims[-1], output_size))
        self.ffn = torch.nn.Sequential(*layers)
        #self.to(dtype)

    def forward(self, x):
        return self.ffn(x)
    
class LlamaPredictor(torch.nn.Module):
    
    def __init__(self, L_max, L, n_embd, n_layer, n_head, vocab_size, dropout_prob):
        super().__init__()
        self.L_max = L_max
        configuration = LlamaConfig(vocab_size=vocab_size,  # Size of the vocabulary
                                    hidden_size=n_embd,   # Dimensionality of the embeddings and hidden states
                                    intermediate_size=n_embd,  # Dimensionality of the feed-forward layer
                                    num_hidden_layers=n_layer,  # Number of hidden layers in the transformer
                                    num_attention_heads=n_head,  # Number of attention heads
                                    hidden_act='relu',  # Activation function
                                    max_position_embeddings=L_max,  # Maximum sequence length
                                    bos_token_id = 0,
                                    eos_token_id = 0,
                                    pad_token_id = 0,
                                    output_hidden_states=True,
                                    attention_dropout = dropout_prob
                                )
        self.GPT = LlamaForCausalLM(configuration)
        self.ffn = torch.nn.Linear(n_embd, 32)
        self.att_msk = torch.zeros(L_max+1).to(torch.int32)
        self.att_msk[:L] += 1
        self.att_msk[-1] += 1
        
    def forward(self, measure):
        attention_mask = self.att_msk.unsqueeze(0).expand(measure.shape[0], -1).to(device=measure.device)
        hidden_states = self.GPT(input_ids=measure, attention_mask=attention_mask).hidden_states[-1]
        out = self.ffn(hidden_states) # (batch, seq_length, 32)
        out = out[:,-1,:].view(-1, 2, 16) # (batch, 32)
        A_real = out[:,0]
        A_imag = out[:,1]
        A = (A_real + 1j * A_imag).view(-1, 4, 4)
        rho = A.mT.conj() @ A
        rho /= torch.diagonal(rho, dim1=-2, dim2=-1).sum(-1).view(-1, 1, 1)
        return rho.to(torch.complex128)