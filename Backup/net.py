
from torchinfo import summary
import torch
import torch.nn as nn

class LSTMNet(nn.Module):
    def __init__(self, vocab_size=20, embed_dim=300, hidden_dim=512, num_layers=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.encoder = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.decoder = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        embed = self.embedding(x)
        out, hidden = self.encoder(embed)
        out = self.decoder(out)
        out = out.view(-1, out.size(2))
        return out, hidden

summary(
    LSTMNet(),
    (1, 100),
    dtypes=[torch.long],
    verbose=2,
    col_width=16,
    col_names=["kernel_size", "output_size", "num_params", "mult_adds"],
    row_settings=["var_names"],
)


from torchviz import make_dot


model = LSTMNet()
dummy_input = torch.randint(0, 20, (1, 100), dtype=torch.long)  # integers in vocab range
logits, hidden = model(dummy_input)                             # forward pass
graph = make_dot(logits, params=dict(model.named_parameters())) # only pass logits
graph.render("LSTMNet_graph", format="png", cleanup=True)

