try:
    from transformer_modules.encoder import Encoder as encoder
except:
    try:
        from model.transformer_modules.encoder import Encoder as encoder
    except:
        from encoder import Encoder as encoder

import torch


class Encoder(encoder):
    def __init__(self, d_model, heads, d_ff = 2048, dropout=0.1, num_layers=1):
        super(Encoder, self).__init__(d_model, heads, d_ff, dropout, num_layers)
        self.class_token = torch.nn.Parameter(torch.randn(1, 1, d_model))
        self.query_embed = torch.nn.Embedding(1, d_model)
        self.vid_embed = torch.nn.Embedding(1, d_model)
        self.linear = torch.nn.Linear(d_model, 1)   

    def forward(self, vid, query, mask=None):
        v_bs, v_seq, v_dim = vid.shape
        vid = vid.view(v_bs*v_seq, v_dim)
        query = query.squeeze(1)
        # Add query and video embeddings
        query = query + self.query_embed.weight
        vid = vid + self.vid_embed.weight
        query = query.unsqueeze(1)
        vid = vid.view(v_bs, v_seq, v_dim)
        # Add the class token to the input
        x = torch.cat((self.class_token.repeat(v_bs,1,1), query, vid), dim=1)
        x = super(Encoder, self).forward(x, mask)
        # Apply the linear layer
        x = self.linear(x[:, 0])
        return x
    
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Encoder(512, 8).to(device)
    vid = torch.rand(8, 10, 512).to(device)
    query = torch.rand(8, 1, 512).to(device)
    output = model(vid, query)
    print(output.shape)

if __name__ == '__main__':
    main()
