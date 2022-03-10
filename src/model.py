import torch
from torch import nn
from torchmetrics.functional import confusion_matrix
import pytorch_lightning as pl

class CSIModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.criterion = nn.BCELoss()
        self.capture_rnn = nn.Sequential(
            nn.Linear(config['capture_input_dim'], config['d_Wa']),
            nn.Tanh(),
            nn.Dropout(config['dropout']),
            nn.LSTM(
                input_size=config['d_Wa'], 
                hidden_size=config['d_lstm'],
                num_layers=1, 
                batch_first=True)
        )
        self.capture_proj = nn.Sequential(
            nn.Linear(config['d_lstm'], config['d_Wr']),
            nn.Tanh(),
            nn.Dropout(config['dropout'])
        )
        self.score = nn.Sequential(
            nn.Linear(config['score_input_dim'], config['d_Wu']),
            nn.Tanh(),
            nn.Linear(config['d_Wu'], config['d_Ws']),
            nn.Sigmoid()
        )
        self.cls = nn.Sequential(
            nn.Linear(config['d_Ws'] + config['d_Wr'], 1),
            nn.Sigmoid()
        )
        
    def configure_optimizers(self):
        all_params = dict(self.named_parameters())
        wd_name = 'score.0.weight'
        wd_params = all_params[wd_name]
        del all_params[wd_name]
        return torch.optim.Adam(
            [
                {'params':  wd_params, 'weight_decay': self.config['weight_decay']}, 
                {'params': list(all_params.values())},
            ], 
            lr=self.config['lr']
        )
        

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)    
        
    def forward(self, x_capture, x_score):
        hc, (_, _) = self.capture_rnn(x_capture.float())
        hc = self.capture_proj(hc[:, -1])
        hs = self.score(x_score.float()).mean(dim=1)
        h = torch.cat([hc, hs], dim=1)
        return self.cls(h)
    
    def step(self, batch, mode='train'):
        x_capture, x_score, labels = batch
        labels = labels[:, None].float()
        logits = self.forward(x_capture, x_score)
        loss = self.criterion(logits, labels)
        
        preds = logits.clone()
        preds[preds >=0.5] = 1
        preds[preds < 0.5] = 0
        acc = (preds == labels).sum() / labels.shape[0]
        tn, fn, fp, tp = confusion_matrix(logits, labels.int(), num_classes=2, threshold=0.5).flatten()

        self.log(f'{mode}_loss', loss.item())
        self.log(f'{mode}_acc', acc.item())
        self.log(f'{mode}_tn', tn.item())
        self.log(f'{mode}_fn', fn.item())
        self.log(f'{mode}_fp', fp.item())
        self.log(f'{mode}_tp', tp.item())
        return {
            'loss':loss, 
            'acc':acc, 
            'tn':tn, 
            'fn':fn, 
            'fp':fp, 
            'tp':tp
        }
    
    def training_step(self, batch, batch_idx):
        return self.step(batch)
    
    def test_step(self, batch, batch_idx):
        return self.step(batch, mode='test')
    
    def validation_step(self, batch, batch_idx):
        return self.step(batch, mode='val')
    