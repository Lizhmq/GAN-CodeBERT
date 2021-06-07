import logging
import os

import torch
from torch import nn
from transformers import RobertaModel


logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)



class BaseModel(nn.Module):

    def __init__(self, hidden, classes, bio_classes, lambd, args, device):
        super().__init__()
        self.args = args
        self.lambd = lambd
        self.device = device
        self.classes = classes
        self.bio_classes = bio_classes
        self.dropout = nn.Dropout(0.1)
        self.gen_classifier = nn.Linear(hidden, classes)
        self.bio_classifier = nn.Linear(hidden, bio_classes)
        self.bert = RobertaModel.from_pretrained(self.args.pretrain_dir)
        self._init_cls_weight()


    def _init_cls_weight(self, initializer_range=0.02):
        for layer in (self.gen_classifier, self.bio_classifier):
            layer.weight.data.normal_(mean=0.0, std=initializer_range)
            if layer.bias is not None:
                layer.bias.data.zero_()
    

    def save_pretrained(self, path):
        self.bert.save_pretrained(path)
        torch.save(self.gen_classifier.state_dict(), os.path.join(path, "gen_cls.bin"))
        torch.save(self.bio_classifier.state_dict(), os.path.join(path, "bio_cls.bin"))
        torch.save(self.crf.state_dict(), os.path.join(path, "crf.bin"))
        
    def from_pretrained(self, path):
        self.bert = RobertaModel.from_pretrained(path)
        self.gen_classifier.load_state_dict(torch.load(os.path.join(path, "gen_cls.bin"), map_location=self.device))
        self.bio_classifier.load_state_dict(torch.load(os.path.join(path, "bio_cls.bin"), map_location=self.device))
        self.crf.load_state_dict(torch.load(os.path.join(path, "crf.bin"), map_location=self.device))
        return self

    def forward1(self, input_ids, input_mask, y=None):
        output = self.bert(input_ids, input_mask)[0]
        output0 = self.dropout(output[:, 0, :])
        logits0 = self.gen_classifier(output0)
        if y == None:
            return logits0
        else:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits0, y).item()
            
            pred = torch.argmax(logits0, dim=1).cpu().numpy()
            gold = y.cpu().numpy()
            return pred, gold, loss

    def forward2(self, input_ids, y=None, input_mask=None, bio_mask=None):
        bio_mask = (bio_mask == 1)
        sequence_output = self.bert(input_ids, input_mask)[0]
        batch_size, max_len, feat_dim = sequence_output.shape
        valid_output = sequence_output.masked_select(bio_mask.unsqueeze(-1))
        valid_output = valid_output.view(-1, feat_dim)
        logits = self.bio_classifier(valid_output)
        pred = torch.tensor(self.crf.decode(logits.unsqueeze(0))[0], device=self.device)
        return pred
    
    def forward(self, input_ids, y=None, input_mask=None, labels=None, bio_mask=None):
        sequence_output = self.bert(input_ids, input_mask)[0]
        output0 = self.dropout(sequence_output[:, 0, :])
        batch_size, max_len, feat_dim = sequence_output.shape

        logits0 = self.gen_classifier(output0)
        if y != None:
            loss_fct0 = nn.CrossEntropyLoss()
            loss0 = loss_fct0(logits0, y)
            return loss0
        else:
            return logits0


def build_model(args, bio_size, load_path=None):
    model = BaseModel(768, 2, bio_size, args.lambd, args, args.device)
    if load_path is not None:
        model = model.from_pretrained(load_path).to(args.device)
    return model
    