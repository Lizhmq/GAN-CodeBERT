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

    def __init__(self, hidden, classes, args, device):
        super().__init__()
        self.args = args
        self.device = device
        self.classes = classes
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden, classes)
        self.bert = RobertaModel.from_pretrained(self.args.pretrain_dir)
        self._init_cls_weight()


    def _init_cls_weight(self, initializer_range=0.02):
        for layer in (self.classifier,):
            layer.weight.data.normal_(mean=0.0, std=initializer_range)
            if layer.bias is not None:
                layer.bias.data.zero_()
    

    def save_pretrained(self, path):
        self.bert.save_pretrained(path)
        torch.save(self.classifier.state_dict(), os.path.join(path, "cls.bin"))
        
    def from_pretrained(self, path):
        self.bert = RobertaModel.from_pretrained(path)
        self.classifier.load_state_dict(torch.load(os.path.join(path, "cls.bin"), map_location=self.device))
        return self

    def forward(self, input_ids, y=None, input_mask=None):
        sequence_output = self.bert(input_ids, input_mask)[0]
        output0 = self.dropout(sequence_output[:, 0, :])
        batch_size, max_len, feat_dim = sequence_output.shape
        logits0 = self.classifier(output0)
        if y != None:
            loss_fct0 = nn.CrossEntropyLoss()
            loss0 = loss_fct0(logits0, y)
            return loss0
        else:
            return logits0


def build_model(args, load_path=None):
    model = BaseModel(768, 2, args, args.device)
    if load_path is not None:
        model = model.from_pretrained(load_path).to(args.device)
    return model
    