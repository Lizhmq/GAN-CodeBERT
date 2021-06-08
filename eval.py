import argparse
import logging
import os
import random

import numpy as np
import torch
from transformers import (AdamW, RobertaTokenizer, get_linear_schedule_with_warmup)
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from sklearn.metrics import classification_report, accuracy_score
from torch.utils.tensorboard import SummaryWriter
from dataset import ClassifierDataset
from model import BaseModel, build_model


logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)



def valid_acc(args, model, tokenizer, eval_dataset, prefix="", eval_when_training=False):
    eval_output_dir = args.output_dir

    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)

    eval_sampler = SequentialSampler(
        eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, drop_last=True)

    model.eval()

    predicts = []
    golds = []
    for batch in tqdm(eval_dataloader):
        x, y = batch
        golds.extend(y)
        x = x.to(args.device)
        y = y.to(args.device)
        x_mask = x.ne(tokenizer.pad_token_id).to(x)

        with torch.no_grad():
            output = model(x, None, x_mask)
            predicts.extend(torch.argmax(output, dim=1).cpu().numpy())

    print(classification_report(golds, predicts, digits=4))
    acc = accuracy_score(golds, predicts)
    results = {"accuracy": acc}
    return results




def main():

    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data path.")
    ## Other parameters
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--pretrain_dir", default="", type=str,
                        help="The dir of pretrained model.")
    parser.add_argument("--block_size", default=384, type=int,
                        help="Optional input sequence length after tokenization."
                             "The training dataset will be truncated in block of this size for training."
                             "Default to the model max input length for single sentence inputs (take into account special tokens).")
    parser.add_argument("--eval_batch_size", default=12, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    args = parser.parse_args()
    args.local_rank = -1
    args.device = torch.device("cuda", 0)
    
    model_list = ["./save/orig/checkpoint-600-0.59", \
                    "./save/orig/checkpoint-1200-0.6335", \
                    "./save/orig/checkpoint-1800-0.6438", \
                    "./save/orig/checkpoint-2400-0.618"]
    for _, model_path in enumerate(model_list):
        args.pretrain_dir = model_path
        tokenizer = RobertaTokenizer.from_pretrained(args.pretrain_dir)
        test_dataset = ClassifierDataset(tokenizer, args, logger, file_name="test.pkl", block_size=512)
        model = build_model(args, args.pretrain_dir)
        valid_acc(args, model, tokenizer, test_dataset)
        

if __name__ == "__main__":
    main()