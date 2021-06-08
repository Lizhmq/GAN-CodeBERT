import os
import pickle
import torch
from torch.utils.data import Dataset


class ClassifierDataset(Dataset):
    def __init__(self, tokenizer, args, logger, file_name='xxx.pkl', block_size=512):
        if args.local_rank == -1:
            local_rank = 0
            world_size = 1
        else:
            local_rank = args.local_rank
            world_size = torch.distributed.get_world_size()

        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        cached_file = os.path.join(args.output_dir, file_name[:-4] +"_blocksize_%d" %
                                   (block_size)+"_wordsize_%d" % (world_size)+"_rank_%d" % (local_rank))
        if os.path.exists(cached_file):
            with open(cached_file, 'rb') as handle:
                datas = pickle.load(handle)
                self.inputs, self.labels = datas["x"], datas["y"]
        else:
            self.inputs = []
            self.labels = []
            datafile = os.path.join(args.data_dir, file_name)
            inputs, labels = pickle.load(open(datafile, "rb"))

            length = len(inputs)

            for idx, (data, label) in enumerate(zip(inputs, labels)):
                if idx % world_size == local_rank:
                    code = " ".join(data)
                    code_tokens = tokenizer.tokenize(code)[:block_size-2]
                    code_tokens = [tokenizer.cls_token] + \
                        code_tokens + [tokenizer.sep_token]
                    code_ids = tokenizer.convert_tokens_to_ids(code_tokens)
                    padding_length = block_size - len(code_ids)
                    code_ids += [tokenizer.pad_token_id] * padding_length
                    self.inputs.append(code_ids)
                    self.labels.append(label)

                # if idx % (length // 10) == 0:
                #     percent = idx / (length//10) * 10
                #     logger.warning("Rank %d, load %d" % (local_rank, percent))

            with open(cached_file, 'wb') as handle:
                pickle.dump({"x": self.inputs, "y": self.labels},
                            handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, item):
        return torch.tensor(self.inputs[item]), torch.tensor(self.labels[item])
