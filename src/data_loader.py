import logging
import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizer
from create_dataset import MOSI, MOSEI, PAD

# 设置 Logger
logger = logging.getLogger(__name__)

class MSADataset(Dataset):
    def __init__(self, config):
        self.config = config
        
        # 动态加载数据集
        if "mosi" in str(config.dataset).lower():
            dataset = MOSI(config)
        elif "mosei" in str(config.dataset).lower():
            dataset = MOSEI(config)
        else:
            raise ValueError("Dataset not defined correctly")

        self.data, self.word2id, _ = dataset.get_data(config.mode)
        self.len = len(self.data)
        config.word2id = self.word2id

    @property
    def tva_dim(self):
        t_dim = 768
        return t_dim, self.data[0][0][1].shape[1], self.data[0][0][2].shape[1]

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len

def get_loader(args, config, shuffle=True):
    bert_tokenizer = BertTokenizer.from_pretrained(args.bert_path, do_lower_case=True)
    personality_tokenizer = BertTokenizer.from_pretrained(args.personality_bert_path)

    dataset = MSADataset(config)
    config.data_len = len(dataset)
    config.tva_dim = dataset.tva_dim

    if config.mode == 'train':
        args.n_train = len(dataset)
    elif config.mode == 'valid':
        args.n_valid = len(dataset)
    elif config.mode == 'test':
        args.n_test = len(dataset)

    def custom_pad_sequence(sequences, target_len, batch_first=False, padding_value=0.0):
        max_size = target_len
        trailing_dims = sequences[0].size()[1:]
        if batch_first:
            out_dims = (len(sequences), max_size) + trailing_dims
        else:
            out_dims = (max_size, len(sequences)) + trailing_dims
        out_tensor = sequences[0].new_full(out_dims, padding_value)
        for i, tensor in enumerate(sequences):
            length = tensor.size(0)
            if batch_first:
                out_tensor[i, :length, ...] = tensor
            else:
                out_tensor[:length, i, ...] = tensor
        return out_tensor

    def collate_fn(batch):
        batch = sorted(batch, key=lambda x: len(x[0][3]), reverse=True)
        v_lens, a_lens, labels, ids = [], [], [], []

        for sample in batch:
            if len(sample[0]) > 4: # Aligned
                v_lens.append(torch.tensor([sample[0][4]], dtype=torch.int))
                a_lens.append(torch.tensor([sample[0][5]], dtype=torch.int))
            else: # Non-aligned
                v_lens.append(torch.tensor([len(sample[0][3])], dtype=torch.int))
                a_lens.append(torch.tensor([len(sample[0][3])], dtype=torch.int))
            labels.append(torch.from_numpy(sample[1]))
            ids.append(sample[2])

        vlens = torch.cat(v_lens)
        alens = torch.cat(a_lens)
        labels = torch.cat(labels, dim=0)

        if labels.size(1) == 7:
            labels = labels[:, 0][:, None]

        sentences = pad_sequence([torch.LongTensor(sample[0][0]) for sample in batch], padding_value=PAD)
        visual = custom_pad_sequence([torch.FloatTensor(sample[0][1]) for sample in batch], target_len=vlens.max().item())
        acoustic = custom_pad_sequence([torch.FloatTensor(sample[0][2]) for sample in batch], target_len=alens.max().item())

        SENT_LEN = 50
        bert_details = []
        personality_details = []

        for sample in batch:
            text = " ".join(sample[0][3])
            encoded_bert_sent = bert_tokenizer.encode_plus(
                text, max_length=SENT_LEN, add_special_tokens=True, truncation=True, padding='max_length')
            personality_bert_sent = personality_tokenizer.encode_plus(
                text, max_length=SENT_LEN, add_special_tokens=True, truncation=True, padding='max_length')
            bert_details.append(encoded_bert_sent)
            personality_details.append(personality_bert_sent)

        bert_sentences = torch.LongTensor([sample["input_ids"] for sample in bert_details])
        bert_sentence_types = torch.LongTensor([sample["token_type_ids"] for sample in bert_details])
        bert_sentence_att_mask = torch.LongTensor([sample["attention_mask"] for sample in bert_details])

        p_bert_sentences = torch.LongTensor([sample["input_ids"] for sample in personality_details])
        p_bert_sentence_types = torch.LongTensor([sample["token_type_ids"] for sample in personality_details])
        p_bert_sentence_att_mask = torch.LongTensor([sample["attention_mask"] for sample in personality_details])
        personality_bert_feature = [p_bert_sentences, p_bert_sentence_types, p_bert_sentence_att_mask]

        lengths = torch.LongTensor([len(sample[0][0]) for sample in batch])

        if (vlens <= 0).sum() > 0:
            vlens[vlens == 0] = 1

        return sentences, visual, vlens, acoustic, alens, labels, lengths, bert_sentences, bert_sentence_types, bert_sentence_att_mask, ids, personality_bert_feature

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=4,
        collate_fn=collate_fn,
        generator=torch.Generator()
    )

    return data_loader