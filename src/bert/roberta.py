"""
 Jason's Extension
 RoBERTa module for Knowledge Graph Embedding
"""


import torch
from torch import nn
from transformers import RobertaTokenizerFast, RobertaModel

MAX_LENGTH = 50

class RoBertaEmbedding(nn.Module):
    def __init__(self) -> None:
        super(RoBertaEmbedding, self).__init__()

        self.bert = RobertaModel.from_pretrained('roberta-base')
        self.tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base', add_prefix_space=True)

        # freeze roberta for now
        for param in self.bert.parameters():
            param.requires_grad = False

    def forward(self, head_entity, relation, tail_entity):
        batch_tokens = []
        batch_segment_ids = []
        for h, r, t in zip(head_entity, relation, tail_entity):
            h_token = self.tokenizer.tokenize(h)
            r_token = self.tokenizer.tokenize(r)
            t_token = self.tokenizer.tokenize(t)
            tokens = ["[CLS]"] + h_token + ["[SEP]"] + r_token + ["[SEP]"] + t_token + ["[SEP]"]
            segment_ids = [-1] + len(h_token)*[0] + [-1] + len(r_token)*[1] + [-1] + len(t_token)*[2] + [-1]
            for _ in range(MAX_LENGTH - len(segment_ids)):
                segment_ids.append(-1)
            batch_segment_ids.append(segment_ids)
            batch_tokens.append(tokens)

        
        batch_tokens = self.tokenizer.batch_encode_plus(
                batch_tokens,
                max_length = MAX_LENGTH,
                add_special_tokens=False,
                padding='max_length', 
                truncation=True,
                return_tensors='pt',
                is_split_into_words=True
            )
        
        for t in batch_tokens:
            batch_tokens[t] = batch_tokens[t].to(self.bert.device)

        outputs = self.bert(**batch_tokens)

        last_hidden_states = outputs[0]

        batch_segment_ids = torch.tensor(batch_segment_ids).unsqueeze(-1).to(self.bert.device)

        # Extract the embeddings of head entity by span means
        head_entity_embeddings = (last_hidden_states * (batch_segment_ids == 0)).sum(dim=1) / (batch_segment_ids == 0).sum(dim=1)
    
        # Extract the embeddings of relation by span means
        relation_embeddings = (last_hidden_states * (batch_segment_ids == 1)).sum(dim=1) / (batch_segment_ids == 1).sum(dim=1)

        # Extract the embeddings of tail entity by span means
        tail_entity_embeddings = (last_hidden_states * (batch_segment_ids == 2)).sum(dim=1) / (batch_segment_ids == 2).sum(dim=1)

        return head_entity_embeddings, relation_embeddings, tail_entity_embeddings




