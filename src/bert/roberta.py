"""
 Jason's Extension
 RoBERTa module for Knowledge Graph Embedding
"""


import torch
from torch import nn
from transformers import RobertaTokenizerFast, RobertaModel, RobertaForSequenceClassification
import torch.nn.functional as F


MAX_LENGTH = 50

class RoBertaEmbedding(nn.Module):
    def __init__(self, labels) -> None:
        super(RoBertaEmbedding, self).__init__()

        self.bert = RobertaForSequenceClassification.from_pretrained('roberta-base', hidden_dropout_prob=0.2, num_labels=labels)
        self.tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base', add_prefix_space=True)

        # # freeze roberta for now
        for name, param in self.bert.named_parameters():
            if name.split('.')[0] == 'classifier' or (name.split('.')[3] in ('4','5')):
            # if name.split('.')[0] == 'classifier' or (name.split('.')[3] == '23' and name.split('.')[4] in ('output','intermediate')):
                param.requires_grad = True
            else:
                param.requires_grad = False
            # if name.split('.')[0] == 'embeddings':
            #     param.requires_grad = False
            # else:
            #     param.requires_grad = True


    def forward(self, head_entity, relation):
        batch_tokens = []
        for h, r in zip(head_entity, relation):
            h_token = self.tokenizer.tokenize(h.replace('_',' '))
            r_token = self.tokenizer.tokenize(r.replace('_',' '))
            tokens = ["[CLS]"] + h_token + ["[SEP]"] + r_token + ["[SEP]"]
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

        return F.sigmoid(outputs.logits)

        # last_hidden_states = outputs[0]

        # return last_hidden_states[:,0,:]

        # batch_segment_ids = torch.tensor(batch_segment_ids).unsqueeze(-1).to(self.bert.device)

        # # Extract the embeddings of head entity by span means
        # head_entity_embeddings = (last_hidden_states * (batch_segment_ids == 0)).sum(dim=1) / (batch_segment_ids == 0).sum(dim=1)
    
        # # Extract the embeddings of relation by span means
        # relation_embeddings = (last_hidden_states * (batch_segment_ids == 1)).sum(dim=1) / (batch_segment_ids == 1).sum(dim=1)

        # # Extract the embeddings of tail entity by span means
        # tail_entity_embeddings = (last_hidden_states * (batch_segment_ids == 2)).sum(dim=1) / (batch_segment_ids == 2).sum(dim=1)

        # return head_entity_embeddings, relation_embeddings, tail_entity_embeddings




