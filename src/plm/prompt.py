from openprompt.plms import load_plm
from openprompt.data_utils import InputExample

def get_template():
    return '{"placeholder":"text_a"} is related to {"mask"} through the relationship of {"placeholder":"text_b"}.'


def get_verbalizer(id2entity):
    res = []
    for i in range(len(id2entity)):
        entity = id2entity[i].lower().replace('_',' ')
        res.append([entity])
    return res


def wrap_data(head_entity, relation):
    data = []
    for h, r in zip(head_entity, relation):
        data.append(InputExample(text_a = h.replace('_',' '),text_b = r.replace('_',' '),guid = 1))
    return data


def to_device(data, device):
    for key in ["input_ids", "attention_mask", "decoder_input_ids", "loss_ids"]:
        if key in data:
            data[key] = data[key].to(device)
    return data