import torch


def torch_stack_dicts(list_of_dicts):
    keys = list_of_dicts[0].keys()
    output_dict = {}
    for k in keys:
        output_dict[k] = torch.stack([x[k] for x in list_of_dicts]).mean()
    return output_dict


def sanitize_dict(d):
    for k in list(d.keys()):
        if type(d[k]) not in (str, int, float, bool):
            d[k] = str(d[k])
    return d
