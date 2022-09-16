import torch
bart = torch.hub.load('pytorch/fairseq', 'bart.large')
bart.eval()  # disable dropout (or leave in train mode to finetune)

