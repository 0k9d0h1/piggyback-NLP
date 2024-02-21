from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch
import numpy as np
import torch.nn as nn
from modnets.layers import Binarizer

model = torch.load(
    "../checkpoint_unlabeled/BERT_piggyback_realnews,s2orc,pubmed_8.842957496643066,4.367776843989007,9.557445526123047.pt", map_location=torch.device('cpu'))["model"]

for module in model.modules():
    if "ElementWise" in str(type(module)):
        print(module.masks["1"])
        b1 = Binarizer.apply(module.masks["1"])
        module.masks["1"].requires_grad = False
        mean_le = module.masks["1"][module.masks["1"].le(5e-3)].mean()
        mean_gt = module.masks["1"][module.masks["1"].gt(5e-3)].mean()
