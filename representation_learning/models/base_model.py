import torch
import torch.nn as nn
from tqdm import tqdm

AVAILABLE_MODELS = {
    "efficientnetb0": "efficientnetb0",  # Value is the module name; the class inside must be named 'Model'
}

class ModelBase(nn.Module):
    def __init__(self, device):
        super(ModelBase, self).__init__()
        self.device = device

    def prepare_inference(self):
        self.model.eval()
        self.model = self.model.to(self.device)
    
    def prepare_train(self):
        self.model.train()
        self.model = self.model.to(self.device)

    def batch_inference(self, batched_samples):
        embeds = []
        for batch in tqdm(batched_samples, desc=" processing batches", position=0, leave=False):
            embedding = self.__call__(batch)
            if embedding.dim() == 1:
                embedding = embedding.unsqueeze(0)
            embeds.append(embedding)
        return torch.cat(embeds, axis=0)
