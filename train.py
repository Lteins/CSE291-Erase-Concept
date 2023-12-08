import json
import torch
from torchvision import transforms
from PIL import Image
# Load Resnet50 weights
from torchvision.models import resnet50, ResNet50_Weights
from torch.nn import functional as F
from StableDiffuser import StableDiffuser
from finetuning import FineTunedModel
import torch
from tqdm import tqdm
import datetime
import torchvision
import torchvision.transforms as T
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data import sampler
from torch.utils.data import TensorDataset
from PIL import Image
from diffusers import AutoencoderKL
import fire, os, glob
import classifier

class ObjectDetector():
    def __init__(self, detector_type):
        self.dtype = torch.FloatTensor
        if (torch.cuda.is_available()):
            self.dtype = torch.cuda.FloatTensor
        # transfer learning on top of ResNet (only replacing final FC layer)
        self.device = "cuda:1"
        if detector_type == "car":
            self.model = classifier.CarModel() 
        elif detector_type == "van gogh":
            self.model = classifier.VanGoghModel()
        self.model.to(self.device)

    def get_input_grad(self, x): 
        x_var = Variable(x.type(self.dtype).to(self.device), requires_grad=True)
        prob = self.model(x_var)
        prob.backward()
        return x_var.grad

def train(prompt, save_path, iterations=150, negative_guidance=1, lr=0.015, nsteps = 50):
    save_path = os.path.join(save_path, prompt.replace(" ", "_"))
    modules = "unet$"
    freeze_modules=[]
    diffuser = StableDiffuser(scheduler='DDIM').to('cuda:1')
    diffuser.train()
    finetuner = FineTunedModel(diffuser, modules, frozen_modules=freeze_modules)
    params = list(finetuner.parameters())
    criteria = torch.nn.MSELoss()

    pbar = tqdm(range(iterations))

    with torch.no_grad():
        # neutral_text_embeddings = diffuser.get_text_embeddings([''],n_imgs=1)
        positive_text_embeddings = diffuser.get_text_embeddings([prompt],n_imgs=1)

    del diffuser.vae
    del diffuser.text_encoder
    del diffuser.tokenizer
    del diffuser.safety_checker

    torch.cuda.empty_cache()

    detector = ObjectDetector(prompt)
    optimizer = torch.optim.SGD(params, lr=lr)
    for i in pbar:
        with torch.no_grad():
            diffuser.set_scheduler_timesteps(50)

            optimizer.zero_grad()

            diffuse_iter = torch.randint(40, nsteps-1, (1,)).item()
            latents = diffuser.get_initial_latents(1, 512, 1)
            with finetuner:
                latents_steps, _ = diffuser.diffusion(
                    latents,
                    positive_text_embeddings,
                    start_iteration=0,
                    end_iteration=diffuse_iter,
                    guidance_scale=3,
                    show_progress=False,
                )
            
            with finetuner:
                ref_latents = diffuser.predict_noise(diffuse_iter, latents_steps[0], positive_text_embeddings, guidance_scale=1)


        with finetuner:
            negative_latents = diffuser.predict_noise(diffuse_iter, latents_steps[0], positive_text_embeddings, guidance_scale=7)
        
        input_x = latents_steps[0]    
        detector_grad = detector.get_input_grad(input_x)        
        loss = criteria(negative_latents.float(), ref_latents.detach().float() + 40*(detector_grad)) #loss = criteria(e_n, e_0) works the best try 5000 epochs
        loss.backward()
        #print(loss.item())
        gradient = params[0].grad      
        optimizer.step()
        if i % 10 == 0 and i != 0:
            # delete preivous checkpoint
            for fname in glob.glob(save_path + f'mid_checkpoint.pt.*'):
                os.remove(fname)
            torch.save(
                finetuner.state_dict(), 
                save_path + f'mid_checkpoint.pt.{i}'
            )

    torch.save(finetuner.state_dict(), save_path + f'erase.pt')
    torch.cuda.empty_cache()

if __name__ == '__main__':    
    fire.Fire(train)