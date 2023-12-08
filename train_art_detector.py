from StableDiffuser import StableDiffuser
from finetuning import FineTunedModel
import torch
from tqdm import tqdm
import datetime
import torchvision.transforms as T
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from diffusers import AutoencoderKL

def create_model(num_artists):
    import torchvision
    # transfer learning on top of ResNet (only replacing final FC layer)
    model_conv = torchvision.models.resnet18(pretrained=True)
    # Parameters of newly constructed modules have requires_grad=True by default
    for param in model_conv.parameters():
        param.requires_grad = False
    num_ftrs = model_conv.fc.in_features
    model_conv.fc = nn.Linear(num_ftrs, num_artists)
    # load the pre-trained weights
    model_conv.load_state_dict(torch.load('./detector/artist/artist_ckp/state_dict.dat.von_gogh'))
    return model_conv

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


class RGBConverter(nn.Module):
    def __init__(self):
        super(RGBConverter, self).__init__()
        # Magic number used in the detector
        mean_resnet = np.array([0.485, 0.456, 0.406])
        std_resnet = np.array([0.229, 0.224, 0.225])
        self.val_transform = T.Compose([T.Resize(224), T.Normalize(mean_resnet, std_resnet)])
    
    def toRGB(self, RGBA, background=(255,255,255)):
        _, D, R, C = RGBA.shape
        if D == 3:
            return RGBA
        RGB = torch.zeros((1, 3, R, C), dtype=torch.float32)
        R, G, B, A = RGBA[0].split(1, dim=0)
        A = A.float() / 255
        RGB[0, 0,:,:] = R.squeeze() * A.squeeze() + (1 - A.squeeze()) * background[0]
        RGB[0, 1,:,:] = G.squeeze() * A.squeeze() + (1 - A.squeeze()) * background[1]
        RGB[0, 2,:,:] = B.squeeze() * A.squeeze() + (1 - A.squeeze()) * background[2]
        return RGB

    def forward(self, input):
        min = torch.min(input.detach())
        max = torch.max(input.detach())
        input = (input-min)/(max-min)*255
        #input = self.toRGB(input)
        input = self.val_transform(input.squeeze())
        return input


class ArtModel(nn.Module):

    def __init__(self):
        super(ArtModel, self).__init__()
        self.rgb = RGBConverter()
        self.vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae").to("cuda:1")
        self.classifier = create_model(5)
        self.classifier.eval()
        for param in self.classifier.parameters():
            param.requires_grad = False
    def forward(self, x):
        x = self.vae.decode(1 / self.vae.config.scaling_factor * x).sample
        x = self.rgb(x)
        x = self.classifier(x.unsqueeze(0))
        return x

class ObjectDetector():
    def __init__(self):
        self.dtype = torch.FloatTensor
        if (torch.cuda.is_available()):
            self.dtype = torch.cuda.FloatTensor
        # transfer learning on top of ResNet (only replacing final FC layer)
        self.model = ArtModel()
        
        self.model.to("cuda:1")
        self.device = "cuda:1"
        # self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

    def get_input_grad(self, x): 
        x_var = Variable(x.type(self.dtype).to(self.device), requires_grad=True)
        resnet_output = self.model(x_var)
        prob = resnet_output[0][2]
        prob.backward()
        return x_var.grad

prompt="Alfred Sisley"
modules = ".*attn2$"
iterations=200
negative_guidance=1
lr=0.015
save_path="tmp/test"
freeze_modules=[]


nsteps = 50

diffuser = StableDiffuser(scheduler='DDIM').to('cuda:1')
diffuser.train()


finetuner = FineTunedModel(diffuser, modules, frozen_modules=freeze_modules)

params = list(finetuner.parameters())
criteria = torch.nn.MSELoss()


pbar = tqdm(range(iterations))

with torch.no_grad():
    positive_text_embeddings = diffuser.get_text_embeddings([prompt],n_imgs=1)

# del diffuser.vae
# del diffuser.text_encoder
# del diffuser.tokenizer
del diffuser.safety_checker

torch.cuda.empty_cache()

# del detector, optimizer
detector = ObjectDetector()
optimizer = torch.optim.SGD(params, lr=lr)

print("BEGIN TRAIN")
diffuser.train()
for i in pbar:
    with torch.no_grad():
        diffuser.set_scheduler_timesteps(60)

        optimizer.zero_grad()

        diffuse_iter = torch.randint(1, nsteps-1, (1,)).item()

        latents = diffuser.get_initial_latents(1, 512, 1)
        # print("LATENT SIZE: ", latents.size())
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
        negative_latents = diffuser.predict_noise(diffuse_iter, latents_steps[0], positive_text_embeddings, guidance_scale=1)
    
    input_x = latents_steps[0]    
    detector_grad = detector.get_input_grad(input_x)        
    loss = criteria(negative_latents.float(), ref_latents.detach().float() + 10*(detector_grad))
    loss.backward()
    optimizer.step()


    # if i % 10 == 0 and i != 0:
    #     now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    #     torch.save(
    #         finetuner.state_dict(), 
    #         save_path + f'_checkpoint_{i}_{now_str}.pt'
    #     )

now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
torch.save(finetuner.state_dict(), save_path + f'_{now_str}.pt')


torch.cuda.empty_cache()