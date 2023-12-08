import copy
import re
import torch
import util

class FineTunedModel(torch.nn.Module):

    def __init__(self,
                 model,
                 modules,
                 frozen_modules=[]
                 ):
        #print(model, modules, frozen_modules)
        super().__init__()

        if isinstance(modules, str):
            modules = [modules]

        self.model = model
        self.ft_modules = {}
        self.orig_modules = {}

        util.freeze(self.model)

        for module_name, module in model.named_modules():
            for ft_module_regex in modules:

                match = re.search(ft_module_regex, module_name)
                
                if match is not None:

                    ft_module = copy.deepcopy(module)
                    
                    self.orig_modules[module_name] = module
                    self.ft_modules[module_name] = ft_module

                    util.unfreeze(ft_module)

                    print(f"=> Finetuning {module_name}")
       
                    for ft_module_name, module in ft_module.named_modules():

                        ft_module_name = f"{module_name}.{ft_module_name}"

                        for freeze_module_name in frozen_modules:

                            match = re.search(freeze_module_name, ft_module_name)

                            if match:
                                print(f"=> Freezing {ft_module_name}")
                                util.freeze(module)

        self.ft_modules_list = torch.nn.ModuleList(self.ft_modules.values())
        self.orig_modules_list = torch.nn.ModuleList(self.orig_modules.values())


    @classmethod
    def from_checkpoint(cls, model, checkpoint, frozen_modules=[]):

        if isinstance(checkpoint, str):
            checkpoint = torch.load(checkpoint)

        modules = [f"{key}$" for key in list(checkpoint.keys())]

        ftm = FineTunedModel(model, modules, frozen_modules=frozen_modules)
        ftm.load_state_dict(checkpoint)

        return ftm

        
    def __enter__(self):

        for key, ft_module in self.ft_modules.items():
            util.set_module(self.model, key, ft_module)

    def __exit__(self, exc_type, exc_value, tb):

        for key, module in self.orig_modules.items():
            util.set_module(self.model, key, module)

    def parameters(self):

        parameters = []

        for ft_module in self.ft_modules.values():

            parameters.extend(list(ft_module.parameters()))

        return parameters

    def state_dict(self):

        state_dict = {key: module.state_dict() for key, module in self.ft_modules.items()}

        return state_dict

    def load_state_dict(self, state_dict):

        for key, sd in state_dict.items():
            
            self.ft_modules[key].load_state_dict(sd)
