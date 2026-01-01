import torch 
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
from .RainNet import RainNet
from .utils import guided_filter

class Harmonizer:
    def __init__(self,model_path,device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = RainNet(input_nc=3, output_nc=3, ngf=64).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        self.img_size = 512

        self.transforms = transforms.Compose([
            transforms.Resize((self.img_size,self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
        ])

        self.mask_transforms = transforms.Compose([
            transforms.Resize((self.img_size,self.img_size)),
            transforms.ToTensor()
        ])
    def predict(self,composite,mask):
        """
        predict 的 Docstring
        
        :param composite: PIL Image
        :param mask: PIL Image
        """
        orig_w,orig_h = composite.size

        comp_tensor = self.transforms(composite).unsqueeze(0).to(self.device)
        mask_tensor = self.mask_transforms(mask).unsqueeze(0).to(self.device)

        with torch.no_grad():
            fake_tensor = self.model(comp_tensor,mask_tensor)

            fake_resized = F.interpolate(fake_tensor, size=(orig_h, orig_w),mode = 'bicubic',align_corners=True)
            fake_01 = (fake_resized * 0.5) + 0.5

            guide_np = np.array(composite).astype(np.float32)/255.0
            src_np = fake_01.squeeze(0).permute(1,2,0).cpu().numpy()

            refined_np = guided_filter(guide_np,src_np,radius=100,eps=1e-3)

            refined_tensor = torch.from_numpy(refined_np).permute(2,0,1).unsqueeze(0).to(self.device)

            mask_orig_tensor = transforms.ToTensor()(mask).unsqueeze(0).to(self.device)    
            comp_orig_tensor = transforms.ToTensor()(composite).unsqueeze(0).to(self.device)

            output_tensor = refined_tensor * mask_orig_tensor + comp_orig_tensor * (1 - mask_orig_tensor)
            return output_tensor.cpu()