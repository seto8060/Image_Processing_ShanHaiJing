from rain_harmonizer import Harmonizer
import torchvision
from PIL import Image
path = "/root/imgPro/rainNet/rainNet/checkpoints/netG_epoch_80.pth" # *.pth 模型权重路径
model = Harmonizer(model_path = path)

comp_path = './composite/7.jpg'
mask_path = './mask/7.png'

comp = Image.open(comp_path).convert('RGB')
mask = Image.open(mask_path).convert('L')

print("穿 越 中...")
result = model.predict(comp,mask)

torchvision.utils.save_image(result,'./result.png')
print('欢迎来到 ***')

