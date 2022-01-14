import torch
import torchvision.transforms as transforms
import os
import numpy as np
from PIL import Image
import model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
relayPATH = "./GAN/checkpoint/" + os.listdir("./GAN/checkpoint")[-1]
testInputDIR = "./GAN/test_input"
testOutputDIR = "./GAN/test_output"

with torch.no_grad():
    checkpoint = torch.load(relayPATH)
    
    genM2P = model.Gen().to(DEVICE)
    genP2M = model.Gen().to(DEVICE)
    genM2P.load_state_dict(checkpoint['State_dict'][0])
    genP2M.load_state_dict(checkpoint['State_dict'][1])
    
    genP2M.eval()
    
    test_transforms = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    test_data = os.listdir(testInputDIR)
    image_list = []
    for i in test_data:
        imagePhoto = Image.open(testInputDIR + "/" + i).convert('RGB')
        imagePhoto = test_transforms(imagePhoto)
        image_list.append(imagePhoto)
    
    Tensor = torch.cuda.FloatTensor if DEVICE == "cuda" else torch.FloatTensor
    image_tensor = torch.stack(image_list, 0).type(Tensor)
    fake_image_list = genP2M(image_tensor).detach().cpu()
        
    for i, fake_image in enumerate(fake_image_list):
        img_arr = fake_image.squeeze().permute(1, 2, 0).numpy() * 255
        img_arr = img_arr.astype(np.uint8)
        
        img = transforms.ToPILImage()(img_arr)
        save_filename = testOutputDIR+"/"+test_data[i]
        img.save(save_filename)