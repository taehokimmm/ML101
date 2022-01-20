import torch
import os
from torchvision.utils import save_image
from torch.utils.data import DataLoader
import model, dataset
import shutil

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
checkDIR = "./GAN/song/checkpoints"
relayPATH = checkDIR + "/" + os.listdir(checkDIR)[-1]
#testInputDIR = "./GAN/song/test_input"
testInputDIR = "./GAN/dataset/photo_jpg"
testOutputDIR = "./GAN/song/test_output"
batch_size = 1


for file in os.scandir(testOutputDIR):
    os.remove(file.path)

with torch.no_grad():
    checkpoint = torch.load(relayPATH)
    genP2M = model.Gen().to(DEVICE)
    genP2M.load_state_dict(checkpoint['model_state'][1])
    genP2M.eval()
    
    test_dataset = dataset.imageDataset(None, testInputDIR, mode='Test')
    test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False)
    
    for i, P_imgs in enumerate(test_loader):
        P_imgs = P_imgs.to(DEVICE)
        fake_monet = genP2M(P_imgs).detach().cpu()
        for j in range(min(batch_size, len(test_dataset)-i*batch_size)):
            fake_monet_batch = fake_monet[j] / 2.0 + 0.5
            save_image(fake_monet_batch, testOutputDIR + "/" + test_dataset.data_photo[i*batch_size+j])
        if (i+1) % 1000 == 0:
            print(f'{i+1} images finished')

shutil.make_archive("./GAN/song/images", 'zip', testOutputDIR)