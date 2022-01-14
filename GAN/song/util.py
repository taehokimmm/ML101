import torch
import numpy as np
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

def img_show(img):
    img = img / 2.0 + 0.5
    npimg = img.detach().cpu().numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def sample_images(genM2P, genP2M, real_M, real_P, DEVICE, figside=1.5):
    Tensor = torch.cuda.FloatTensor if DEVICE == "cuda" else torch.FloatTensor
    genM2P.eval()
    genP2M.eval()
    
    real_M = real_M.type(Tensor)
    fake_P = genM2P(real_M).detach()
    real_P = real_P.type(Tensor)
    fake_M = genP2M(real_P).detach()
    
    nrows = real_M.size(0)
    real_M = make_grid(real_M, nrow=nrows, normalize=True)
    fake_P = make_grid(fake_P, nrow=nrows, normalize=True)
    real_P = make_grid(real_P, nrow=nrows, normalize=True)
    fake_M = make_grid(fake_M, nrow=nrows, normalize=True)
    
    image_grid = torch.cat((real_M, fake_P, real_P, fake_M), 1).cpu().permute(1, 2, 0)
    
    plt.figure(figsize=(figside*nrows, figside*4))
    plt.imshow(image_grid)
    plt.axis('off')
    plt.show()
    

def save_checkpoint(epoch, model_list, optimizer_list, filename):
    model_state, optimizer_state = [], []
    for m in model_list:
        model_state.append(m.state_dict())
    for o in optimizer_list:
        optimizer_state.append(o.state_dict())
    state = {
        'Epoch' : epoch,
        'State_dict' : model_state,
        'optimizer': optimizer_state
    }
    torch.save(state, filename)