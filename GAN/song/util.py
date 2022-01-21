import torch
import numpy as np
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import seaborn as sns

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
    plt.show(block=False)
    plt.pause(2)
    plt.close()


def save_checkpoint(epoch, model_list, optimizer_list, scheduler_list, filename):
    model_state, optimizer_state, scheduler_state = [], [], []
    for m in model_list:
        model_state.append(m.state_dict())
    for o in optimizer_list:
        optimizer_state.append(o.state_dict())
    for s in scheduler_list:
        scheduler_state.append(s.state_dict())
    state = {
        'epoch' : epoch,
        'model_state' : model_state,
        'optimizer_state': optimizer_state,
        'scheduler_state': scheduler_state
    }
    torch.save(state, filename)
    
    
def loss_graph(gen_loss_list, M_dis_loss_list, P_dis_loss_list):
    dis_loss = np.vstack((M_dis_loss_list, P_dis_loss_list)).T

    sns.set()

    plt.figure(figsize=(10, 10))

    plt.subplot(2, 1, 1)
    plt.plot(dis_loss)
    plt.legend(['M_dis', 'P_dis'])
    plt.title("Discriminator loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")

    plt.subplot(2, 1, 2)
    plt.plot(gen_loss_list)
    plt.legend(['gen'])
    plt.title("Generator loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")

    plt.tight_layout()
    plt.savefig("./loss_graph.jpg")
    plt.show()