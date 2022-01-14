import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import dataset, model, util
import itertools
import os
import gc
gc.collect()
torch.cuda.empty_cache()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 1
lr_init = 2e-4
n_epoches = 200
start_epoch = 0
lambda_weight = 10
relay_learning = False
monetDIR = "./GAN/dataset/monet_jpg"
photoDIR = "./GAN/dataset/photo_jpg"

if relay_learning:
    relayPATH = "./GAN/song/checkpoints/" + os.listdir("./GAN/song/checkpoints")[-1]

genM2P = model.Gen().to(DEVICE).apply(model.initialize_weight)
genP2M = model.Gen().to(DEVICE).apply(model.initialize_weight)
disM = model.Dis().to(DEVICE).apply(model.initialize_weight)
disP = model.Dis().to(DEVICE).apply(model.initialize_weight)

opGen = torch.optim.Adam(itertools.chain(genM2P.parameters(), genP2M.parameters()), lr=lr_init, betas=(0.5, 0.999))
opDisM = torch.optim.Adam(disM.parameters(), lr=lr_init, betas=(0.5, 0.999))
opDisP = torch.optim.Adam(disP.parameters(), lr=lr_init, betas=(0.5, 0.999))

lr_scheduler_Gen = model.ConstantandLinearlyDecay(opGen, start_decay=100, total=n_epoches)
lr_scheduler_DisM = model.ConstantandLinearlyDecay(opDisM, start_decay=100, total=n_epoches)
lr_scheduler_DisP = model.ConstantandLinearlyDecay(opDisP, start_decay=100, total=n_epoches)

train_dataset = dataset.imageDataset(monetDIR, photoDIR, 777)
train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)

adv_loss_func = nn.MSELoss().to(DEVICE)
cyc_loss_func = nn.L1Loss().to(DEVICE)
id_loss_func = nn.L1Loss().to(DEVICE)

if relay_learning:
    checkpoint = torch.load(relayPATH)
    genM2P.load_state_dict(checkpoint['State_dict'][0])
    genP2M.load_state_dict(checkpoint['State_dict'][1])
    disM.load_state_dict(checkpoint['State_dict'][2])
    disP.load_state_dict(checkpoint['State_dict'][3])
    
    opGen.load_state_dict(checkpoint['optimizer'][0])
    opDisM.load_state_dict(checkpoint['optimizer'][1])
    opDisP.load_state_dict(checkpoint['optimizer'][2])
    
    start_epoch = checkpoint['Epoch'] + 1
    
Tensor = torch.cuda.FloatTensor if DEVICE == "cuda" else torch.FloatTensor
out_shape = [batch_size, 1, 16, 16]
ones = torch.ones(out_shape).type(Tensor)
zeros = torch.ones(out_shape).type(Tensor)

for epoch in range(start_epoch, n_epoches):
    genM2P.train()
    genP2M.train()
    disM.train()
    disP.train()
    
    for M_imgs, P_imgs in train_loader:
        torch.cuda.empty_cache()
        M_imgs = M_imgs.to(DEVICE)
        P_imgs = P_imgs.to(DEVICE)
        
        M_identity = genP2M(M_imgs)
        M2P_trans = genM2P(M_imgs)
        M_restored = genP2M(M2P_trans)
        
        P_identity = genM2P(P_imgs)
        P2M_trans = genP2M(P_imgs)
        P_restored = genM2P(P2M_trans)
        
        #generator loss
        opGen.zero_grad()
        
        MP_pred = disP(M2P_trans)
        M_adv_loss = adv_loss_func(MP_pred, ones)
        PM_pred = disM(P2M_trans)
        P_adv_loss = adv_loss_func(PM_pred, ones)
        adv_loss = (M_adv_loss + P_adv_loss) / 2
        
        M_cyc_loss = cyc_loss_func(M_restored, M_imgs)
        P_cyc_loss = cyc_loss_func(P_restored, P_imgs)
        cyc_loss = (M_cyc_loss + P_cyc_loss) / 2
        
        M_id_loss = id_loss_func(M_identity, M_imgs)
        P_id_loss = id_loss_func(P_identity, P_imgs)
        id_loss = (M_id_loss + P_id_loss) / 2
        
        gen_loss = adv_loss + lambda_weight * cyc_loss + lambda_weight * .5 * id_loss
        
        gen_loss.backward()
        opGen.step()
        
        
        #discriminator M
        opDisM.zero_grad()
        
        M_real_loss = adv_loss_func(disM(M_imgs), ones).to(DEVICE)
        M_fake_loss = adv_loss_func(disM(P2M_trans.detach()), zeros).to(DEVICE)
        M_dis_loss = (M_real_loss + M_fake_loss) / 2
        
        M_dis_loss.backward()
        opDisM.step()
        
        
        #discriminator P
        opDisP.zero_grad()
        
        P_real_loss = adv_loss_func(disP(P_imgs), ones).to(DEVICE)
        P_fake_loss = adv_loss_func(disP(M2P_trans.detach()), zeros).to(DEVICE)
        P_dis_loss = (P_real_loss + P_fake_loss) / 2
        
        P_dis_loss.backward()
        opDisP.step()
        
        
    lr_scheduler_Gen.step()
    lr_scheduler_DisM.step()
    lr_scheduler_DisP.step()
    
    print(f'[Epoch {epoch+1}/{n_epoches}]')
    for file in os.scandir("./GAN/song/checkpoints"):
        os.remove(file.path)
    util.save_checkpoint(epoch, [genM2P, genP2M, disM, disP], [opGen, opDisM, opDisP], "./GAN/song/checkpoints/model_epoch_{}.pt".format(epoch+1))
    
    if (epoch+1) % 20 == 0 or epoch == 0:
        #test_real_M, test_real_P = next(iter(train_loader))
        #util.sample_images(genM2P, genP2M, test_real_M, test_real_P, DEVICE)
        
        print(f'[Generator loss: {gen_loss.item()} | identity: {id_loss.item()} adversarial: {adv_loss.item()} cycle: {cyc_loss.item()}]')
        print(f'[DiscriminatorM loss: {M_dis_loss.item()} | DiscriminatorP loss : {P_dis_loss.item()}')