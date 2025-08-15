import os, csv, math, torch, matplotlib.pyplot as plt
import torchvision.transforms as T, torchvision.utils as vutils
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm
import torch.nn as nn
from torch import autocast, amp

# ---------------- CONFIG ----------------
DATA_DIR   = "data/good"                 # unchanged
OUT_DIR    = "outputs/fake_good_v6"
CKPT_DIR   = f"{OUT_DIR}/checkpoints"
LOSS_PLOT  = f"{OUT_DIR}/loss_plot.png"
CSV_LOG    = f"{OUT_DIR}/losses.csv"

RES        = 1024
EPOCHS     = 20
NZ, NGF, NDF = 100, 64, 64              # ↑ capacity
BATCH_CPU  = 2
LR, BETA1  = 2e-4, 0.5
TOP_N      = 6                          # 6 real + 6 fake preview

TARGET_H   = 884
PAD_TOP    = (RES - TARGET_H) // 2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP     = DEVICE.type == "cuda"
scaler  = amp.GradScaler(enabled=AMP)

os.makedirs(OUT_DIR,  exist_ok=True); os.makedirs(CKPT_DIR, exist_ok=True)

# -------------- data ----------------
class Pad1024:
    def __call__(self, img):
        w,h = img.size
        t=(RES-h)//2; b=RES-h-t; l=(RES-w)//2; r=RES-w-l
        return T.functional.pad(img,(l,t,r,b),padding_mode='reflect')

transform = T.Compose([
    T.Grayscale(),
    Pad1024(),
    T.ToTensor(),
    T.Normalize((0.5,), (0.5,))
])

loader = DataLoader(
    datasets.ImageFolder("data", transform),
    batch_size=BATCH_CPU if DEVICE.type=="cpu" else BATCH_CPU*4,
    shuffle=True, pin_memory=True)

# -------------- nets ----------------
def sn(conv): return nn.utils.spectral_norm(conv)

def make_generator():
    ups = int(math.log2(RES) - 2)          # 6 doublings
    layers = [nn.ConvTranspose2d(NZ, NGF*16,4,1,0,bias=False),
              nn.BatchNorm2d(NGF*16), nn.ReLU(True)]
    out_c = NGF*16
    for _ in range(ups):
        in_c, out_c = out_c, max(NGF, out_c//2)
        layers += [nn.ConvTranspose2d(in_c,out_c,4,2,1,bias=False),
                   nn.BatchNorm2d(out_c), nn.ReLU(True)]
    layers += [nn.ConvTranspose2d(out_c,1,3,1,1,bias=False), nn.Tanh()]
    return nn.Sequential(*layers)

def make_discriminator():
    downs = int(math.log2(RES) - 2)
    layers, in_c = [], 1
    for i in range(downs):
        out_c = min(NDF*2**i, NDF*8)
        layers += [sn(nn.Conv2d(in_c,out_c,4,2,1,bias=False))]
        if i: layers += [nn.BatchNorm2d(out_c)]
        layers += [nn.LeakyReLU(0.2,True)]
        in_c = out_c
    layers += [sn(nn.Conv2d(in_c,1,4,1,0,bias=False))]
    return nn.Sequential(*layers)

G, D = make_generator().to(DEVICE), make_discriminator().to(DEVICE)
optG = torch.optim.Adam(G.parameters(), LR, betas=(BETA1,0.999))
optD = torch.optim.Adam(D.parameters(), LR, betas=(BETA1,0.999))

# -------- hinge loss helpers --------
def d_hinge(real, fake):
    return (torch.relu(1-real).mean() + torch.relu(1+fake).mean())*0.5
def g_hinge(fake):
    return -fake.mean()

G_log, D_log = [], []

# -------------- TRAIN --------------
for epoch in range(1, EPOCHS+1):
    g_sum = d_sum = 0.
    for real,_ in tqdm(loader, desc=f"Epoch {epoch}/{EPOCHS}"):
        real = real.to(DEVICE, non_blocking=True)
        b    = real.size(0)

        # ----- D -----
        optD.zero_grad()
        z = torch.randn(b, NZ,1,1, device=DEVICE)
        with autocast("cuda", enabled=AMP):
            fake = G(z)
            d_real, d_fake = D(real), D(fake.detach())
            d_loss = d_hinge(d_real, d_fake)
        scaler.scale(d_loss).backward(); scaler.step(optD); scaler.update()

        # ----- G -----
        optG.zero_grad()
        with autocast("cuda", enabled=AMP):
            g_loss = g_hinge(D(fake))
        scaler.scale(g_loss).backward(); scaler.step(optG); scaler.update()

        g_sum += g_loss.item(); d_sum += d_loss.item()

    G_log.append(g_sum/len(loader)); D_log.append(d_sum/len(loader))

    # ----- preview -----
    with torch.no_grad():
        z = torch.randn(TOP_N, NZ,1,1, device=DEVICE)
        fake_batch = G(z).cpu()

        fake_v = (fake_batch[:,:,PAD_TOP:PAD_TOP+TARGET_H,:] * 0.5 + 0.5)
        real_v = (real.cpu()[:TOP_N,:,PAD_TOP:PAD_TOP+TARGET_H,:] * 0.5 + 0.5)

        b = min(fake_v.size(0), real_v.size(0))
        comp = torch.cat((real_v[:b], fake_v[:b]), dim=2)        # vertical
        vutils.save_image(comp, f"{OUT_DIR}/side_by_side_epoch_{epoch:03}.png",
                          nrow=b, normalize=False)
        vutils.save_image(fake_v[:b], f"{OUT_DIR}/epoch_{epoch:03}.png",
                          nrow=b, normalize=False)

    # checkpoint every 2 epochs
    if epoch % 2 == 0:
        torch.save(G.state_dict(), f"{CKPT_DIR}/generator_epoch_{epoch:03}.pt")

# ------------ FINISH ------------
torch.save(G.state_dict(), f"{OUT_DIR}/generator_good_v6.pt")
with open(CSV_LOG,'w',newline='') as f:
    csv.writer(f).writerows([("epoch","G","D")] +
                            [(i+1,G_log[i],D_log[i]) for i in range(len(G_log))])

plt.plot(G_log,label='G'); plt.plot(D_log,label='D')
plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend()
plt.tight_layout(); plt.savefig(LOSS_PLOT); plt.close()
print("✅ v6 training complete – outputs in", OUT_DIR)
