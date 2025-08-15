import os, csv, math, torch, matplotlib.pyplot as plt
import torchvision.transforms as T, torchvision.utils as vutils
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm
from torch import autocast                     # <- modern AMP ctx-manager

# ----------------------------- CONFIG ---------------------------------
DATA_DIR       = "data/good"
OUT_DIR        = "outputs/fake_good_v4"
CKPT_DIR       = f"{OUT_DIR}/checkpoints"
LOSS_PLOT      = f"{OUT_DIR}/loss_plot.png"
CSV_LOG        = f"{OUT_DIR}/losses.csv"

STAGES         = [(256,8),(512,10),(1024,12)]  # (resolution , epochs)
NZ, NGF, NDF   = 100, 32, 32                   # light for CPU
BATCH_BASE     = 2                             # CPU safe; GPU auto-ups
LR, BETA1      = 2e-4, 0.5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP    = DEVICE.type == "cuda"
# ----------------------------------------------------------------------

#GOAL: a 256, 512, 1024 resolution dcgan each, epoch 1-9, 10-19, 20-30 respectively. POC for increase for each #

os.makedirs(OUT_DIR,  exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

def pad_to(res):
    class _Pad:
        def __call__(self, img):
            w,h = img.size
            t = (res-h)//2; b = res-h-t; l = (res-w)//2; r = res-w-l
            return T.functional.pad(img,(l,t,r,b),padding_mode='reflect')
    return _Pad()

# ---------------- Dynamic DCGAN blocks --------------------------------
import torch.nn as nn

def make_generator(res: int) -> nn.Module:
    """Seed 1×1 → 4×4, then log2(res/4) strided-2 up-convs."""
    ups = int(math.log2(res) - 2)              # 256 → 6 doublings
    layers = []

    # 1. Seed block: 1×1 → 4×4
    out_c = NGF * 16
    layers += [nn.ConvTranspose2d(NZ, out_c, 4, 1, 0, bias=False),
               nn.BatchNorm2d(out_c), nn.ReLU(True)]

    # 2. Doubling blocks
    for i in range(ups):
        in_c  = out_c
        out_c = max(NGF, in_c // 2)
        layers += [nn.ConvTranspose2d(in_c, out_c, 4, 2, 1, bias=False),
                   nn.BatchNorm2d(out_c), nn.ReLU(True)]

    # 3. Final 3×3 to smooth & keep size
    layers += [nn.ConvTranspose2d(out_c, 1, 3, 1, 1, bias=False), nn.Tanh()]
    return nn.Sequential(*layers)

def make_discriminator(res: int) -> nn.Module:
    downs = int(math.log2(res) - 2)            # mirror of generator
    layers, in_c = [], 1
    for i in range(downs):
        out_c = min(NDF * 2 ** i, NDF * 8)
        layers += [nn.Conv2d(in_c, out_c, 4, 2, 1, bias=False)]
        if i: layers += [nn.BatchNorm2d(out_c)]
        layers += [nn.LeakyReLU(0.2, True)]
        in_c = out_c
    layers += [nn.Conv2d(in_c, 1, 4, 1, 0, bias=False), nn.Sigmoid()]
    return nn.Sequential(*layers)

criterion = nn.BCELoss(reduction='mean')
scaler    = torch.cuda.amp.GradScaler(enabled=AMP)

# -------------------------- TRAIN -------------------------------------
G_losses, D_losses, total_epoch = [], [], 0

for RES, EPOCHS in STAGES:
    tfm = T.Compose([T.Grayscale(), pad_to(RES),
                     T.ToTensor(), T.Normalize((0.5,), (0.5,))])
    dl  = DataLoader(
            datasets.ImageFolder("data", tfm),
            batch_size = BATCH_BASE if DEVICE.type=="cpu" else BATCH_BASE*4,
            shuffle    = True, pin_memory=True)

    netG, netD = make_generator(RES).to(DEVICE), make_discriminator(RES).to(DEVICE)
    optG = torch.optim.Adam(netG.parameters(), LR, (BETA1, 0.999))
    optD = torch.optim.Adam(netD.parameters(), LR, (BETA1, 0.999))
    fixed_noise = torch.randn(next(iter(dl))[0].size(0), NZ,1,1, device=DEVICE)
    crop_top    = (RES-884)//2 if RES==1024 else 0

    for e in range(1, EPOCHS+1):
        g_running, d_running = 0.0, 0.0
        for real,_ in tqdm(dl, desc=f"Epoch {total_epoch+e}/{sum(s[1] for s in STAGES)}"):
            real = real.to(DEVICE, non_blocking=True)
            bsz  = real.size(0)
            valid = torch.ones_like(netD(real),  device=DEVICE)
            fake_ = torch.zeros_like(valid)

            # --- Train D ---
            optD.zero_grad()
            with autocast(device_type='cuda', enabled=AMP):
                d_real = criterion(netD(real), valid)
                noise  = torch.randn(bsz, NZ,1,1, device=DEVICE)
                fake   = netG(noise)
                d_fake = criterion(netD(fake.detach()), fake_)
                d_loss = 0.5*(d_real + d_fake)
            scaler.scale(d_loss).backward(); scaler.step(optD); scaler.update()

            # --- Train G ---
            optG.zero_grad()
            with autocast(device_type='cuda', enabled=AMP):
                g_loss = criterion(netD(fake), valid)
            scaler.scale(g_loss).backward(); scaler.step(optG); scaler.update()

            g_running += g_loss.item(); d_running += d_loss.item()

        G_losses.append(g_running/len(dl)); D_losses.append(d_running/len(dl))

        # --- Visual & save ---
        # ---------- visual save (replace old block) ----------
        with torch.no_grad():
            # regenerate noise to match *current* real batch
            noise      = torch.randn(real.size(0), NZ, 1, 1, device=DEVICE)
            fake       = netG(noise).cpu()

            if RES == 1024:                      # 1024 × 1024 stage
                fake_vis = fake[:,:,crop_top:crop_top+884,:]
                real_vis = real.cpu()[:,:,crop_top:crop_top+884,:]
            else:                                # smaller stages → upscale just for viewing
                fake_vis = torch.nn.functional.interpolate(fake,(884,1024))
                real_vis = torch.nn.functional.interpolate(real.cpu(),(884,1024))

            # de-norm to [0,1]
            fake_vis = fake_vis*0.5 + 0.5
            real_vis = real_vis*0.5 + 0.5

            # --- make sure shapes match on (B,C,W) before cat on height (dim=2) ---
            b = min(real_vis.size(0), fake_vis.size(0))
            c = min(real_vis.size(1), fake_vis.size(1))
            w = min(real_vis.size(3), fake_vis.size(3))
            real_vis = real_vis[:b, :c, :, :w]
            fake_vis = fake_vis[:b, :c, :, :w]

            comp = torch.cat((real_vis, fake_vis), dim=2)   # vertical stack
            vutils.save_image(comp, f"{OUT_DIR}/side_{total_epoch+e:03}.png", normalize=False)
        # ------------------------------------------------------`

        if (total_epoch+e) % 5 == 0:
            torch.save(netG.state_dict(), f"{CKPT_DIR}/G_{total_epoch+e:03}.pt")

    total_epoch += EPOCHS

# -------------------- FINISH --------------------
torch.save(netG.state_dict(), f"{OUT_DIR}/generator_good_v4.pt")
with open(CSV_LOG,'w',newline='') as f:
    csv.writer(f).writerows([("epoch","G","D")] + [(i+1,G_losses[i],D_losses[i]) for i in range(len(G_losses))])

plt.plot(G_losses,label='G'); plt.plot(D_losses,label='D')
plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()
plt.tight_layout(); plt.savefig(LOSS_PLOT); plt.close()
print("✅ Finished – model & logs written to", OUT_DIR)
