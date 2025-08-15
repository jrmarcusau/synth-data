# ===================== defect_overlay_unet_v12.py =====================
"""
Conditional GAN (Pix2Pix) for synthesising defect overlays on wafer
cross-sections.

Folders written
---------------
outputs/defect_overlay_cpu_v12/
    ├─ defect_maps_grid.png
    ├─ side_by_side_01.png …
    └─ ckpt_v12/{G,D}_epoch_01.pt …
"""
# ---------------------------------------------------------------------
#                                imports
# ---------------------------------------------------------------------
import os, math
from pathlib import Path
from functools import lru_cache

import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.utils as vutils
from PIL import Image
from tqdm import tqdm

# ---------------------------------------------------------------------
#                            configuration
# ---------------------------------------------------------------------
ROOT_GOOD, ROOT_DEFECT = "data/good", "data/defect"
RES_TRAIN, RES_FULL    = 256, 1024
EPOCHS,  BATCH,  LR    = 50, 4, 2e-4          # GAN defaults
BETA1                   = 0.5

THRESH_ABS  = 0.12
λ_L1, λ_GAN = 100.0, 1.0

PREVIEW_N   = 4

OUT_DIR  = "outputs/defect_overlay_cpu_v12"
CKPT_DIR = f"{OUT_DIR}/ckpt_v12"
os.makedirs(CKPT_DIR, exist_ok=True)

torch.set_num_threads(max(os.cpu_count()-1, 1))
DEV = torch.device("cpu")

# ---------------------------------------------------------------------
#                        transforms (no aug)
# ---------------------------------------------------------------------
class PadSquare:
    def __init__(self, size): self.size = size
    def __call__(self, img):
        w, h = img.size
        side = max(w, h, self.size)
        pad  = [(side-w)//2, (side-h)//2,
                side-w-(side-w)//2, side-h-(side-h)//2]
        return T.functional.pad(img, pad, padding_mode='reflect')

_xform = T.Compose([
    T.Grayscale(),
    PadSquare(RES_TRAIN),
    T.Resize((RES_TRAIN, RES_TRAIN),
             interpolation=T.InterpolationMode.BILINEAR),
    T.ToTensor(),
    T.Normalize((0.5,), (0.5,))
])

# ---------------------------------------------------------------------
#                     helpers & fixed-threshold mask
# ---------------------------------------------------------------------
@lru_cache(maxsize=1)
def _gauss_kernel(ch=1,k=5,σ=1.0):
    ax=torch.arange(k)-k//2
    g=torch.exp(-(ax**2)/(2*σ**2))
    ker=(g[:,None]@g[None,:]).float(); ker/=ker.sum()
    return ker[None,None].repeat(ch,1,1,1)

def make_overlay(defect,bg):
    delta=defect-bg
    mask =(delta.abs()>THRESH_ABS).float()
    # 3×3 closing → 5×5 opening (same as v9)
    mask=F.max_pool2d(mask,3,1,1); mask=-F.max_pool2d(-mask,3,1,1)
    mask=-F.max_pool2d(-mask,5,1,2); mask=F.max_pool2d(mask,5,1,2)
    return mask*delta, mask

# ---------------------------------------------------------------------
#                           dataset
# ---------------------------------------------------------------------
ALLOWED={".png",".jpg",".jpeg",".bmp",".tif",".tiff"}
def list_imgs(root): return [str(p) for p in Path(root).rglob('*')
                             if p.suffix.lower() in ALLOWED]

def quick_vec(path,side=64):
    img=Image.open(path).convert('L').resize((side,side),Image.BILINEAR)
    return torch.from_numpy(np.asarray(img,dtype=np.float32)/255.)

class OverlayDS(Dataset):
    def __init__(self):
        self.good, self.defect = list_imgs(ROOT_GOOD), list_imgs(ROOT_DEFECT)
        if not self.good or not self.defect:
            raise RuntimeError("empty data folders")
        g_vecs=torch.stack([quick_vec(p) for p in self.good])
        self.pairs=[]
        for d in self.defect:
            d_vec=quick_vec(d)
            idx=torch.argmin(((g_vecs-d_vec)**2).view(len(self.good),-1).mean(1))
            self.pairs.append((d, self.good[idx]))
        self.N=len(self.pairs)
    def __len__(self): return self.N
    def __getitem__(self,i):
        d_path,g_path=self.pairs[i]
        defect=_xform(Image.open(d_path).convert("L"))
        good  =_xform(Image.open(g_path).convert("L"))
        ov,_  =make_overlay(defect,good)
        return good, ov

# ---------------------------------------------------------------------
#                 generator (5-level U-Net, in-ch=2, out-ch=1)
# ---------------------------------------------------------------------
def down(i,o): return nn.Sequential(nn.Conv2d(i,o,4,2,1,bias=False),
                                    nn.BatchNorm2d(o), nn.LeakyReLU(0.2,True))
def up(i,o):   return nn.Sequential(nn.ConvTranspose2d(i,o,4,2,1,bias=False),
                                    nn.BatchNorm2d(o), nn.ReLU(True))

class G_UNet(nn.Module):
    def __init__(self, ch=64):
        super().__init__()
        self.e1=down(2,ch)
        self.e2=down(ch,ch*2)
        self.e3=down(ch*2,ch*4)
        self.e4=down(ch*4,ch*8)
        self.e5=down(ch*8,ch*8)
        self.d5=up(ch*8,ch*8)
        self.d4=up(ch*16,ch*4)
        self.d3=up(ch*8 ,ch*2)
        self.d2=up(ch*4 ,ch)
        self.d1=nn.ConvTranspose2d(ch*2,1,4,2,1)
    def forward(self,x):
        e1=self.e1(x); e2=self.e2(e1); e3=self.e3(e2); e4=self.e4(e3); e5=self.e5(e4)
        d5=self.d5(e5)
        d4=self.d4(torch.cat([d5,e4],1))
        d3=self.d3(torch.cat([d4,e3],1))
        d2=self.d2(torch.cat([d3,e2],1))
        out=self.d1(torch.cat([d2,e1],1))
        return torch.tanh(out)  # [-1,1]

# ---------------------------------------------------------------------
#                discriminator (Patch-GAN, in-ch=2)
# ---------------------------------------------------------------------
def disc_block(i,o,norm=True):
    layers=[nn.Conv2d(i,o,4,2,1,bias=not norm)]
    if norm: layers.append(nn.BatchNorm2d(o))
    layers.append(nn.LeakyReLU(0.2,True))
    return layers

class D_Patch(nn.Module):
    def __init__(self,ch=64):
        super().__init__()
        seq = []
        seq += disc_block(2, ch, norm=False)   # 256 →128
        seq += disc_block(ch, ch*2)            # 128 →64
        seq += disc_block(ch*2, ch*4)          # 64  →32
        seq += disc_block(ch*4, ch*8)          # 32  →16
        seq.append(nn.Conv2d(ch*8, 1, 4, 1, 1))# 16  →15
        self.model = nn.Sequential(*seq)
    def forward(self,x): return self.model(x)   # logits

# ---------------------------------------------------------------------
#                       one-off defect map grid
# ---------------------------------------------------------------------
def save_defect_grid(ds,out_path,n=16):
    maps=[ds[i][1]*0.5+0.5 for i in range(min(n,len(ds)))]
    vutils.save_image(vutils.make_grid(maps,nrow=4,padding=2),
                      out_path,normalize=False)

# ---------------------------------------------------------------------
#                          training routine
# ---------------------------------------------------------------------
def main():
    ds=OverlayDS(); save_defect_grid(ds,f"{OUT_DIR}/defect_maps_grid.png")
    loader=DataLoader(ds,BATCH,shuffle=True,num_workers=0)

    netG, netD = G_UNet().to(DEV), D_Patch().to(DEV)
    optG = torch.optim.Adam(netG.parameters(), LR, betas=(BETA1,0.999))
    optD = torch.optim.Adam(netD.parameters(), LR, betas=(BETA1,0.999))

    BCE = nn.BCEWithLogitsLoss()
    L1  = nn.L1Loss()

    for ep in range(1,EPOCHS+1):
        for bg, ov_gt in tqdm(loader, desc=f"Epoch {ep}/{EPOCHS}", leave=False):
            bg, ov_gt = bg.to(DEV), ov_gt.to(DEV)
            # ------------------ prepare inputs -----------------------
            z  = torch.randn_like(bg[:,:1])         # 1-ch noise
            gin = torch.cat([bg, z], 1)             # 2 channels

            # ========================================================
            #                     update D
            # ========================================================
            netD.zero_grad()
            ov_fake = netG(gin).detach()
            real_pair = torch.cat([bg, ov_gt ], 1)
            fake_pair = torch.cat([bg, ov_fake], 1)

            pred_real = netD(real_pair)
            pred_fake = netD(fake_pair)
            label_real = torch.ones_like(pred_real)
            label_fake = torch.zeros_like(pred_fake)

            loss_D = 0.5*(BCE(pred_real, label_real) +
                           BCE(pred_fake, label_fake))
            loss_D.backward()
            optD.step()

            # ========================================================
            #                     update G
            # ========================================================
            netG.zero_grad()
            ov_fake = netG(gin)
            fake_pair = torch.cat([bg, ov_fake], 1)
            pred_fake = netD(fake_pair)

            gan_loss = BCE(pred_fake, label_real)
            l1_loss  = L1(ov_fake, ov_gt) * λ_L1
            loss_G   = λ_GAN*gan_loss + l1_loss
            loss_G.backward()
            optG.step()

        # ---------- epoch summary ----------
        print(f"Ep {ep:02}:  L_D={loss_D.item():.3f}  "
              f"L_G={loss_G.item():.3f}  "
              f"mean|ov|={ov_fake.abs().mean():.3f}")

        # ---------- preview grid ----------
        with torch.no_grad():
            z = torch.randn(PREVIEW_N,1,RES_TRAIN,RES_TRAIN,device=DEV)
            bg_sample, _ = next(iter(loader))
            bg_sample = bg_sample.to(DEV)[:PREVIEW_N]
            gin = torch.cat([bg_sample, z],1)
            ov_pred = netG(gin)
            # upscale for 1024 row 4
            ov_full  = F.interpolate(ov_pred,(RES_FULL,RES_FULL),
                                      mode='bilinear',align_corners=False)
            bg_full  = F.interpolate(bg_sample,(RES_FULL,RES_FULL),
                                      mode='bilinear',align_corners=False)
            synth    = torch.clamp(bg_full+ov_full,-1,1)

            white = torch.ones_like(bg_full)
            true  = torch.clamp(white + F.interpolate(ov_gt[:PREVIEW_N],
                           (RES_FULL,RES_FULL),mode='bilinear',
                           align_corners=False), -1,1)
            pred  = torch.clamp(white + ov_full, -1,1)
            defect= torch.clamp(bg_full + F.interpolate(ov_gt[:PREVIEW_N],
                           (RES_FULL,RES_FULL),mode='bilinear',
                           align_corners=False), -1,1)

            rows=[defect,true,pred,synth]
            preview=torch.cat([vutils.make_grid(r,nrow=PREVIEW_N,padding=2)
                               for r in rows], dim=1)
            vutils.save_image(preview*0.5+0.5,
                              f"{OUT_DIR}/side_by_side_{ep:02}.png",
                              normalize=False)

        # checkpoints
        torch.save(netG.state_dict(), f"{CKPT_DIR}/G_epoch_{ep:02}.pt")
        torch.save(netD.state_dict(), f"{CKPT_DIR}/D_epoch_{ep:02}.pt")

    print("✅ training complete – see", OUT_DIR)

# ---------------------------------------------------------------------
if __name__ == "__main__":
    main()
# =====================================================================
