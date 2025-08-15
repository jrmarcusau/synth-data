# ===================== defect_overlay_unet_v13.py =====================
"""
Pix2Pix-style conditional GAN for defect overlays – hinge loss + SN.

Folders written
---------------
outputs/defect_overlay_cpu_v13/
    ├─ defect_maps_grid.png
    ├─ side_by_side_epoch_01.png …
    ├─ loss_plot.png
    ├─ losses.csv
    └─ ckpt_v13/{G,D}_epoch_05.pt …
"""
# ---------------------------------------------------------------------
import os, math, csv, torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from functools import lru_cache
import torchvision.transforms as T
import torchvision.utils as vutils
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

# ----------------------------- config --------------------------------
ROOT_GOOD, ROOT_DEFECT = "data/good", "data/defect"
RES_TRAIN, RES_FULL    = 256, 1024
EPOCHS,  BATCH,  LR    = 50, 4, 2e-4
BETA1                   = 0.5
NZ                      = 1            # one noise channel (broadcast)
λ_L1, λ_GAN             = 25.0, 1.0

OUT_DIR  = "outputs/defect_overlay_cpu_v13"
CKPT_DIR = f"{OUT_DIR}/ckpt_v13"
LOSS_PLOT= f"{OUT_DIR}/loss_plot.png"
CSV_LOG  = f"{OUT_DIR}/losses.csv"
os.makedirs(CKPT_DIR, exist_ok=True)

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(max(os.cpu_count()-1, 1))

# --------------------------- transforms ------------------------------
class PadSquare:
    def __init__(self, size): self.size = size
    def __call__(self, img):
        w,h = img.size
        side = max(w,h,self.size)
        pad  = [(side-w)//2,(side-h)//2,
                side-w-(side-w)//2, side-h-(side-h)//2]
        return T.functional.pad(img,pad,padding_mode='reflect')

_xform = T.Compose([
    T.Grayscale(),
    PadSquare(RES_TRAIN),
    T.Resize((RES_TRAIN,RES_TRAIN),
             interpolation=T.InterpolationMode.BILINEAR),
    T.ToTensor(),
    T.Normalize((0.5,), (0.5,))
])

# ----------------------- mask / overlay helper -----------------------
THRESH_ABS = 0.12
@lru_cache(maxsize=1)
def _overlay_kernel(k):                   # rectangular morph kernel
    return torch.ones(1,1,k,k,device=DEV)

def make_overlay(defect,bg):
    delta = defect - bg
    mask  = (delta.abs() > THRESH_ABS).float()
    # 3×3 closing
    mask = F.max_pool2d(mask,3,1,1); mask=-F.max_pool2d(-mask,3,1,1)
    # 5×5 opening
    mask = -F.max_pool2d(-mask,5,1,2); mask = F.max_pool2d(mask,5,1,2)
    return mask*delta, mask

# ----------------------------- dataset -------------------------------
ALLOWED={".png",".jpg",".jpeg",".bmp",".tif",".tiff"}
def list_imgs(root): return [str(p) for p in Path(root).rglob('*')
                             if p.suffix.lower() in ALLOWED]

def quick_vec(path,side=64):
    img = Image.open(path).convert('L').resize((side,side),
                                               Image.BILINEAR)
    return torch.from_numpy(np.asarray(img,dtype=np.float32)/255.)

class OverlayDS(Dataset):
    def __init__(self):
        self.good, self.defect = list_imgs(ROOT_GOOD), list_imgs(ROOT_DEFECT)
        if not self.good or not self.defect:
            raise RuntimeError("empty data folders")
        g_vecs = torch.stack([quick_vec(p) for p in self.good])
        self.pairs=[]
        for d in self.defect:
            d_vec=quick_vec(d)
            idx=torch.argmin(((g_vecs-d_vec)**2)
                             .view(len(self.good),-1).mean(1))
            self.pairs.append((d,self.good[idx]))
    def __len__(self): return len(self.pairs)
    def __getitem__(self,i):
        d_path,g_path = self.pairs[i]
        defect=_xform(Image.open(d_path).convert("L"))
        good  =_xform(Image.open(g_path).convert("L"))
        ov,_  =make_overlay(defect,good)
        return good, ov

# ----------------------------- networks ------------------------------
def sn(layer): return nn.utils.spectral_norm(layer)       # shorthand

def down(i,o,norm=True):          # discriminator block
    layers=[sn(nn.Conv2d(i,o,4,2,1,bias=not norm))]
    if norm: layers.append(nn.BatchNorm2d(o))
    layers.append(nn.LeakyReLU(0.2,True))
    return layers

class D_Patch(nn.Module):
    def __init__(self,ch=32):     # ↓ half channels for tiny data
        super().__init__()
        seq=[]
        seq+=down(2, ch, norm=False)
        seq+=down(ch, ch*2)
        seq+=down(ch*2, ch*4)
        seq+=down(ch*4, ch*8)
        seq.append(sn(nn.Conv2d(ch*8,1,4,1,1,bias=False)))
        self.model=nn.Sequential(*seq)
    def forward(self,x): return self.model(x).mean()   # global hinge score

def g_block(i,o,up=True,last=False):
    if up:
        layers=[nn.ConvTranspose2d(i,o,4,2,1,bias=False)]
    else:
        layers=[nn.Conv2d(i,o,4,2,1,bias=False)]
    if not last: layers+=[nn.BatchNorm2d(o), nn.ReLU(True)]
    return layers

class G_UNet(nn.Module):
    def __init__(self,ch=64,in_ch=1+NZ):   # bg + noise
        super().__init__()
        self.e1=nn.Sequential(*g_block(in_ch,ch,up=False))
        self.e2=nn.Sequential(*g_block(ch,   ch*2,up=False))
        self.e3=nn.Sequential(*g_block(ch*2, ch*4,up=False))
        self.e4=nn.Sequential(*g_block(ch*4, ch*8,up=False))
        self.e5=nn.Sequential(*g_block(ch*8, ch*8,up=False))
        self.d5=nn.Sequential(*g_block(ch*8, ch*8))
        self.d4=nn.Sequential(*g_block(ch*16, ch*4))
        self.d3=nn.Sequential(*g_block(ch*8,  ch*2))
        self.d2=nn.Sequential(*g_block(ch*4,  ch))
        self.d1=nn.Sequential(sn(nn.ConvTranspose2d(ch*2,1,4,2,1,bias=False)),
                              nn.Tanh())
    def forward(self,x):
        e1=self.e1(x); e2=self.e2(e1); e3=self.e3(e2)
        e4=self.e4(e3); e5=self.e5(e4)
        d5=self.d5(e5)
        d4=self.d4(torch.cat([d5,e4],1))
        d3=self.d3(torch.cat([d4,e3],1))
        d2=self.d2(torch.cat([d3,e2],1))
        out=self.d1(torch.cat([d2,e1],1))
        return out

# ----------------------------- losses --------------------------------
def d_hinge(real,fake):
    return 0.5*(F.relu(1-real).mean()+F.relu(1+fake).mean())
def g_hinge(fake): return -fake.mean()
L1 = nn.L1Loss()

# ------------------------ tiny DiffAugment ---------------------------
def diff_aug(x):
    if torch.rand(1)<0.5: x=torch.flip(x,dims=[3])
    x = x + 0.05*torch.randn_like(x)
    return x.clamp(-1,1)

# -------------------------- training --------------------------------
def save_defect_grid(ds,path,n=16):
    maps=[ds[i][1]*0.5+0.5 for i in range(min(n,len(ds)))]
    vutils.save_image(vutils.make_grid(maps,nrow=4,padding=2),
                      path,normalize=False)

def main():
    ds=OverlayDS(); save_defect_grid(ds,f"{OUT_DIR}/defect_maps_grid.png")
    loader=DataLoader(ds,BATCH,shuffle=True,num_workers=0)

    G, D = G_UNet().to(DEV), D_Patch().to(DEV)
    optG = torch.optim.Adam(G.parameters(), LR, betas=(BETA1,0.999))
    optD = torch.optim.Adam(D.parameters(), LR, betas=(BETA1,0.999))

    G_log, D_log = [], []

    for ep in range(1,EPOCHS+1):
        g_sum=d_sum=0.
        for bg,ov_gt in tqdm(loader,desc=f"Epoch {ep}/{EPOCHS}",leave=False):
            bg,ov_gt=bg.to(DEV),ov_gt.to(DEV)
            B = bg.size(0)
            # ------------ noise channel (broadcast) ---------------
            z = torch.randn(B,1,1,1,device=DEV).expand(-1,1,RES_TRAIN,RES_TRAIN)
            gin = torch.cat([bg,z],1)

            # -------------------- D -------------------------------
            optD.zero_grad()
            with torch.no_grad():
                ov_fake = G(gin)
            real_pair = torch.cat([bg, ov_gt ],1)
            fake_pair = torch.cat([bg, ov_fake],1)
            d_real = D(diff_aug(real_pair))
            d_fake = D(diff_aug(fake_pair))
            loss_D = d_hinge(d_real,d_fake)
            loss_D.backward(); optD.step()

            # -------------------- G -------------------------------
            optG.zero_grad()
            ov_fake = G(gin)
            fake_pair = torch.cat([bg, ov_fake],1)
            g_adv = g_hinge(D(diff_aug(fake_pair)))
            g_l1  = L1(ov_fake,ov_gt)*λ_L1
            loss_G = g_adv + g_l1
            loss_G.backward(); optG.step()

            g_sum+=loss_G.item(); d_sum+=loss_D.item()

        G_log.append(g_sum/len(loader)); D_log.append(d_sum/len(loader))
        print(f"Ep {ep:02}  G={G_log[-1]:.3f}  D={D_log[-1]:.3f}")

        # ---------------- preview -------------------------------
        with torch.no_grad():
            z = torch.randn(B,1,1,1,device=DEV).expand(-1,1,RES_TRAIN,RES_TRAIN)
            gin = torch.cat([bg, z],1)
            ov_pred = G(gin)[:4]
            bg_samp = bg[:4]
            ov_gt_samp = ov_gt[:4]

            ov_full  = F.interpolate(ov_pred,(RES_FULL,RES_FULL),
                                     mode='bilinear',align_corners=False)
            bg_full  = F.interpolate(bg_samp,(RES_FULL,RES_FULL),
                                     mode='bilinear',align_corners=False)
            synth    = torch.clamp(bg_full+ov_full,-1,1)

            white = torch.ones_like(bg_full)
            true  = torch.clamp(white+F.interpolate(ov_gt_samp,(RES_FULL,RES_FULL),
                                                    mode='bilinear',align_corners=False),-1,1)
            pred  = torch.clamp(white+ov_full,-1,1)
            defect= torch.clamp(bg_full+F.interpolate(ov_gt_samp,(RES_FULL,RES_FULL),
                                                      mode='bilinear',align_corners=False),-1,1)
            rows=[defect,true,pred,synth]
            preview=torch.cat([vutils.make_grid(r,nrow=4,padding=2)
                               for r in rows], dim=1)
            vutils.save_image(preview*0.5+0.5,
                              f"{OUT_DIR}/side_by_side_epoch_{ep:02}.png",
                              normalize=False)

        # ---- checkpoints & logs ----
        if ep % 5 == 0:
            torch.save(G.state_dict(),f"{CKPT_DIR}/G_epoch_{ep:02}.pt")
            torch.save(D.state_dict(),f"{CKPT_DIR}/D_epoch_{ep:02}.pt")

    # final save + plot
    torch.save(G.state_dict(),f"{CKPT_DIR}/G_final.pt")
    with open(CSV_LOG,'w',newline='') as f:
        csv.writer(f).writerows([("epoch","G","D")] +
                                [(i+1,G_log[i],D_log[i]) for i in range(len(G_log))])
    plt.plot(G_log,label='G'); plt.plot(D_log,label='D')
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend()
    plt.tight_layout(); plt.savefig(LOSS_PLOT); plt.close()
    print("✅ v13 training complete – outputs in", OUT_DIR)

# ---------------------------------------------------------------------
if __name__=="__main__":
    main()
# =====================================================================
