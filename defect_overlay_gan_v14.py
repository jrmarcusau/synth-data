# ===================== defect_overlay_unet_v14.py =====================
"""
Conditional GAN for defect overlays
— hinge loss, spectral-norm, DiffAugment
— checkerboard-free generator (nearest-neighbor upsample + 3×3 conv)
All preprocessing, mask extraction (τ = 0.12), preview layout, and file
structure remain unchanged.

Outputs
-------
outputs/defect_overlay_cpu_v14/
    ├ defect_maps_grid.png
    ├ side_by_side_epoch_01.png …
    ├ losses.csv
    ├ loss_plot.png
    └ ckpt_v14/{G,D}_epoch_05.pt …
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
NZ                      = 1          # 1-channel broadcast noise
λ_L1                    = 75.0       # will drop later
OUT_DIR  = "outputs/defect_overlay_cpu_v14"
CKPT_DIR = f"{OUT_DIR}/ckpt_v14"
CSV_LOG  = f"{OUT_DIR}/losses.csv"
LOSS_PLOT= f"{OUT_DIR}/loss_plot.png"
os.makedirs(CKPT_DIR, exist_ok=True)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(max(os.cpu_count()-1, 1))

# --------------------------- transforms ------------------------------
class PadSquare:
    def __init__(self,size): self.size=size
    def __call__(self,img):
        w,h = img.size; side=max(w,h,self.size)
        pad=[(side-w)//2,(side-h)//2,
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
def make_overlay(defect,bg):
    delta = defect - bg
    mask  = (delta.abs()>THRESH_ABS).float()
    mask  = F.max_pool2d(mask,3,1,1); mask=-F.max_pool2d(-mask,3,1,1)
    mask  = -F.max_pool2d(-mask,5,1,2); mask = F.max_pool2d(mask,5,1,2)
    return mask*delta, mask

# ----------------------------- dataset -------------------------------
ALLOWED={".png",".jpg",".jpeg",".bmp",".tif",".tiff"}
def list_imgs(root): return [str(p) for p in Path(root).rglob('*')
                             if p.suffix.lower() in ALLOWED]
def quick_vec(path,side=64):
    img=Image.open(path).convert('L').resize((side,side),Image.BILINEAR)
    return torch.from_numpy(np.asarray(img,dtype=np.float32)/255.)
class OverlayDS(Dataset):
    def __init__(self):
        good, defect = list_imgs(ROOT_GOOD), list_imgs(ROOT_DEFECT)
        if not good or not defect: raise RuntimeError("empty folders")
        g_vecs=torch.stack([quick_vec(p) for p in good])
        self.pairs=[]
        for d in defect:
            d_vec=quick_vec(d)
            idx=torch.argmin(((g_vecs-d_vec)**2).view(len(good),-1).mean(1))
            self.pairs.append((d, good[idx]))
    def __len__(self): return len(self.pairs)
    def __getitem__(self,i):
        d,g = self.pairs[i]
        defect=_xform(Image.open(d).convert("L"))
        good  =_xform(Image.open(g).convert("L"))
        ov,_  = make_overlay(defect,good)
        return good, ov

# ----------------------------- networks ------------------------------
def sn(layer): return nn.utils.spectral_norm(layer)
# -- discriminator (Patch-GAN, hinge, SN, ch=64) --
def d_block(i,o,norm=True):
    m=[sn(nn.Conv2d(i,o,4,2,1,bias=not norm))]
    if norm: m.append(nn.BatchNorm2d(o))
    m.append(nn.LeakyReLU(0.2,True)); return m
class D_Patch(nn.Module):
    def __init__(self,ch=64):
        super().__init__()
        seq=[]
        seq+=d_block(2,   ch,   norm=False)
        seq+=d_block(ch,  ch*2)
        seq+=d_block(ch*2,ch*4)
        seq+=d_block(ch*4,ch*8)
        seq.append(sn(nn.Conv2d(ch*8,1,4,1,1,bias=False)))
        self.net=nn.Sequential(*seq)
    def forward(self,x): return self.net(x).mean()
# -- generator: resize-conv U-Net (checkerboard-free) --
class Up(nn.Module):
    def __init__(self,i,o):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv=sn(nn.Conv2d(i,o,3,1,1,bias=False))
        self.bn=nn.BatchNorm2d(o); self.act=nn.ReLU(True)
    def forward(self,x): return self.act(self.bn(self.conv(self.up(x))))
class G_UNet(nn.Module):
    def __init__(self,ch=64,in_ch=1+NZ):
        super().__init__()
        self.e1=nn.Sequential(sn(nn.Conv2d(in_ch,ch,4,2,1,bias=False)),
                              nn.LeakyReLU(0.2,True))
        self.e2=nn.Sequential(sn(nn.Conv2d(ch,ch*2,4,2,1,bias=False)),
                              nn.BatchNorm2d(ch*2),nn.LeakyReLU(0.2,True))
        self.e3=nn.Sequential(sn(nn.Conv2d(ch*2,ch*4,4,2,1,bias=False)),
                              nn.BatchNorm2d(ch*4),nn.LeakyReLU(0.2,True))
        self.e4=nn.Sequential(sn(nn.Conv2d(ch*4,ch*8,4,2,1,bias=False)),
                              nn.BatchNorm2d(ch*8),nn.LeakyReLU(0.2,True))
        self.e5=nn.Sequential(sn(nn.Conv2d(ch*8,ch*8,4,2,1,bias=False)),
                              nn.BatchNorm2d(ch*8),nn.LeakyReLU(0.2,True))
        self.d5=Up(ch*8 ,ch*8)
        self.d4=Up(ch*16,ch*4)
        self.d3=Up(ch*8 ,ch*2)
        self.d2=Up(ch*4 ,ch)
        self.out=nn.Sequential(
            nn.Upsample(scale_factor=2,mode='nearest'),
            sn(nn.Conv2d(ch*2,1,3,1,1,bias=False)),
            nn.Tanh())
    def forward(self,x):
        e1=self.e1(x); e2=self.e2(e1); e3=self.e3(e2); e4=self.e4(e3)
        e5=self.e5(e4)
        d5=self.d5(e5)
        d4=self.d4(torch.cat([d5,e4],1))
        d3=self.d3(torch.cat([d4,e3],1))
        d2=self.d2(torch.cat([d3,e2],1))
        return self.out(torch.cat([d2,e1],1))

# ------------------------ losses & DiffAug ---------------------------
def d_hinge(real,fake): return 0.5*(F.relu(1-real)+F.relu(1+fake)).mean()
def g_hinge(fake): return -fake.mean()
L1 = nn.L1Loss()
def diff_aug(x):
    if torch.rand(1)<0.5: x=torch.flip(x,[3])
    x=x+0.05*torch.randn_like(x); return x.clamp(-1,1)

# -------------------------- helpers ---------------------------------
def grid_defects(ds,path,n=16):
    imgs=[ds[i][1]*0.5+0.5 for i in range(min(n,len(ds)))]
    vutils.save_image(vutils.make_grid(imgs,nrow=4,padding=2),
                      path,normalize=False)

# ----------------------------- train --------------------------------
def main():
    ds=OverlayDS(); grid_defects(ds,f"{OUT_DIR}/defect_maps_grid.png")
    loader=DataLoader(ds,BATCH,shuffle=True,num_workers=0)
    G,D = G_UNet().to(DEV), D_Patch().to(DEV)
    optG=torch.optim.Adam(G.parameters(), LR, betas=(BETA1,0.999))
    optD=torch.optim.Adam(D.parameters(), LR, betas=(BETA1,0.999))
    G_log,D_log=[],[]
    for ep in range(1,EPOCHS+1):
        g_tot=d_tot=0.
        if ep==30:                                              # drop L1 later
            global λ_L1; λ_L1=25.0
        for bg,ov_gt in tqdm(loader,desc=f"Ep {ep}/{EPOCHS}",leave=False):
            bg,ov_gt=bg.to(DEV),ov_gt.to(DEV); B=bg.size(0)
            z=2*torch.randn(B,1,1,1,device=DEV).expand(-1,1,RES_TRAIN,RES_TRAIN)
            gin=torch.cat([bg,z],1)
            # ------- D -------
            optD.zero_grad()
            with torch.no_grad(): ov_fake=G(gin)
            d_r=D(diff_aug(torch.cat([bg,ov_gt],1)))
            d_f=D(diff_aug(torch.cat([bg,ov_fake],1)))
            loss_D=d_hinge(d_r,d_f); loss_D.backward(); optD.step()
            # ------- G -------
            optG.zero_grad()
            ov_fake=G(gin)
            g_adv=g_hinge(D(diff_aug(torch.cat([bg,ov_fake],1))))
            g_l1=L1(ov_fake,ov_gt)*λ_L1
            loss_G=g_adv+g_l1; loss_G.backward(); optG.step()
            g_tot+=loss_G.item(); d_tot+=loss_D.item()
        G_log.append(g_tot/len(loader)); D_log.append(d_tot/len(loader))
        print(f"Ep {ep:02}  G={G_log[-1]:.3f}  D={D_log[-1]:.3f}")
        # -------------- preview --------------
        with torch.no_grad():
            z=2*torch.randn(B,1,1,1,device=DEV).expand(-1,1,RES_TRAIN,RES_TRAIN)
            ov=G(torch.cat([bg,z],1))[:4]; bg4=bg[:4]; gt4=ov_gt[:4]
            ovF=F.interpolate(ov,(RES_FULL,RES_FULL),mode='bilinear',align_corners=False)
            bgF=F.interpolate(bg4,(RES_FULL,RES_FULL),mode='bilinear',align_corners=False)
            synth=torch.clamp(bgF+ovF,-1,1)
            white=torch.ones_like(bgF)
            gt  =torch.clamp(white+F.interpolate(gt4,(RES_FULL,RES_FULL),
                                                 mode='bilinear',align_corners=False),-1,1)
            pred=torch.clamp(white+ovF,-1,1)
            defect=torch.clamp(bgF+F.interpolate(gt4,(RES_FULL,RES_FULL),
                                                 mode='bilinear',align_corners=False),-1,1)
            grid=torch.cat([vutils.make_grid(r,nrow=4,padding=2)
                            for r in [defect,gt,pred,synth]],dim=1)
            vutils.save_image(grid*0.5+0.5,
                              f"{OUT_DIR}/side_by_side_epoch_{ep:02}.png",
                              normalize=False)
        # checkpoint
        if ep%5==0:
            torch.save(G.state_dict(),f"{CKPT_DIR}/G_epoch_{ep:02}.pt")
            torch.save(D.state_dict(),f"{CKPT_DIR}/D_epoch_{ep:02}.pt")
    torch.save(G.state_dict(),f"{CKPT_DIR}/G_final.pt")
    with open(CSV_LOG,'w',newline='') as f:
        csv.writer(f).writerows([("epoch","G","D")]+[(i+1,G_log[i],D_log[i]) for i in range(len(G_log))])
    plt.plot(G_log,label='G'); plt.plot(D_log,label='D')
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend(); plt.tight_layout()
    plt.savefig(LOSS_PLOT); plt.close()
    print("✅ v14 training complete — see", OUT_DIR)

# ---------------------------------------------------------------------
if __name__=="__main__":
    main()
# =====================================================================
