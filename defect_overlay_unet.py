# ==================== defect_overlay_cpu.py ===========================
"""
CPU-only, fast-train overlay model:
  • train at 256 px   (set RES_TRAIN = 512 if you want)
  • keep GAN preview at 1024 px
  • Tiny-U-Net 3 levels, 16 base chans
"""

import os, csv, random, math
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T, torchvision.utils as vutils
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

# ---------- CONFIG ----------------------------------------------------
ROOT_GOOD   = "data/good"
ROOT_DEFECT = "data/defect"
GEN_WEIGHTS = "outputs/fake_good_v6/generator_good_v6.pt"

RES_TRAIN   = 256          # 256 or 512   ← tiny U-Net input/output
RES_FULL    = 1024         # background GAN native size (stay 1024)
EPOCHS      = 10
BATCH       = 4            # CPU: keep small enough to fit RAM
LR          = 1e-3         # a bit higher for CPU
TV_W        = 0.0          # turn off for speed
THR         = 0.08
PREVIEW_N   = 4

OUT_DIR     = "outputs/defect_overlay_cpu"
CKPT_DIR    = f"{OUT_DIR}/ckpt"
os.makedirs(OUT_DIR, exist_ok=True); os.makedirs(CKPT_DIR, exist_ok=True)

torch.set_num_threads(max(os.cpu_count() - 1, 1))
DEV = torch.device("cpu")

# ---------- transforms ------------------------------------------------
class PadSquare:
    def __init__(self, size): self.size=size
    def __call__(self,img):
        w,h=img.size; side=max(w,h,self.size)
        pad=[(side-w)//2,(side-h)//2,side-w-(side-w)//2,side-h-(side-h)//2]
        return T.functional.pad(img,pad,padding_mode='reflect')

xform = T.Compose([
    T.Grayscale(),
    PadSquare(RES_TRAIN),
    T.Resize((RES_TRAIN, RES_TRAIN), interpolation=T.InterpolationMode.BILINEAR),
    T.ToTensor(),
    T.Normalize((0.5,), (0.5,))
])

ALLOWED={'.png','.jpg','.jpeg','.bmp','.tif','.tiff'}
def files(root): return [str(p) for p in Path(root).rglob('*')
                         if p.suffix.lower() in ALLOWED]

@torch.no_grad()
def make_overlay(defect,bg):
    delta=defect-bg
    mask=(delta.abs()>THR).float()
    return mask*delta,mask

# ---------- dataset ---------------------------------------------------
class OverlayDS(Dataset):
    def __init__(self):
        self.good=files(ROOT_GOOD); self.bad=files(ROOT_DEFECT)
        if not self.good or not self.bad:
            raise RuntimeError("empty data/good or data/defect")
        self.N=max(len(self.good),len(self.bad))
    def __len__(s): return s.N
    def __getitem__(s,idx):
        g=xform(Image.open(random.choice(s.good)).convert('L'))
        d=xform(Image.open(random.choice(s.bad )).convert('L'))
        ov,mask=make_overlay(d,g)
        return g,ov,mask

# ---------- background GAN (unchanged) --------------------------------
NZ, NGF = 100, 64
def make_generator():
    layers=[nn.ConvTranspose2d(NZ, NGF*16,4,1,0,bias=False),
            nn.BatchNorm2d(NGF*16), nn.ReLU(True)]
    out_c=NGF*16
    for _ in range(int(math.log2(RES_FULL)-2)):
        in_c,out_c=out_c,max(NGF,out_c//2)
        layers += [nn.ConvTranspose2d(in_c,out_c,4,2,1,bias=False),
                   nn.BatchNorm2d(out_c), nn.ReLU(True)]
    layers += [nn.ConvTranspose2d(out_c,1,3,1,1,bias=False), nn.Tanh()]
    return nn.Sequential(*layers)

Gbg=make_generator().to(DEV).eval()
Gbg.load_state_dict(torch.load(GEN_WEIGHTS,map_location=DEV))

@torch.no_grad()
def sample_bg(n):
    z=torch.randn(n,NZ,1,1,device=DEV)
    bg=Gbg(z)                               # 1×1024×1024
    if RES_TRAIN!=RES_FULL:
        bg_small=F.interpolate(bg,size=(RES_TRAIN,RES_TRAIN),
                               mode='bilinear',align_corners=False)
    else:
        bg_small=bg
    return bg, bg_small                     # full-res and downsampled

# ---------- Tiny-U-Net (3 levels) -------------------------------------
def down(i,o): return nn.Sequential(nn.Conv2d(i,o,4,2,1,bias=False),
                                    nn.BatchNorm2d(o), nn.LeakyReLU(0.2,True))
def up(i,o):   return nn.Sequential(nn.ConvTranspose2d(i,o,4,2,1,bias=False),
                                    nn.BatchNorm2d(o), nn.ReLU(True))

class TinyUNet(nn.Module):
    def __init__(s,ch=16):
        super().__init__()
        s.e1=down(1,ch); s.e2=down(ch,ch*2); s.e3=down(ch*2,ch*4)
        s.d3=up(ch*4,ch*2); s.d2=up(ch*4,ch)
        s.d1=nn.ConvTranspose2d(ch*2,2,4,2,1)
    def forward(s,x):
        e1=s.e1(x); e2=s.e2(e1); e3=s.e3(e2)
        d3=s.d3(e3); d2=s.d2(torch.cat([d3,e2],1))
        d1=s.d1(torch.cat([d2,e1],1))
        mask=torch.sigmoid(d1[:,:1]); delta=torch.tanh(d1[:,1:])
        return mask*delta,mask

# ---------- train loop -------------------------------------------------
def tv(m): return F.l1_loss(m[:,:,:,1:],m[:,:,:,:-1]) + \
                 F.l1_loss(m[:,:,1:,:],m[:,:,:-1,:])

def main():
    ds=OverlayDS()
    loader=DataLoader(ds,BATCH,shuffle=True,num_workers=0)

    net=TinyUNet().to(DEV)
    opt=torch.optim.Adam(net.parameters(),LR)
    L1,BCE=nn.L1Loss(),nn.BCELoss()
    for ep in range(1,EPOCHS+1):
        for bg,ov_gt,mask_gt in tqdm(loader,desc=f"Ep {ep}/{EPOCHS}"):
            bg,ov_gt,mask_gt=[t.to(DEV) for t in (bg,ov_gt,mask_gt)]
            ov_pred,mask_pred=net(bg)
            loss=L1(ov_pred,ov_gt)+BCE(mask_pred,mask_gt)
            if TV_W: loss += TV_W*tv(mask_pred)
            opt.zero_grad(); loss.backward(); opt.step()

        # -------- GAN preview (1024 px) -------------------------------
        with torch.no_grad():
            bg_full,bg_small=sample_bg(PREVIEW_N)
            ov_small,_=net(bg_small)
            ov_full=F.interpolate(ov_small,size=(RES_FULL,RES_FULL),
                                  mode='bilinear',align_corners=False)
            synth=torch.clamp(bg_full+ov_full,-1,1)
            comp=torch.cat([bg_full,synth],dim=2)
            vutils.save_image(comp*0.5+0.5,
                              f"{OUT_DIR}/side_by_side_{ep:02}.png",
                              nrow=PREVIEW_N)

        torch.save(net.state_dict(),f"{CKPT_DIR}/tiny_{ep:02}.pt")

    print("✅ CPU training finished. Previews in", OUT_DIR)

# ----------------------------------------------------------------------
if __name__=="__main__":
    main()
# ======================================================================
