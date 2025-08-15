# ===================== defect_overlay_cpu_v5.py ======================
"""
FAST CPU-only overlay trainer
‒ 256 px centre-crop
‒ Tiny-U-Net (2-level, 16->32 ch)
‒ 1-epoch warm-up on positives
‒ Pos/neg half-batches (neg = bg1–bg2)
‒ 4-row 1024 px preview each epoch
"""

import os, math, random
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T, torchvision.utils as vutils
from PIL import Image, ImageFilter
from tqdm import tqdm

# ---------------- CONFIG ----------------
ROOT_GOOD   = "data/good"
ROOT_DEFECT = "data/defect"
GEN_W       = "outputs/fake_good_v6/generator_good_v6.pt"

RES_TRAIN   = 256
RES_FULL    = 1024
EPOCHS      = 12
WARM_EPOCHS = 1
BATCH       = 4
LR          = 1e-3

THRESH      = 0.05
EDGE_W      = 0.7
SPARSE_W    = 1e-3
NEG_W       = 0.2
PREVIEW_N   = 4

OUT_DIR  = "outputs/defect_overlay_cpu_v5"
CKPT_DIR = f"{OUT_DIR}/ckpt_v5"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

torch.set_num_threads(max(os.cpu_count()-1,1))
DEV = torch.device("cpu")

# -------------- transforms --------------
class CenterCrop256:
    def __call__(self, img):
        w,h = img.size
        left = (w-RES_TRAIN)//2
        top  = (h-RES_TRAIN)//2
        return img.crop((left, top, left+RES_TRAIN, top+RES_TRAIN))

xform = T.Compose([
    T.Grayscale(),
    CenterCrop256(),
    T.ToTensor(),
    T.Normalize((0.5,), (0.5,))
])

ALLOWED={".png",".jpg",".jpeg",".bmp",".tif",".tiff"}
def imgs(root): return [str(p) for p in Path(root).rglob('*')
                        if p.suffix.lower() in ALLOWED]

@torch.no_grad()
def overlay(defect,bg):
    d = defect - bg
    m = (d.abs()>THRESH).float()
    if m.sum():
        mimg=T.ToPILImage()(m).filter(ImageFilter.MaxFilter(3)).filter(ImageFilter.MinFilter(3))
        m=T.ToTensor()(mimg)
    return m*d,m

GOOD_REAL, DEFECT_REAL = imgs(ROOT_GOOD), imgs(ROOT_DEFECT)
if not GOOD_REAL or not DEFECT_REAL:
    raise RuntimeError("data/good or data/defect empty")

# -------------- GAN for preview ---------
NZ, NGF = 100,64
def make_gen():
    layers=[nn.ConvTranspose2d(NZ,NGF*16,4,1,0,bias=False),
            nn.BatchNorm2d(NGF*16), nn.ReLU(True)]
    oc=NGF*16
    for _ in range(int(math.log2(RES_FULL))-2):
        ic,oc=oc,max(NGF,oc//2)
        layers+=[nn.ConvTranspose2d(ic,oc,4,2,1,bias=False),
                 nn.BatchNorm2d(oc),nn.ReLU(True)]
    layers+=[nn.ConvTranspose2d(oc,1,3,1,1,bias=False),nn.Tanh()]
    return nn.Sequential(*layers)

Gbg=make_gen().to(DEV).eval()
Gbg.load_state_dict(torch.load(GEN_W,map_location=DEV))

@torch.no_grad()
def gan_pair(n):
    z=torch.randn(n,NZ,1,1,device=DEV)
    full=Gbg(z)
    small=F.interpolate(full,(RES_TRAIN,RES_TRAIN),mode="bilinear",align_corners=False)
    return full,small

def rand_good():
    if random.random()<0.5:
        return xform(Image.open(random.choice(GOOD_REAL)).convert('L'))
    _,s=gan_pair(1); return s[0]

# -------------- dataset -----------------
class DS(Dataset):
    def __len__(s): return max(len(GOOD_REAL),len(DEFECT_REAL))
    def __getitem__(s,_):
        if random.random()<0.5:
            bg=rand_good()
            defect=xform(Image.open(random.choice(DEFECT_REAL)).convert('L'))
            ov,mask=overlay(defect,bg); lbl=1
        else:
            bg=rand_good(); ov=mask=torch.zeros_like(bg); lbl=0
        return bg,ov,mask,lbl

# -------------- Tiny U-Net 2lvl ---------
def down(i,o): return nn.Sequential(nn.Conv2d(i,o,4,2,1,bias=False),
                                    nn.BatchNorm2d(o),nn.LeakyReLU(0.2,True))
def up(i,o):   return nn.Sequential(nn.ConvTranspose2d(i,o,4,2,1,bias=False),
                                    nn.BatchNorm2d(o),nn.ReLU(True))
class UNet(nn.Module):
    def __init__(s,ch=16):
        super().__init__()
        s.e1=down(1,ch); s.e2=down(ch,ch*2)
        s.d2=up(ch*2,ch); s.d1=nn.ConvTranspose2d(ch*2,2,4,2,1)
    def forward(s,x):
        e1=s.e1(x); e2=s.e2(e1); d2=s.d2(e2)
        d1=s.d1(torch.cat([d2,e1],1))
        mask=torch.sigmoid(d1[:,:1]); delta=torch.tanh(d1[:,1:])
        return mask*delta,mask

# -------------- helpers -----------------
SOBX=torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]],dtype=torch.float32).view(1,1,3,3)/8
SOBY=SOBX.transpose(2,3)
def gmag(x): return torch.sqrt(F.conv2d(x,SOBX,padding=1)**2+
                               F.conv2d(x,SOBY,padding=1)**2+1e-6)

# -------------- train -------------------
net=UNet().to(DEV); opt=torch.optim.Adam(net.parameters(),LR)
L1=nn.L1Loss()
loader=DataLoader(DS(),BATCH,shuffle=True,num_workers=0)

for ep in range(1,EPOCHS+1):
    warm = ep<=WARM_EPOCHS
    for bg,ov_gt,mask_gt,lbl in tqdm(loader,desc=f"ep{ep}/{EPOCHS}"):
        if warm: lbl=torch.ones_like(lbl)
        bg,ov_gt,lbl=[t.to(DEV) for t in (bg,ov_gt,lbl)]
        ov_pred,mask_pred=net(bg)
        pos=lbl==1; neg=lbl==0
        loss=torch.tensor(0.,device=DEV)
        if pos.any():
            pp,pt=ov_pred[pos],ov_gt[pos]
            pm=mask_pred[pos]
            loss+=L1(pp,pt)+EDGE_W*L1(gmag(pp),gmag(pt))+SPARSE_W*pm.mean()
        if neg.any():
            np,nm=ov_pred[neg],mask_pred[neg]
            loss+=NEG_W*(torch.abs(np).mean()+nm.mean())
        opt.zero_grad(); loss.backward(); opt.step()

    # ---- preview ----
    with torch.no_grad():
        bgF,bgS=gan_pair(PREVIEW_N)
        ovS,_=net(bgS)
        ovF=F.interpolate(ovS,(RES_FULL,RES_FULL),mode="bilinear",align_corners=False)
        synth=torch.clamp(bgF+ovF,-1,1)
        real_defects=[]
        ds=DS()
        while len(real_defects)<PREVIEW_N:
            b,o,_,l=ds[0]
            if l==1: real_defects.append((b,o))
        rb=torch.stack([p[0] for p in real_defects]).to(DEV)
        ro=torch.stack([p[1] for p in real_defects]).to(DEV)
        rbF=F.interpolate(rb,(RES_FULL,RES_FULL),mode="bilinear",align_corners=False)
        roF=F.interpolate(ro,(RES_FULL,RES_FULL),mode="bilinear",align_corners=False)
        real = torch.clamp(rbF+roF,-1,1)
        white=torch.ones_like(rbF)
        trueW=torch.clamp(white+roF,-1,1); predW=torch.clamp(white+ovF,-1,1)
        grid=torch.cat([vutils.make_grid(r,nrow=PREVIEW_N,padding=2)
                       for r in (real,trueW,predW,synth)],1)
        vutils.save_image(grid*0.5+0.5,f"{OUT_DIR}/side_by_side_{ep:02}.png",normalize=False)
    torch.save(net.state_dict(),f"{CKPT_DIR}/unet_{ep:02}.pt")
    print("epoch",ep,"done")

print("✅ training complete – see",OUT_DIR)
# =====================================================================
