# ===================== defect_overlay_unet_v15.py =====================
"""
Conditional GAN for defect overlays
Aggressive anti-collapse version:
    • no tanh (clamp ±0.5)
    • 8×8 tiled noise
    • λ_L1 100→50→10 schedule
    • R1 gradient penalty (0.001)
    • checkerboard-free resize-conv generator
All preprocessing, τ=0.12 masking, white preview rows, and file layout
unchanged.
"""
# ---------------------------------------------------------------------
import os, math, csv, torch, numpy as np, matplotlib.pyplot as plt
from pathlib import Path
import torchvision.transforms as T, torchvision.utils as vutils
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn, torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

# ---------------------------- config ---------------------------------
ROOT_GOOD, ROOT_DEFECT = "data/good", "data/defect"
RES_TRAIN, RES_FULL    = 256, 1024
EPOCHS,  BATCH,  LR    = 50, 4, 2e-4
BETA1                   = 0.5
R1_GAMMA                = 0.001          # gradient penalty weight
NZ                      = 1              # noise chan
OUT_DIR  = "outputs/defect_overlay_cpu_v15"
CKPT_DIR = f"{OUT_DIR}/ckpt_v15"
CSV_LOG  = f"{OUT_DIR}/losses.csv"
LOSS_PLOT= f"{OUT_DIR}/loss_plot.png"
os.makedirs(CKPT_DIR, exist_ok=True)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(max(os.cpu_count()-1, 1))

# --------------------------- transforms ------------------------------
class PadSquare:
    def __init__(self,s): self.s=s
    def __call__(self,img):
        w,h=img.size; side=max(w,h,self.s)
        pad=[(side-w)//2,(side-h)//2,
             side-w-(side-w)//2, side-h-(side-h)//2]
        return T.functional.pad(img,pad,padding_mode='reflect')

_xform=T.Compose([
    T.Grayscale(),
    PadSquare(RES_TRAIN),
    T.Resize((RES_TRAIN,RES_TRAIN),
             interpolation=T.InterpolationMode.BILINEAR),
    T.ToTensor(),
    T.Normalize((0.5,),(0.5,))
])

# ----------------------- mask / overlay ------------------------------
THRESH_ABS=0.12
def make_overlay(defect,bg):
    d=defect-bg
    m=(d.abs()>THRESH_ABS).float()
    m=F.max_pool2d(m,3,1,1); m=-F.max_pool2d(-m,3,1,1)
    m=-F.max_pool2d(-m,5,1,2); m=F.max_pool2d(m,5,1,2)
    return m*d, m

# ----------------------------- dataset -------------------------------
ALLOWED={".png",".jpg",".jpeg",".bmp",".tif",".tiff"}
def list_imgs(root): return [str(p) for p in Path(root).rglob('*')
                             if p.suffix.lower() in ALLOWED]
def quick_vec(path,side=64):
    img=Image.open(path).convert('L').resize((side,side),Image.BILINEAR)
    return torch.from_numpy(np.asarray(img,dtype=np.float32)/255.)
class OverlayDS(Dataset):
    def __init__(self):
        good,defect=list_imgs(ROOT_GOOD),list_imgs(ROOT_DEFECT)
        if not good or not defect: raise RuntimeError("empty data")
        g_vecs=torch.stack([quick_vec(p) for p in good])
        self.pairs=[]
        for d in defect:
            d_vec=quick_vec(d)
            idx=torch.argmin(((g_vecs-d_vec)**2).view(len(good),-1).mean(1))
            self.pairs.append((d,good[idx]))
    def __len__(self): return len(self.pairs)
    def __getitem__(self,i):
        d,g=self.pairs[i]
        defect=_xform(Image.open(d).convert('L'))
        good  =_xform(Image.open(g).convert('L'))
        ov,_  =make_overlay(defect,good)
        return good,ov

# ---------------------------- networks -------------------------------
def sn(l): return nn.utils.spectral_norm(l)
# discriminator
def d_block(i,o,norm=True):
    m=[sn(nn.Conv2d(i,o,4,2,1,bias=not norm))]
    if norm:m.append(nn.BatchNorm2d(o))
    m.append(nn.LeakyReLU(0.2,True)); return m
class D(nn.Module):
    def __init__(self,ch=64):
        super().__init__()
        seq=[]
        seq+=d_block(2,ch,False)
        seq+=d_block(ch,ch*2)
        seq+=d_block(ch*2,ch*4)
        seq+=d_block(ch*4,ch*8)
        seq.append(sn(nn.Conv2d(ch*8,1,4,1,1,bias=False)))
        self.net=nn.Sequential(*seq)
    def forward(self,x,return_features=False):
        feats=[]
        out=x
        for layer in self.net:
            out=layer(out)
            feats.append(out)
        return (out.mean(),feats[-2]) if return_features else out.mean()
# generator (resize-conv UNet)
class Up(nn.Module):
    def __init__(self,i,o):
        super().__init__()
        self.up=nn.Upsample(scale_factor=2,mode='nearest')
        self.conv=sn(nn.Conv2d(i,o,3,1,1,bias=False))
        self.bn=nn.BatchNorm2d(o); self.act=nn.ReLU(True)
    def forward(self,x): return self.act(self.bn(self.conv(self.up(x))))
class G(nn.Module):
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
            sn(nn.Conv2d(ch*2,1,3,1,1,bias=False)))
    def forward(self,x):
        e1=self.e1(x); e2=self.e2(e1); e3=self.e3(e2); e4=self.e4(e3)
        e5=self.e5(e4)
        d5=self.d5(e5)
        d4=self.d4(torch.cat([d5,e4],1))
        d3=self.d3(torch.cat([d4,e3],1))
        d2=self.d2(torch.cat([d3,e2],1))
        out=self.out(torch.cat([d2,e1],1))
        return torch.clamp(out, -0.5, 0.5)

# ------------- losses, DiffAug, logging helpers ----------------------
def d_hinge(real,fake): return 0.5*(F.relu(1-real)+F.relu(1+fake))
def g_hinge(fake): return -fake.mean()
L1=nn.L1Loss()
def diff_aug(x):
    if torch.rand(1)<0.5: x=torch.flip(x,[3])
    x=x+0.05*torch.randn_like(x); return x.clamp(-1,1)
def grid_defects(ds,path,n=16):
    imgs=[ds[i][1]*0.5+0.5 for i in range(min(n,len(ds)))]
    vutils.save_image(vutils.make_grid(imgs,nrow=4,padding=2),
                      path,normalize=False)

# ------------------------------- train -------------------------------
def main():
    ds=OverlayDS(); grid_defects(ds,f"{OUT_DIR}/defect_maps_grid.png")
    loader=DataLoader(ds,BATCH,shuffle=True,num_workers=0)
    netG,netD=G().to(DEV),D().to(DEV)
    optG=torch.optim.Adam(netG.parameters(),LR,betas=(BETA1,0.999))
    optD=torch.optim.Adam(netD.parameters(),LR,betas=(BETA1,0.999))
    G_log,D_log=[],[]
    for ep in range(1,EPOCHS+1):
        if ep<5:  curr_L1=100
        elif ep<20:curr_L1=50
        else:      curr_L1=10
        g_tot=d_tot=0.
        for bg,ov_gt in tqdm(loader,desc=f"E{ep}/{EPOCHS}",leave=False):
            bg,ov_gt=bg.to(DEV),ov_gt.to(DEV); B=bg.size(0)
            z=torch.randn(B,1,8,8,device=DEV)
            z=F.interpolate(z,(RES_TRAIN,RES_TRAIN),mode='nearest')
            gin=torch.cat([bg,z],1)

            # ---- D ----
            with torch.no_grad():
                ov_fake=netG(gin)
            optD.zero_grad()
            real_pair=torch.cat([bg, ov_gt], 1).detach().requires_grad_(True)
            fake_pair=torch.cat([bg,ov_fake.detach()],1)
            d_real,feat_real=netD(diff_aug(real_pair),return_features=True)
            d_fake         =netD(diff_aug(fake_pair))
            loss_D = d_hinge(d_real,d_fake).mean()
            # R1 gradient penalty
            grad=torch.autograd.grad(d_real,real_pair,
                       torch.ones_like(d_real),create_graph=True)[0]
            loss_D += R1_GAMMA* (grad.pow(2).view(B,-1).sum(1).mean())
            loss_D.backward(); optD.step()

            # ---- G ----
            optG.zero_grad()
            ov_fake=netG(gin)
            fake_pair=torch.cat([bg,ov_fake],1)
            g_adv=g_hinge(netD(diff_aug(fake_pair)))
            g_l1 = L1(ov_fake,ov_gt)*curr_L1
            loss_G = g_adv + g_l1
            loss_G.backward(); optG.step()

            g_tot+=loss_G.item(); d_tot+=loss_D.item()

        G_log.append(g_tot/len(loader)); D_log.append(d_tot/len(loader))
        print(f"Ep {ep:02}  G={G_log[-1]:.3f}  D={D_log[-1]:.3f}")

        # ---------- preview ----------
        with torch.no_grad():
            z=torch.randn(B,1,8,8,device=DEV)
            z=F.interpolate(z,(RES_TRAIN,RES_TRAIN),mode='nearest')
            ov=netG(torch.cat([bg,z],1))[:4]; bg4=bg[:4]; gt4=ov_gt[:4]
            ovF=F.interpolate(ov,(RES_FULL,RES_FULL),mode='bilinear',align_corners=False)
            bgF=F.interpolate(bg4,(RES_FULL,RES_FULL),mode='bilinear',align_corners=False)
            synth=torch.clamp(bgF+ovF,-1,1)
            white=torch.ones_like(bgF)
            gt  =torch.clamp(white+F.interpolate(gt4,(RES_FULL,RES_FULL),
                                                 mode='bilinear',
                                                 align_corners=False),-1,1)
            pred=torch.clamp(white+ovF,-1,1)
            defect=torch.clamp(bgF+F.interpolate(gt4,(RES_FULL,RES_FULL),
                                                 mode='bilinear',
                                                 align_corners=False),-1,1)
            grid=torch.cat([vutils.make_grid(r,nrow=4,padding=2)
                            for r in [defect,gt,pred,synth]],dim=1)
            vutils.save_image(grid*0.5+0.5,
                              f"{OUT_DIR}/side_by_side_epoch_{ep:02}.png",
                              normalize=False)

        if ep%5==0:
            torch.save(netG.state_dict(),f"{CKPT_DIR}/G_epoch_{ep:02}.pt")
            torch.save(netD.state_dict(),f"{CKPT_DIR}/D_epoch_{ep:02}.pt")

    torch.save(netG.state_dict(),f"{CKPT_DIR}/G_final.pt")
    with open(CSV_LOG,'w',newline='') as f:
        csv.writer(f).writerows([("epoch","G","D")]+[(i+1,G_log[i],D_log[i]) for i in range(len(G_log))])
    plt.plot(G_log,label='G'); plt.plot(D_log,label='D')
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend()
    plt.tight_layout(); plt.savefig(LOSS_PLOT); plt.close()
    print("✅ v15 training complete — see", OUT_DIR)

# ---------------------------------------------------------------------
if __name__=="__main__":
    main()
# =====================================================================
