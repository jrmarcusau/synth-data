# ===================== defect_overlay_cpu_v4.py ======================
"""
CPU-only overlay trainer (no hard ROI).

How this version works
----------------------
• Each batch is 50 % **positive** (real defect image) and 50 % **negative**
  (difference between two good images ⇒ overlay target = 0).
• The net is penalised if it predicts anything on negative samples,
  so it learns that background variation is NOT a defect.
• Tiny-U-Net (3 levels, 16 ch) trained at 256 px (quick).
• Loss =  L1 + EDGE + SPARSE
           + BCE(mask,0)  on negatives.
• Four-row preview saved each epoch at 1024 px:
      1) real defect (good + true overlay)
      2) true overlay on white
      3) predicted overlay on white
      4) predicted overlay on GAN background
"""

import os, csv, math, random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.utils as vutils
from PIL import Image, ImageFilter
from tqdm import tqdm
import matplotlib.pyplot as plt

# ----------------------- CONFIG --------------------------------------
ROOT_GOOD   = "data/good"
ROOT_DEFECT = "data/defect"
GEN_WEIGHTS = "outputs/fake_good_v6/generator_good_v6.pt"

RES_TRAIN   = 256          # 256 = fast; 512 = sharper
RES_FULL    = 1024         # GAN native size
EPOCHS      = 20
BATCH       = 4
LR          = 1e-3

THRESH      = 0.12         # delta threshold for mask
SPARSE_W    = 5e-3         # mask sparsity
EDGE_W      = 1.0          # edge consistency
NEG_W       = 1.0          # strength of negative-sample penalty
TV_W        = 0.0          # TV off by default

PREVIEW_N   = 4            # columns in preview grid

OUT_DIR     = "outputs/defect_overlay_cpu_v4"
CKPT_DIR    = f"{OUT_DIR}/ckpt_v4"
os.makedirs(OUT_DIR,  exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

torch.set_num_threads(max(os.cpu_count() - 1, 1))
DEV = torch.device("cpu")

# --------------------- TRANSFORMS & HELPERS --------------------------
class PadSquare:
    def __init__(self, size): self.size = size
    def __call__(self, img):
        w,h = img.size
        side = max(w, h, self.size)
        pad  = [(side-w)//2, (side-h)//2,
                side-w - (side-w)//2, side-h - (side-h)//2]
        return T.functional.pad(img, pad, padding_mode='reflect')

xform = T.Compose([
    T.Grayscale(),
    PadSquare(RES_TRAIN),
    T.Resize((RES_TRAIN, RES_TRAIN),
             interpolation=T.InterpolationMode.BILINEAR),
    T.ToTensor(),
    T.Normalize((0.5,), (0.5,))      # → [-1,1]
])

ALLOWED = {".png",".jpg",".jpeg",".bmp",".tif",".tiff"}
def list_imgs(root): return [str(p) for p in Path(root).rglob('*')
                             if p.suffix.lower() in ALLOWED]

@torch.no_grad()
def build_overlay(defect, bg):
    """Return (overlay, mask) from a defect image & its clean background."""
    delta = defect - bg
    mask  = (delta.abs() > THRESH).float()

    # 3×3 closing to clean speckles
    if mask.sum() > 0:
        mask_img = T.ToPILImage()(mask)
        mask_img = mask_img.filter(ImageFilter.MaxFilter(3))  # dilate
        mask_img = mask_img.filter(ImageFilter.MinFilter(3))  # erode
        mask     = T.ToTensor()(mask_img)

    return mask * delta, mask

# ------------------------ BACKGROUND GAN -----------------------------
NZ, NGF = 100, 64
def make_gen():
    layers=[nn.ConvTranspose2d(NZ, NGF*16,4,1,0,bias=False),
            nn.BatchNorm2d(NGF*16), nn.ReLU(True)]
    oc = NGF*16
    for _ in range(int(math.log2(RES_FULL)-2)):
        ic, oc = oc, max(NGF, oc//2)
        layers += [nn.ConvTranspose2d(ic, oc, 4, 2, 1, bias=False),
                   nn.BatchNorm2d(oc), nn.ReLU(True)]
    layers += [nn.ConvTranspose2d(oc,1,3,1,1,bias=False), nn.Tanh()]
    return nn.Sequential(*layers)

Gbg = make_gen().to(DEV).eval()
Gbg.load_state_dict(torch.load(GEN_WEIGHTS, map_location=DEV))

@torch.no_grad()
def sample_gan(n):
    z  = torch.randn(n, NZ, 1, 1, device=DEV)
    bg_full  = Gbg(z)
    bg_small = F.interpolate(bg_full, (RES_TRAIN, RES_TRAIN),
                             mode="bilinear", align_corners=False)
    return bg_full, bg_small

def random_good():
    """Return a good image tensor (real or GAN) at training resolution."""
    if random.random() < 0.5 and GOOD_REAL:
        return xform(Image.open(random.choice(GOOD_REAL)).convert('L'))
    _, bg_small = sample_gan(1)
    return bg_small[0]

# -------------------------- DATASET ----------------------------------
GOOD_REAL   = list_imgs(ROOT_GOOD)
DEFECT_REAL = list_imgs(ROOT_DEFECT)
if not GOOD_REAL or not DEFECT_REAL:
    raise RuntimeError("data/good or data/defect is empty.")

class OverlayDS(Dataset):
    def __len__(self): return max(len(GOOD_REAL), len(DEFECT_REAL))
    def __getitem__(self, _):
        if random.random() < 0.5:
            # ---------- positive sample ----------
            bg  = random_good()
            defect_img = xform(Image.open(random.choice(DEFECT_REAL)).convert('L'))
            overlay, mask = build_overlay(defect_img, bg)
            label = 1
        else:
            # ---------- negative sample ----------
            bg  = random_good()
            bg2 = random_good()
            overlay = torch.zeros_like(bg)
            mask    = torch.zeros_like(bg)
            label   = 0
        return bg, overlay, mask, label

# ---------------------- Tiny-U-Net (3-level) -------------------------
def down(i,o): return nn.Sequential(nn.Conv2d(i,o,4,2,1,bias=False),
                                    nn.BatchNorm2d(o), nn.LeakyReLU(0.2,True))
def up(i,o):   return nn.Sequential(nn.ConvTranspose2d(i,o,4,2,1,bias=False),
                                    nn.BatchNorm2d(o), nn.ReLU(True))
class TinyUNet(nn.Module):
    def __init__(self, ch=16):
        super().__init__()
        self.e1=down(1,ch); self.e2=down(ch,ch*2); self.e3=down(ch*2,ch*4)
        self.d3=up(ch*4,ch*2); self.d2=up(ch*4,ch); self.d1=nn.ConvTranspose2d(ch*2,2,4,2,1)
    def forward(self,x):
        e1=self.e1(x); e2=self.e2(e1); e3=self.e3(e2)
        d3=self.d3(e3); d2=self.d2(torch.cat([d3,e2],1))
        d1=self.d1(torch.cat([d2,e1],1))
        mask  = torch.sigmoid(d1[:,:1])
        delta = torch.tanh(   d1[:,1:])
        return mask*delta, mask

# -------------------------- Loss helpers -----------------------------
SOBEL_X = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=torch.float32).view(1,1,3,3)/8
SOBEL_Y = SOBEL_X.transpose(2,3)
SOBEL_X, SOBEL_Y = SOBEL_X.to(DEV), SOBEL_Y.to(DEV)

def grad_mag(img):
    gx = F.conv2d(img, SOBEL_X, padding=1)
    gy = F.conv2d(img, SOBEL_Y, padding=1)
    return torch.sqrt(gx**2 + gy**2 + 1e-6)

def tv_loss(m):
    return F.l1_loss(m[:,:,:,1:], m[:,:,:,:-1]) + F.l1_loss(m[:,:,1:,:], m[:,:,:-1,:])

# ---------------------------- TRAIN ----------------------------------
ds = OverlayDS()
loader = DataLoader(ds, BATCH, shuffle=True, num_workers=0)

net = TinyUNet().to(DEV)
opt = torch.optim.Adam(net.parameters(), LR)
L1 = nn.L1Loss()

for ep in range(1, EPOCHS+1):
    for bg, ov_tgt, mask_tgt, lbl in tqdm(loader, desc=f"Epoch {ep}/{EPOCHS}"):
        bg, ov_tgt, mask_tgt, lbl = bg.to(DEV), ov_tgt.to(DEV), mask_tgt.to(DEV), lbl.to(DEV)
        ov_pred, mask_pred = net(bg)

        pos_idx = lbl == 1
        neg_idx = lbl == 0

        loss = 0.0
        if pos_idx.any():
            p_ov_pred  = ov_pred[pos_idx]
            p_ov_tgt   = ov_tgt[pos_idx]
            p_mask_pred = mask_pred[pos_idx]
            loss += ( L1(p_ov_pred, p_ov_tgt) +
                      EDGE_W * L1(grad_mag(p_ov_pred), grad_mag(p_ov_tgt)) +
                      SPARSE_W * p_mask_pred.mean())
        if neg_idx.any():
            n_ov_pred  = ov_pred[neg_idx]
            n_mask_pred= mask_pred[neg_idx]
            loss += NEG_W * ( torch.abs(n_ov_pred).mean() + n_mask_pred.mean() )

        if TV_W: loss += TV_W * tv_loss(mask_pred)

        opt.zero_grad(); loss.backward(); opt.step()

    # ----------------------- preview ---------------------------------
    with torch.no_grad():
        bg_full, bg_small = sample_gan(PREVIEW_N)
        ov_pred_small,_   = net(bg_small)
        ov_pred_full = F.interpolate(ov_pred_small, (RES_FULL,RES_FULL),
                                     mode="bilinear", align_corners=False)
        synth_full = torch.clamp(bg_full + ov_pred_full, -1, 1)

        # get a real defect batch for display
        real_bg, real_ov, _ = [], [], []
        while len(real_bg) < PREVIEW_N:
            b, o, m, l = ds[random.randrange(len(ds))]
            if l == 1:
                real_bg.append(b); real_ov.append(o)
        real_bg = torch.stack(real_bg).to(DEV)
        real_ov = torch.stack(real_ov).to(DEV)
        real_bg_full = F.interpolate(real_bg, (RES_FULL,RES_FULL),
                                     mode="bilinear", align_corners=False)
        real_ov_full = F.interpolate(real_ov, (RES_FULL,RES_FULL),
                                     mode="bilinear", align_corners=False)
        real_defect = torch.clamp(real_bg_full + real_ov_full, -1, 1)

        white = torch.ones_like(real_bg_full)
        true_on_white = torch.clamp(white + real_ov_full, -1, 1)
        pred_on_white = torch.clamp(white + ov_pred_full, -1, 1)

        rows  = [real_defect, true_on_white, pred_on_white, synth_full]
        grids = [vutils.make_grid(r, nrow=PREVIEW_N, padding=2) for r in rows]
        preview = torch.cat(grids, dim=1)
        vutils.save_image(preview*0.5+0.5,
                          f"{OUT_DIR}/side_by_side_{ep:02}.png",
                          normalize=False)

    torch.save(net.state_dict(), f"{CKPT_DIR}/unet_{ep:02}.pt")
    print(f"Epoch {ep} done – preview & ckpt saved.")

print("✅ v4 training finished – check", OUT_DIR)
# =====================================================================
