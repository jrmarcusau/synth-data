import os, csv, math, torch, matplotlib.pyplot as plt
import torchvision.transforms as T, torchvision.utils as vutils
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm
from torch import autocast, amp
import torch.nn as nn

# ------------- CONFIG -------------
DATA_DIR   = "data/good"                 # keep original layout
OUT_DIR    = "outputs/fake_good_v4"
CKPT_DIR   = f"{OUT_DIR}/checkpoints"
LOSS_PLOT  = f"{OUT_DIR}/loss_plot.png"
CSV_LOG    = f"{OUT_DIR}/losses.csv"

RES        = 1024                        # single full-res stage
EPOCHS     = 20
NZ, NGF, NDF = 100, 32, 32
BATCH_CPU  = 2
LR, BETA1  = 2e-4, 0.5
TOP_N      = 6                           # 6 real + 6 fake in preview

TARGET_H   = 884                         # original wafer height
PAD_TOP    = (RES - TARGET_H) // 2       # 70 pixels

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP     = DEVICE.type == "cuda"
scaler  = amp.GradScaler(enabled=AMP)
criterion = nn.BCELoss()

os.makedirs(OUT_DIR,  exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

# ---------- pad transform ----------
class PadTo1024:
    def __call__(self, img):
        w, h = img.size
        t = (RES - h) // 2; b = RES - h - t
        l = (RES - w) // 2; r = RES - w - l
        return T.functional.pad(img, (l, t, r, b), padding_mode="reflect")

transform = T.Compose([
    T.Grayscale(),
    PadTo1024(),
    T.ToTensor(),
    T.Normalize((0.5,), (0.5,))
])

dloader = DataLoader(
    datasets.ImageFolder("data", transform),
    batch_size=BATCH_CPU if DEVICE.type == "cpu" else BATCH_CPU * 4,
    shuffle=True,
    pin_memory=True
)

# ---------- DCGAN nets ----------
def make_generator():
    ups = int(math.log2(RES) - 2)               # 6 doublings 4→8→…→1024
    layers = [nn.ConvTranspose2d(NZ, NGF * 16, 4, 1, 0, bias=False),
              nn.BatchNorm2d(NGF * 16), nn.ReLU(True)]
    out_c = NGF * 16
    for _ in range(ups):
        in_c, out_c = out_c, max(NGF, out_c // 2)
        layers += [nn.ConvTranspose2d(in_c, out_c, 4, 2, 1, bias=False),
                   nn.BatchNorm2d(out_c), nn.ReLU(True)]
    layers += [nn.ConvTranspose2d(out_c, 1, 3, 1, 1, bias=False), nn.Tanh()]
    return nn.Sequential(*layers)

def make_discriminator():
    downs = int(math.log2(RES) - 2)
    layers, in_c = [], 1
    for i in range(downs):
        out_c = min(NDF * 2 ** i, NDF * 8)
        layers += [nn.Conv2d(in_c, out_c, 4, 2, 1, bias=False)]
        if i: layers += [nn.BatchNorm2d(out_c)]
        layers += [nn.LeakyReLU(0.2, True)]
        in_c = out_c
    layers += [nn.Conv2d(in_c, 1, 4, 1, 0, bias=False), nn.Sigmoid()]
    return nn.Sequential(*layers)

netG, netD = make_generator().to(DEVICE), make_discriminator().to(DEVICE)
optG = torch.optim.Adam(netG.parameters(), LR, (BETA1, 0.999))
optD = torch.optim.Adam(netD.parameters(), LR, (BETA1, 0.999))

G_log, D_log = [], []

# ------------- TRAIN -------------
for epoch in range(1, EPOCHS + 1):
    g_sum = d_sum = 0.0
    for real, _ in tqdm(dloader, desc=f"Epoch {epoch}/{EPOCHS}"):
        real = real.to(DEVICE, non_blocking=True)
        bsz  = real.size(0)

        # ----- D -----
        optD.zero_grad()
        with autocast("cuda", enabled=AMP):
            out_real = netD(real)
            valid    = torch.ones_like(out_real)
            d_real   = criterion(out_real, valid)

            z    = torch.randn(bsz, NZ, 1, 1, device=DEVICE)
            fake = netG(z)
            out_fake = netD(fake.detach())
            d_fake   = criterion(out_fake, torch.zeros_like(out_fake))
            d_loss   = 0.5 * (d_real + d_fake)
        scaler.scale(d_loss).backward(); scaler.step(optD); scaler.update()

        # ----- G -----
        optG.zero_grad()
        with autocast("cuda", enabled=AMP):
            g_loss = criterion(netD(fake), torch.ones_like(out_fake))
        scaler.scale(g_loss).backward(); scaler.step(optG); scaler.update()

        g_sum += g_loss.item(); d_sum += d_loss.item()

    G_log.append(g_sum / len(dloader)); D_log.append(d_sum / len(dloader))

        # -------- preview (6 real | 6 fake) --------
    with torch.no_grad():
        z = torch.randn(TOP_N, NZ, 1, 1, device=DEVICE)
        fake_batch = netG(z).cpu()

        fake_vis = fake_batch[:, :, PAD_TOP:PAD_TOP+TARGET_H, :]
        real_vis = real.cpu()[:, :, PAD_TOP:PAD_TOP+TARGET_H, :]

        fake_vis = fake_vis * 0.5 + 0.5
        real_vis = real_vis * 0.5 + 0.5

        # --- NEW: ensure both have the same leading dimension ---
        b = min(real_vis.size(0), fake_vis.size(0))
        fake_vis, real_vis = fake_vis[:b], real_vis[:b]
        # --------------------------------------------------------

        comp = torch.cat((real_vis, fake_vis), dim=2)  # vertical stack
        vutils.save_image(comp,
                          f"{OUT_DIR}/side_by_side_epoch_{epoch:03}.png",
                          nrow=b,
                          normalize=False)
        vutils.save_image(fake_vis,
                          f"{OUT_DIR}/epoch_{epoch:03}.png",
                          nrow=b,
                          normalize=False)

    # ----- checkpoint every 2 epochs -----
    if epoch % 2 == 0:
        torch.save(netG.state_dict(), f"{CKPT_DIR}/generator_epoch_{epoch:03}.pt")

# ------------- FINISH -------------
torch.save(netG.state_dict(), f"{OUT_DIR}/generator_good_v4.pt")
with open(CSV_LOG, "w", newline="") as f:
    csv.writer(f).writerows([("epoch", "G", "D")] +
                            [(i + 1, G_log[i], D_log[i]) for i in range(len(G_log))])

plt.plot(G_log, label="G"); plt.plot(D_log, label="D")
plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend()
plt.tight_layout(); plt.savefig(LOSS_PLOT); plt.close()
print("✅ Training complete – outputs in", OUT_DIR)
