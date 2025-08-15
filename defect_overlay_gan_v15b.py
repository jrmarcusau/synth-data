# ================= defect_overlay_unet_v15b.py =================
"""
Resume v15 training from epoch 50 → continue through epoch 100.
Requires the v15 checkpoints:
    outputs/defect_overlay_cpu_v15/ckpt_v15/G_epoch_50.pt
    outputs/defect_overlay_cpu_v15/ckpt_v15/D_epoch_50.pt
Everything else (model, losses, DiffAug, λ_L1 schedule, etc.) unchanged.
"""
import torch, os, csv, matplotlib.pyplot as plt
import torchvision.transforms as T, torchvision.utils as vutils

from defect_overlay_gan_v15 import (  # re-use all definitions from v15
    OverlayDS, G, D, diff_aug, d_hinge, g_hinge,
    L1, make_overlay, RES_TRAIN, RES_FULL,
    R1_GAMMA, PadSquare, _xform, BATCH, LR, BETA1,
    OUT_DIR, CKPT_DIR, LOSS_PLOT, CSV_LOG, grid_defects)

DEV   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
START = 101                # resume after epoch 50
END   = 200               # finish at epoch 100 (inclusive)

# --------- data loader (unchanged) ----------
from torch.utils.data import DataLoader
ds      = OverlayDS()
loader  = DataLoader(ds, BATCH, shuffle=True, num_workers=0)

# --------- rebuild nets & load checkpoints --
netG, netD = G().to(DEV), D().to(DEV)
netG.load_state_dict(torch.load(f"{CKPT_DIR}/G_epoch_100.pt", map_location=DEV))
netD.load_state_dict(torch.load(f"{CKPT_DIR}/D_epoch_100.pt", map_location=DEV))

optG = torch.optim.Adam(netG.parameters(), LR, betas=(BETA1,0.999))
optD = torch.optim.Adam(netD.parameters(), LR, betas=(BETA1,0.999))

# --------- continue logs if they exist ------
G_log, D_log = [], []
if os.path.isfile(CSV_LOG):
    with open(CSV_LOG) as f:
        next(f)  # header
        for row in csv.reader(f): G_log.append(float(row[1])); D_log.append(float(row[2]))

# --------------- training loop -------------
for ep in range(START, END+1):
    # λ_L1 schedule (same rules)
    if ep < 5:      curr_L1 = 100
    elif ep < 20:   curr_L1 = 50
    elif ep < 100:  curr_L1 = 10
    elif ep < 180:  curr_L1 = 5
    else:           curr_L1 = 2

    g_tot = d_tot = 0.
    for bg, ov_gt in loader:
        bg, ov_gt = bg.to(DEV), ov_gt.to(DEV)
        B = bg.size(0)

        # 8×8 tiled noise
        z = torch.randn(B, 1, 8, 8, device=DEV)
        z = torch.nn.functional.interpolate(z, (RES_TRAIN, RES_TRAIN), mode='nearest')
        gin = torch.cat([bg, z], 1)

        # ----- Discriminator -----
        optD.zero_grad(set_to_none=True)
        with torch.no_grad():
            ov_fake = netG(gin)
        real_pair = torch.cat([bg, ov_gt], 1).detach().requires_grad_(True)
        fake_pair = torch.cat([bg, ov_fake.detach()], 1)

        d_real = netD(diff_aug(real_pair))
        d_fake = netD(diff_aug(fake_pair))
        loss_D = d_hinge(d_real, d_fake).mean()

        grad = torch.autograd.grad(d_real, real_pair,
                                   torch.ones_like(d_real),
                                   create_graph=True)[0]
        loss_D += R1_GAMMA * grad.pow(2).view(B, -1).sum(1).mean()
        loss_D.backward(); optD.step()

        # ----- Generator ----------
        optG.zero_grad(set_to_none=True)
        ov_fake = netG(gin)
        fake_pair = torch.cat([bg, ov_fake], 1)
        g_adv = g_hinge(netD(diff_aug(fake_pair)))
        g_l1  = L1(ov_fake, ov_gt) * curr_L1
        loss_G = g_adv + g_l1
        loss_G.backward(); optG.step()

        g_tot += loss_G.item(); d_tot += loss_D.item()

    G_log.append(g_tot / len(loader)); D_log.append(d_tot / len(loader))
    print(f"Ep {ep:03}  G={G_log[-1]:.3f}  D={D_log[-1]:.3f}")

    # ---------- preview (every epoch) ----------
    with torch.no_grad():
        z = torch.randn(B, 1, 8, 8, device=DEV)
        z = torch.nn.functional.interpolate(z, (RES_TRAIN, RES_TRAIN),
                                            mode='nearest')
        ov = netG(torch.cat([bg, z], 1))[:4]
        bg4, gt4 = bg[:4], ov_gt[:4]

        ovF = torch.nn.functional.interpolate(
                ov, (RES_FULL, RES_FULL), mode='bilinear', align_corners=False)
        bgF = torch.nn.functional.interpolate(
                bg4, (RES_FULL, RES_FULL), mode='bilinear', align_corners=False)
        synth = torch.clamp(bgF + ovF, -1, 1)

        white = torch.ones_like(bgF)
        gt   = torch.clamp(white + torch.nn.functional.interpolate(
                        gt4, (RES_FULL, RES_FULL), mode='bilinear',
                        align_corners=False), -1, 1)
        pred = torch.clamp(white + ovF, -1, 1)
        defect = torch.clamp(bgF + torch.nn.functional.interpolate(
                            gt4, (RES_FULL, RES_FULL), mode='bilinear',
                            align_corners=False), -1, 1)

        grid = torch.cat([vutils.make_grid(r, nrow=4, padding=2)
                        for r in [defect, gt, pred, synth]], dim=1)
        vutils.save_image(grid * 0.5 + 0.5,
                        f"{OUT_DIR}/side_by_side_epoch_{ep:03}.png",
                        normalize=False)

    if ep % 5 == 0:
        torch.save(netG.state_dict(), f"{CKPT_DIR}/G_epoch_{ep:03}.pt")
        torch.save(netD.state_dict(), f"{CKPT_DIR}/D_epoch_{ep:03}.pt")

# -------------- update logs & plot ----------
with open(CSV_LOG, 'w', newline='') as f:
    csv.writer(f).writerows([("epoch", "G", "D")] +
                            [(i+1, G_log[i], D_log[i]) for i in range(len(G_log))])

plt.plot(G_log, label='G'); plt.plot(D_log, label='D')
plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend()
plt.tight_layout(); plt.savefig(LOSS_PLOT); plt.close()
print("✅ v15b training complete – epochs 101-200 done.")
# ==============================================================
