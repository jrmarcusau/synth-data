import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

# ====================
# Dataset: Only good images
# ====================
class GoodDataset(Dataset):
    def __init__(self, good_dir, transform=None):
        self.paths = sorted([os.path.join(good_dir, f) for f in os.listdir(good_dir)])
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("L")
        if self.transform:
            img = self.transform(img)
        return img

# ====================
# Sobel Edge Loss
# ====================
class SobelEdge(nn.Module):
    def __init__(self):
        super().__init__()
        kx = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        ky = torch.tensor([[1,2,1],[0,0,0],[-1,-2,-1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.weight_x = nn.Parameter(kx, requires_grad=False)
        self.weight_y = nn.Parameter(ky, requires_grad=False)

    def forward(self, x):
        gx = F.conv2d(x, self.weight_x, padding=1)
        gy = F.conv2d(x, self.weight_y, padding=1)
        return torch.sqrt(gx**2 + gy**2 + 1e-6)

# ====================
# U-Net Generator
# ====================
class UNetGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.down1 = self.block(1, 64)
        self.down2 = self.block(64, 128)
        self.down3 = self.block(128, 256)
        self.down4 = self.block(256, 512)
        self.middle = nn.Sequential(
            self.block(512, 512),
            nn.Dropout(0.5)
        )
        self.up4 = self.upblock(1024, 256)
        self.up3 = self.upblock(512, 128)
        self.up2 = self.upblock(256, 64)
        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, 1, 4, 2, 1),
            nn.Tanh()
        )

    def block(self, in_c, out_c):
        return nn.Sequential(nn.Conv2d(in_c, out_c, 4, 2, 1), nn.BatchNorm2d(out_c), nn.ReLU())

    def upblock(self, in_c, out_c):
        return nn.Sequential(nn.ConvTranspose2d(in_c, out_c, 4, 2, 1), nn.BatchNorm2d(out_c), nn.ReLU())

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        m = self.middle(d4)

        def match(x, y): return F.interpolate(x, size=y.shape[2:], mode="bilinear", align_corners=False)

        u4 = self.up4(torch.cat([match(m, d4), d4], dim=1))
        u3 = self.up3(torch.cat([match(u4, d3), d3], dim=1))
        u2 = self.up2(torch.cat([match(u3, d2), d2], dim=1))
        out = self.final(torch.cat([match(u2, d1), d1], dim=1))
        return out

# ====================
# Patch Discriminator
# ====================
class PatchDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2),
            nn.Conv2d(256, 1, 4, 1, 1), nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

# ====================
# Utility
# ====================
def denorm(x): return x * 0.5 + 0.5

def save_visual(img, recon, epoch, out_dir):
    comp = torch.cat([denorm(img), denorm(recon)], dim=0)
    utils.save_image(comp, os.path.join(out_dir, f"recon_epoch_{epoch:02d}.png"), nrow=4)

# ====================
# Training Loop
# ====================
def train():
    good_dir = "./data/good"
    out_dir = "./outputs/fake_good_v1"
    ckpt_dir = "./outputs/checkpoints_good_v1"
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_size = 256
    batch_size = 8

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    dataset = GoodDataset(good_dir, transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    G = UNetGenerator().to(device)
    D = PatchDiscriminator().to(device)
    sobel = SobelEdge().to(device)

    opt_G = torch.optim.Adam(G.parameters(), lr=1e-4)
    opt_D = torch.optim.Adam(D.parameters(), lr=1e-4)

    l1_loss = nn.L1Loss()
    adv_loss = nn.BCELoss()

    loss_G_list, loss_D_list = [], []

    for epoch in range(1, 41):
        pbar = tqdm(loader, desc=f"[Epoch {epoch}/40]")
        for img in pbar:
            img = img.to(device)
            recon = G(img)

            pred_fake = D(recon)
            pred_real = D(img)

            valid = torch.ones_like(pred_real)
            fake = torch.zeros_like(pred_fake)

            # Generator Loss
            opt_G.zero_grad()
            loss_G_adv = adv_loss(pred_fake, valid)
            loss_G_l1 = 0.1 * l1_loss(recon, img)
            loss_G_edge = 0.1 * l1_loss(sobel(recon), sobel(img))
            loss_G = loss_G_adv + loss_G_l1 + loss_G_edge
            loss_G.backward()
            opt_G.step()

            # Discriminator Loss
            opt_D.zero_grad()
            loss_D_real = adv_loss(pred_real, valid)
            loss_D_fake = adv_loss(D(recon.detach()), fake)
            loss_D = 0.5 * (loss_D_real + loss_D_fake)
            loss_D.backward()
            opt_D.step()

            pbar.set_postfix(loss_G=loss_G.item(), loss_D=loss_D.item())

        loss_G_list.append(loss_G.item())
        loss_D_list.append(loss_D.item())
        with torch.no_grad():
            save_visual(img[:4], G(img[:4]), epoch, out_dir)

        if epoch % 10 == 0:
            torch.save(G.state_dict(), os.path.join(ckpt_dir, f"G_epoch_{epoch}.pt"))
            torch.save(D.state_dict(), os.path.join(ckpt_dir, f"D_epoch_{epoch}.pt"))

    plt.plot(loss_G_list, label="Generator")
    plt.plot(loss_D_list, label="Discriminator")
    plt.title("Loss Curve (Good GAN v1)")
    plt.legend()
    plt.savefig(os.path.join(out_dir, "loss_curve_good_v1.png"))
    plt.close()

if __name__ == "__main__":
    train()
