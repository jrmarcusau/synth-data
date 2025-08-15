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
# Dataset
# ====================
class PairedDefectDataset(Dataset):
    def __init__(self, good_dir, defect_dir, transform=None):
        self.good_paths = sorted([os.path.join(good_dir, f) for f in os.listdir(good_dir)])
        self.defect_paths = sorted([os.path.join(defect_dir, f) for f in os.listdir(defect_dir)])
        self.transform = transform

    def __len__(self):
        return min(len(self.good_paths), len(self.defect_paths))

    def __getitem__(self, idx):
        good = Image.open(self.good_paths[idx]).convert("L")
        defect = Image.open(self.defect_paths[idx]).convert("L")
        if self.transform:
            good = self.transform(good)
            defect = self.transform(defect)
        return good, defect

# ====================
# Sobel Edge Module
# ====================
class SobelEdge(nn.Module):
    def __init__(self):
        super().__init__()
        kernel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        kernel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.weight_x = nn.Parameter(kernel_x, requires_grad=False)
        self.weight_y = nn.Parameter(kernel_y, requires_grad=False)

    def forward(self, x):
        gx = F.conv2d(x, self.weight_x, padding=1)
        gy = F.conv2d(x, self.weight_y, padding=1)
        return torch.sqrt(gx ** 2 + gy ** 2 + 1e-6)

# ====================
# Simple Feature Extractor (Perceptual)
# ====================
class SimplePerceptual(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 8, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(8, 16, 3, 1, 1), nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)

# ====================
# U-Net Generator with Dropout
# ====================
class UNetGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.down1 = self.conv_block(1, 64)
        self.down2 = self.conv_block(64, 128)
        self.down3 = self.conv_block(128, 256)
        self.down4 = self.conv_block(256, 512)
        self.middle = nn.Sequential(
            self.conv_block(512, 512),
            nn.Dropout(0.5)
        )
        self.up4 = self.up_block(1024, 256)
        self.up3 = self.up_block(512, 128)
        self.up2 = self.up_block(256, 64)
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(128, 1, 4, 2, 1),
            nn.Tanh()
        )

    def conv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 4, 2, 1),
            nn.BatchNorm2d(out_c),
            nn.ReLU()
        )

    def up_block(self, in_c, out_c):
        return nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c, 4, 2, 1),
            nn.BatchNorm2d(out_c),
            nn.ReLU()
        )

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        m = self.middle(d4)

        def match_size(a, b):
            return F.interpolate(a, size=b.shape[2:], mode="bilinear", align_corners=False)

        u4 = self.up4(torch.cat([match_size(m, d4), d4], dim=1))
        u3 = self.up3(torch.cat([match_size(u4, d3), d3], dim=1))
        u2 = self.up2(torch.cat([match_size(u3, d2), d2], dim=1))
        out = self.up1(torch.cat([match_size(u2, d1), d1], dim=1))
        return out

# ====================
# PatchGAN Discriminator
# ====================
class PatchDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(2, 64, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2),
            nn.Conv2d(256, 1, 4, 1, 1), nn.Sigmoid()
        )

    def forward(self, x, y):
        return self.model(torch.cat([x, y], dim=1))

# ====================
# Utilities
# ====================
def denorm(x):
    return x * 0.5 + 0.5

def save_visual(good, generated, epoch, out_dir, tag="step"):
    comp = torch.cat([denorm(good), denorm(generated)], dim=0)
    utils.save_image(comp, os.path.join(out_dir, f"{tag}_epoch_{epoch:02d}.png"), nrow=4)

# ====================
# Training
# ====================
def train():
    good_dir = "./data/good"
    defect_dir = "./data/defect"
    out_dir = "./outputs/fake_defects_v10"
    ckpt_dir = "./outputs/checkpoints_v10"
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    img_size = 256
    batch_size = 8
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    dataset = PairedDefectDataset(good_dir, defect_dir, transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    G = UNetGenerator().to(device)
    D = PatchDiscriminator().to(device)
    sobel = SobelEdge().to(device)
    perceptual = SimplePerceptual().to(device)

    pixel_loss = nn.L1Loss()
    adv_loss = nn.BCELoss()
    feature_loss = nn.L1Loss()

    opt_G = torch.optim.Adam(G.parameters(), lr=1e-4)
    opt_D = torch.optim.Adam(D.parameters(), lr=1e-4)

    loss_G_list, loss_D_list = [], []

    for epoch in range(1, 51):
        pbar = tqdm(loader, desc=f"[Epoch {epoch}/50]")
        for good, defect in pbar:
            good, defect = good.to(device), defect.to(device)

            fake_defect = G(good)
            pred_fake = D(good, fake_defect)
            pred_real = D(good, defect)

            valid = torch.ones_like(pred_fake)
            fake = torch.zeros_like(pred_fake)

            # Generator
            opt_G.zero_grad()
            loss_adv = adv_loss(pred_fake, valid)
            loss_pixel = 0.1 * pixel_loss(fake_defect, defect)
            loss_edge = 0.1 * pixel_loss(sobel(fake_defect), sobel(defect))
            loss_feat = 0.1 * feature_loss(perceptual(fake_defect), perceptual(defect))
            loss_G = loss_adv + loss_pixel + loss_edge + loss_feat
            loss_G.backward()
            opt_G.step()

            # Discriminator
            opt_D.zero_grad()
            loss_real = adv_loss(pred_real, valid)
            loss_fake = adv_loss(D(good, fake_defect.detach()), fake)
            loss_D = 0.5 * (loss_real + loss_fake)
            loss_D.backward()
            opt_D.step()

            pbar.set_postfix(loss_G=loss_G.item(), loss_D=loss_D.item())

        loss_G_list.append(loss_G.item())
        loss_D_list.append(loss_D.item())

        with torch.no_grad():
            save_visual(good[:4], G(good[:4]), epoch, out_dir, tag="gan")

        if epoch % 10 == 0:
            torch.save(G.state_dict(), os.path.join(ckpt_dir, f"gen_epoch_{epoch}.pt"))
            torch.save(D.state_dict(), os.path.join(ckpt_dir, f"disc_epoch_{epoch}.pt"))

    # Plot loss
    plt.plot(loss_G_list, label="Generator")
    plt.plot(loss_D_list, label="Discriminator")
    plt.title("Loss Curve (v10)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(out_dir, "loss_curve_v10.png"))
    plt.close()

if __name__ == "__main__":
    train()
