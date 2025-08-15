import os
import random
import csv
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision import utils
from tqdm import tqdm

# ----------------------
# 📁 Dataset (clean + random defect)
# ----------------------
class DefectDataset(Dataset):
    def __init__(self, good_dir, defect_dir, transform=None):
        self.clean_paths = sorted([os.path.join(good_dir, f) for f in os.listdir(good_dir)])
        self.defect_paths = sorted([os.path.join(defect_dir, f) for f in os.listdir(defect_dir)])
        self.transform = transform

    def __len__(self):
        return len(self.clean_paths)

    def __getitem__(self, idx):
        clean = Image.open(self.clean_paths[idx]).convert('L')
        defect = Image.open(random.choice(self.defect_paths)).convert('L')

        if self.transform:
            clean = self.transform(clean)
            defect = self.transform(defect)

        return clean, defect

# ----------------------
# 🧠 UNet Generator
# ----------------------
class UNetGenerator(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()

        def down(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 4, 2, 1),
                nn.BatchNorm2d(out_c),
                nn.LeakyReLU(0.2)
            )

        def up(in_c, out_c):
            return nn.Sequential(
                nn.ConvTranspose2d(in_c, out_c, 4, 2, 1),
                nn.BatchNorm2d(out_c),
                nn.ReLU()
            )

        self.down1 = down(1, 64)
        self.down2 = down(64, 128)
        self.down3 = down(128, 256)

        self.up2 = up(256, 128)
        self.up1 = up(128 + 128, 64)
        self.final = nn.Sequential(
            nn.ConvTranspose2d(64 + 64, out_channels, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        d1 = self.down1(x)    # 128x128
        d2 = self.down2(d1)   # 64x64
        d3 = self.down3(d2)   # 32x32

        u2 = self.up2(d3)     # 64x64
        u1 = self.up1(torch.cat([u2, d2], dim=1))  # 128x128
        out = self.final(torch.cat([u1, d1], dim=1))  # 256x256
        return out

# ----------------------
# 🧪 PatchGAN Discriminator
# ----------------------
class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels=2):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2),
            nn.Conv2d(128, 1, 4, 1, 1), nn.Sigmoid()
        )

    def forward(self, x, y):
        inp = torch.cat([x, y], dim=1)  # (B, 2, H, W)
        return self.model(inp)

# ----------------------
# 🎯 Training
# ----------------------
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("outputs/fake_defects_v2", exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # grayscale normalization
    ])

    dataset = DefectDataset('./data/good', './data/defect', transform)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    G = UNetGenerator().to(device)
    D = PatchDiscriminator().to(device)

    adversarial_loss = nn.BCELoss()
    pixel_loss = nn.L1Loss()

    opt_G = torch.optim.Adam(G.parameters(), lr=1e-4, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=1e-4, betas=(0.5, 0.999))

    loss_G_list = []
    loss_D_list = []

    for epoch in range(1, 51):
        pbar = tqdm(loader, desc=f"Epoch {epoch}/50", unit="batch")

        for clean, defect in pbar:
            clean, defect = clean.to(device), defect.to(device)

            # Prepare labels that match discriminator output size
            # Use dummy forward pass to get shape once
            with torch.no_grad():
                temp_out = D(clean, G(clean).detach())
            valid = torch.ones_like(temp_out, device=device)
            fake = torch.zeros_like(temp_out, device=device)

            # Generator
            opt_G.zero_grad()
            fake_defect = G(clean)
            pred_fake = D(clean, fake_defect)
            loss_G_adv = adversarial_loss(pred_fake, valid)
            loss_G_l1 = pixel_loss(fake_defect, clean)
            loss_G = loss_G_adv + 50 * loss_G_l1
            loss_G.backward()
            opt_G.step()

            # Discriminator
            opt_D.zero_grad()
            pred_real = D(clean, defect)
            pred_fake = D(clean, fake_defect.detach())
            loss_D_real = adversarial_loss(pred_real, valid)
            loss_D_fake = adversarial_loss(pred_fake, fake)
            loss_D = (loss_D_real + loss_D_fake) * 0.5
            loss_D.backward()
            opt_D.step()

            pbar.set_postfix(loss_D=loss_D.item(), loss_G=loss_G.item())

        # Save loss
        loss_G_list.append(loss_G.item())
        loss_D_list.append(loss_D.item())

        # Save visual results
        G.eval()
        with torch.no_grad():
            val_input = clean[:4]
            val_output = G(val_input)
            comparison = torch.cat([val_input, val_output], dim=0) * 0.5 + 0.5
            utils.save_image(comparison, f"outputs/fake_defects_v2/epoch_{epoch:02d}.png", nrow=4)
        G.train()

    # Save model
    torch.save(G.state_dict(), "outputs/generator_final_v2.pth")
    print("✅ Training done. Model saved as generator_final_v2.pth")

    # Save losses to CSV
    with open("outputs/training_log.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'loss_G', 'loss_D'])
        for e in range(len(loss_G_list)):
            writer.writerow([e + 1, loss_G_list[e], loss_D_list[e]])

    # Plot losses
    plt.figure()
    plt.plot(loss_G_list, label='Generator Loss')
    plt.plot(loss_D_list, label='Discriminator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.savefig("outputs/loss_plot.png")
    print("📊 Loss plot saved to outputs/loss_plot.png")

if __name__ == "__main__":
    train()
