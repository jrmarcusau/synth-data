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
from torch.nn.utils import spectral_norm

# ----------------------
# 📁 Dataset with stronger defect augmentation
# ----------------------
class DefectDataset(Dataset):
    def __init__(self, good_dir, defect_dir, transform=None):
        self.clean_paths = sorted([os.path.join(good_dir, f) for f in os.listdir(good_dir)])
        self.defect_paths = sorted([os.path.join(defect_dir, f) for f in os.listdir(defect_dir)])
        self.transform = transform

        self.defect_transform = transforms.Compose([
            transforms.RandomRotation(5),
            transforms.RandomResizedCrop(256, scale=(0.9, 1.0)),
            transforms.ColorJitter(contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    def __len__(self):
        return len(self.clean_paths)

    def __getitem__(self, idx):
        clean = Image.open(self.clean_paths[idx]).convert('L')
        defect = Image.open(random.choice(self.defect_paths)).convert('L')
        if self.transform:
            clean = self.transform(clean)
        defect = self.defect_transform(defect)
        return clean, defect

# ----------------------
# 🧠 UNet Generator with dropout
# ----------------------
class UNetGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = nn.Dropout2d(0.2)

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
        self.down4 = down(256, 512)

        self.up3 = up(512, 256)
        self.up2 = up(512, 128)
        self.up1 = up(256, 64)

        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, 1, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)

        u3 = self.up3(self.dropout(d4))
        u2 = self.up2(torch.cat([u3, d3], dim=1))
        u1 = self.up1(torch.cat([u2, d2], dim=1))
        out = self.final(torch.cat([u1, d1], dim=1))
        return out

# ----------------------
# 🔍 Discriminator with SpectralNorm
# ----------------------
class PatchDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        def block(in_c, out_c, stride=2):
            return nn.Sequential(
                spectral_norm(nn.Conv2d(in_c, out_c, 4, stride, 1)),
                nn.LeakyReLU(0.2)
            )

        self.model = nn.Sequential(
            block(2, 64),
            block(64, 128),
            block(128, 256),
            block(256, 512),
            nn.Conv2d(512, 1, 4, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x, y):
        return self.model(torch.cat([x, y], dim=1))

# ----------------------
# 🧪 Feature Matching Loss
# ----------------------
class FeatureMatchingLoss(nn.Module):
    def __init__(self, discriminator):
        super().__init__()
        self.D = discriminator
        self.criterion = nn.L1Loss()

    def forward(self, real_pair, fake_pair):
        loss = 0
        x_real = real_pair[0].detach()
        x_fake = fake_pair[0]
        y_real = real_pair[1].detach()
        y_fake = fake_pair[1]

        f_real = []
        f_fake = []
        x = torch.cat([x_real, y_real], dim=1)
        y = torch.cat([x_fake, y_fake], dim=1)
        for layer in self.D.model[:-2]:
            x = layer(x)
            y = layer(y)
            f_real.append(x)
            f_fake.append(y)

        for fr, ff in zip(f_real, f_fake):
            loss += self.criterion(ff, fr)
        return loss

# ----------------------
# 🎯 Training
# ----------------------
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("outputs/fake_defects_v3", exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    dataset = DefectDataset('./data/good', './data/defect', transform)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    G = UNetGenerator().to(device)
    D = PatchDiscriminator().to(device)
    FM = FeatureMatchingLoss(D).to(device)

    bce = nn.BCELoss()
    l1 = nn.L1Loss()

    opt_G = torch.optim.Adam(G.parameters(), lr=1e-4, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=1e-4, betas=(0.5, 0.999))

    loss_G_list = []
    loss_D_list = []

    for epoch in range(1, 51):
        pbar = tqdm(loader, desc=f"Epoch {epoch}/50", unit="batch")
        for i, (clean, defect) in enumerate(pbar):
            clean, defect = clean.to(device), defect.to(device)
            fake_defect = G(clean)
            valid = torch.ones_like(D(clean, defect))
            fake = torch.zeros_like(D(clean, fake_defect))

            # --- Discriminator (train every 2 steps) ---
            if i % 2 == 0:
                opt_D.zero_grad()
                pred_real = D(clean, defect)
                pred_fake = D(clean, fake_defect.detach())
                loss_D_real = bce(pred_real, valid)
                loss_D_fake = bce(pred_fake, fake)
                loss_D = 0.5 * (loss_D_real + loss_D_fake)
                loss_D.backward()
                opt_D.step()

            # --- Generator ---
            opt_G.zero_grad()
            pred_fake = D(clean, fake_defect)
            loss_G_adv = bce(pred_fake, valid)
            loss_G_l1 = l1(fake_defect, clean)
            loss_G_fm = FM((clean, defect), (clean, fake_defect))
            loss_G = loss_G_adv + 2 * loss_G_l1 + 1 * loss_G_fm
            loss_G.backward()
            opt_G.step()

            pbar.set_postfix(loss_D=loss_D.item(), loss_G=loss_G.item())

        loss_G_list.append(loss_G.item())
        loss_D_list.append(loss_D.item())

        G.eval()
        with torch.no_grad():
            val_input = clean[:4]
            val_output = G(val_input)
            comparison = torch.cat([val_input, val_output], dim=0) * 0.5 + 0.5
            utils.save_image(comparison, f"outputs/fake_defects_v3/epoch_{epoch:02d}.png", nrow=4)
        G.train()

        # Save checkpoints every 10 epochs
        if epoch % 10 == 0:
            torch.save(G.state_dict(), f"outputs/generator_checkpoint_v3_epoch{epoch:02d}.pth")

    # Final save
    torch.save(G.state_dict(), "outputs/generator_final_v3.pth")

    # Log loss
    with open("outputs/training_log_v3.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'loss_G', 'loss_D'])
        for e in range(len(loss_G_list)):
            writer.writerow([e + 1, loss_G_list[e], loss_D_list[e]])

    # Plot loss
    plt.figure()
    plt.plot(loss_G_list, label='Generator Loss')
    plt.plot(loss_D_list, label='Discriminator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve v3')
    plt.legend()
    plt.savefig("outputs/loss_plot_v3.png")

    print("✅ Training complete. Final and checkpoint models saved.")

if __name__ == "__main__":
    train()