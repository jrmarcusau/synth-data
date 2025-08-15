import os
import csv
import random
import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import matplotlib.pyplot as plt
from tqdm import tqdm


# === Masked Defect Dataset ===
class MaskedDefectDataset(Dataset):
    def __init__(self, good_dir, defect_dir, transform=None):
        self.good_paths = sorted([os.path.join(good_dir, f) for f in os.listdir(good_dir)])
        self.defect_paths = sorted([os.path.join(defect_dir, f) for f in os.listdir(defect_dir)])
        self.transform = transform

    def __len__(self):
        return min(len(self.good_paths), len(self.defect_paths))

    def __getitem__(self, idx):
        clean = Image.open(self.good_paths[idx]).convert("RGB")
        defect = Image.open(self.defect_paths[idx]).convert("RGB")
        if self.transform:
            clean = self.transform(clean)
            defect = self.transform(defect)
        mask = generate_random_mask(clean.size(1), clean.size(2))
        return clean, defect, mask


# === Random Soft Mask Generator ===
def generate_random_mask(height, width):
    mask = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(mask)
    for _ in range(random.randint(1, 3)):
        x, y = random.randint(0, width), random.randint(0, height)
        r = random.randint(10, 40)
        draw.ellipse((x - r, y - r, x + r, y + r), fill=255)
    mask = transforms.ToTensor()(mask)  # shape: [1, H, W]
    return mask


# === Generator (UNet-style) ===
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        def down(in_c, out_c): return nn.Sequential(
            nn.Conv2d(in_c, out_c, 4, 2, 1), nn.BatchNorm2d(out_c), nn.ReLU())
        def up(in_c, out_c): return nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c, 4, 2, 1), nn.BatchNorm2d(out_c), nn.ReLU())

        self.encoder = nn.Sequential(
            down(4, 64), down(64, 128), down(128, 256)
        )
        self.middle = nn.Sequential(nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(), nn.Dropout2d(0.2))
        self.decoder = nn.Sequential(
            up(256, 128), up(128, 64),
            nn.ConvTranspose2d(64, 3, 4, 2, 1), nn.Tanh()
        )

    def forward(self, x, mask):
        x_masked = torch.cat([x, mask], dim=1)  # (B, 4, H, W)
        x = self.encoder(x_masked)
        x = self.middle(x)
        return self.decoder(x)


# === Discriminator (PatchGAN) ===
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        def block(in_c, out_c): return nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(in_c, out_c, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.model = nn.Sequential(
            block(6, 64), block(64, 128), block(128, 256),
            nn.Conv2d(256, 1, 4, padding=1)
        )

    def forward(self, clean, defect_or_fake):
        x = torch.cat([clean, defect_or_fake], dim=1)  # (B, 6, H, W)
        return self.model(x)


# === Training ===
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("outputs/fake_defects_v4", exist_ok=True)
    os.makedirs("outputs/checkpoints_v4", exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    dataset = MaskedDefectDataset("data/good", "data/defect", transform)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    G = Generator().to(device)
    D = Discriminator().to(device)

    opt_G = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))

    adversarial_loss = nn.BCEWithLogitsLoss()
    pixel_loss = nn.L1Loss()

    loss_G_list, loss_D_list = [], []

    for epoch in range(1, 51):
        pbar = tqdm(loader, desc=f"Epoch {epoch}/50", unit="batch")
        for i, (clean, defect, mask) in enumerate(pbar):
            clean, defect, mask = clean.to(device), defect.to(device), mask.to(device)

            # === Train Generator ===
            opt_G.zero_grad()
            fake_defect = G(clean, mask)
            pred_fake = D(clean, fake_defect)
            valid = torch.ones_like(pred_fake, device=device)

            loss_G_adv = adversarial_loss(pred_fake, valid)
            loss_G_l1 = pixel_loss(fake_defect * mask, defect * mask)
            loss_G = loss_G_adv + 2 * loss_G_l1
            loss_G.backward()
            opt_G.step()

            # === Train Discriminator (every 2 steps) ===
            if i % 2 == 0:
                opt_D.zero_grad()
                pred_real = D(clean, defect)
                pred_fake = D(clean, fake_defect.detach())
                loss_D_real = adversarial_loss(pred_real, valid)
                loss_D_fake = adversarial_loss(pred_fake, torch.zeros_like(pred_fake, device=device))
                loss_D = 0.5 * (loss_D_real + loss_D_fake)
                loss_D.backward()
                opt_D.step()
            else:
                loss_D = torch.tensor(0.0)

            pbar.set_postfix(loss_G=loss_G.item(), loss_D=loss_D.item())

        loss_G_list.append(loss_G.item())
        loss_D_list.append(loss_D.item())

        # === Save output images ===
        G.eval()
        with torch.no_grad():
            test_input = clean[:4]
            test_mask = mask[:4]
            test_output = G(test_input, test_mask)
            comparison = torch.cat([test_input, test_output], dim=0) * 0.5 + 0.5
            utils.save_image(comparison, f"outputs/fake_defects_v4/epoch_{epoch:02d}.png", nrow=4)
        G.train()

        # === Checkpointing ===
        if epoch % 10 == 0:
            torch.save(G.state_dict(), f"outputs/checkpoints_v4/generator_epoch{epoch:02d}.pt")

    # === Final save ===
    torch.save(G.state_dict(), "outputs/checkpoints_v4/generator_final_v4.pt")

    # === Log losses ===
    with open("outputs/checkpoints_v4/loss_log_v4.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'loss_G', 'loss_D'])
        for e in range(len(loss_G_list)):
            writer.writerow([e + 1, loss_G_list[e], loss_D_list[e]])

    plt.figure()
    plt.plot(loss_G_list, label='Generator Loss')
    plt.plot(loss_D_list, label='Discriminator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve v4')
    plt.legend()
    plt.savefig("outputs/checkpoints_v4/loss_plot_v4.png")
    print("✅ Training complete. All outputs and checkpoints saved.")


if __name__ == "__main__":
    train()
