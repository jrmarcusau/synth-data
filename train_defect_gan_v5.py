import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, utils, models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

# ---------------------------
# Configs
# ---------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 256
CROP_SIZE = 128
BATCH_SIZE = 8
EPOCHS = 50
SAVE_EVERY = 10
GOOD_DIR = './data/good'
DEFECT_DIR = './data/defect'
OUT_DIR = 'outputs/fake_defects_v5'
CKPT_DIR = 'outputs/checkpoints_v5'

# ---------------------------
# Utilities
# ---------------------------
def load_image(path):
    return Image.open(path).convert('RGB')

def random_mask(size):
    mask = torch.zeros(size, size)
    cx, cy = random.randint(32, size - 32), random.randint(32, size - 32)
    rw, rh = random.randint(10, 30), random.randint(10, 30)
    mask[max(0, cy - rh):min(size, cy + rh), max(0, cx - rw):min(size, cx + rw)] = 1.0
    return mask.unsqueeze(0)

# ---------------------------
# Dataset
# ---------------------------
class DefectPatchDataset(Dataset):
    def __init__(self, good_dir, defect_dir, transform=None):
        self.good_paths = sorted([os.path.join(good_dir, f) for f in os.listdir(good_dir)])
        self.defect_paths = sorted([os.path.join(defect_dir, f) for f in os.listdir(defect_dir)])
        self.transform = transform

    def __len__(self):
        return len(self.good_paths)

    def __getitem__(self, idx):
        clean_img = load_image(self.good_paths[idx])
        clean = transforms.functional.resize(clean_img, (IMG_SIZE, IMG_SIZE))
        if self.defect_paths:
            defect = load_image(random.choice(self.defect_paths))
            defect = transforms.functional.resize(defect, (IMG_SIZE, IMG_SIZE))
        else:
            defect = clean

        if self.transform:
            clean = self.transform(clean)
            defect = self.transform(defect)

        # Random crop
        i = random.randint(0, IMG_SIZE - CROP_SIZE)
        j = random.randint(0, IMG_SIZE - CROP_SIZE)
        clean_crop = clean[:, i:i+CROP_SIZE, j:j+CROP_SIZE]
        defect_crop = defect[:, i:i+CROP_SIZE, j:j+CROP_SIZE]

        mask = random_mask(CROP_SIZE)
        return clean_crop, defect_crop, mask

# ---------------------------
# Generator
# ---------------------------
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.down1 = nn.Sequential(nn.Conv2d(4, 64, 4, 2, 1), nn.ReLU())
        self.down2 = nn.Sequential(nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU())
        self.up1 = nn.Sequential(nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU())
        self.up2 = nn.Sequential(nn.ConvTranspose2d(64, 3, 4, 2, 1), nn.Tanh())

    def forward(self, x, mask):
        x = torch.cat([x, mask], dim=1)
        d1 = self.down1(x)
        d2 = self.down2(d1)
        u1 = self.up1(d2)
        u2 = self.up2(u1)
        return u2

# ---------------------------
# Discriminator
# ---------------------------
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        def block(in_c, out_c): return nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(in_c, out_c, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.model = nn.Sequential(
            block(6, 64),
            block(64, 128),
            block(128, 256),
            nn.Conv2d(256, 1, 4, 1, 1)
        )

    def forward(self, clean, defect):
        x = torch.cat([clean, defect], dim=1)
        return self.model(x)

# ---------------------------
# VGG Feature Extractor
# ---------------------------
class VGGFeatures(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(pretrained=True).features[:9]
        self.vgg = vgg
        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.vgg(x)

# ---------------------------
# Training Loop
# ---------------------------
def train():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    dataset = DefectPatchDataset(GOOD_DIR, DEFECT_DIR, transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    G = Generator().to(DEVICE)
    D = Discriminator().to(DEVICE)
    VGG = VGGFeatures().to(DEVICE)

    opt_G = torch.optim.Adam(G.parameters(), lr=1e-4, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=1e-4, betas=(0.5, 0.999))

    adv_loss = nn.BCEWithLogitsLoss()
    l1_loss = nn.L1Loss()

    loss_G_list, loss_D_list = [], []

    for epoch in range(1, EPOCHS+1):
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{EPOCHS}")
        for i, (clean, defect, mask) in enumerate(pbar):
            clean, defect, mask = clean.to(DEVICE), defect.to(DEVICE), mask.to(DEVICE)
            fake = G(clean, mask)

            # Train Discriminator
            if i % 2 == 0:
                D.zero_grad()
                real_pred = D(clean, defect)
                fake_pred = D(clean, fake.detach())
                valid = torch.ones_like(real_pred)
                fake_label = torch.zeros_like(fake_pred)
                loss_D = (adv_loss(real_pred, valid) + adv_loss(fake_pred, fake_label)) * 0.5
                loss_D.backward()
                opt_D.step()
            else:
                loss_D = torch.tensor(0.0)

            # Train Generator
            G.zero_grad()
            fake_pred = D(clean, fake)
            loss_G_adv = adv_loss(fake_pred, valid)
            loss_G_l1 = l1_loss(fake, defect)
            loss_G_feat = l1_loss(VGG(fake), VGG(defect))
            loss_G = loss_G_adv + 50 * loss_G_l1 + 10 * loss_G_feat
            loss_G.backward()
            opt_G.step()

            pbar.set_postfix(loss_D=loss_D.item(), loss_G=loss_G.item())

        loss_D_list.append(loss_D.item())
        loss_G_list.append(loss_G.item())

        # Save samples
        G.eval()
        with torch.no_grad():
            sample = clean[:4]
            mask_sample = mask[:4]
            fake_sample = G(sample, mask_sample)
            grid = torch.cat([sample, fake_sample], dim=0) * 0.5 + 0.5
            utils.save_image(grid, f"{OUT_DIR}/epoch_{epoch:02d}.png", nrow=4)
        G.train()

        # Save checkpoint
        if epoch % SAVE_EVERY == 0:
            torch.save(G.state_dict(), f"{CKPT_DIR}/G_epoch{epoch:02d}.pt")
            torch.save(D.state_dict(), f"{CKPT_DIR}/D_epoch{epoch:02d}.pt")

    # Plot losses
    plt.figure()
    plt.plot(loss_G_list, label="Generator Loss")
    plt.plot(loss_D_list, label="Discriminator Loss")
    plt.title("Training Loss Curve v5")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f"{OUT_DIR}/loss_curve_v5.png")

if __name__ == "__main__":
    train()
