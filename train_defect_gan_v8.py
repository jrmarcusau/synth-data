import os
import torch
import torch.nn as nn
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
        good = Image.open(self.good_paths[idx]).convert("RGB")
        defect = Image.open(self.defect_paths[idx]).convert("RGB")
        if self.transform:
            good = self.transform(good)
            defect = self.transform(defect)
        return good, defect

# ====================
# Models
# ====================
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 4, 2, 1), nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(6, 64, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2),
            nn.Conv2d(128, 1, 4, 1, 1), nn.Sigmoid()
        )

    def forward(self, x, y):
        return self.model(torch.cat([x, y], dim=1))

# ====================
# Utilities
# ====================
def denorm(x):
    return x * 0.5 + 0.5  # from [-1,1] to [0,1]

def save_visual(good, generated, epoch, out_dir, tag="step"):
    comp = torch.cat([denorm(good), denorm(generated)], dim=0)
    utils.save_image(comp, os.path.join(out_dir, f"{tag}_epoch_{epoch:02d}.png"), nrow=4)

# ====================
# Training
# ====================
def train():
    good_dir = "./data/good"
    defect_dir = "./data/defect"
    out_dir = "./outputs/fake_defects_v8"
    ckpt_dir = "./outputs/checkpoints_v8"
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    img_size = 256
    batch_size = 8
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    dataset = PairedDefectDataset(good_dir, defect_dir, transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    G = Generator().to(device)
    D = Discriminator().to(device)

    pixel_loss = nn.L1Loss()
    adv_loss = nn.BCELoss()

    # -----------------------
    # 1. Pretrain Generator
    # -----------------------
    print("🔧 Pretraining Generator (G(x) ≈ x)")
    opt_pre = torch.optim.Adam(G.parameters(), lr=1e-4)
    for epoch in range(1, 6):
        pbar = tqdm(loader, desc=f"[Pretrain {epoch}/5]")
        for good, _ in pbar:
            good = good.to(device)
            recon = G(good)
            loss = pixel_loss(recon, good)

            opt_pre.zero_grad()
            loss.backward()
            opt_pre.step()
            pbar.set_postfix(loss=loss.item())

        with torch.no_grad():
            save_visual(good[:4], G(good[:4]), epoch, out_dir, tag="pretrain")

    torch.save(G.state_dict(), os.path.join(ckpt_dir, "pretrained_generator_v8.pt"))

    # -----------------------
    # 2. GAN Phase
    # -----------------------
    print("🎯 GAN training with defects")
    opt_G = torch.optim.Adam(G.parameters(), lr=1e-4)
    opt_D = torch.optim.Adam(D.parameters(), lr=1e-4)

    loss_G_list, loss_D_list = [], []

    for epoch in range(1, 51):
        pbar = tqdm(loader, desc=f"[GAN Epoch {epoch}/50]")
        for good, defect in pbar:
            good, defect = good.to(device), defect.to(device)

            fake_defect = G(good)
            pred_fake = D(good, fake_defect)
            valid = torch.ones_like(pred_fake)
            fake = torch.zeros_like(pred_fake)

            # Train Generator
            opt_G.zero_grad()
            loss_G_adv = adv_loss(pred_fake, valid)
            loss_G_l1 = pixel_loss(fake_defect, defect)
            loss_G = loss_G_adv + 10 * loss_G_l1
            loss_G.backward()
            opt_G.step()

            # Train Discriminator
            opt_D.zero_grad()
            pred_real = D(good, defect)
            pred_fake_detach = D(good, fake_defect.detach())
            loss_D_real = adv_loss(pred_real, valid)
            loss_D_fake = adv_loss(pred_fake_detach, fake)
            loss_D = 0.5 * (loss_D_real + loss_D_fake)
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

    # Plot loss curve
    plt.plot(loss_G_list, label="Generator")
    plt.plot(loss_D_list, label="Discriminator")
    plt.title("Loss Curve (v8)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(out_dir, "loss_curve_v8.png"))
    plt.close()

# ====================
# Main
# ====================
if __name__ == "__main__":
    train()
