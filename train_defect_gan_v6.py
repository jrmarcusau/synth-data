import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

# === CONFIG ===
IMG_SIZE = 256
BATCH_SIZE = 8
TOTAL_EPOCHS = 50
PRETRAIN_EPOCHS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GOOD_DIR = "./data/good"
DEFECT_DIR = "./data/defect"
SAVE_DIR = "outputs/fake_defects_v6"
CKPT_DIR = "outputs/checkpoints_v6"

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

# === DATASET ===
class DefectDataset(Dataset):
    def __init__(self, good_dir, defect_dir, transform=None):
        self.good_paths = sorted([os.path.join(good_dir, f) for f in os.listdir(good_dir)])
        self.defect_paths = sorted([os.path.join(defect_dir, f) for f in os.listdir(defect_dir)])
        self.transform = transform

    def __len__(self):
        return min(len(self.good_paths), len(self.defect_paths))

    def __getitem__(self, idx):
        clean = Image.open(self.good_paths[idx]).convert('RGB')
        defect = Image.open(self.defect_paths[idx]).convert('RGB')
        if self.transform:
            clean = self.transform(clean)
            defect = self.transform(defect)
        return clean, defect

# === MASK GEN ===
def get_defect_mask(clean, defect, threshold=0.1):
    diff = torch.abs(defect - clean)
    mask = (torch.mean(diff, dim=1, keepdim=True) > threshold).float()
    return mask

# === MODELS ===
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

class PatchDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(6, 64, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2),
            nn.Conv2d(256, 1, 4, 1, 1), nn.Sigmoid()
        )

    def forward(self, x, y):
        return self.net(torch.cat([x, y], dim=1))

# === TRAINING ===
def train(pretrain=False):
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    dataset = DefectDataset(GOOD_DIR, DEFECT_DIR, transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    G = Generator().to(DEVICE)
    D = PatchDiscriminator().to(DEVICE)

    l1_loss = nn.L1Loss()
    adv_loss = nn.BCELoss()

    opt_G = torch.optim.Adam(G.parameters(), lr=1e-4)
    opt_D = torch.optim.Adam(D.parameters(), lr=1e-4)

    loss_G_list, loss_D_list = [], []

    total_epochs = PRETRAIN_EPOCHS if pretrain else TOTAL_EPOCHS
    print(f"{'Pretraining' if pretrain else 'Full GAN training'} for {total_epochs} epochs...")

    for epoch in range(1, total_epochs + 1):
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs}", unit="batch")

        for clean, defect in pbar:
            clean, defect = clean.to(DEVICE), defect.to(DEVICE)

            # === Generator ===
            opt_G.zero_grad()
            fake_defect = G(clean)

            if pretrain:
                g_loss = l1_loss(fake_defect, clean)  # identity learning
            else:
                pred_fake = D(clean, fake_defect)
                pred_real = D(clean, defect)

                valid = torch.ones_like(pred_real)
                fake = torch.zeros_like(pred_fake)

                mask = get_defect_mask(clean, defect)
                weighted_l1 = torch.mean(mask * torch.abs(fake_defect - defect))

                g_loss = adv_loss(pred_fake, valid) + 100 * weighted_l1

            g_loss.backward()
            opt_G.step()

            # === Discriminator ===
            if not pretrain:
                opt_D.zero_grad()
                pred_real = D(clean, defect)
                pred_fake = D(clean, fake_defect.detach())
                d_loss = 0.5 * (adv_loss(pred_real, valid) + adv_loss(pred_fake, fake))
                d_loss.backward()
                opt_D.step()
                loss_D_list.append(d_loss.item())
            else:
                d_loss = torch.tensor(0)

            loss_G_list.append(g_loss.item())
            pbar.set_postfix(loss_D=d_loss.item(), loss_G=g_loss.item())

        # === Visualization ===
        G.eval()
        with torch.no_grad():
            val_input = clean[:4]
            val_output = G(val_input)
            comparison = torch.cat([val_input, val_output], dim=0) * 0.5 + 0.5
            tag = "pretrain" if pretrain else "epoch"
            utils.save_image(comparison, f"{SAVE_DIR}/{tag}_{epoch:02d}.png", nrow=4)
        G.train()

        # === Checkpoint ===
        if not pretrain and (epoch % 10 == 0 or epoch == TOTAL_EPOCHS):
            torch.save(G.state_dict(), f"{CKPT_DIR}/G_epoch{epoch:02d}.pt")
            torch.save(D.state_dict(), f"{CKPT_DIR}/D_epoch{epoch:02d}.pt")

    # === Loss Curve ===
    plt.figure()
    plt.plot(loss_G_list, label="Generator Loss")
    if not pretrain:
        plt.plot(loss_D_list, label="Discriminator Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("v6 Training Loss" + (" (Pretrain)" if pretrain else ""))
    plt.legend()
    plt.savefig(f"{SAVE_DIR}/loss_curve_{'pretrain' if pretrain else 'gan'}.png")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrain", action="store_true", help="Run identity pretraining phase")
    args = parser.parse_args()
    train(pretrain=args.pretrain)
