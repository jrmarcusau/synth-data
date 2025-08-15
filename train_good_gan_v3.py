import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dset
import torchvision.utils as vutils
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

# === Config ===
TARGET_SIZE = (1024, 884)
PADDED_SIZE = 1024
BATCH_SIZE = 8
NZ = 100
NGF = 64
NDF = 64
NC = 1
EPOCHS = 30
LR = 0.0002
BETA1 = 0.5

DATA_DIR = 'data/good'
OUTPUT_DIR = 'outputs/fake_good_v3'
CHECKPOINT_DIR = 'outputs/checkpoints_good_v3'
LOSS_PLOT_PATH = os.path.join(OUTPUT_DIR, 'loss_plot.png')

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# === Pad transform to 1024x1024 (no cropping) ===
class PadToSquare1024:
    def __call__(self, img):
        w, h = img.size
        pad_top = (PADDED_SIZE - h) // 2
        pad_bottom = PADDED_SIZE - h - pad_top
        pad_left = (PADDED_SIZE - w) // 2
        pad_right = PADDED_SIZE - w - pad_left
        return transforms.functional.pad(img, (pad_left, pad_top, pad_right, pad_bottom), padding_mode='reflect')

transform = transforms.Compose([
    transforms.Grayscale(),
    PadToSquare1024(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = dset.ImageFolder(root='data', transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)

# === Generator (1024x1024 output) ===
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(NZ, NGF * 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(NGF * 16),
            nn.ReLU(True),
            nn.ConvTranspose2d(NGF * 16, NGF * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NGF * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(NGF * 8, NGF * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NGF * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(NGF * 4, NGF * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NGF * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(NGF * 2, NGF, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NGF),
            nn.ReLU(True),
            nn.ConvTranspose2d(NGF, NC, 16, 16, 0, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)

# === Discriminator (1024x1024 input) ===
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(NC, NDF, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(NDF, NDF * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NDF * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(NDF * 2, NDF * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NDF * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(NDF * 4, NDF * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NDF * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(NDF * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

# === Init models & loss ===
netG = Generator().to(DEVICE)
netD = Discriminator().to(DEVICE)
criterion = nn.BCELoss()

optimizerD = torch.optim.Adam(netD.parameters(), lr=LR, betas=(BETA1, 0.999))
optimizerG = torch.optim.Adam(netG.parameters(), lr=LR, betas=(BETA1, 0.999))

G_losses, D_losses = [], []

# === Training ===
for epoch in range(EPOCHS):
    for i, data in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")):
        real_images = data[0].to(DEVICE, non_blocking=True)
        b_size = real_images.size(0)

        # --- Train D ---
        netD.zero_grad()
        output_real = netD(real_images)
        label_real = torch.full_like(output_real, 1.0, device=DEVICE)
        errD_real = criterion(output_real, label_real)
        errD_real.backward()

        noise = torch.randn(b_size, NZ, 1, 1, device=DEVICE)
        fake = netG(noise)
        output_fake = netD(fake.detach())
        label_fake = torch.full_like(output_fake, 0.0, device=DEVICE)
        errD_fake = criterion(output_fake, label_fake)
        errD_fake.backward()
        optimizerD.step()

        # --- Train G ---
        netG.zero_grad()
        output_gen = netD(fake)
        label_gen = torch.full_like(output_gen, 1.0, device=DEVICE)
        errG = criterion(output_gen, label_gen)
        errG.backward()
        optimizerG.step()

    # --- Log losses ---
    G_losses.append(errG.item())
    D_losses.append(errD_real.item() + errD_fake.item())

    with torch.no_grad():
        fixed_noise = torch.randn(real_images.size(0), NZ, 1, 1, device=DEVICE)
        fake_imgs = netG(fixed_noise).detach().cpu()

        # Crop 1024x1024 → 1024x884 (center crop)
        top_crop = (PADDED_SIZE - TARGET_SIZE[1]) // 2
        cropped_imgs = fake_imgs[:, :, top_crop:top_crop+TARGET_SIZE[1], :]
        real_crop = real_images[:, :, top_crop:top_crop+TARGET_SIZE[1], :].cpu()

        # De-normalize both
        fake_vis = cropped_imgs * 0.5 + 0.5
        real_vis = real_crop * 0.5 + 0.5

        vutils.save_image(fake_vis, f"{OUTPUT_DIR}/epoch_{epoch+1:03}.png", normalize=False)

        comparison = torch.cat((real_vis, fake_vis), dim=2)
        vutils.save_image(comparison, f"{OUTPUT_DIR}/side_by_side_epoch_{epoch+1:03}.png", normalize=False)

    # --- Save checkpoint every 5 epochs ---
    if (epoch + 1) % 5 == 0:
        torch.save(netG.state_dict(), f"{CHECKPOINT_DIR}/generator_epoch_{epoch+1:03}.pt")

# === Final save ===
torch.save(netG.state_dict(), os.path.join(OUTPUT_DIR, "generator_good_v3.pt"))

# === Plot losses ===
plt.figure(figsize=(10, 5))
plt.title("Generator and Discriminator Loss")
plt.plot(G_losses, label="Generator")
plt.plot(D_losses, label="Discriminator")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.savefig(LOSS_PLOT_PATH)
plt.close()
