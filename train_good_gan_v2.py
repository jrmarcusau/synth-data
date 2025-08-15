import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dset
import torchvision.utils as vutils
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image

# === Config ===
TARGET_SIZE = (1024, 884)  # final image size: (W, H)
PADDED_SIZE = 1024         # square input for GAN
BATCH_SIZE = 4             # reduce if memory is tight
NZ = 100
NGF = 64
NDF = 64
NC = 1
EPOCHS = 30
LR = 0.0002
BETA1 = 0.5

DATA_DIR = 'data/good'
OUTPUT_DIR = 'outputs/fake_good_v2'
CHECKPOINT_DIR = 'outputs/checkpoints_good_v2'
LOSS_PLOT_PATH = 'outputs/loss_plot.png'

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# === Transform: Pad to 1024x1024 without cropping ===
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
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# === Generator (1024x1024 output) ===
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(NZ, NGF * 16, 4, 1, 0, bias=False),     # 1x1 -> 4x4
            nn.BatchNorm2d(NGF * 16),
            nn.ReLU(True),
            nn.ConvTranspose2d(NGF * 16, NGF * 8, 4, 2, 1, bias=False), # 4x4 -> 8x8
            nn.BatchNorm2d(NGF * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(NGF * 8, NGF * 4, 4, 2, 1, bias=False),  # 8x8 -> 16x16
            nn.BatchNorm2d(NGF * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(NGF * 4, NGF * 2, 4, 2, 1, bias=False),  # 16x16 -> 32x32
            nn.BatchNorm2d(NGF * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(NGF * 2, NGF, 4, 2, 1, bias=False),      # 32x32 -> 64x64
            nn.BatchNorm2d(NGF),
            nn.ReLU(True),
            nn.ConvTranspose2d(NGF, NC, 16, 16, 0, bias=False),         # 64x64 -> 1024x1024
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)

# === Discriminator (1024x1024 input) ===
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(NC, NDF, 4, 2, 1, bias=False),       # 1024 -> 512
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(NDF, NDF * 2, 4, 2, 1, bias=False),  # 512 -> 256
            nn.BatchNorm2d(NDF * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(NDF * 2, NDF * 4, 4, 2, 1, bias=False), # -> 128
            nn.BatchNorm2d(NDF * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(NDF * 4, NDF * 8, 4, 2, 1, bias=False), # -> 64
            nn.BatchNorm2d(NDF * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(NDF * 8, 1, 4, 1, 0, bias=False),       # -> 61x61
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

# === Initialize models ===
netG = Generator().to(DEVICE)
netD = Discriminator().to(DEVICE)
criterion = nn.BCELoss()

optimizerD = torch.optim.Adam(netD.parameters(), lr=LR, betas=(BETA1, 0.999))
optimizerG = torch.optim.Adam(netG.parameters(), lr=LR, betas=(BETA1, 0.999))

G_losses, D_losses = [], []

# === Training Loop ===
for epoch in range(EPOCHS):
    for i, data in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")):
        real_images = data[0].to(DEVICE)
        b_size = real_images.size(0)

        # === Train Discriminator ===
        netD.zero_grad()
        output_real = netD(real_images)
        label_real = torch.full(output_real.shape, 1.0, dtype=torch.float, device=DEVICE)
        errD_real = criterion(output_real, label_real)
        errD_real.backward()

        noise = torch.randn(b_size, NZ, 1, 1, device=DEVICE)
        fake = netG(noise)
        output_fake = netD(fake.detach())
        label_fake = torch.full(output_fake.shape, 0.0, dtype=torch.float, device=DEVICE)
        errD_fake = criterion(output_fake, label_fake)
        errD_fake.backward()
        optimizerD.step()

        # === Train Generator ===
        netG.zero_grad()
        output_gen = netD(fake)
        label_gen = torch.full(output_gen.shape, 1.0, dtype=torch.float, device=DEVICE)
        errG = criterion(output_gen, label_gen)
        errG.backward()
        optimizerG.step()

    # === Logging ===
    G_losses.append(errG.item())
    D_losses.append(errD_real.item() + errD_fake.item())

    with torch.no_grad():
        # Match fake sample size to real batch for fair comparison
        fixed_noise = torch.randn(real_images.size(0), NZ, 1, 1, device=DEVICE)
        fake_imgs = netG(fixed_noise).detach().cpu()

        # Crop 1024x1024 → 1024x884 (center crop vertically)
        top_crop = (PADDED_SIZE - TARGET_SIZE[1]) // 2
        cropped_imgs = fake_imgs[:, :, top_crop:top_crop+TARGET_SIZE[1], :]

        vutils.save_image(cropped_imgs, f"{OUTPUT_DIR}/epoch_{epoch+1:03}.png", normalize=True)

        real_crop = real_images[:, :, top_crop:top_crop+TARGET_SIZE[1], :].cpu()
        comparison = torch.cat((real_crop, cropped_imgs), dim=2)  # vertical stack
        vutils.save_image(comparison, f"{OUTPUT_DIR}/side_by_side_epoch_{epoch+1:03}.png")

    # === Save checkpoint every 5 epochs ===
    if (epoch + 1) % 5 == 0:
        torch.save(netG.state_dict(), f"{CHECKPOINT_DIR}/generator_epoch_{epoch+1:03}.pt")

# === Save final model ===
torch.save(netG.state_dict(), os.path.join(OUTPUT_DIR, "generator_good_v2.pt"))

# === Plot Losses ===
plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss")
plt.plot(G_losses, label="Generator")
plt.plot(D_losses, label="Discriminator")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.savefig(LOSS_PLOT_PATH)
plt.close()
