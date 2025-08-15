import os
import random
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader

# --- Dataset ---
class DefectDataset(Dataset):
    def __init__(self, good_dir, defect_dir, transform=None):
        self.good_paths = sorted([os.path.join(good_dir, f) for f in os.listdir(good_dir)])
        self.defect_paths = sorted([os.path.join(defect_dir, f) for f in os.listdir(defect_dir)])
        self.transform = transform

    def __len__(self):
        return len(self.good_paths)

    def __getitem__(self, idx):
        clean_path = self.good_paths[idx]
        defect_path = random.choice(self.defect_paths)

        clean = Image.open(clean_path).convert('L')
        defect = Image.open(defect_path).convert('L')

        if self.transform:
            clean = self.transform(clean)
            defect = self.transform(defect)

        return clean, defect

# --- Generator ---
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 64, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.ConvTranspose2d(64, 1, 4, stride=2, padding=1), nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)

# --- Discriminator ---
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(2, 64, 4, stride=2, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, stride=2, padding=1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(128 * 64 * 64, 1), nn.Sigmoid()
        )

    def forward(self, x, y):
        inp = torch.cat([x, y], dim=1)
        return self.model(inp)

# --- Training setup ---
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_size = 256
    os.makedirs("outputs/fake_defects", exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # grayscale
    ])

    dataset = DefectDataset('./data/good', './data/defect', transform)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    G = Generator().to(device)
    D = Discriminator().to(device)
    adversarial_loss = nn.BCELoss()
    pixel_loss = nn.L1Loss()

    optimizer_G = torch.optim.Adam(G.parameters(), lr=1e-4)
    optimizer_D = torch.optim.Adam(D.parameters(), lr=1e-4)

    for epoch in range(1, 51):
        for i, (clean, defect) in enumerate(loader):
            clean, defect = clean.to(device), defect.to(device)
            valid = torch.ones(clean.size(0), 1).to(device)
            fake = torch.zeros(clean.size(0), 1).to(device)

            # Train Generator
            optimizer_G.zero_grad()
            fake_defect = G(clean)
            pred_fake = D(clean, fake_defect)
            loss_G_adv = adversarial_loss(pred_fake, valid)
            loss_G_pix = pixel_loss(fake_defect, defect)
            loss_G = loss_G_adv + 100 * loss_G_pix  # pixel loss weighted
            loss_G.backward()
            optimizer_G.step()

            # Train Discriminator
            optimizer_D.zero_grad()
            pred_real = D(clean, defect)
            pred_fake = D(clean, fake_defect.detach())
            loss_D = (adversarial_loss(pred_real, valid) + adversarial_loss(pred_fake, fake)) / 2
            loss_D.backward()
            optimizer_D.step()

        print(f"[Epoch {epoch}/50] Loss_D: {loss_D.item():.4f}, Loss_G: {loss_G.item():.4f}")

        # Save sample outputs every 5 epochs
        if epoch % 5 == 0:
            G.eval()
            with torch.no_grad():
                sample_input = next(iter(loader))[0].to(device)
                fake_output = G(sample_input)
                fake_output = fake_output * 0.5 + 0.5  # unnormalize
                utils.save_image(fake_output, f"outputs/fake_defects/epoch_{epoch}.png", nrow=4)
            G.train()

    # Save final model
    torch.save(G.state_dict(), "outputs/generator_final.pth")
    print("Training complete. Final model saved.")

if __name__ == "__main__":
    train()
