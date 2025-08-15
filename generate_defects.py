import os
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, utils
from torch.utils.data import DataLoader, Dataset

# --- Same Generator as training ---
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

# --- Inference Dataset ---
class InferenceDataset(Dataset):
    def __init__(self, img_dir, transform):
        self.img_paths = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir)])
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert('L')
        img_tensor = self.transform(img)
        return img_tensor, os.path.basename(self.img_paths[idx])

# --- Main inference ---
def generate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("outputs/inference_output_v2", exist_ok=True)

    # Load generator
    G = Generator().to(device)
    G.load_state_dict(torch.load("outputs/generator_final_v2.pth", map_location=device))
    G.eval()

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    dataset = InferenceDataset("./data/inference_input", transform)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    with torch.no_grad():
        for img_tensor, filename in loader:
            img_tensor = img_tensor.to(device)
            output = G(img_tensor)
            output = output * 0.5 + 0.5  # unnormalize to [0,1]
            save_path = os.path.join("outputs/inference_output", filename[0])
            utils.save_image(output, save_path)

    print("✅ Inference complete. Images saved to outputs/inference_output")

if __name__ == "__main__":
    generate()
