# requirements: pip install torch torchvision
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ----- Encoder / Decoder / VAE -----
class Encoder(nn.Module):
    def __init__(self, in_dim=784, hid=400, z_dim=20):
        super().__init__()
        self.fc = nn.Linear(in_dim, hid)
        self.mu = nn.Linear(hid, z_dim)
        self.logvar = nn.Linear(hid, z_dim)

    def forward(self, x):
        # x: (B,1,28,28) -> (B,784)
        x = x.view(x.size(0), -1)
        h = F.relu(self.fc(x))
        return self.mu(h), self.logvar(h)

class Decoder(nn.Module):
    def __init__(self, z_dim=20, hid=400, out_dim=784):
        super().__init__()
        self.fc1 = nn.Linear(z_dim, hid)
        self.fc2 = nn.Linear(hid, out_dim)

    def forward(self, z):
        h = F.relu(self.fc1(z))
        x_hat = torch.sigmoid(self.fc2(h))  # MNIST → [0,1]
        return x_hat.view(z.size(0), 1, 28, 28)

class VAE(nn.Module):
    def __init__(self, z_dim=20):
        super().__init__()
        self.enc = Encoder(z_dim=z_dim)
        self.dec = Decoder(z_dim=z_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def forward(self, x):
        mu, logvar = self.enc(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.dec(z)
        return x_hat, mu, logvar

# ----- Loss -----
def vae_loss(x, x_hat, mu, logvar, beta=1.0, recon="bce"):
    if recon == "bce":
        # BCE는 입력/출력이 [0,1]일 때 적합
        rec = F.binary_cross_entropy(x_hat, x, reduction="sum")
    else:
        rec = F.mse_loss(x_hat, x, reduction="sum")
    # KL(q||p) = -0.5 * Σ(1 + logσ² - μ² - σ²)
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return rec + beta * kl, rec, kl

# ----- Data -----
transform = transforms.Compose([transforms.ToTensor()])
train_set = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_set  = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=2)
test_loader  = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=2)

# ----- Train -----
def train_epoch(model, opt, loader):
    model.train()
    tot = rec_tot = kl_tot = 0.0
    for x, _ in loader:
        x = x.to(DEVICE)
        x_hat, mu, logvar = model(x)
        loss, rec, kl = vae_loss(x, x_hat, mu, logvar, beta=1.0, recon="bce")
        opt.zero_grad(); loss.backward(); opt.step()
        tot += loss.item(); rec_tot += rec.item(); kl_tot += kl.item()
    n = len(loader.dataset)
    return tot/n, rec_tot/n, kl_tot/n

@torch.no_grad()
def eval_epoch(model, loader):
    model.eval()
    tot = rec_tot = kl_tot = 0.0
    for x, _ in loader:
        x = x.to(DEVICE)
        x_hat, mu, logvar = model(x)
        loss, rec, kl = vae_loss(x, x_hat, mu, logvar, beta=1.0, recon="bce")
        tot += loss.item(); rec_tot += rec.item(); kl_tot += kl.item()
    n = len(loader.dataset)
    return tot/n, rec_tot/n, kl_tot/n

def main():
    torch.manual_seed(0)
    model = VAE(z_dim=20).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    EPOCHS = 1
    for ep in range(1, EPOCHS+1):
        tr = train_epoch(model, opt, train_loader)
        ev = eval_epoch(model, test_loader)
        print(f"[{ep:02d}] train loss={tr[0]:.4f} (rec={tr[1]:.4f}, kl={tr[2]:.4f}) | "
              f"test loss={ev[0]:.4f} (rec={ev[1]:.4f}, kl={ev[2]:.4f})")

    # 샘플 생성 (z~N(0,I))
    with torch.no_grad():
        z = torch.randn(64, 20, device=DEVICE)
        samples = model.dec(z).cpu()  # (64,1,28,28)
        # 저장(선택)
        import torchvision.utils as vutils
        vutils.save_image(samples, "vae_samples_1.png", nrow=8)

if __name__ == "__main__":
    main()
