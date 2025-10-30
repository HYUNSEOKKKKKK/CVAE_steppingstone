from __future__ import annotations
import math
import os
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ============================
# 1) Data & utilities (6D: dx,dy,dz,dw,dh,dyaw_deg)
# ============================

@dataclass
class Step6:
    dx: float   # 다음 스톤 중심의 x 변화 (m)
    dy: float   # 다음 스톤 중심의 y 변화 (m)
    dz: float   # 다음 스톤 중심의 z 변화 (m)
    dw: float   # 다음 스톤 가로(폭) 변화 (m)
    dh: float   # 다음 스톤 세로(길이) 변화 (m)
    dyaw: float # 다음 스톤 z축 기준 yaw 변화 (deg)  ← 추가

    def to_tensor(self) -> torch.Tensor:
        return torch.tensor([self.dx, self.dy, self.dz, self.dw, self.dh, self.dyaw], dtype=torch.float32)

    @staticmethod
    def from_tensor(t: torch.Tensor) -> "Step6":
        t = t.detach().cpu().float()
        return Step6(dx=float(t[0]), dy=float(t[1]), dz=float(t[2]),
                     dw=float(t[3]), dh=float(t[4]), dyaw=float(t[5]))


def wrap_deg(x: torch.Tensor) -> torch.Tensor:
    """torch 텐서를 [-180, 180)로 래핑"""
    return (x + 180.0) % 360.0 - 180.0

def wrap_deg_f(x: float) -> float:
    """파이썬 float용 [-180, 180) 래핑"""
    return (x + 180.0) % 360.0 - 180.0

def angle_ar_blend(prev_deg: float, new_deg: float, alpha: float) -> float:
    """
    각도(도)용 AR(1) 블렌딩. prev에 대해 new의 최단각 차이를 따라 alpha만큼 이동.
    반환은 [-180,180)으로 래핑.
    """
    delta = wrap_deg_f(new_deg - prev_deg)        # 최단각 차이
    out = prev_deg + alpha * delta
    return wrap_deg_f(out)

# ==== Normalization helpers for Step6 (dx,dy,dz,dw,dh,dyaw_deg) ====
# 대략적 스케일로 나눠 [-1,1] 근처. 마지막 항은 각도(°) 스케일.
SCALE6 = torch.tensor([0.2, 0.2, 0.1, 0.11, 0.11, 180.0], dtype=torch.float32)  # 45° 기준 (환경에 맞게 조정)

def normalize_step_tensor(x: torch.Tensor) -> torch.Tensor:
    # x: (..., 6)
    return x / SCALE6.to(x.device)

def denormalize_step_tensor(xn: torch.Tensor) -> torch.Tensor:
    return xn * SCALE6.to(xn.device)

def normalize_cond(cond: torch.Tensor) -> torch.Tensor:
    # cond = [prev2(6), prev1(6)] -> 총 12
    return torch.cat([
        normalize_step_tensor(cond[..., :6]),
        normalize_step_tensor(cond[..., 6:12])
    ], dim=-1)

def denormalize_output(xn: torch.Tensor) -> torch.Tensor:
    # 모델 출력(6D delta)을 원 단위로 복원
    return denormalize_step_tensor(xn)


# ============================
# SyntheticStepDeltaDataset (6D 버전)
# ============================
class SyntheticStepDeltaDataset(Dataset):
    """
    (prev2, prev1) → current 의 6D 델타 표본을 만듭니다.
    각 표본:
      cond ∈ R^12 = [prev2(6), prev1(6)]
      target ∈ R^6 = current(6)
    누적용 초기 절대 상태(seed)는 학습엔 필요없지만, 시각화/샘플링에서 사용하세요.
    """
    def __init__(self, length: int = 20000, seed: int = 0):
        super().__init__()
        g = random.Random(seed)
        self.samples: List[Tuple[torch.Tensor, torch.Tensor]] = []

        def sample_delta(prev: Optional[Step6], prev2: Optional[Step6]) -> Step6:
            # 기본 균등 + 연속성(약한 AR)
            dx   = g.uniform(0.10, 0.30)
            dy   = g.uniform(-0.10, 0.10)
            dz   = g.uniform(-0.05, 0.05)
            dw   = g.uniform(-0.06, 0.05)
            dh   = g.uniform(-0.06, 0.05)
            dyaw = g.uniform(-90, 90)   # °

            if prev:
                # 약한 관성(이전 델타와 섞기) — 위치/크기: 선형, 각도: 최단각 기반
                alpha = 1.0
                dx   = alpha*dx   + (1-alpha)*prev.dx
                dy   = alpha*dy   + (1-alpha)*prev.dy
                dz   = alpha*dz   + (1-alpha)*prev.dz
                dw   = alpha*dw   + (1-alpha)*prev.dw
                dh   = alpha*dh   + (1-alpha)*prev.dh
                dyaw = angle_ar_blend(prev.dyaw, dyaw, alpha)  # ← 각도 전용 AR 블렌드
            else:
                dyaw = wrap_deg_f(dyaw)

            return Step6(dx, dy, dz, dw, dh, dyaw)

        for _ in range(length):
            d2 = sample_delta(None, None)
            d1 = sample_delta(d2, None)
            d0 = sample_delta(d1, d2)
            cond   = torch.cat([d2.to_tensor(), d1.to_tensor()])  # 12D
            target = d0.to_tensor()                               # 6D
            self.samples.append((cond, target))

    def __len__(self):  return len(self.samples)
    def __getitem__(self, idx: int):  return self.samples[idx]


# 2) CVAE model
# ============================

class MLP(nn.Module):
    def __init__(self, sizes: Iterable[int], activation=nn.ReLU, out_activation=None):
        super().__init__()
        sizes = list(sizes)
        layers: List[nn.Module] = []
        for i in range(len(sizes) - 2):
            layers += [nn.Linear(sizes[i], sizes[i+1]), activation()]
        layers.append(nn.Linear(sizes[-2], sizes[-1]))
        if out_activation is not None:
            layers.append(out_activation())
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)


class CVAE(nn.Module):
    def __init__(self, cond_dim: int = 12, x_dim: int = 6, z_dim: int = 8):
        super().__init__()
        # Encoder: [512, 128] -> (mu, logvar)
        self.enc = MLP([cond_dim + x_dim, 512, 128])
        self.mu = nn.Linear(128, z_dim)
        self.logvar = nn.Linear(128, z_dim)
        # Decoder: [128, 512] -> x'
        self.dec_in = nn.Linear(cond_dim + z_dim, 128)
        self.dec_h = nn.ReLU()
        self.dec = MLP([128, 512, x_dim])

    def encode(self, x, c):
        h = self.enc(torch.cat([x, c], dim=-1))
        return self.mu(h), self.logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, c):
        h = self.dec_h(self.dec_in(torch.cat([z, c], dim=-1)))
        return self.dec(h)

    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        x_rec = self.decode(z, c)
        return x_rec, mu, logvar


class Cvaeloss(nn.Module):
    def __init__(self, kld_weight: float = 0.04, angle_last: bool = True, angle_scale_deg: float = 45.0):
        super().__init__()
        self.kld_weight = kld_weight
        self.angle_last = angle_last
        self.angle_scale_deg = angle_scale_deg
        self.mse = nn.MSELoss()

    @staticmethod
    def wrap_deg_tensor(x: torch.Tensor) -> torch.Tensor:
        # [-180, 180) 래핑
        return (x + 180.0) % 360.0 - 180.0

    def forward(self, x_rec, x, mu, logvar):
        if not self.angle_last:
            mse = self.mse(x_rec, x)
        else:
            # 1) 비각도 성분(앞 5개) MSE
            mse_lin = self.mse(x_rec[..., :-1], x[..., :-1])

            # 2) 각도 성분(마지막 1개) — 정규화 풀고, 각도 차이 래핑 후 재정규화
            pred_deg = x_rec[..., -1] * self.angle_scale_deg
            targ_deg = x[..., -1]    * self.angle_scale_deg
            diff_deg = self.wrap_deg_tensor(pred_deg - targ_deg)
            diff_norm = diff_deg / self.angle_scale_deg  # 정규화 스케일로 복귀
            mse_ang = torch.mean(diff_norm ** 2)

            mse = mse_lin + mse_ang

        # KL
        kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return mse + self.kld_weight * kld, mse, kld



# ============================
# 3) Training
# ============================

def train_cvae(
    model: CVAE,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    epochs: int = 50,
    lr: float = 2e-4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ckpt_path: str = "cvae_mapgen.pt",
):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = Cvaeloss(kld_weight=0.04, angle_last=True, angle_scale_deg=float(SCALE6[-1]))

    scaler = torch.cuda.amp.GradScaler(enabled=(device.startswith("cuda")))

    best_val = float("inf")
    for ep in range(1, epochs + 1):
        model.train()
        total = 0.0
        for cond, target in train_loader:
            # 배치 루프 안에서
            cond = cond.to(device)
            target = target.to(device)

            cond_n = normalize_cond(cond)
            target_n = normalize_step_tensor(target)

            with torch.cuda.amp.autocast(enabled=(device.startswith("cuda"))):
                pred_n, mu, lv = model(target_n, cond_n)  # ★ 정규화 공간에서 학습
                loss, mse, kld = criterion(pred_n, target_n, mu, lv)

            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total += float(loss.detach().cpu()) * cond.size(0)
        avg = total / len(train_loader.dataset)

        val = avg
        if val_loader is not None:
            model.eval()
            tt = 0.0
            with torch.no_grad():
                for cond, target in val_loader:
                    cond = cond.to(device)
                    target = target.to(device)

                    cond_n = normalize_cond(cond)
                    target_n = normalize_step_tensor(target)

                    pred_n, mu, lv = model(target_n, cond_n)
                    loss, _, _ = criterion(pred_n, target_n, mu, lv)

                    tt += float(loss.detach().cpu()) * cond.size(0)
            val = tt / len(val_loader.dataset)

        print(f"[Epoch {ep:03d}] train_loss={avg:.5f} val_loss={val:.5f}")
        if val < best_val:
            best_val = val
            torch.save({"model": model.state_dict(), "val": val}, ckpt_path)
            print(f"  ↳ saved best checkpoint to {ckpt_path}")


# ============================
# 4) Inference / sampling
# ============================
import math

def sample_next(
    model: CVAE,
    dnm2: Step6, dnm1: Step6,
    a: float = 0.0,                 # z ~ N(0, (1+a) I)
    z: Optional[torch.Tensor] = None,
    device: str = "cpu",
) -> Step6:
    """
    이전 2스텝(dnm2, dnm1)을 조건으로 1개 샘플 생성.
    - a: 분산 스케일 조절 (cov = (1+a) I). a >= -1 권장.
    - z를 직접 주면 그대로 사용(스케일 적용 안 함).
    - 출력의 마지막 성분(dyaw[deg])은 [-180,180)로 래핑.
    """
    model.eval()
    with torch.no_grad():
        # (1) 조건 준비: [prev2(6), prev1(6)] -> (1,12)
        cond = torch.cat([dnm2.to_tensor(), dnm1.to_tensor()]).to(device)[None, :]
        cond_n = normalize_cond(cond)  # (1,12)

        # (2) 잠재 샘플 z 준비
        if z is None:
            zdim = model.mu.out_features
            # cov = (1+a) I  →  std = sqrt(max(1+a, 0))
            scale = math.sqrt(max(1.0 + float(a), 0.0))
            z = torch.randn(1, zdim, device=device) * scale
        else:
            z = z.to(device)

        # (3) 디코딩 (정규화 → 원 단위)
        x_hat_n = model.decode(z, cond_n)     # (1,6) normalized
        x_hat   = denormalize_output(x_hat_n) # (1,6) original units

        # (4) dyaw[deg] 래핑
        x = x_hat[0].clone()
        x[-1] = (x[-1] + 180.0) % 360.0 - 180.0

        return Step6.from_tensor(x)



from typing import List, Tuple, Optional

# --- yaw 래핑 유틸 ([-180,180))
def wrap_deg_f(x: float) -> float:
    return (x + 180.0) % 360.0 - 180.0


# ============================
# 1) Generate sequence (with variance scale 'a')
# ============================
def generate_sequence(
    model: CVAE,
    d0: Step6, d1: Step6,
    steps: int = 20,
    a: float = 0.0,              # ← z ~ N(0, (1+a) I)
    device: str = "cpu",
) -> List[Step6]:
    """
    초기 두 델타(d0,d1)를 시드로 하여, 매 스텝마다 sample_next로 1개씩 생성.
    a > 0 → 다양성↑, a=0 → 기본, -1 < a < 0 → 보수적.
    """
    seq = [d0, d1]
    for _ in range(steps):
        nxt = sample_next(model, seq[-2], seq[-1], a=a, device=device)
        seq.append(nxt)
    return seq


# ============================
# 2) Deltas -> absolute (centers, sizes, [yaws])
# ============================
def deltas_to_absolute(
    seq: List[Step6],
    center0: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    size0: Tuple[float, float] = (0.35, 0.25),
    yaw0_deg: float = 0.0,              # ← 초기 yaw (deg)
    return_yaw: bool = True,            # ← True면 yaw 리스트도 반환
    min_size: float = 0.05,             # ← w,h 하한 가드
) -> Tuple[
    List[Tuple[float, float, float]],
    List[Tuple[float, float]],
    Optional[List[float]]
]:
    """
    6D 델타 시퀀스를 절대 좌표/크기/(선택)yaw로 누적.
    반환:
      centers: [(x,y,z), ...]  (seed 포함)
      sizes:   [(w,h),   ...]  (seed 포함)
      yaws:    [yaw_deg, ...]  (seed 포함; return_yaw=False면 None)
    """
    x, y, z = center0
    w, h    = size0
    yaw_deg = wrap_deg_f(yaw0_deg)

    centers = [(x, y, z)]
    sizes   = [(w, h)]
    yaws    = [yaw_deg]

    # 시드 두 개 제외하고 누적 (seq[2:]부터 적용)
    for d in seq[2:]:
        x   += d.dx
        y   += d.dy
        z   += d.dz
        w    = max(min_size, w + d.dw)
        h    = max(min_size, h + d.dh)
        yaw_deg = wrap_deg_f(yaw_deg + d.dyaw)  # ← yaw 누적 후 래핑

        centers.append((x, y, z))
        sizes.append((w, h))
        yaws.append(yaw_deg)

    if return_yaw:
        return centers, sizes, yaws
    else:
        return centers, sizes, None




# ============================
# 6) Visualization helpers
# ============================

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np


def _Rx(a):
    ca, sa = math.cos(a), math.sin(a)
    return np.array([[1,0,0],[0,ca,-sa],[0,sa,ca]], dtype=float)

def _Ry(a):
    ca, sa = math.cos(a), math.sin(a)
    return np.array([[ca,0,sa],[0,1,0],[-sa,0,ca]], dtype=float)

def _Rz(a):
    ca, sa = math.cos(a), math.sin(a)
    return np.array([[ca,-sa,0],[sa,ca,0],[0,0,1]], dtype=float)


import matplotlib.colors as mcolors

def plot_sequence_3d_deltas(
    seq: List[Step6],
    center0=(0,0,0),
    size0=(0.35,0.25),
    thickness: float = 0.04,
    title="Generated (3D)",
    yaw0_deg: float = 0.0,
    tilt_x_deg: float = 0.0,
    tilt_y_deg: float = 0.0,
    # --- 새로 추가: 시작/마지막 스톤 색상 (연한 톤)
    start_color="lightgreen",   # 1번 스톤
    goal_color="#FFB3B3",       # 마지막 스톤(연한 빨강)
    equalize_axes: bool = True,
):
    # yaw 포함해서 누적
    centers, sizes, yaws = deltas_to_absolute(
        seq, center0=center0, size0=size0, yaw0_deg=yaw0_deg, return_yaw=True
    )

    # --- z 정규화(그레이스케일용) ---
    zs = [c[2] for c in centers[1:]] if len(centers) > 1 else [0.0]
    zmin, zmax = (min(zs), max(zs))
    eps = 1e-12
    import matplotlib as mpl
    norm = mpl.colors.Normalize(vmin=zmin-0.1, vmax=zmax+0.1 if abs(zmax - zmin) > eps else zmin + 1.0)
    cmap = plt.cm.Greys

    fig = plt.figure(figsize=(7,5))
    ax = fig.add_subplot(111, projection='3d'); ax.set_title(title)
    ax.view_init(elev=35, azim=-135)
    ax.grid(False)

    faces, fcols = [], []

    N = len(centers) - 1  # 플로팅하는 스톤 수(시드 제외)

    # seed(0) 제외하고 1~N 스톤을 회전 포함해 생성
    for i, (ctr, sz, yaw_deg) in enumerate(zip(centers[1:], sizes[1:], yaws[1:]), start=1):
        cx, cy, cz = ctr
        w, h = sz

        # 회전된 상면 꼭짓점 (4x3, 순서: CCW)
        top = stone_corners_3d(
            center=(cx, cy, cz),
            yaw_deg=yaw_deg,
            tilt_x_deg=tilt_x_deg,
            tilt_y_deg=tilt_y_deg,
            size=(w, h)
        )
        # 하면: 월드 z축으로 thickness 만큼 아래
        bottom = top - np.array([0, 0, thickness], dtype=float)

        # --- 색상 선택: 1번/마지막은 지정색, 나머지는 높이-그레이스케일 ---
        if i == 1:
            c = mcolors.to_rgba(start_color, alpha=0.9)
        elif i == N:
            c = mcolors.to_rgba(goal_color, alpha=0.9)
        else:
            c = cmap(norm(cz))

        # 상/하면 추가 (하면은 법선 맞추려 뒤집음)
        faces += [top, bottom[::-1]]
        fcols += [c, c]

        # 옆면(4개 면)
        for k in range(4):
            k2 = (k + 1) % 4
            side = np.vstack([top[k], top[k2], bottom[k2], bottom[k]])
            faces.append(side); fcols.append(c)

        # 라벨(스텝 번호)
        # ax.text(cx, cy, cz + thickness*0.55, f"{i}", ha='center', va='bottom', fontsize=8)

    pc = Poly3DCollection(faces, facecolors=fcols, edgecolors='k', linewidths=0.5, alpha=0.9)
    # 면의 깊이 정렬 방식(가려짐 완화에 도움)
    pc.set_zsort('min')
    ax.add_collection3d(pc)

    xs = [c[0] for c in centers]; ys = [c[1] for c in centers]; zs_all = [c[2] for c in centers]
    xr = (min(xs)-0.8, max(xs)+0.8); yr = (min(ys)-0.8, max(ys)+0.8); zr = (min(zs_all)-0.3, max(zs_all)+0.3)
    ax.set_xlim(*xr); ax.set_ylim(*yr); ax.set_zlim(*zr)
    ax.set_box_aspect((xr[1]-xr[0], yr[1]-yr[0], zr[1]-zr[0]))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.zaxis.set_major_locator(MaxNLocator(nbins=5))

    # --- colorbar (그레이스케일용; 1번/마지막은 예외 색) ---
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('z [m] (height)')

    plt.tight_layout(); plt.show()



def stone_corners_3d(center: tuple[float,float,float], yaw_deg: float, tilt_x_deg: float, tilt_y_deg: float,
                      size=(0.35,0.25)) -> np.ndarray:
    """Return 4x3 array of stone corners in world frame (order around the rectangle)."""
    cx, cy, cz = center
    w, h = (size[0],size[1])
    local = np.array([[-w/2, -h/2, 0.0], [ w/2, -h/2, 0.0], [ w/2,  h/2, 0.0], [-w/2,  h/2, 0.0]], dtype=float)
    R = _Rz(math.radians(yaw_deg)) @ _Rx(math.radians(tilt_x_deg)) @ _Ry(math.radians(tilt_y_deg))
    pts = (local @ R.T) + np.array([[cx, cy, cz]])
    return pts





# ============================
# 7) Main (yaw 포함 6D 설정)
# ============================

def main():
    # --------- [A] 공통 세팅 / 하이퍼파라미터 ---------
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 데이터셋/로더
    DATA_LEN   = 20000      # 합성 데이터 개수
    SEED       = 123        # 재현용 시드
    BATCH_SIZE = 512
    EPOCHS     = 50
    LR         = 2e-4

    # CVAE 차원 (6D delta = [dx,dy,dz,dw,dh,dyaw], cond = prev2(6)+prev1(6)=12)
    COND_DIM = 12
    X_DIM    = 6
    Z_DIM    = 8

    # 시퀀스 생성 파라미터
    STEPS_GEN   = 10        # generate_sequence로 생성할 길이
    A_VARIANCE  = 0.5      # z ~ N(0, (1+a) I) 의 a (다양성↑: a>0, 보수적: -1<a<0)
    YAW0_DEG    = 0.0       # 절대 누적 시작 yaw (deg)
    START_CENTER= (0.0, 0.0, 0.0)  # 초기 스톤 중심
    START_SIZE  = (0.30, 0.30)     # 초기 스톤 크기 (w,h)
    THICKNESS   = 0.04      # 3D 블록 두께

    CKPT_PATH = "cvae_mapgen.pt"

    # --------- [B] 데이터셋 준비 (6D 버전) ---------
    ds = SyntheticStepDeltaDataset(length=DATA_LEN, seed=SEED)
    n_train = int(0.9 * len(ds))
    train_ds, val_ds = torch.utils.data.random_split(ds, [n_train, len(ds) - n_train])
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # --------- [C] 모델 구성 (6D에 맞춤) ---------
    model = CVAE(cond_dim=COND_DIM, x_dim=X_DIM, z_dim=Z_DIM)

    # --------- [D] 학습 ---------
    train_cvae(
        model, train_loader, val_loader,
        epochs=EPOCHS, lr=LR, device=device, ckpt_path=CKPT_PATH
    )

    # --------- [E] 최고 성능 체크포인트 로드 ---------
    ckpt = torch.load(CKPT_PATH, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.to(device)

    # --------- [F] 시드 델타 2개 설정 (필요에 따라 조정) ---------
    # d0/d1은 "델타"이므로, 절대 위치가 아니라 다음 스텝으로의 변화량(단위 m, deg)
    d0 = Step6(0.00, 0.00, 0.00, 0.00, 0.00, 0.00)  # 첫 변화량
    d1 = Step6(0.20, 0.00, 0.00, 0.00, 0.00, 0.00)  # 살짝 x 전진

    # --------- [G] 시퀀스 생성 (a로 샘플 다양성 조절) ---------
    seq = generate_sequence(
        model, d0, d1,
        steps=STEPS_GEN, a=A_VARIANCE, device=device
    )

    # --------- [H] 통계 출력 (6D 모두) ---------
    vals = {
        "dx [m]":   [p.dx   for p in seq[2:]],
        "dy [m]":   [p.dy   for p in seq[2:]],
        "dz [m]":   [p.dz   for p in seq[2:]],
        "dw [m]":   [p.dw   for p in seq[2:]],
        "dh [m]":   [p.dh   for p in seq[2:]],
        "dyaw [°]": [p.dyaw for p in seq[2:]],
    }
    for k, v in vals.items():
        if len(v) > 0:
            print(f"[gen] {k}: min={min(v):.3f} mean={sum(v)/len(v):.3f} max={max(v):.3f}")

    # --------- [I] 시각화 (yaw 반영)
    plot_sequence_3d_deltas(
        seq,
        center0=START_CENTER,
        size0=START_SIZE,
        thickness=THICKNESS,
        yaw0_deg=YAW0_DEG,  # 누적 시작 yaw
        title="Generated stepping stone",
        equalize_axes= True
    )


if __name__ == "__main__":
    main()
