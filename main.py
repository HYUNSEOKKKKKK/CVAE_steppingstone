from __future__ import annotations
import math
import os
import random
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from matplotlib.patches import Patch
from matplotlib.colors import Normalize
import json, math
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Iterable
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def vae_seq_to_terrain_json(seq: List["Step6"],
                            center0=(0.0,0.0,0.0),
                            size0=(0.30,0.30),
                            yaw0_deg=0.0,
                            terrain_key: str = "vae_run",
                            json_path: str | Path = "vae_terrain.json",
                            use_from: int = 2,
                            copy_counts: List[int] | None = None):
    """
    CVAE seq -> load_terrain() 호환 JSON 파일 생성
    - use_from=2 : 시드 델타 2개(d0,d1)를 제외한 실제 생성분만 지형에 반영
    - copy_counts: 필요 없으면 None. 쓰고 싶으면 len(stones)와 동일 길이로.
    """
    centers, sizes, yaws = deltas_to_absolute_6d(seq, center0, size0, yaw0_deg)

    stones: List[Dict[str, Any]] = []
    for i, ((cx,cy,cz), (w,h), yaws) in enumerate(zip(centers[use_from:], sizes[use_from:], yaws[use_from:]), start=1):
        stones.append({
            "name": f"stone_{i}",
            "center": [float(cx), float(cy), float(cz)],  # (x,y,z)
            "width":  float(w),
            "height": float(h),
            "yaw_deg": float(yaws),
            # NOTE: 스키마에는 yaw 없음. 필요하면 아래 확장 팁 참고.
        })

    # 파일 구조: { "<terrain_key>": {"stones": [...], "copy_counts": [...] } }
    obj = {
        terrain_key: {
            "stones": stones
        }
    }
    if copy_counts is not None:
        if len(copy_counts) != len(stones):
            raise ValueError(f"copy_counts length {len(copy_counts)} != stones {len(stones)}")
        obj[terrain_key]["copy_counts"] = [int(x) for x in copy_counts]

    json_path = Path(json_path)
    # 기존 파일이 있으면 합쳐서 갱신(동일 키는 덮어씀)
    if json_path.exists():
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                base = json.load(f)
            if not isinstance(base, dict):
                base = {}
        except Exception:
            base = {}
        base[terrain_key] = obj[terrain_key]
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(base, f, ensure_ascii=False, indent=2)
    else:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)

    return str(json_path), terrain_key



# 네가 쓰는 Step6: (dx,dy,dz,dw,dh,dyaw[deg]) 라고 가정
# class Step6: ...  # 이미 있겠지만, 여기서는 속성만 사용

def deltas_to_absolute_6d(seq: List["Step6"],
                          center0: Tuple[float,float,float],
                          size0: Tuple[float,float],
                          yaw0_deg: float = 0.0):
    """
    6D 델타 -> 절대치 (완전 독립 버전)
    - x,y: dx,dy를 회전 없이 월드 좌표계에서 그대로 누적
    - z: dz만 누적
    - yaw: dyaw만 누적 (x,y에 영향 없음)
    - w,h: dw,dh만 누적 (yaw/xy에 영향 없음)
    반환 길이: len(seq)
    """
    x, y, z = map(float, center0)
    w, h = map(float, size0)
    yaw_deg = float(yaw0_deg)

    centers, sizes, yaws = [], [], []
    MIN_W, MIN_H = 0.05, 0.05

    for s in seq:
        # 위치: 회전 영향 없이 독립
        x += float(s.dx)
        y += float(s.dy)
        z += float(s.dz)

        # 크기: 독립
        w = max(MIN_W, w + float(s.dw))
        h = max(MIN_H, h + float(s.dh))

        # 방향: 독립
        yaw_deg += float(s.dyaw)

        centers.append((x, y, z))
        sizes.append((w, h))
        yaws.append(yaw_deg)

    return centers, sizes, yaws

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
            dx   = g.uniform(0.30, 0.60)
            dy   = g.uniform(-0.15, 0.15)
            dz   = g.uniform(-0.1, 0.1)
            dw   = g.uniform(-0.06, 0.05)
            dh   = g.uniform(-0.06, 0.05)
            dyaw = g.uniform(-90, 90)   # °-90,90

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


# -------- rotation (z-axis) --------
def _Rz_deg(deg: float) -> np.ndarray:
    a = math.radians(deg)
    ca, sa = math.cos(a), math.sin(a)
    return np.array([[ca, -sa, 0.0],
                     [sa,  ca, 0.0],
                     [0.0, 0.0, 1.0]], dtype=float)

# -------- 3D box vertices (centered, rotated by yaw_deg) --------
def make_box_vertices(center: Tuple[float,float,float],
                      w: float, h: float, thickness: float,
                      yaw_deg: float) -> np.ndarray:
    """
    반환: (8,3) 꼭짓점. 순서는 아래 z-면 4개 + 위 z-면 4개
    내부 인덱스:
      아래면: 0-1-2-3 (시계/반시계 상관 없이 일관성 유지)
      윗면:   4-5-6-7 (아래면 각 꼭짓점 + z_off)
    """
    cx, cy, cz = center
    # 원점 기준 축정렬 직사각형의 평면 코너(아래면 z=0)
    half = np.array([[ -w/2, -h/2, 0.0],
                     [  w/2, -h/2, 0.0],
                     [  w/2,  h/2, 0.0],
                     [ -w/2,  h/2, 0.0]], dtype=float)

    # z축 회전
    Rz = _Rz_deg(yaw_deg)
    base_xy = (Rz @ half.T).T   # (4,3)

    # 아래면/윗면 z 오프셋
    z0 = cz - thickness/2
    z1 = cz + thickness/2

    bottom = base_xy + np.array([cx, cy, z0])
    top    = base_xy + np.array([cx, cy, z1])

    V = np.vstack([bottom, top])  # (8,3)
    return V

# -------- faces from vertices --------
def box_faces_from_vertices(V: np.ndarray) -> List[List[np.ndarray]]:
    """
    V: (8,3) 0..3=bottom, 4..7=top
    반환: 각 면을 이루는 꼭짓점 리스트들의 리스트 (Poly3DCollection용)
    """
    b0, b1, b2, b3 = V[0], V[1], V[2], V[3]
    t0, t1, t2, t3 = V[4], V[5], V[6], V[7]

    faces = [
        [b0, b1, b2, b3],  # bottom
        [t0, t1, t2, t3],  # top
        [b0, b1, t1, t0],  # side 1
        [b1, b2, t2, t1],  # side 2
        [b2, b3, t3, t2],  # side 3
        [b3, b0, t0, t3],  # side 4
    ]
    return faces

# -------- equal aspect for 3D --------
def set_axes_equal(ax):
    """3D 축에서 x,y,z 스케일 동일하게 맞춤"""
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    # z_limits = ax.get_zlim3d()
    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    # z_range = abs(z_limits[1] - z_limits[0])
    max_range = max([x_range, y_range])

    x_middle = np.mean(x_limits)
    y_middle = np.mean(y_limits)
    # z_middle = np.mean(z_limits)

    ax.set_xlim3d([x_middle - max_range/2, x_middle + max_range/2])
    ax.set_ylim3d([y_middle - max_range/2, y_middle + max_range/2])
    ax.set_zlim3d([-1.0, 1.0])


def plot_stones_3d_from_json(json_path: str | Path,
                             terrain_key: str,
                             thickness: float = 0.04,
                             face_alpha: float = 0.35,
                             edgecolor: str = 'k',
                             ax=None):
    json_path = Path(json_path)
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    block = data[terrain_key]
    stones: List[Dict[str,Any]] = block["stones"]

    created_ax = False
    if ax is None:
        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(111, projection='3d')
        created_ax = True

    n = len(stones)
    centers_z = np.array([s["center"][2] for s in stones], dtype=float)

    # 가운데 스톤 z 범위 (컬러바용)
    use_colorbar = False
    if n >= 3:
        mid_z = centers_z[1:-1]
        z_min, z_max = float(mid_z.min()), float(mid_z.max())
        use_colorbar = z_max > z_min
        if not use_colorbar:
            # 모두 같은 높이면 컬러바 대신 단일 회색으로 처리
            z_min, z_max = z_min - 0.5, z_max + 0.5
    else:
        # 스톤 2개 이하면 컬러바 생략
        z_min, z_max = 0.0, 1.0

    cmap = plt.get_cmap('gray')
    norm = Normalize(vmin=z_min-0.01, vmax=z_max+0.01)

    def gray_by_z(z):
        if n < 3:
            return (0.6, 0.6, 0.6, 1.0)
        # cm(gray) + Normalize로 일관된 컬러바 사용
        return cmap(norm(z))

    all_xyz = []
    for idx, s in enumerate(stones):
        cx, cy, cz = s["center"]
        w = float(s["width"])
        h = float(s["height"])
        yaw = float(s.get("yaw_deg", 0.0))

        # 색상 규칙: 첫=초록, 마지막=빨강, 가운데=z기반 그레이
        if idx == 0:
            facecolor = (0.0, 0.7, 0.2)      # 초록
        elif idx == n - 1:
            facecolor = (0.9, 0.1, 0.1)      # 빨강
        else:
            facecolor = gray_by_z(cz)

        V = make_box_vertices((cx,cy,cz), w, h, thickness, yaw)
        faces = box_faces_from_vertices(V)
        poly = Poly3DCollection(
            faces, alpha=face_alpha, edgecolor=edgecolor, facecolor=facecolor
        )
        ax.add_collection3d(poly)
        all_xyz.append(V)

        # # yaw 방향 화살표 (색을 스톤 색과 맞춤)
        # arrow_len = 0.5 * max(w, h)
        # dir_vec = (_Rz_deg(yaw) @ np.array([arrow_len, 0.0, 0.0])).reshape(3)
        # ax.quiver(cx, cy, cz, dir_vec[0], dir_vec[1], 0.0,
        #           length=1.0, normalize=False, color=facecolor)

    # 축 범위 자동 설정 (NumPy 2.0 호환)
    if all_xyz:
        P = np.vstack(all_xyz)  # (N*8, 3)
        rng_x = np.ptp(P[:, 0])
        rng_y = np.ptp(P[:, 1])
        rng_z = np.ptp(P[:, 2])
        pad = 0.1 * max(rng_x, rng_y, rng_z, 1e-6)

        ax.set_xlim(P[:, 0].min() - pad, P[:, 0].max() + pad)
        ax.set_ylim(P[:, 1].min() - pad, P[:, 1].max() + pad)
        ax.set_zlim(P[:, 2].min() - pad, P[:, 2].max() + pad)


    set_axes_equal(ax)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f" Generated Stepping Stones(VAE) ")
    ax.grid(False)
    ax.view_init(elev=30, azim=-110)
    # 범례(패치) + 가운데 스톤 그레이스케일 컬러바
    legend_patches = []
    if n >= 1:
        legend_patches.append(Patch(facecolor=(0.0, 0.7, 0.2), edgecolor='k', label='First stone'))
    if n >= 2:
        legend_patches.append(Patch(facecolor=(0.9, 0.1, 0.1), edgecolor='k', label='Last stone'))
    if n >= 3:
        # 가운데 스톤들은 컬러바로 설명
        if legend_patches:
            ax.legend(handles=legend_patches, loc='upper right')
        else:
            ax.legend(handles=[
                Patch(facecolor=(0.0, 0.7, 0.2), edgecolor='k', label='First stone'),
                Patch(facecolor=(0.9, 0.1, 0.1), edgecolor='k', label='Last stone'),
            ], loc='upper right')

        if use_colorbar:
            # 이전 플롯처럼 오른쪽에 높이 범례(컬러바) 추가
            # (Axes가 하나라면 fig를 ax.figure로 안전하게 획득)
            fig = ax.figure
            mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
            mappable.set_array([])  # Matplotlib 경고 방지용
            cbar = fig.colorbar(mappable, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Height (z)', rotation=90)
        else:
            # 모든 z가 동일하면 텍스트로 표기
            ax.text2D(0.98, 0.94, "Middle stones: constant height",
                      transform=ax.transAxes, ha='right', va='top')
    else:
        if legend_patches:
            ax.legend(handles=legend_patches, loc='upper right')

    if created_ax:
        plt.tight_layout()
        plt.show()


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
    STEPS_GEN   = 4        # generate_sequence로 생성할 길이
    A_VARIANCE  = 0.0      # z ~ N(0, (1+a) I) 의 a (다양성↑: a>0, 보수적: -1<a<0)
    YAW0_DEG    = 0.0       # 절대 누적 시작 yaw (deg)
    START_CENTER= (0.0, 2.5, 0.0)  # 초기 스톤 중심
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
    d1 = Step6(0.40, 0.00, 0.00, 0.00, 0.00, 0.00)  # 살짝 x 전진

    # --------- [G] 시퀀스 생성 (a로 샘플 다양성 조절) ---------
    seq = generate_sequence(
        model, d0, d1,
        steps=STEPS_GEN, a=A_VARIANCE, device=device
    )

    # --------- [G2] CVAE_steppingstone 결과를 terrain JSON으로 저장 ---------
    json_path, terrain_key = vae_seq_to_terrain_json(
        seq,
        center0=START_CENTER,
        size0=START_SIZE,
        yaw0_deg=YAW0_DEG,
        terrain_key="vae_run",  # 원하면 런마다 다른 이름 사용
        json_path="vae_terrain.json",  # 경로 고정/변경 자유
        use_from=2,  # d0,d1 제외
        copy_counts=None  # 필요 시 [1]*N 등으로 전달
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
    plot_stones_3d_from_json("vae_terrain.json", terrain_key="vae_run",
                             thickness=0.04, face_alpha=0.95, edgecolor='k')


if __name__ == "__main__":
    main()
