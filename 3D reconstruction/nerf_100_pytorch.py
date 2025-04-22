import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import math


class NerfModel(nn.Module):
    def __init__(self, embedding_dim_pos=10, embedding_dim_direction=4, hidden_dim=128):   
        super(NerfModel, self).__init__()
        
        self.block1 = nn.Sequential(nn.Linear(embedding_dim_pos * 6 + 3, hidden_dim), nn.ReLU(),
                                    nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                    nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                    nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), )
        # density estimation
        self.block2 = nn.Sequential(nn.Linear(embedding_dim_pos * 6 + hidden_dim + 3, hidden_dim), nn.ReLU(),
                                    nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                    nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                    nn.Linear(hidden_dim, hidden_dim + 1), )
        # color estimation
        self.block3 = nn.Sequential(nn.Linear(embedding_dim_direction * 6 + hidden_dim + 3, hidden_dim // 2), nn.ReLU(), )
        self.block4 = nn.Sequential(nn.Linear(hidden_dim // 2, 3), nn.Sigmoid(), )

        self.embedding_dim_pos = embedding_dim_pos
        self.embedding_dim_direction = embedding_dim_direction
        self.relu = nn.ReLU()

    @staticmethod
    def positional_encoding(x, L):
        out = [x]
        for j in range(L):
            out.append(torch.sin(2 ** j *math.pi * x))
            out.append(torch.cos(2 ** j *math.pi * x))
        return torch.cat(out, dim=1)

    def forward(self, o, d):
        emb_x = self.positional_encoding(o, self.embedding_dim_pos) # emb_x: [batch_size, embedding_dim_pos * 6]
        emb_d = self.positional_encoding(d, self.embedding_dim_direction) # emb_d: [batch_size, embedding_dim_direction * 6]
        h = self.block1(emb_x) # h: [batch_size, hidden_dim]
        tmp = self.block2(torch.cat((h, emb_x), dim=1)) # tmp: [batch_size, hidden_dim + 1]
        h, sigma = tmp[:, :-1], self.relu(tmp[:, -1]) # h: [batch_size, hidden_dim], sigma: [batch_size]
        h = self.block3(torch.cat((h, emb_d), dim=1)) # h: [batch_size, hidden_dim // 2]
        c = self.block4(h) # c: [batch_size, 3]
        return c, sigma


def compute_accumulated_transmittance(alphas):
    accumulated_transmittance = torch.cumprod(alphas, 1)
    return torch.cat((torch.ones((accumulated_transmittance.shape[0], 1), device=alphas.device),
                      accumulated_transmittance[:, :-1]), dim=-1)

@torch.no_grad()
def render_test(fine_model, coarse_model, dataset, hn=0, hf=0.5, chunk_size=10, img_index=0, nb_bins=192, H=400, W=400):
    ray_origins = dataset[img_index * H * W: (img_index + 1) * H * W, :3]
    ray_directions = dataset[img_index * H * W: (img_index + 1) * H * W, 3:6]
    data = []
    for i in range(int(np.ceil(H / chunk_size))):
        ro_chunk = ray_origins[i * W * chunk_size: (i + 1) * W * chunk_size].to(fine_model.block1[0].weight.device)
        rd_chunk = ray_directions[i * W * chunk_size: (i + 1) * W * chunk_size].to(fine_model.block1[0].weight.device)
        _, rgb = render_rays_hierarchical(coarse_model, fine_model, ro_chunk, rd_chunk, hn, hf)
        data.append(rgb)
    img = torch.cat(data).detach().cpu().numpy().reshape(H, W, 3)
    plt.imshow(img)
    plt.axis('off')
    plt.savefig(f'novel_views/img_{img_index}.png', bbox_inches='tight')
    plt.close()


def render_rays_hierarchical(coarse_model, fine_model, ray_origins, ray_directions,
                             hn=0, hf=1, nb_coarse=64, nb_fine=128):
    device = ray_origins.device
    batch_size = ray_origins.shape[0]

    # --- Coarse pass ---
    t_coarse = torch.linspace(hn, hf, nb_coarse, device=device).expand(batch_size, nb_coarse)
    mid = (t_coarse[:, :-1] + t_coarse[:, 1:]) / 2.
    lower = torch.cat((t_coarse[:, :1], mid), -1)
    upper = torch.cat((mid, t_coarse[:, -1:]), -1)
    u = torch.rand(t_coarse.shape, device=device)
    t_coarse = lower + (upper - lower) * u
    delta_coarse = torch.cat((t_coarse[:, 1:] - t_coarse[:, :-1],
                              torch.tensor([1e10], device=device).expand(batch_size, 1)), -1)

    x_coarse = ray_origins.unsqueeze(1) + t_coarse.unsqueeze(2) * ray_directions.unsqueeze(1)
    ray_d_exp = ray_directions.unsqueeze(1).expand_as(x_coarse)

    c_coarse, sigma_coarse = coarse_model(x_coarse.reshape(-1, 3), ray_d_exp.reshape(-1, 3))
    c_coarse = c_coarse.reshape(batch_size, nb_coarse, 3)
    sigma_coarse = sigma_coarse.reshape(batch_size, nb_coarse)

    alpha = 1 - torch.exp(-sigma_coarse * delta_coarse)
    weights = compute_accumulated_transmittance(1 - alpha).unsqueeze(2) * alpha.unsqueeze(2)
    rgb_coarse = (weights * c_coarse).sum(dim=1)
    weight_sum = weights.sum(-1).sum(-1)
    rgb_coarse += 1 - weight_sum.unsqueeze(-1)

    # --- Importance sampling for fine pass ---
    with torch.no_grad():
        weights_pdf = (weights.squeeze(-1) + 1e-5)
        pdf = weights_pdf / torch.sum(weights_pdf, dim=-1, keepdim=True)
        cdf = torch.cumsum(pdf, dim=-1)
        cdf = torch.cat([torch.zeros_like(cdf[:, :1]), cdf], dim=-1)

        u_fine = torch.rand(batch_size, nb_fine, device=device)
        inds = torch.searchsorted(cdf, u_fine, right=True)
        below = torch.clamp(inds - 1, min=0)
        above = torch.clamp(inds, max=cdf.shape[-1] - 1)
        inds_g = torch.stack([below, above], dim=-1)

        cdf_g = torch.gather(cdf.unsqueeze(1).expand(-1, nb_fine, -1), 2, inds_g)
        bins = torch.linspace(hn, hf, nb_coarse, device=device)
        bins_g = torch.gather(bins.unsqueeze(0).unsqueeze(1).expand(-1, nb_fine, -1), 2, inds_g)

        denom = cdf_g[:, :, 1] - cdf_g[:, :, 0]
        denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
        t_fine = bins_g[:, :, 0] + (u_fine - cdf_g[:, :, 0]) / denom * (bins_g[:, :, 1] - bins_g[:, :, 0])

    t_all = torch.cat([t_coarse, t_fine], dim=-1)
    t_all, _ = torch.sort(t_all, dim=-1)
    delta = torch.cat((t_all[:, 1:] - t_all[:, :-1],
                       torch.tensor([1e10], device=device).expand(batch_size, 1)), -1)

    x_fine = ray_origins.unsqueeze(1) + t_all.unsqueeze(2) * ray_directions.unsqueeze(1)
    ray_d_exp = ray_directions.unsqueeze(1).expand_as(x_fine)

    c_fine, sigma_fine = fine_model(x_fine.reshape(-1, 3), ray_d_exp.reshape(-1, 3))
    c_fine = c_fine.reshape(batch_size, -1, 3)
    sigma_fine = sigma_fine.reshape(batch_size, -1)

    alpha = 1 - torch.exp(-sigma_fine * delta)
    weights = compute_accumulated_transmittance(1 - alpha).unsqueeze(2) * alpha.unsqueeze(2)
    rgb_fine = (weights * c_fine).sum(dim=1)
    weight_sum = weights.sum(-1).sum(-1)
    rgb_fine += 1 - weight_sum.unsqueeze(-1)

    return rgb_coarse, rgb_fine

def train_hierarchical(coarse_model, fine_model, optimizer, scheduler, data_loader, device='cpu', hn=0, hf=1, H=400, W=400, epochs=10):
    for _ in tqdm(range(epochs)):
        for batch in data_loader:
            ray_origins = batch[:, :3].to(device)
            ray_directions = batch[:, 3:6].to(device)
            ground_truth = batch[:, 6:].to(device)
            rgb_coarse, rgb_fine = render_rays_hierarchical(coarse_model, fine_model, ray_origins, ray_directions, hn, hf)
            loss = ((rgb_fine - ground_truth) ** 2).mean()
            # Coarse loss uses detached GT for matching length
            loss_coarse = ((rgb_coarse - ground_truth.detach()) ** 2).mean()
            loss += loss_coarse
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
        for img_index in range(2):
            render_test(fine_model, coarse_model, testing_dataset, hn, hf, img_index=img_index, H=H, W=W)



if __name__ == '__main__':
    device = 'cuda'
    
    training_dataset = torch.from_numpy(np.load('training_data.pkl', allow_pickle=True))
    testing_dataset = torch.from_numpy(np.load('testing_data.pkl', allow_pickle=True))

    coarse_model = NerfModel(hidden_dim=256).to(device)
    fine_model = NerfModel(hidden_dim=256).to(device)
    optimizer = torch.optim.Adam(list(coarse_model.parameters()) + list(fine_model.parameters()), lr=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[4, 8], gamma=0.5)
    data_loader = DataLoader(training_dataset, batch_size=1024, shuffle=True)

    train_hierarchical(coarse_model, fine_model, optimizer, scheduler, data_loader, device=device, hn=2, hf=6, epochs=16, H=400, W=400)
