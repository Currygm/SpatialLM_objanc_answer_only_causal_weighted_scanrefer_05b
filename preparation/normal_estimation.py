"""Original radius-neighborhood PCA normals, also supporting CPU."""
import numpy as np
def _estimate_normals(xyz: np.ndarray, device: str, radius: float, max_nn: int) -> np.ndarray:
    import torch
    from scipy.spatial import cKDTree

    coordinates = np.ascontiguousarray(xyz, dtype=np.float32)
    tree = cKDTree(coordinates)
    distances, indices = tree.query(
        coordinates,
        k=max_nn,
        distance_upper_bound=radius,
        workers=-1,
    )
    device = device.lower()
    output = np.empty_like(coordinates)
    chunk_size = 32768
    for start in range(0, len(coordinates), chunk_size):
        end = min(start + chunk_size, len(coordinates))
        chunk_indices = indices[start:end]
        valid = np.isfinite(distances[start:end]) & (chunk_indices < len(coordinates))
        safe_indices = np.where(valid, chunk_indices, 0)
        neighbors = torch.as_tensor(
            coordinates[safe_indices], device=device, dtype=torch.float32
        )
        mask = torch.as_tensor(valid, device=device, dtype=torch.float32)
        counts = mask.sum(dim=1, keepdim=True)
        mean = (neighbors * mask.unsqueeze(-1)).sum(dim=1) / counts.clamp_min(1.0)
        centered = (neighbors - mean.unsqueeze(1)) * mask.unsqueeze(-1)
        covariance = torch.einsum("nki,nkj->nij", centered, centered)
        covariance = covariance / (counts - 1.0).clamp_min(1.0).unsqueeze(-1)
        _, eigenvectors = torch.linalg.eigh(covariance)
        normals = eigenvectors[:, :, 0]
        sparse = counts[:, 0] < 3
        normals[sparse] = normals.new_tensor([0.0, 0.0, 1.0])
        output[start:end] = normals.cpu().numpy()
    return output
