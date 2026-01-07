import torch
import numpy as np
from typing import Tuple, List, Optional


class LayeredTreeGenerator:
    """分层点生成器（树状展开）

    生成规则：
    - 第一层：初始点固定为坐标原点（0 向量）
    - 每一层对上一层的每个点，分别在该点的单位球上采样两个子点（child = parent + unit_vector）
    - 对于长度>=3 的链 x1->x2->x3 要求 dot(x3-x1, x2-x1) >= 0（cos >= 0），保证子树向外展开
    - 在每次生成新一层后，计算当前所有点之间的两两距离，若 |dist - 1| <= tol 则记为一条需要连线的点对
    - 最终返回所有生成点的位置和不超过 max_pair_num 的点对（截断以保证批次大小一致）

    返回：
    - p_batch: Tensor (num_points, dim)
    - pairs: LongTensor (2, num_pairs) 对应点索引
    """

    def __init__(self,
                 dim: int = 2,
                 device: str = "cpu",
                 tol: float = 1e-7,
                 seed: Optional[int] = None):
        self.dim = dim
        self.device = torch.device(device)
        self.tol = tol
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

    def _sample_unit_vector(self, n: int) -> torch.Tensor:
        v = torch.randn(n, self.dim, device=self.device)
        v = v / (v.norm(dim=1, keepdim=True) + 1e-12)
        return v

    def generate(self, k: int, max_pair_num: int, init_point: Optional[torch.Tensor] = None,
                 max_attempts: int = 50) -> Tuple[torch.Tensor, torch.LongTensor]:
        """生成 k 层（包含第一层），返回点集与点对关系。

        Args:
            k: 层数（>=1），第一层只有一个点（原点）
            max_pair_num: 返回的最大点对数量（多则截断）
            init_point: （已忽略）初始点固定为原点
            max_attempts: 为满足角度约束重采样的最大尝试次数

        Returns:
            p_batch: Tensor (num_points, dim)
            pairs: LongTensor (2, num_pairs)
        """
        assert k >= 1, "k must be >= 1"

        points: List[torch.Tensor] = []
        layers: List[List[int]] = []  # 每层点在 points 列表中的索引

        # 第一层：初始化点（固定为原点）
        p0 = torch.zeros(1, self.dim, device=self.device, dtype=torch.float32)
        points.append(p0[0])
        layers.append([0])

        # 逐层生成
        for layer_idx in range(2, k + 1):
            prev_indices = layers[-1]
            new_indices = []

            for idx in prev_indices:
                parent = points[idx]

                # 为该 parent 生成两个 child
                children_needed = 2
                attempts = 0
                generated = 0
                while generated < children_needed:
                    attempts += 1
                    if attempts > max_attempts * children_needed:
                        # 安全退出，防止死循环
                        break

                    v = self._sample_unit_vector(1)[0]
                    child = parent + v

                    # 相对于原点的角度约束：dot(child - x0, parent - x0) >= 0
                    # 由于 x0 为原点，条件简化为 dot(child, parent) >= 0
                    ok = True
                    if (child @ parent) < 0:
                        ok = False

                    if ok:
                        points.append(child)
                        new_indices.append(len(points) - 1)
                        generated += 1

            layers.append(new_indices)

        # 合并为 tensor
        p_batch = torch.stack(points, dim=0)

        # 计算所有点对中满足 |dist - 1| <= tol 的对
        # 仅 i<j
        n = p_batch.shape[0]
        if n == 0:
            return p_batch, torch.zeros((2, 0), dtype=torch.long)

        # pairwise distances
        diff = p_batch.unsqueeze(1) - p_batch.unsqueeze(0)  # (n,n,d)
        dists = torch.norm(diff, dim=2)
        mask = torch.abs(dists - 1.0) <= self.tol

        # remove diagonal
        mask.fill_diagonal_(False)

        src, dst = torch.nonzero(mask, as_tuple=True)
        # keep only i<j to avoid duplicate undirected pairs
        keep = src < dst
        src = src[keep]
        dst = dst[keep]

        if src.numel() == 0:
            pairs = torch.zeros((2, 0), dtype=torch.long)
        else:
            pairs = torch.stack([src, dst], dim=0)

        # 截断至 max_pair_num
        if pairs.shape[1] > max_pair_num:
            pairs = pairs[:, :max_pair_num]

        return p_batch, pairs


if __name__ == "__main__":
    gen = LayeredTreeGenerator(dim=2, device="cpu", seed=42)
    pts, pairs = gen.generate(k=4, max_pair_num=100)
    print('num_points=', pts.shape[0])
    print('num_pairs=', pairs.shape[1])
