import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import TransformerEncoderLayer
from torch.utils.checkpoint import checkpoint


class PatchEmbedding(nn.Module):
    """将图像分割成块并嵌入到向量空间"""

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        # 输入形状: (B, C, H, W)
        x = self.proj(x)  # (B, E, H/P, W/P)
        x = x.flatten(2)  # (B, E, N)
        x = x.transpose(1, 2)  # (B, N, E)
        return x


class VisionTransformer(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=1000,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
    ):
        super().__init__()

        # 分块嵌入
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_chans, embed_dim)
        n_patches = self.patch_embed.n_patches

        # CLS token和位置编码
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))

        # Transformer编码器
        self.blocks = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    d_model=embed_dim,
                    nhead=num_heads,
                    dim_feedforward=int(embed_dim * mlp_ratio),
                    dropout=0.1,
                    activation="gelu",
                    norm_first=True,  # Pre-LN结构
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 512),      
            nn.GELU(),
            nn.Dropout(0.6),                
            nn.Linear(512, num_classes)
        )
        self.depth = depth

        # 初始化参数
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    def forward(self, x):
        B, _, _, _ = x.shape

        # 分块嵌入
        x = self.patch_embed(x)  # (B, n_patches, embed_dim)
        
        pos_embed = self.pos_embed

        # 添加CLS token
        cls_token = self.cls_token.expand(B, -1, -1)  # (B, 1, embed_dim)
        x = torch.cat([cls_token, x], dim=1)  # (B, n_patches+1, embed_dim)

        # 添加位置编码
        x += pos_embed

        # 调整维度以适应Transformer输入 (seq_len, B, embed_dim)
        x = x.permute(1, 0, 2)

        # 通过Transformer编码器
        for i, block in enumerate(self.blocks):
            if self.training and i > 2:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
            # Each block has shape : (len of seq + cls token, batch size, hidden dim size)
            # print(f'Transformer {i + 1} layers output shape is :',x.shape)

        # 恢复维度并应用LayerNorm
        x = x.permute(1, 0, 2)  # (B, seq_len, embed_dim)
        x = self.norm(x)

        # 取CLS token输出并分类
        cls_out = x[:, 0]
        return self.head(cls_out)


class FocalLoss(nn.Module):
    """This is a FocalLoss class which can caculate
    Focal Loss.
    
    Focal loss = -alpha_t * (1 - p_t)^gamma * log(p_t)

    where p_t is the predicted probability of the target class and
    alpha_t is the weight for the target class.

    Args:
        alpha: float, optional
            The weight for the target class. Default is 0.25.
        gamma: float, optional
            The focusing parameter. Default is 2.0.
    """
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        BCE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        return loss.mean()

def l1_regularization(model, lambda_l1=0.01):
    """
    Computes the L1 regularization loss for a given PyTorch model.

    L1 regularization adds a penalty equal to the absolute value of the magnitude
    of coefficients to the loss function. This encourages the model to keep the
    weights small, promoting sparsity in the model parameters.

    The L1 regularization loss is calculated as:
    L1_loss = lambda_l1 * sum(abs(param)) for all parameters in the model.

    Args:
        model (torch.nn.Module): The PyTorch model for which to compute the L1 regularization loss.
        lambda_l1 (float, optional): The regularization strength (lambda). Defaults to 0.01.

    Returns:
        torch.Tensor: The L1 regularization loss.

    Raises:
        TypeError: If `model` is not an instance of `torch.nn.Module`.

    Example:
        >>> import torch
        >>> import torch.nn as nn
        >>> model = nn.Linear(10, 1)  # A simple linear model
        >>> l1_loss = l1_regularization(model, lambda_l1=0.05)
        >>> print(l1_loss)
        tensor(0.1234)  # Example output

    Note:
        This function assumes that the model's parameters are accessible via `model.parameters()`.
        The regularization strength `lambda_l1` controls the trade-off between the original loss
        and the regularization term. A higher value of `lambda_l1` will result in stronger regularization.

    See Also:
        torch.nn.Module: The base class for all neural network modules in PyTorch.
        torch.nn.utils.weight_norm: A utility for applying weight normalization to model parameters.
    """
    l1_loss = 0
    for param in model.parameters():
        l1_loss += torch.sum(torch.abs(param))
    return lambda_l1 * l1_loss

if __name__ == "__main__":
    vit = VisionTransformer(
        img_size=224,
        patch_size=16,
        num_classes=1000,
        embed_dim=768,
        depth=12,
        num_heads=12,
    )
    x = torch.randn(2, 3, 224, 224)
    out = vit(x)
    print(out.shape)  # shoudl be [2, 1000]
    print(vit.modules)
