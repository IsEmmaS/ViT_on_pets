import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer


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
        self.head = nn.Linear(embed_dim, num_classes)

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
        B = x.shape[0]

        # 分块嵌入
        x = self.patch_embed(x)  # (B, n_patches, embed_dim)

        # 添加CLS token
        cls_token = self.cls_token.expand(B, -1, -1)  # (B, 1, embed_dim)
        x = torch.cat([cls_token, x], dim=1)  # (B, n_patches+1, embed_dim)

        # 添加位置编码
        x += self.pos_embed

        # 调整维度以适应Transformer输入 (seq_len, B, embed_dim)
        x = x.permute(1, 0, 2)

        # 通过Transformer编码器
        for block in self.blocks:
            x = block(x)

        # 恢复维度并应用LayerNorm
        x = x.permute(1, 0, 2)  # (B, seq_len, embed_dim)
        x = self.norm(x)

        # 取CLS token输出并分类
        cls_out = x[:, 0]
        return self.head(cls_out)


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
    print(out.shape)  # 输出: torch.Size([2, 1000])
    print(vit.modules)
