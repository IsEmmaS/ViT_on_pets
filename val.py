import argparse
import torch
import os

from tqdm import tqdm
from dataloader import create_dataloaders
from model import VisionTransformer


def val(args):
    _, val_loader = create_dataloaders(
        data_dir=args.data_dir,
        img_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VisionTransformer(
        img_size=args.image_size,
        patch_size=args.patch_size,
        num_classes=args.num_classes,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
    )
    model.to(device)
    model.eval()
    if os.path.exists("best_model.pth"):
        model.load_state_dict(
            torch.load("best_model.pth", map_location=device, weights_only=True)
        )
        print("Loaded pretrained weights from best_model.pth")

    with torch.no_grad():
        val_loader_tqdm = tqdm(val_loader, desc="Validating", leave=True)

        total = 0
        correct = 0

        for images, labels in val_loader_tqdm:
            images, labels = images.to(device), labels.to(device)

            with torch.amp.autocast(device_type='cuda', enabled=True):
                output = model(images)

            _, pred = output.max(1)
            correct += pred.eq(labels).sum().item()

            total += labels.size(0)
        val_acc = correct / total
    print(f"Val [correct / total = val acc]:[{correct} / {total} = {val_acc:.4f} %]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, default="images", help="Path to the data directory")
    parser.add_argument("--image_size", type=int, default=224, help="Image size")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading")

    parser.add_argument("--patch_size", type=int, default=16, help="Patch size")
    parser.add_argument("--num_classes", type=int, default=37, help="Number of classes")
    parser.add_argument("--embed_dim", type=int, default=768, help="Embedding dimension")
    parser.add_argument("--depth", type=int, default=32, help="Number of Transformer blocks")
    parser.add_argument("--num_heads", type=int, default=12, help="Number of attention heads")
    parser.add_argument("--mlp_ratio", type=float, default=4.0, help="MLP layer ratio")

    args = parser.parse_args()

    val(args)
