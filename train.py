import torch
import torch.nn as nn
import torch.optim as optim

import argparse


from tqdm import tqdm
from dataloader import create_dataloaders
from model import VisionTransformer, FocalLoss, l1_regularization


def train(args):
    # 加载数据
    train_loader, val_loader = create_dataloaders(
        data_dir=args.data_dir,
        img_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # 检查 GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 定义模型
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

    # 定义损失函数、优化器和学习率调度器
    criterion = FocalLoss(alpha=0.5, gamma=1.5)
    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-7
    )

    # 混合精度训练支持
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)

    # training loop
    best_val_acc = 0.0
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        train_loader_tqdm = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{args.epochs}",
            leave=False,
            dynamic_ncols=True,
        )

        for images, labels in train_loader_tqdm:
            images, labels = images.to(device), labels.to(device)

            with torch.amp.autocast("cuda", enabled=args.amp):
                outputs = model(images)
                f_loss = criterion(outputs, labels)
                l1_loss = l1_regularization(model, lambda_l1=args.lambda_l1)
                loss = f_loss + l1_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            train_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            postfix = {
                "loss": f"{loss.item():.4f}",
                "lr": f'{optimizer.param_groups[0]["lr"]:.2e}',
            }
            train_loader_tqdm.set_postfix(postfix)

        scheduler.step()

        avg_train_loss = train_loss / len(train_loader.dataset)
        train_accuracy = correct / total

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            val_loader_tqdm = tqdm(val_loader, desc='Validating', leave=False)
            for images, labels in val_loader_tqdm:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                postfix = {
                'loss': f'{loss.item():.4f}',
                }
                val_loader_tqdm.set_postfix(postfix)

        avg_val_loss = val_loss / len(val_loader.dataset)
        val_accuracy = correct / total

        print(
            f"Epoch [{epoch+1}/{args.epochs}] | "
            f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_accuracy:.4f} | "
            f"L1 Loss: {l1_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.4f} |"
            f"Lr: {optimizer.param_groups[0]['lr']:.2e}"
        )

        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            torch.save(model.state_dict(), f"best_model.pth")

    print(f"Best validation accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="images", help="Path to the data directory")
    parser.add_argument("--image_size", type=int, default=224, help="Image size")
    parser.add_argument("--patch_size", type=int, default=16, help="Patch size")
    parser.add_argument("--num_classes", type=int, default=37, help="Number of classes")
    parser.add_argument("--embed_dim", type=int, default=768, help="Embedding dimension")
    parser.add_argument("--depth", type=int, default=32, help="Number of Transformer blocks")
    parser.add_argument("--num_heads", type=int, default=12, help="Number of attention heads")
    parser.add_argument("--mlp_ratio", type=float, default=4.0, help="MLP layer ratio")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading")
    parser.add_argument("--lr", type=float, default=3e-5, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision training")
    parser.add_argument("--lambda_l1", type=float, default=1e-3, help="Strangth of L1 regu")
    args = parser.parse_args()

    train(args)
