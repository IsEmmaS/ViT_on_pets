import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import argparse
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from model import VisionTransformer
from dataloader import myDataset
from tqdm import tqdm


def m_train(args):
    if args.local_rank >= 0:
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend="nccl")
        ddp = True
    else:
        ddp = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_files = [
        f for f in os.listdir(args.data_dir) if f.endswith((".jpg", ".jpeg", ".png"))
    ]
    all_file_paths = [os.path.join(args.data_dir, f) for f in all_files]

    train_size = int(0.8 * len(all_file_paths))
    train_files = all_file_paths[:train_size]
    val_files = all_file_paths[train_size:]

    train_transform = transforms.Compose(
        [
            transforms.Resize((args.images_size, args.images_size)),
            transforms.RandomResizedCrop(args.images_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.Resize((args.images_size, args.images_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_dataset = myDataset(train_files, transform=train_transform)
    test_dataset = myDataset(val_files, transform=val_transform)

    # 数据加载器
    if ddp:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        train_loader = DataLoader(train_dataset, batch_size=32, sampler=train_sampler)
    else:
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000)

    model = VisionTransformer().to(device)
    criterion = nn.CrossEntropyLoss()
    if ddp:
        model = DDP(model, device_ids=[args.local_rank])
        gpu_num = torch.distributed.get_world_size()
    else:
        gpu_num = 1

    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr * gpu_num, weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)

    # train loop
    best_test_acc = 0.0
    for epoch in range(args.epochs):
        if ddp:
            train_sampler.set_epoch(epoch)
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
        for data, target in train_loader_tqdm:
            data, target = data.to(device), target.to(device)

            with torch.amp.autocast("cuda", enabled=args.amp):
                output = model(data)
                loss = criterion(output, target)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            train_loss += loss.item() * data.size(0)
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

            postfix = {
                "loss": f"{loss.item():.4f}",
                "lr": f"{optimizer.param_groups[0]['lr']}",
            }
            train_loader_tqdm.set_postfix(postfix)

        scheduler.step()

        avg_train_loss = train_loss / len(train_loader.dataset)
        train_acc = correct / total

    # test loop
    test_loss = 0.0
    correct = 0
    total = 0

    if not ddp or (ddp and dist.get_rank() == 0):
        model.eval()
        with torch.no_grad():
            test_loader_tqdm = tqdm(test_loader, desc="Testing···", leave=False)
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)

                test_loss += criterion(output, target).item() * data.size(0)
                _, pred = output.max(1)
                total += data.size(0)
                correct += pred.eq(target).sum().item()

                postfix = {"loss": f"{loss.item():.4f}"}
                test_loader_tqdm.set_postfix(postfix)

        avg_test_loss = test_loss / len(test_loader.dataset)
        test_acc = correct / total

        print(
            f"Epoch [{epoch+1}/{args.epochs}] | "
            f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {avg_test_loss:.4f} | Val Acc: {test_acc:.4f}"
            f"Lr: {optimizer.param_groups[0]['lr']:.2e}"
        )
    if not ddp or (ddp and dist.get_rank() == 0):
        if test_acc > best_test_acc:
            torch.save(model.state_dict(), "best_model.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument(
        "--data_dir", type=str, default="images", help="Path to the data directory"
    )
    parser.add_argument("--image_size", type=int, default=224, help="Image size")
    parser.add_argument("--patch_size", type=int, default=16, help="Patch size")
    parser.add_argument("--num_classes", type=int, default=37, help="Number of classes")
    parser.add_argument(
        "--embed_dim", type=int, default=768, help="Embedding dimension"
    )
    parser.add_argument(
        "--depth", type=int, default=32, help="Number of Transformer blocks"
    )
    parser.add_argument(
        "--num_heads", type=int, default=12, help="Number of attention heads"
    )
    parser.add_argument("--mlp_ratio", type=float, default=4.0, help="MLP layer ratio")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of workers for data loading"
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument(
        "--amp", action="store_true", help="Enable mixed precision training"
    )
    args = parser.parse_args()

    m_train(args)
