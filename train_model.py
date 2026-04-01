import argparse
import json
import logging
import os
import time
from datetime import timedelta

import torch
import torch.nn as nn
import torchvision.transforms as transforms

from src.data.dataset import HAM10000Dataset
from src.data.sampler import BalancedBatchSampler, UnderSampler
from src.engine.train import train_one_epoch
from src.engine.evaluate import evaluate
from src.loss.criterion import OhemCrossEntropy, RecallCrossEntropy, FocalLoss, labels_to_class_weights
from src.models.skin_cancer_classifier import SkinLesionClassifier

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger()


def parse_args():
    parser = argparse.ArgumentParser(description='HAM10000 skin lesion classifier')

    # Config file — loaded first; individual CLI flags override any key inside it
    parser.add_argument('--config', default=None, help='Path to JSON config file')

    # Data
    parser.add_argument('--data_path', default=None, help='Root directory of the HAM10000 dataset')
    parser.add_argument('--metadata_csv', default='HAM10000_metadata.csv')
    parser.add_argument('--image_dirs', nargs='+', default=None,
                        help='Image sub-directories (auto-detected when omitted)')
    parser.add_argument('--val_split', type=float, default=0.2)

    # Model
    parser.add_argument('--model', default='resnet50',
                        help='Backbone name: resnet18/50/101, efficientnet_b*, convnext_*')
    parser.add_argument('--num_classes', type=int, default=7)
    parser.add_argument('--pretrained', action='store_true', default=True)
    parser.add_argument('--input_size', type=int, default=224)

    # Training
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda')

    # Loss
    parser.add_argument('--log_interval', type=int, default=100,
                        help='Print training progress every N iterations')

    # Loss
    parser.add_argument('--loss', default='cross_entropy',
                        choices=['cross_entropy', 'weighted_cross_entropy', 'ohem', 'recall_ce', 'focal'])
    parser.add_argument('--ifw', action='store_true',
                        help='Apply inverse-frequency class weights to the loss')

    # Sampler
    parser.add_argument('--sampler', default=None, choices=['balanced', 'undersample'])
    parser.add_argument('--undersample_rate', type=float, default=0.2,
                        help='Fraction of majority-class samples kept when undersampling')

    # Checkpoint / output
    parser.add_argument('--output_dir', default='output')
    parser.add_argument('--checkpoint', default=None,
                        help='Path to a checkpoint to resume training from')
    parser.add_argument('--start_epoch', type=int, default=0)

    args, _ = parser.parse_known_args()
    if args.config:
        with open(args.config) as f:
            cfg = json.load(f)
        parser.set_defaults(**cfg)

    args = parser.parse_args()
    if args.data_path is None:
        parser.error('--data_path is required (provide it via --config or directly on the CLI)')
    return args


def train(args):
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device: {device}')

    # ── Transforms ───────────────────────────────────────────────────────────
    train_transform = transforms.Compose([
        transforms.Resize((args.input_size, args.input_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((args.input_size, args.input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # ── Datasets ─────────────────────────────────────────────────────────────
    train_dataset = HAM10000Dataset(
        args.data_path, metadata_csv=args.metadata_csv,
        split='train', val_split=args.val_split, random_state=args.seed,
        transform=train_transform, image_dirs=args.image_dirs["trainval"] if args.image_dirs else None,
    )
    val_dataset = HAM10000Dataset(
        args.data_path, metadata_csv=args.metadata_csv,
        split='val', val_split=args.val_split, random_state=args.seed,
        transform=val_transform, image_dirs=args.image_dirs["trainval"] if args.image_dirs else None,
    )
    test_dataset = HAM10000Dataset(
        args.data_path, metadata_csv=args.test_metadata_csv,
        split='test', val_split=None, random_state=args.seed,
        transform=val_transform, image_dirs=args.image_dirs["test"] if args.image_dirs else None,
    )
    
    logging.info(f'Train size: {len(train_dataset)}  |  Val size: {len(val_dataset)}  |  Test size: {len(test_dataset)}')

    # ── Samplers ─────────────────────────────────────────────────────────────
    if args.sampler == 'balanced':
        train_sampler = BalancedBatchSampler(train_dataset)
    elif args.sampler == 'undersample':
        train_sampler = UnderSampler(train_dataset, under_sample_rate=args.undersample_rate)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, sampler=train_sampler,
        batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, drop_last=False,
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, drop_last=False,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    num_classes = args.num_classes  
    model = SkinLesionClassifier(
        backbone_name=args.model, num_classes=num_classes, pretrained=args.pretrained,
    ).to(device)
    logging.info(f'Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}')

    # ── Loss ──────────────────────────────────────────────────────────────────
    if args.ifw:
        class_weights = labels_to_class_weights(
            train_dataset.samples, num_classes=num_classes,
        ).to(device)
        logging.info(f'Class weights (IFW): {class_weights.cpu().tolist()}')
    else:
        class_weights = None

    if args.loss == 'ohem':
        criterion = OhemCrossEntropy(weight=class_weights)
    elif args.loss == 'recall_ce':
        criterion = RecallCrossEntropy(n_classes=num_classes, weight=class_weights)
    elif args.loss == 'focal':
        criterion = FocalLoss(gamma=2.0, alpha=class_weights)
    elif args.loss == 'weighted_cross_entropy':
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()
    logging.info(f'Criterion: {criterion}')

    # ── Optimizer & LR scheduler ──────────────────────────────────────────────
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ── Output dir & config snapshot ─────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)
    logger.addHandler(logging.FileHandler(os.path.join(args.output_dir, 'training.log')))
    logging.info(f'Saving outputs to: {args.output_dir}')

    # ── Resume from checkpoint ────────────────────────────────────────────────
    if args.checkpoint and os.path.isfile(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        args.start_epoch = ckpt.get('epoch', 0) + 1
        logging.info(f"Resumed from '{args.checkpoint}' (epoch {args.start_epoch - 1})")

    # ── Training loop ─────────────────────────────────────────────────────────
    best_macro_f1 = 0.0
    best_epoch    = 0
    start_time    = time.time()

    logging.info(f'Starting training for {args.epochs} epochs')
    for epoch in range(args.start_epoch, args.epochs):
        current_lr = optimizer.param_groups[0]['lr']
        logging.info(f'\nEpoch {epoch + 1}/{args.epochs}  lr={current_lr:.2e}')
        logging.info('-' * 40)

        train_stats = train_one_epoch(model, criterion, train_loader, optimizer, device, epoch,
                                      log_interval=args.log_interval)
        val_stats   = evaluate(model, criterion, val_loader, device)
        scheduler.step()

        logging.info(f"  train  loss={train_stats['loss']:.4f}  acc={train_stats['accuracy']:.2f}%")
        logging.info(f"  val    loss={val_stats['loss']:.4f}  acc={val_stats['accuracy']:.2f}%")
        logging.info(f"  macro_f1={val_stats['macro_f1']:.4f}  balanced_acc={val_stats['balanced_acc']:.4f}")
        logging.info(f"  malignant_sensitivity={val_stats['malignant_acc']:.4f}  "
                     f"benign_specificity={val_stats['benign_acc']:.4f}")

        # ── Save best checkpoint (selected by macro F1) ───────────────────────
        if val_stats['macro_f1'] > best_macro_f1:
            best_macro_f1 = val_stats['macro_f1']
            best_epoch    = epoch
            best_path = os.path.join(args.output_dir, 'best_model.pth')
            torch.save({
                'epoch':                epoch,
                'model_state_dict':     model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'macro_f1':             val_stats['macro_f1'],
                'balanced_acc':         val_stats['balanced_acc'],
                'malignant_acc':        val_stats['malignant_acc'],
            }, best_path)
            logging.info(f"  => New best  macro_f1={best_macro_f1:.4f}  saved to {best_path}")

        torch.save({
            'epoch':                epoch,
            'model_state_dict':     model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, os.path.join(args.output_dir, 'last_checkpoint.pth'))

        scalar_val = {k: (v.item() if hasattr(v, 'item') else v)
                      for k, v in val_stats.items()
                      if not isinstance(v, (dict, list)) and not hasattr(v, '__len__')}
        log_entry = {
            'epoch': epoch,
            **{f'train_{k}': v for k, v in train_stats.items()},
            **{f'val_{k}':   v for k, v in scalar_val.items()},
        }
        with open(os.path.join(args.output_dir, 'log.jsonl'), 'a') as f:
            f.write(json.dumps(log_entry) + '\n')

        logging.info(f"  best so far: macro_f1={best_macro_f1:.4f}  (epoch {best_epoch + 1})")

    total_time = str(timedelta(seconds=int(time.time() - start_time)))
    logging.info(f'\nTraining done in {total_time}')
    logging.info(f'Best macro F1: {best_macro_f1:.4f}  (epoch {best_epoch + 1}) on validation dataset')
    
    
    # ── Heldout test set loop ─────────────────────────────────────────────────────────
    test_stats   = evaluate(model, criterion, test_loader, device)
    logging.info(f"\nTest set results:")
    logging.info(f"  loss={test_stats['loss']:.4f}  acc={test_stats['accuracy']:.2f}%")
    logging.info(f"  macro_f1={test_stats['macro_f1']:.4f}  balanced_acc={test_stats['balanced_acc']:.4f}")
    logging.info(f"  malignant_sensitivity={test_stats['malignant_acc']:.4f}  benign_specificity={test_stats['benign_acc']:.4f}")
    
    return

if __name__ == '__main__':
    args = parse_args()
    train(args)