
import os
import argparse
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

# project imports
from model.utils import DataLoader
from model.localizer_cnn import CNNFeatureExtractor, LocalKNN

def build_localizer_bank_for_existing_loader(args, train_batch, log_dir):
    """
    Rebuild the localizer memory bank using an existing DataLoader (to avoid re-reading data)
    and the same save path scheme as training. This keeps behavior identical to the original
    inline implementation.
    """
    print('[Localizer] Building normal patch memory bank...')
    extractor = CNNFeatureExtractor(backbone_name=args.local_backbone).cuda()
    knn = LocalKNN(max_items=args.local_max_items).cuda()

    extractor.eval()
    with torch.no_grad():
        for idx, (imgs) in enumerate(train_batch):
            if idx % args.local_stride != 0:  # sampling to save time/memory
                continue
            x = imgs.cuda(non_blocking=True)
            # main stream is in [-1, 1]; map back to [0, 1]
            if getattr(args, 'method', 'pred') == 'pred':
                cur = (x[:, 12:12 + 3] + 1) / 2.0
            else:
                cur = (x[:, :3] + 1) / 2.0
            feats = extractor(cur)
            knn.enqueue(feats)

    bank_path = os.path.join(log_dir, 'localizer_bank.pt')
    torch.save({
        'keys': knn.keys.half(),      # save half precision to reduce size
        'backbone': args.local_backbone,
        'out_dim': extractor.out_dim
    }, bank_path)
    print(f'[Localizer] Bank saved to: {bank_path}')
    return bank_path

def _setup_cuda_from_args(args):
    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    if getattr(args, "gpus", None) is None:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(args.gpus)

def _make_train_loader(args):
    train_folder = os.path.join(args.dataset_path, args.dataset_type, "training", "frames")
    if not os.path.exists(train_folder):
        raise FileNotFoundError(f"Training frames folder not found: {train_folder}")
    train_dataset = DataLoader(
        train_folder,
        transforms.Compose([transforms.ToTensor()]),
        resize_height=args.h, resize_width=args.w, time_step=args.t_length - 1
    )
    train_batch = data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, drop_last=True
    )
    return train_batch

def _default_log_dir(args):
    return os.path.join("./exp", args.dataset_type, args.method, args.exp_dir)

def main():
    parser = argparse.ArgumentParser(description="Build Localizer Memory Bank (standalone)")
    # runtime / device
    parser.add_argument('--gpus', nargs='+', type=str, help='gpus')
    parser.add_argument('--num_workers', type=int, default=2, help='number of workers for the loader')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size for building the bank')

    # dataset / project
    parser.add_argument('--dataset_type', type=str, default='ped2', help='type of dataset: ped2, avenue, shanghai')
    parser.add_argument('--dataset_path', type=str, default='./dataset', help='directory of data')
    parser.add_argument('--exp_dir', type=str, default='log', help='experiment subdir inside exp')
    parser.add_argument('--method', type=str, default='pred', help='pred or recon (must match training)')
    parser.add_argument('--h', type=int, default=256, help='height of input images')
    parser.add_argument('--w', type=int, default=256, help='width of input images')
    parser.add_argument('--t_length', type=int, default=5, help='length of the frame sequences (must match training)')

    # localizer-specific (same defaults as Train.py)
    parser.add_argument('--local_backbone', type=str, default='resnet50', help='timm backbone name')
    parser.add_argument('--local_max_items', type=int, default=50000, help='max patch items in localizer memory')
    parser.add_argument('--local_stride', type=int, default=10, help='use 1/stride of training samples to build bank')

    # optional override for where to save
    parser.add_argument('--out_dir', type=str, default=None, help='override output dir; defaults to exp/<dataset>/<method>/<exp_dir>')

    args = parser.parse_args()

    _setup_cuda_from_args(args)

    # build loader (no training required)
    train_batch = _make_train_loader(args)

    # resolve save path (match Train.py)
    log_dir = args.out_dir if args.out_dir else _default_log_dir(args)
    os.makedirs(log_dir, exist_ok=True)

    # run builder
    build_localizer_bank_for_existing_loader(args, train_batch, log_dir)

if __name__ == "__main__":
    main()
