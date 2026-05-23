import sys
import os
import re
import argparse
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.tools import *
from network import AlexNet, ResNet
from TransformerModel.modeling import VisionTransformer, VIT_CONFIGS

torch.multiprocessing.set_sharing_strategy('file_system')

TRAIN_MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "train-models")
DATABASE_INDEX_DIR = os.path.join(os.path.dirname(__file__), "..", "database_index")

# Pattern: {dataset}_{info}_{net_print}_Bit{bit}-{type}.{ext}
MODEL_FILENAME_RE = re.compile(
    r"^(?P<dataset>.+?)_(?P<info>.+?)_(?P<net_print>ViT-B_\d+|AlexNet|ResNet\w*)_Bit(?P<bit>\d+)-(?P<kind>BestModel|IntermediateModel)\.(?P<ext>pt|pth)$"
)


def list_model_files():
    files = sorted(
        f for f in os.listdir(TRAIN_MODELS_DIR)
        if MODEL_FILENAME_RE.match(f)
    )
    return files


def parse_model_filename(filename):
    m = MODEL_FILENAME_RE.match(filename)
    if not m:
        raise ValueError(f"Cannot parse model filename: {filename}")
    return {
        "dataset": m.group("dataset"),
        "info": m.group("info"),
        "net_print": m.group("net_print"),
        "bit": int(m.group("bit")),
        "kind": m.group("kind"),
    }


def build_config(parsed):
    net_print = parsed["net_print"]
    if "ViT" in net_print:
        net = VisionTransformer
        model_type = net_print
        pretrained_dir = f"pretrainedVIT/{net_print}.npz"
    elif net_print == "AlexNet":
        net = AlexNet
        model_type = None
        pretrained_dir = None
    else:
        net = ResNet
        model_type = None
        pretrained_dir = None

    config = {
        "dataset": parsed["dataset"],
        "net": net,
        "net_print": net_print,
        "model_type": model_type,
        "pretrained_dir": pretrained_dir,
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        "batch_size": 64,
        "resize_size": 256,
        "crop_size": 224,
        "info": parsed["info"],
        "bit_list": [parsed["bit"]],
    }
    config = config_dataset(config)
    return config


def load_model(config, bit):
    device = config["device"]
    num_classes = config["n_class"]
    net_print = config["net_print"]

    if "ViT" in net_print:
        vit_config = VIT_CONFIGS[config["model_type"]]
        net = config["net"](
            vit_config,
            config["crop_size"],
            zero_head=True,
            num_classes=num_classes,
            hash_bit=bit,
        ).to(device)
    else:
        net = config["net"](hash_bit=bit).to(device)

    return net


def select_model_interactive(files):
    print("\nDanh sách model trong train-models/:")
    for i, f in enumerate(files):
        print(f"  [{i}] {f}")
    while True:
        try:
            idx = int(input(f"\nChọn model (0-{len(files)-1}): "))
            if 0 <= idx < len(files):
                return files[idx]
        except (ValueError, KeyboardInterrupt):
            pass
        print("Lựa chọn không hợp lệ, thử lại.")


def main():
    parser = argparse.ArgumentParser(description="Build database index from a trained hashing model")
    parser.add_argument("--model", type=str, default=None,
                        help="Tên file model trong train-models/ (ví dụ: cifar10_CSQ_ViT-B_32_Bit64-BestModel.pth)")
    parser.add_argument("--list", action="store_true", help="Liệt kê các model có sẵn rồi thoát")
    args = parser.parse_args()

    files = list_model_files()
    if not files:
        print(f"Không tìm thấy model nào trong {TRAIN_MODELS_DIR}")
        sys.exit(1)

    if args.list:
        print("Model có sẵn:")
        for f in files:
            print(f"  {f}")
        return

    if args.model:
        if args.model not in files:
            print(f"Model '{args.model}' không tồn tại trong train-models/")
            print("Dùng --list để xem danh sách model có sẵn.")
            sys.exit(1)
        chosen = args.model
    else:
        chosen = select_model_interactive(files)

    model_path = os.path.join(TRAIN_MODELS_DIR, chosen)
    parsed = parse_model_filename(chosen)
    bit = parsed["bit"]

    print(f"\nModel   : {chosen}")
    print(f"Dataset : {parsed['dataset']}  |  Method: {parsed['info']}  |  Backbone: {parsed['net_print']}  |  Bits: {bit}")

    config = build_config(parsed)
    device = config["device"]

    print("\nLoading model...")
    net = load_model(config, bit)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    net.load_state_dict(checkpoint["net"] if "net" in checkpoint else checkpoint)
    net.eval()

    print("Loading database...")
    _, _, database_loader, _, _, _ = get_data(config)

    db_codes, db_labels, db_indices = [], [], []

    print("Hashing entire database...")
    with torch.no_grad():
        for images, labels, inds in tqdm(database_loader):
            outputs = net(images.to(device))
            db_codes.append(torch.sign(outputs).cpu())
            db_labels.append(labels.cpu())
            db_indices.append(inds.cpu())

    db_codes = torch.cat(db_codes, dim=0)
    db_labels = torch.cat(db_labels, dim=0)
    db_indices = torch.cat(db_indices, dim=0)

    out_dir = os.path.join(DATABASE_INDEX_DIR, f"{parsed['dataset']}_{parsed['info']}_{parsed['net_print']}_Bit{bit}")
    os.makedirs(out_dir, exist_ok=True)

    torch.save(db_codes, os.path.join(out_dir, "db_codes.pt"))
    torch.save(db_labels, os.path.join(out_dir, "db_labels.pt"))
    torch.save(db_indices, os.path.join(out_dir, "db_indices.pt"))

    print("\nSaved:")
    print(f"  {out_dir}/db_codes.pt   {db_codes.shape}")
    print(f"  {out_dir}/db_labels.pt  {db_labels.shape}")
    print(f"  {out_dir}/db_indices.pt {db_indices.shape}")


if __name__ == "__main__":
    main()
