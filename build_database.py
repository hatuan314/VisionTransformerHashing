from utils.tools import *
from network import *
from TransformerModel.modeling import VisionTransformer, VIT_CONFIGS
import torch
from tqdm import tqdm
import os

torch.multiprocessing.set_sharing_strategy('file_system')


def get_config():
    config = {
        "dataset": "cifar10",
        "net": VisionTransformer,
        "net_print": "ViT-B_32",
        "model_type": "ViT-B_32",
        "pretrained_dir": "pretrainedVIT/ViT-B_32.npz",
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        "batch_size": 64,
        "resize_size": 256,
        "crop_size": 224,
        "info": "CSQ",
        "lambda": 0.0001,
        "bit_list": [64],
    }
    config = config_dataset(config)
    return config


def load_model(config, bit, model_path):
    device = config["device"]
    num_classes = config["n_class"]
    vit_config = VIT_CONFIGS[config["model_type"]]

    net = config["net"](
        vit_config,
        config["crop_size"],
        zero_head=True,
        num_classes=num_classes,
        hash_bit=bit
    ).to(device)

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if "net" in checkpoint:
        net.load_state_dict(checkpoint["net"])
    else:
        net.load_state_dict(checkpoint)

    net.eval()
    return net


def main():
    config = get_config()
    device = config["device"]
    bit = 64
    model_path = "cifar10_CSQ_ViT-B_32_Bit64-BestModel.pth"

    print("Loading model...")
    net = load_model(config, bit, model_path)

    print("Loading database...")
    _, _, database_loader, _, _, _ = get_data(config)

    db_codes = []
    db_labels = []
    db_indices = []

    print("Hashing entire database...")
    with torch.no_grad():
        for images, labels, inds in tqdm(database_loader):
            images_gpu = images.to(device)
            outputs = net(images_gpu)
            codes = torch.sign(outputs).cpu()

            db_codes.append(codes)
            db_labels.append(labels.cpu())
            db_indices.append(inds.cpu())

    db_codes = torch.cat(db_codes, dim=0)      # [54000, 64]
    db_labels = torch.cat(db_labels, dim=0)    # [54000, 10]
    db_indices = torch.cat(db_indices, dim=0)  # [54000]

    os.makedirs("database_index", exist_ok=True)

    torch.save(db_codes, "database_index/db_codes.pt")
    torch.save(db_labels, "database_index/db_labels.pt")
    torch.save(db_indices, "database_index/db_indices.pt")

    print("Saved:")
    print("database_index/db_codes.pt", db_codes.shape)
    print("database_index/db_labels.pt", db_labels.shape)
    print("database_index/db_indices.pt", db_indices.shape)


if __name__ == "__main__":
    main()