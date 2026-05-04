from utils.tools import *
from network import *
from TransformerModel.modeling import VisionTransformer, VIT_CONFIGS
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image

torch.multiprocessing.set_sharing_strategy('file_system')

CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


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


def image_to_hash(net, image_tensor, device):
    with torch.no_grad():
        image_tensor = image_tensor.unsqueeze(0).to(device)
        output = net(image_tensor)
        binary_code = torch.sign(output).cpu()
    return binary_code


def hamming_distance(query_code, database_codes):
    return 0.5 * (database_codes.shape[1] - query_code @ database_codes.t()).squeeze(0)


def unnormalize(img_tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = img_tensor.cpu() * std + mean
    return torch.clamp(img, 0, 1)


def to_bits_01(code_tensor):
    return ((code_tensor + 1) / 2).int().tolist()


def get_query_transform(config):
    return transforms.Compose([
        transforms.Resize(config["resize_size"]),
        transforms.CenterCrop(config["crop_size"]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])


def main():
    config = get_config()
    device = config["device"]
    bit = 64
    model_path = "cifar10_CSQ_ViT-B_32_Bit64-BestModel.pth"

    # đổi tên file ảnh ở đây nếu cần
    query_image_path = "query.jpg"

    print("Loading model...")
    net = load_model(config, bit, model_path)

    print("Loading precomputed database index...")
    db_codes = torch.load("database_index/db_codes.pt")
    db_labels = torch.load("database_index/db_labels.pt")
    db_indices = torch.load("database_index/db_indices.pt")

    print("db_codes:", db_codes.shape)
    print("db_labels:", db_labels.shape)
    print("db_indices:", db_indices.shape)

    print("Loading dataset again for image display...")
    _, _, dataset_loader, _, _, _ = get_data(config)
    dataset_data = dataset_loader.dataset

    print("Loading uploaded query image...")
    pil_img = Image.open(query_image_path).convert("RGB")
    query_tensor = get_query_transform(config)(pil_img)
    query_code = image_to_hash(net, query_tensor, device)

    print("\n================ QUERY INFO ================")
    print("Uploaded image path:", query_image_path)
    print("Query hash code (0/1):")
    print(to_bits_01(query_code.squeeze(0)))

    print("\nComputing Hamming distance...")
    distances = hamming_distance(query_code, db_codes)

    topk = 10
    top_positions = torch.argsort(distances)[:topk]
    top_label_ids = db_labels[top_positions].argmax(dim=1)

    print("\n================ TOP 10 RESULTS ================")
    print("Top 10 labels (id):", top_label_ids.tolist())
    print("Top 10 labels (name):", [CLASS_NAMES[i] for i in top_label_ids.tolist()])
    print("Top 10 distances:", distances[top_positions].tolist())

    print("\n================ TOP 10 HASH CODES ================")
    for rank, pos in enumerate(top_positions, start=1):
        dataset_index = db_indices[pos].item()
        lbl_id = db_labels[pos].argmax().item()
        lbl_name = CLASS_NAMES[lbl_id]
        dist = distances[pos].item()
        hash_bits = to_bits_01(db_codes[pos])

        print(f"\nRank {rank}")
        print(f"Position in db_index: {pos.item()}")
        print(f"Dataset index: {dataset_index}")
        print(f"Label id: {lbl_id}")
        print(f"Label name: {lbl_name}")
        print(f"Hamming distance: {dist}")
        print(f"Hash code (0/1): {hash_bits}")

    plt.figure(figsize=(22, 4))

    # Ảnh query upload từ máy
    plt.subplot(1, topk + 1, 1)
    plt.imshow(pil_img)
    plt.title("Uploaded Query")
    plt.axis("off")

    # 10 ảnh gần nhất trong CIFAR-10
    for i, pos in enumerate(top_positions, start=2):
        dataset_index = db_indices[pos].item()
        img, label, _ = dataset_data[dataset_index]
        lbl_id = label.argmax().item()
        lbl_name = CLASS_NAMES[lbl_id]
        dist = distances[pos].item()

        plt.subplot(1, topk + 1, i)
        plt.imshow(unnormalize(img).permute(1, 2, 0))
        plt.title(f"{lbl_name}\nd={int(dist)}")
        plt.axis("off")

    plt.tight_layout()
    plt.savefig("top10_uploaded_result.png")
    print("\nSaved result to top10_uploaded_result.png")


if __name__ == "__main__":
    main()