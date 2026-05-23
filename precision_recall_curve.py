import argparse
import os
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 28})


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--method", type=str, default="CSQ")
    parser.add_argument("--bit", type=str, default="16")
    parser.add_argument("--models", type=str, nargs="+",
                        default=["AlexNet", "ResNet", "ViT-B_32", "ViT-B_16"])
    parser.add_argument("--legends", type=str, nargs="+", default=None,
                        help="Labels in the legend; same length as --models")
    parser.add_argument("--results_dir", type=str, default="Checkpoints_Results",
                        help="Directory containing the *.txt result logs")
    parser.add_argument("--nested", action="store_true",
                        help="If set, look in <results_dir>/<dataset>/<method>/ instead of <results_dir>/")
    parser.add_argument("--out_dir", type=str, default=".",
                        help="Directory to save the PDF output")
    parser.add_argument("--show", action="store_true", help="Call plt.show() at the end")
    return parser.parse_args()


def main():
    args = parse_args()
    Dataset = args.dataset
    Method = args.method
    Bit = args.bit
    Model = args.models
    Legends = args.legends if args.legends is not None else Model

    if args.nested:
        pathfile = os.path.join(args.results_dir, Dataset, Method) + os.sep
    else:
        pathfile = args.results_dir.rstrip(os.sep) + os.sep
    print("Looking in:", pathfile)

    markers = "DdsPvo*xH1234h"
    model2marker = {m: markers[i] for i, m in enumerate(Model)}

    plt.figure(figsize=(11, 9))
    for model in Model:
        pathfile_model = pathfile + Dataset + "_" + Method + "_" + model + "_Bit" + Bit + ".txt"
        print(pathfile_model)
        if not os.path.exists(pathfile_model):
            print("  -> missing, skip")
            continue
        with open(pathfile_model, 'r') as file_model:
            Lines = file_model.readlines()
        data = None
        for line in Lines:
            if line.find("PR") != -1:
                data = line[line.rfind("|") + 2:-2].split(' ')
        if data is None:
            print("  -> no PR line, skip")
            continue
        P = [float(data[j]) for j in range(len(data)) if j % 2 != 1]
        R = [float(data[j]) for j in range(len(data)) if j % 2 != 0]
        plt.plot(R, P, linestyle="-", marker=model2marker[model],
                 label=model, linewidth=4, markersize=12)
        print("P:", P)
        print("R:", R)

    plt.grid(True)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(Legends)

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, Dataset + "_" + Method + "_Bit" + Bit + "_pr.pdf")
    plt.savefig(out_path)
    print("Saved:", out_path)
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
