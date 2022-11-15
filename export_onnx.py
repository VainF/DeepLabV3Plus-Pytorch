import network
import argparse

import torch


def get_argparser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--num_classes", type=int, required=True)
    parser.add_argument("--output_stride", type=int, required=True)
    parser.add_argument("--img_size", type=int, required=True)
    parser.add_argument(
        "--separable_conv",
        action="store_true",
        default=False,
        help="apply separable conv to decoder and aspp",
    )
    parser.add_argument("--device", type=str, default="cpu")

    return parser


def main():
    opts = get_argparser().parse_args()

    device = torch.device(opts.device)
    print("Device: %s" % device)

    model = network.modeling.__dict__[opts.model](
        num_classes=opts.num_classes, output_stride=opts.output_stride
    )
    if opts.separable_conv and "plus" in opts.model:
        network.convert_to_separable_conv(model.classifier)

    checkpoint = torch.load(opts.weights, map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint["model_state"])

    torch.onnx.export(model, torch.rand(1, 3, opts.img_size, opts.img_size), opts.name)


if __name__ == "__main__":
    main()
