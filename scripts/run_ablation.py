from _common import build_loaders, parse_args
from neuroalign_emeg.engine import fit_loop, make_optimizer, train_finetune_epoch
from neuroalign_emeg.models import NeuroAlignEMEG
from neuroalign_emeg.utils import get_device, load_yaml, set_seed


def main():
    args = parse_args()
    cfg = load_yaml(args.config)
    set_seed(cfg.get("seed", 42))
    device = get_device(cfg.get("device", "auto"))
    model = NeuroAlignEMEG(cfg).to(device)
    train_loader, val_loader = build_loaders(cfg)
    optimizer = make_optimizer(model, cfg)
    fit_loop(model, train_loader, val_loader, optimizer, device, cfg, train_finetune_epoch, cfg.get("output_dir", "outputs/ablation"))


if __name__ == "__main__":
    main()
