"""Stage 4: model definitions.

  custom_cnn       : 1-channel CNN trained from scratch (input mode 'custom')
  resnet18         : ImageNet-pretrained, 3-channel input (input mode 'pretrained')
  efficientnet_b0  : ImageNet-pretrained, 3-channel input (input mode 'pretrained')

Transfer strategies
  frozen   : backbone frozen (BatchNorm kept in eval mode), only the new head trains
  finetune : head + last stage train (ResNet18: layer4 ; EfficientNet-B0: features[7], features[8])

Run a structural check from the project root:   python -m src.models
"""
import pandas as pd
import torch
import torch.nn as nn

from . import config as C
from .utils import set_seed

NUM_CLASSES = 6
MODEL_INPUT_MODE = {"custom_cnn": "custom", "resnet18": "pretrained", "efficientnet_b0": "pretrained"}


# ------------------------------------------------------------------ Model A
class CustomCNN(nn.Module):
    """Conv3x3-BN-ReLU blocks; 2x2 max-pool after all but the last block; GAP; Dropout; FC.

    Design notes
      * 5 blocks / 4 poolings -> ~78 px receptive field on 224x224 input (large Patches need context).
      * Global average pooling instead of flatten: far fewer parameters (small dataset), position-invariant,
        and compatible with Grad-CAM.
      * BatchNorm stabilises training on small data; conv bias omitted (redundant before BN).
      * ReLU is not in-place so Grad-CAM hooks work safely.
    """

    def __init__(self, num_classes=NUM_CLASSES, in_channels=1,
                 channels=(32, 64, 128, 256, 256), dropout=0.4):
        super().__init__()
        blocks, prev = [], in_channels
        for i, ch in enumerate(channels):
            layers = [nn.Conv2d(prev, ch, kernel_size=3, padding=1, bias=False),
                      nn.BatchNorm2d(ch), nn.ReLU(inplace=False)]
            if i < len(channels) - 1:
                layers.append(nn.MaxPool2d(2))
            blocks.append(nn.Sequential(*layers))
            prev = ch
        self.features = nn.Sequential(*blocks)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(prev, num_classes)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01); nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x).flatten(1)
        return self.fc(self.dropout(x))   # logits; softmax is inside CrossEntropyLoss


# ------------------------------------------------------------------ Model B
class TransferNet(nn.Module):
    def __init__(self, arch="resnet18", num_classes=NUM_CLASSES, strategy="frozen",
                 pretrained=True, dropout=0.2):
        super().__init__()
        from torchvision import models
        if strategy not in ("frozen", "finetune"):
            raise ValueError("strategy must be 'frozen' or 'finetune'")
        self.arch, self.strategy = arch, strategy
        if arch == "resnet18":
            w = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            self.net = models.resnet18(weights=w)
            self.net.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(self.net.fc.in_features, num_classes))
        elif arch == "efficientnet_b0":
            w = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
            self.net = models.efficientnet_b0(weights=w)
            in_f = self.net.classifier[1].in_features
            self.net.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_f, num_classes))
        else:
            raise ValueError(f"unknown arch {arch}")
        self._apply_strategy()

    def head_module(self):
        return self.net.fc if self.arch == "resnet18" else self.net.classifier

    def tuned_backbone_modules(self):
        if self.arch == "resnet18":
            return [self.net.layer4]
        return [self.net.features[7], self.net.features[8]]

    def _apply_strategy(self):
        for p in self.net.parameters():
            p.requires_grad = False
        for p in self.head_module().parameters():
            p.requires_grad = True
        if self.strategy == "finetune":
            for m in self.tuned_backbone_modules():
                for p in m.parameters():
                    p.requires_grad = True

    def parameter_groups(self):
        """(head_params, tuned_backbone_params) - lets Stage 5 use a lower LR for the backbone."""
        head = list(self.head_module().parameters())
        head_ids = {id(p) for p in head}
        backbone = [p for p in self.net.parameters() if p.requires_grad and id(p) not in head_ids]
        return head, backbone

    def train(self, mode=True):
        super().train(mode)
        if mode:   # keep fully-frozen BatchNorm layers on their ImageNet statistics
            for m in self.net.modules():
                if isinstance(m, nn.modules.batchnorm._BatchNorm) and not any(p.requires_grad for p in m.parameters()):
                    m.eval()
        return self

    def forward(self, x):
        return self.net(x)


# ------------------------------------------------------------------ helpers
def build_model(name, strategy=None, pretrained=True, dropout=None, num_classes=NUM_CLASSES):
    if name == "custom_cnn":
        return CustomCNN(num_classes=num_classes, dropout=0.4 if dropout is None else dropout)
    if name in ("resnet18", "efficientnet_b0"):
        return TransferNet(name, num_classes, strategy or "frozen", pretrained,
                           0.2 if dropout is None else dropout)
    raise ValueError(f"unknown model {name}")


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def gradcam_target_layer(model):
    """Last convolutional stage - used for Grad-CAM in Stage 7."""
    if isinstance(model, CustomCNN):
        return model.features[-1]
    if model.arch == "resnet18":
        return model.net.layer4[-1]
    return model.net.features[-1]


def receptive_field(cnn):
    rf, jump = 1, 1
    for m in cnn.features.modules():
        if isinstance(m, (nn.Conv2d, nn.MaxPool2d)):
            k = m.kernel_size if isinstance(m.kernel_size, int) else m.kernel_size[0]
            s = m.stride if isinstance(m.stride, int) else m.stride[0]
            rf += (k - 1) * jump
            jump *= s
    return rf


# ------------------------------------------------------------------ check
def main():
    C.ensure_dirs(); set_seed(C.SEED)
    specs = [("custom_cnn", None), ("resnet18", "frozen"), ("resnet18", "finetune"),
             ("efficientnet_b0", "frozen"), ("efficientnet_b0", "finetune")]
    rows, weights_ok = [], True
    for name, strat in specs:
        try:
            model = build_model(name, strat, pretrained=True)
        except Exception as e:
            if weights_ok:
                print(f"WARNING: could not load pretrained weights ({type(e).__name__}: {str(e)[:120]}). "
                      f"Continuing with RANDOM weights for the structural check only.\n")
                weights_ok = False
            model = build_model(name, strat, pretrained=False)
        ch = 1 if MODEL_INPUT_MODE[name] == "custom" else 3
        model.eval()
        with torch.no_grad():
            out = model(torch.randn(2, ch, C.IMAGE_SIZE, C.IMAGE_SIZE))
        total, trainable = count_parameters(model)
        rows.append({"model": name, "strategy": strat or "from scratch", "input_channels": ch,
                     "total_params": total, "trainable_params": trainable,
                     "trainable_%": round(100 * trainable / total, 2), "output_shape": tuple(out.shape),
                     "gradcam_layer": type(gradcam_target_layer(model)).__name__})
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(df.to_string(index=False))
    df.to_csv(C.REPORT_DIR / "model_summary.csv", index=False)
    cnn = CustomCNN()
    print(f"\nCustom CNN receptive field: {receptive_field(cnn)} px (input {C.IMAGE_SIZE}x{C.IMAGE_SIZE})")
    print("Pretrained weights loaded:" , "yes" if weights_ok else "NO (random weights used)")
    print("Saved: outputs\\reports\\model_summary.csv\n\nStage 4 complete.")


if __name__ == "__main__":
    main()
