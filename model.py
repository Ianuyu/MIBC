import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from IpsilateralFusion import IpsiCrossViewFusion
from BilateralFusion import BilateralFusion

class SiameseResNetRuleModel(nn.Module):
    def __init__(self, backbone_name='resnet50', pretrained=True,
                 num_classes=3, architecture='baseline',
                 concate_method='concat'): 
        super().__init__()

        self.backbone_name = backbone_name
        self.architecture = architecture
        self.num_classes = num_classes
        self.concate_method = concate_method

        # ---------------- Backbone ----------------
        if backbone_name == 'resnet50':
            self.backbone = models.resnet50(
                weights=models.ResNet50_Weights.DEFAULT if pretrained else None
            )
            self.feature_dim = 2048
            self.backbone.fc = nn.Identity()

        elif backbone_name == 'resnet18':
            self.backbone = models.resnet18(
                weights=models.ResNet18_Weights.DEFAULT if pretrained else None
            )
            self.feature_dim = 512
            self.backbone.fc = nn.Identity()

        elif backbone_name == 'resnet101':
            self.backbone = models.resnet101(
                weights=models.ResNet101_Weights.DEFAULT if pretrained else None
            )
            self.feature_dim = 2048
            self.backbone.fc = nn.Identity()
            
        elif backbone_name == 'efficientnet_b0':
            self.backbone = models.efficientnet_b0(
                weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            )
            self.feature_dim = 1280
            self.backbone.classifier = nn.Identity()
            
        elif backbone_name == 'efficientnet_b3':
            self.backbone = models.efficientnet_b3(
                weights=models.EfficientNet_B3_Weights.DEFAULT if pretrained else None
            )
            self.feature_dim = 1536
            self.backbone.classifier = nn.Identity()
            
        elif backbone_name == 'efficientnet_b5':
            self.backbone = models.efficientnet_b5(
                weights=models.EfficientNet_B5_Weights.DEFAULT if pretrained else None
            )
            self.feature_dim = 2048
            self.backbone.classifier = nn.Identity()
            
        elif backbone_name == 'convnext_tiny':
            self.backbone = models.convnext_tiny(
                weights=models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
            )
            self.feature_dim = 768
            self.backbone.classifier[2] = nn.Identity()
            
        elif backbone_name == 'convnext_small':
            self.backbone = models.convnext_small(
                weights=models.ConvNeXt_Small_Weights.DEFAULT if pretrained else None
            )
            self.feature_dim = 768
            self.backbone.classifier[2] = nn.Identity()
            
        elif backbone_name == 'convnext_base':
            self.backbone = models.convnext_base(
                weights=models.ConvNeXt_Base_Weights.DEFAULT if pretrained else None
            )
            self.feature_dim = 1024
            self.backbone.classifier[2] = nn.Identity()
            
        else:
            raise ValueError(f"不支援的骨幹網路: {backbone_name}")

        # global pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # ---------------- cross-view module ----------------
        if architecture == "cross_view" or architecture == "ipsi": 
            self.cross_ipsi = IpsiCrossViewFusion(dim=self.feature_dim, heads=4)
        else:
            self.cross_ipsi = None

        if architecture == "cross_view" or architecture == "bi":
            self.bilateral_fusion = BilateralFusion(dim=self.feature_dim)
        else:
            self.bilateral_fusion = None

        # ---------------- classifiers ----------------
        # 四視角 concat → exam classifier
        # pooled: (B, 4, C) → concat: (B, 4*C)
        if self.concate_method == 'concat':
            self.exam_classifier = nn.Linear(self.feature_dim * 4, self.num_classes)

        elif self.concate_method == 'concat_linear':
            self.exam_classifier = nn.Sequential(
                nn.Dropout(p=0.5),
                nn.Linear(self.feature_dim * 4, self.num_classes)
            )

        elif self.concate_method == 'concat_mlp':
            hidden_dim = self.feature_dim  # 例如 2048

            self.exam_classifier = nn.Sequential(
                nn.LayerNorm(self.feature_dim * 4),      # 穩定訓練
                nn.Dropout(p=0.5),
                nn.Linear(self.feature_dim * 4, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.5),
                nn.Linear(hidden_dim, self.num_classes)
            )

    # ---------------- feature extractor ----------------
    def forward_one_view(self, x):
        """處理單張影像提取特徵"""
        # x shape: (Batch_Size * 4, 3, H, W)
        
        if 'resnet' in self.backbone_name:
            x = self.backbone.conv1(x)
            x = self.backbone.bn1(x)
            x = self.backbone.relu(x)
            x = self.backbone.maxpool(x)
            x = self.backbone.layer1(x)
            x = self.backbone.layer2(x)
            x = self.backbone.layer3(x)
            x = self.backbone.layer4(x)
            
        elif 'efficientnet' in self.backbone_name:
            x = self.backbone.features(x)
            
        elif 'convnext' in self.backbone_name:
            x = self.backbone.features(x)
            
        return x  # (Batch, Feature_Dim, H', W')

    # ---------------- forward ----------------
    def forward(self, x):
        # x: (B,4,3,H,W)
        B, V, C, H, W = x.shape

        x = x.view(B * V, C, H, W)
        fmap = self.forward_one_view(x)                    # (B*4,C,Hf,Wf)
        _, C2, Hf, Wf = fmap.shape
        fmap = fmap.view(B, 4, C2, Hf, Wf)                 # (B,4,C,Hf,Wf)

        # ---- cross-attention（可選）----
        if self.architecture == "cross_view":
            feats = {
                "L-CC":  fmap[:, 0],
                "R-CC":  fmap[:, 1],
                "L-MLO": fmap[:, 2],
                "R-MLO": fmap[:, 3],
            }
            feats = self.cross_ipsi(feats)

            h_cm_left, h_cm_right = self.bilateral_fusion(feats["L-CC"], feats["R-CC"])
            h_mc_left, h_mc_right = self.bilateral_fusion(feats["L-MLO"], feats["R-MLO"])

            fmap = torch.stack(
                [h_cm_left, h_cm_right, h_mc_left, h_mc_right], dim=1
            )

        elif self.architecture == "ipsi":
            feats = {
                "L-CC":  fmap[:, 0],
                "R-CC":  fmap[:, 1],
                "L-MLO": fmap[:, 2],
                "R-MLO": fmap[:, 3],
            }
            feats = self.cross_ipsi(feats)

            fmap = torch.stack(
                [feats["L-CC"], feats["R-CC"], feats["L-MLO"], feats["R-MLO"]], dim=1
            )

        elif self.architecture == "bi":
            feats = {
                "L-CC":  fmap[:, 0],
                "R-CC":  fmap[:, 1],
                "L-MLO": fmap[:, 2],
                "R-MLO": fmap[:, 3],
            }
            h_cm_left, h_cm_right = self.bilateral_fusion(feats["L-CC"], feats["R-CC"])
            h_mc_left, h_mc_right = self.bilateral_fusion(feats["L-MLO"], feats["R-MLO"])

            fmap = torch.stack(
                [h_cm_left, h_cm_right, h_mc_left, h_mc_right], dim=1
            )

        # global pooling
        pooled = self.global_pool(fmap)             # (B, 4, C_f, 1, 1)
        pooled = pooled.view(B, 4, self.feature_dim)  # (B, 4, C_f)

        # 四視角 concat → exam-level feature
        exam_feat = pooled.view(B, 4 * self.feature_dim)  # (B, 4*C_f)

        # exam-level logits
        exam_logits = self.exam_classifier(exam_feat)     # (B, num_classes)

        return exam_logits
