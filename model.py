import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from IpsilateralFusion import IpsiCrossViewFusion
from BilateralFusion import BilateralFusion


class SiameseResNetRuleModel(nn.Module):
    def __init__(self, backbone_name='resnet50', pretrained=True,
                 num_classes=3, architecture='baseline',
                 concate_method='concat', decision_rule='max'):
        super().__init__()

        self.backbone_name = backbone_name
        self.architecture = architecture
        self.num_classes = num_classes
        self.concate_method = concate_method
        self.decision_rule = decision_rule

        # ----------------------------------------------------
        # helper: 建一個 backbone，回傳 (model, feature_dim)
        # ----------------------------------------------------
        def build_backbone(name, pretrained_flag):
            if name == 'resnet50':
                m = models.resnet50(
                    weights=models.ResNet50_Weights.DEFAULT if pretrained_flag else None
                )
                feat_dim = 2048
                m.fc = nn.Identity()

            elif name == 'resnet18':
                m = models.resnet18(
                    weights=models.ResNet18_Weights.DEFAULT if pretrained_flag else None
                )
                feat_dim = 512
                m.fc = nn.Identity()

            elif name == 'resnet101':
                m = models.resnet101(
                    weights=models.ResNet101_Weights.DEFAULT if pretrained_flag else None
                )
                feat_dim = 2048
                m.fc = nn.Identity()

            elif name == 'efficientnet_b0':
                m = models.efficientnet_b0(
                    weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained_flag else None
                )
                feat_dim = 1280
                m.classifier = nn.Identity()

            elif name == 'efficientnet_b3':
                m = models.efficientnet_b3(
                    weights=models.EfficientNet_B3_Weights.DEFAULT if pretrained_flag else None
                )
                feat_dim = 1536
                m.classifier = nn.Identity()

            elif name == 'efficientnet_b5':
                m = models.efficientnet_b5(
                    weights=models.EfficientNet_B5_Weights.DEFAULT if pretrained_flag else None
                )
                feat_dim = 2048
                m.classifier = nn.Identity()

            elif name == 'convnext_tiny':
                m = models.convnext_tiny(
                    weights=models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained_flag else None
                )
                feat_dim = 768
                m.classifier[2] = nn.Identity()

            elif name == 'convnext_small':
                m = models.convnext_small(
                    weights=models.ConvNeXt_Small_Weights.DEFAULT if pretrained_flag else None
                )
                feat_dim = 768
                m.classifier[2] = nn.Identity()

            elif name == 'convnext_base':
                m = models.convnext_base(
                    weights=models.ConvNeXt_Base_Weights.DEFAULT if pretrained_flag else None
                )
                feat_dim = 1024
                m.classifier[2] = nn.Identity()

            else:
                raise ValueError(f"不支援的骨幹網路: {name}")

            return m, feat_dim

        # ----------------------------------------------------
        # 🔹 兩個骨幹：一個給 CC 視角，一個給 MLO 視角
        # ----------------------------------------------------
        self.backbone_cc, self.feature_dim = build_backbone(backbone_name, pretrained)
        self.backbone_mlo, _              = build_backbone(backbone_name, pretrained)

        # global pooling (對 feature map 做 global average pooling)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # ---------------- cross-view module ----------------
        if architecture in ["cross_view", "ipsi"]:
            self.cross_ipsi = IpsiCrossViewFusion(dim=self.feature_dim, heads=4)
        else:
            self.cross_ipsi = None

        if architecture in ["cross_view", "bi"]:
            self.bilateral_fusion = BilateralFusion(dim=self.feature_dim)
        else:
            self.bilateral_fusion = None

        # ---------------- classifiers ----------------
        # 單側乳房：2 視角 (CC + MLO) concat → classifier
        if self.concate_method == 'concat':
            self.breast_classifier = nn.Linear(self.feature_dim * 2, self.num_classes)

        elif self.concate_method == 'concat_linear':
            self.breast_classifier = nn.Sequential(
                nn.Dropout(p=0.5),
                nn.Linear(self.feature_dim * 2, self.num_classes)
            )

        elif self.concate_method == 'concat_mlp':
            hidden_dim = self.feature_dim  # 例如 2048

            self.breast_classifier = nn.Sequential(
                nn.LayerNorm(self.feature_dim * 2),
                nn.Dropout(p=0.5),
                nn.Linear(self.feature_dim * 2, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.5),
                nn.Linear(hidden_dim, self.num_classes)
            )
        else:
            raise ValueError(f"不支援的 concate_method: {self.concate_method}")

    # ---------------- feature extractor ----------------
    def forward_one_view(self, x, view_type: str):
        """
        x: (B,3,H,W)
        view_type: 'cc' or 'mlo'
        """
        if view_type == 'cc':
            backbone = self.backbone_cc
        elif view_type == 'mlo':
            backbone = self.backbone_mlo
        else:
            raise ValueError("view_type 必須是 'cc' 或 'mlo'")

        if 'resnet' in self.backbone_name:
            x = backbone.conv1(x)
            x = backbone.bn1(x)
            x = backbone.relu(x)
            x = backbone.maxpool(x)
            x = backbone.layer1(x)
            x = backbone.layer2(x)
            x = backbone.layer3(x)
            x = backbone.layer4(x)

        elif 'efficientnet' in self.backbone_name or 'convnext' in self.backbone_name:
            x = backbone.features(x)

        return x  # (B, C_f, H_f, W_f)

    # ---------------- forward ----------------
    def forward(self, x):
        """
        x: (B,4,3,H,W)
        視角順序假設為 [L-CC, R-CC, L-MLO, R-MLO]
        """

        B, V, C, H, W = x.shape
        assert V == 4, "目前 forward 假設輸入視角數為 4（L-CC, R-CC, L-MLO, R-MLO）"

        x = x.view(B, 4, C, H, W)

        # ----------- CC 視角經過 CC backbone -----------
        x_cc = x[:, :2].contiguous().view(B * 2, C, H, W)       # (B*2,3,H,W)
        fmap_cc = self.forward_one_view(x_cc, view_type='cc')   # (B*2,Cf,Hf,Wf)
        _, Cf, Hf, Wf = fmap_cc.shape
        fmap_cc = fmap_cc.view(B, 2, Cf, Hf, Wf)                # (B,2,Cf,Hf,Wf)

        # ----------- MLO 視角經過 MLO backbone -----------
        x_mlo = x[:, 2:].contiguous().view(B * 2, C, H, W)      # (B*2,3,H,W)
        fmap_mlo = self.forward_one_view(x_mlo, view_type='mlo')# (B*2,Cf,Hf,Wf)
        fmap_mlo = fmap_mlo.view(B, 2, Cf, Hf, Wf)              # (B,2,Cf,Hf,Wf)

        # ----------- 按原順序組回四視角 feature map -----------
        # fmap: (B,4,Cf,Hf,Wf) 對應 [L-CC, R-CC, L-MLO, R-MLO]
        fmap = torch.cat([fmap_cc, fmap_mlo], dim=1)

        # -------- cross/ipsi/bi 模組（如果有開）-----------
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

        # ------------------------------------------------
        # global pooling：對每個視角做 GAP → (B,4,C_f)
        # ------------------------------------------------
        B, V, C2, Hf, Wf = fmap.shape
        fmap_flat = fmap.view(B * V, C2, Hf, Wf)          # (B*4, C_f, Hf, Wf)
        pooled = self.global_pool(fmap_flat)              # (B*4, C_f, 1, 1)
        pooled = pooled.view(B, V, self.feature_dim)      # (B,4,C_f)

        # ------------------------------------------------
        # 四視角特徵組合
        # 左/右乳：同側跨視角 concat
        # CC/MLO：同視角跨乳 concat  （對應論文圖）
        # ------------------------------------------------
        # 同側：用在 L/R 的 breast prediction / rule / max
        l_feat = torch.cat([pooled[:, 0], pooled[:, 2]], dim=1)   # L-CC + L-MLO
        r_feat = torch.cat([pooled[:, 1], pooled[:, 3]], dim=1)   # R-CC + R-MLO

        # 同視角跨乳：對應論文圖中的 CC-branch / MLO-branch
        # cc_feat = torch.cat([pooled[:, 0], pooled[:, 1]], dim=1)  # L-CC + R-CC
        # mlo_feat = torch.cat([pooled[:, 2], pooled[:, 3]], dim=1) # L-MLO + R-MLO

        # ---------------- breast-level logits ----------------
        L_logits = self.breast_classifier(l_feat)   # (B, num_classes)
        R_logits = self.breast_classifier(r_feat)   # (B, num_classes)
        # CC_logits = self.breast_classifier(cc_feat) # (B, num_classes)
        # MLO_logits = self.breast_classifier(mlo_feat)# (B, num_classes)

        # ---------- 轉成機率 ----------
        L_prob = F.softmax(L_logits, dim=1)
        R_prob = F.softmax(R_logits, dim=1)
        # CC_prob = F.softmax(CC_logits, dim=1)
        # MLO_prob = F.softmax(MLO_logits, dim=1)

        # ------------------------------------------------
        # exam-level decision rule
        # ------------------------------------------------
        if self.decision_rule == 'max':
            # 對每個 class 取左右乳較大的機率，再 renormalize
            m = torch.max(L_prob, R_prob)                   # (B, num_classes)
            exam_prob = m / (m.sum(dim=1, keepdim=True) + 1e-8)

        elif self.decision_rule == 'avg':
            # 🔹對應論文圖：CC-branch & MLO-branch 的 average
            m = (L_prob + R_prob) / 2.0                  # (B, num_classes)
            # 其實 m 本身就已經 sum=1，不過為穩定性保留 renorm
            exam_prob = m / (m.sum(dim=1, keepdim=True) + 1e-8)

        elif self.decision_rule == 'rule':
            # ----- 機率版臨床規則 -----
            pL0 = L_prob[:, 0]
            pL1 = L_prob[:, 1]
            pL2 = L_prob[:, 2]

            pR0 = R_prob[:, 0]
            pR1 = R_prob[:, 1]
            pR2 = R_prob[:, 2]

            # 1) exam = 2：至少一側為 2
            exam_p2 = 1.0 - (1.0 - pL2) * (1.0 - pR2)

            # 2) exam = 0：有一側為 0 且另一側非 2
            #    (L=0,R=0), (L=0,R=1), (L=1,R=0)
            exam_p0 = pL0 * pR0 + pL0 * pR1 + pL1 * pR0

            # 3) exam = 1：剩下的機率
            exam_p1 = 1.0 - exam_p0 - exam_p2

            exam_prob = torch.stack([exam_p0, exam_p1, exam_p2], dim=1)  # (B,3)
            exam_prob = torch.clamp(exam_prob, min=1e-8)
            exam_prob = exam_prob / exam_prob.sum(dim=1, keepdim=True)

        else:
            raise ValueError(f"不支援的決策規則: {self.decision_rule}")

        return exam_log_prob, L_prob, R_prob, L_logits, R_logits
