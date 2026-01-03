#!/bin/bash

# ==========================================
# 環境設定 
# ==========================================

# 請將以下路徑改為您電腦上的實際位置
DATA_ROOT="processed_datasets"  # 圖片根目錄
CSV_DIR="csv/three_classes"       # CSV 檔案目錄

TRAIN_CSV="${CSV_DIR}/train_labels.csv"
VAL_CSV="${CSV_DIR}/val_labels.csv"
TEST_CSV="${CSV_DIR}/test_labels.csv"
SAVE_DIR="./simple_concate_4view"

# ==========================================
# 訓練超參數設定
# ==========================================

# 想要跑的模型列表
BACKBONES=("resnet18" "resnet50" "efficientnet_b0" "efficientnet_b3" "convnext_tiny" "convnext_small")
# BACKBONES=("resnet50" "efficientnet_b0" "efficientnet_b5" "convnext_tiny" "convnext_small")

# 想要跑的架構列表
ARCHITECTURES=("baseline" "ipsi" "bi" "cross_view")    # 選項: cross_view, baseline, ipsi, bi

# 想要跑的拼接方式列表
CONCATE_METHODS=("concat" "concat_linear" "concat_mlp")

# 硬體相關參數
BATCH_SIZE=4     
ACCUM_STEPS=8
EPOCHS=50
LR=1e-4
WD=1e-4
IMG_H=1024
IMG_W=512
# 自動生成實驗 ID (包含時間戳記，避免覆蓋)
TIMESTAMP=$(date +"%m%d_%H%M")

# ==========================================
# 開始迴圈訓練
# ==========================================

for BACKBONE in "${BACKBONES[@]}"; do
    for ARCHITECTURE in "${ARCHITECTURES[@]}"; do
        for CONCATE_METHOD in "${CONCATE_METHODS[@]}"; do
            EFFECTIVE_BS=$((BATCH_SIZE * ACCUM_STEPS))
            EXP_ID="${BACKBONE}_${ARCHITECTURE}_${CONCATE_METHOD}_effbs${EFFECTIVE_BS}_${TIMESTAMP}"
            
            echo "========================================================"
            echo "🚀 Starting Training..."
            echo "   Experiment ID: ${EXP_ID}"
            echo "   Backbone:      ${BACKBONE}"
            echo "   Architecture:  ${ARCHITECTURE}"
            echo "   Batch Size:    ${BATCH_SIZE} (Accum: ${ACCUM_STEPS} => Effective: ${EFFECTIVE_BS})"
            echo "========================================================"

            # 執行 Python 腳本
            uv run main2.py \
                --csv_train "${TRAIN_CSV}" \
                --csv_val "${VAL_CSV}" \
                --csv_test "${TEST_CSV}" \
                --root_dir "${DATA_ROOT}" \
                --save_dir "${SAVE_DIR}" \
                --num_classes 3 \
                --experiment_id "${EXP_ID}" \
                --backbone "${BACKBONE}" \
                --architecture "${ARCHITECTURE}" \
                --concate_method "${CONCATE_METHOD}" \
                --batch_size ${BATCH_SIZE} \
                --gradient_accumulation_steps ${ACCUM_STEPS} \
                --img_height ${IMG_H} \
                --img_width ${IMG_W} \
                --num_epochs ${EPOCHS} \
                --lr ${LR} \
                --weight_decay ${WD} \
                --pretrained \
                --mixed_precision \
                --use_class_weights
                
            # 檢查執行結果
            if [ $? -eq 0 ]; then
                echo "✅ Training [${EXP_ID}] Completed Successfully!"
                echo "--------------------------------------------------------"
            else
                echo "❌ Training [${EXP_ID}] Failed."
                echo "--------------------------------------------------------"
                # 遇到錯誤是否要停止？如果不希望停止整個迴圈，請註解掉下面這行 exit 1
                exit 1
            fi
            
            # (選用) 清除 GPU 快取，避免不同實驗間的干擾
            # python -c "import torch; torch.cuda.empty_cache()"
        done
    done
done
