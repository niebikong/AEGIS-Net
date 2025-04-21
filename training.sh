# different noise_percent
for NOISE_PERCENT in $(seq 0.1 0.1 0.5)
do
    echo "Running with noise_percent = $NOISE_PERCENT"

    NOISE_INT=$(echo "($NOISE_PERCENT * 10)/1" | bc)  # 0.3 → 3, 0.6 → 6

    if [ $NOISE_INT -le 4 ]; then          # <= 0.4
        CLASSIFIER_TRAIN_EPOCHS=50
    elif [ $NOISE_INT -le 6 ]; then        # <= 0.6
        CLASSIFIER_TRAIN_EPOCHS=50
    else                                   # > 0.6
        CLASSIFIER_TRAIN_EPOCHS=50
    fi

    echo "Setting classifier_train_epochs = $CLASSIFIER_TRAIN_EPOCHS"

    if /home/ju/Desktop/NetMamba/.venv/bin/python /home/ju/Desktop/NetMamba/PNP/AEGIS-Net/training.py \
        --noise_percent $NOISE_PERCENT \
        --classifier_train_epochs $CLASSIFIER_TRAIN_EPOCHS; then
        echo "Finished running with noise_percent = $NOISE_PERCENT"
    else
        echo "Failed to run with noise_percent = $NOISE_PERCENT"
    fi

    echo "---------------------------------------------"
done
