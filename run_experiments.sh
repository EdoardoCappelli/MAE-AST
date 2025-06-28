!/bin/bash

echo "Avvio della fase di pre-training..."
python pretrain.py --masking_strategy patch --epochs 16

if [ $? -eq 0 ]; then

    echo "Pre-training completato con successo. Avvio della fase di fine-tuning..."
    python finetune.py --dataset esc --masking_strategy patch --pretrained_checkpoint_path './checkpoints/pretraining_schedPoly_lr0.0001_warmup_patch/librispeech&audioset/checkpoint_epoch_16.pth'
    python finetune.py --dataset esc --masking_strategy frame --pretrained_checkpoint_path './checkpoints/pretraining_schedPoly_lr0.0001_warmup_patch/librispeech&audioset/checkpoint_epoch_16.pth'
    python finetune.py --dataset voxceleb --masking_strategy patch --pretrained_checkpoint_path './checkpoints/pretraining_schedPoly_lr0.0001_warmup_patch/librispeech&audioset/checkpoint_epoch_16.pth'
    python finetune.py --dataset voxceleb --masking_strategy frame --pretrained_checkpoint_path './checkpoints/pretraining_schedPoly_lr0.0001_warmup_patch/librispeech&audioset/checkpoint_epoch_16.pth'

    if [ $? -eq 0 ]; then
        echo "Fine-tuning completato con successo."
    else
        echo "Errore durante la fase di fine-tuning."
    fi

else
    echo "Errore durante la fase di pre-training. Il fine-tuning non verrà eseguito."
fi

echo "Completato."
