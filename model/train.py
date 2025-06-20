import os
import torch
from time import time
from model import (
    TransformerModel, TRAIN_TRANSFORMS, TEST_TRANSFORMS,
    PATH_TRAIN_DIR, PATH_TRAIN_LABELS, PATH_TEST_DIR, PATH_TEST_LABELS,
    TRAIN_LOG, CHECKPOINT_PATH, WEIGHTS_PATH, ALPHABET, LENGTH,
    BATCH_SIZE, HIDDEN, ENC_LAYERS, DEC_LAYERS, N_HEADS, DROPOUT,
    DEVICE, SCHUDULER_ON, OPTIMIZER_NAME, PATIENCE, LR, N_EPOCHS,
    CHECKPOINT_FREQ, evaluate, process_data, generate_data,
    TextLoader, TextCollate, log_metrics
)


def text_to_labels(s, char2idx):
    return [char2idx['SOS']] + [char2idx[i] for i in s if i in char2idx.keys()] + [char2idx['EOS']]


def train(model, optimizer, criterion, train_loader):
    model.train()
    epoch_loss = 0
    for src, trg in train_loader:
        src, trg = src.to(DEVICE), trg.to(DEVICE)
        optimizer.zero_grad()
        output = model(src, trg[:-1, :])

        loss = criterion(output.view(-1, output.shape[-1]), torch.reshape(trg[1:, :], (-1,)))
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    return epoch_loss / len(train_loader)


def fit(model, optimizer, scheduler, criterion, train_loader, val_loader, start_epoch=0, end_epoch=24):
    metrics = []
    for epoch in range(start_epoch, end_epoch):
        epoch_metrics = {}
        start_time = time()
        train_loss = train(model, optimizer, criterion, train_loader)
        end_time = time()
        epoch_metrics, _ = evaluate(model, criterion, val_loader)
        epoch_metrics['train_loss'] = train_loss
        epoch_metrics['epoch'] = epoch
        epoch_metrics['time'] = end_time - start_time
        epoch_metrics['lr'] = optimizer.param_groups[0]["lr"]
        metrics.append(epoch_metrics)
        log_metrics(epoch_metrics, TRAIN_LOG)
        if scheduler is not None:
            scheduler.step(epoch_metrics['loss'])
    return metrics


if __name__ == '__main__':
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    random = torch.Generator().manual_seed(42)

    char2idx = {char: idx for idx, char in enumerate(ALPHABET)}
    idx2char = {idx: char for idx, char in enumerate(ALPHABET)}

    # === Загрузка данных ===
    print("Загружаем обучающие данные...")
    train_img2label, _, _ = process_data(PATH_TRAIN_DIR, PATH_TRAIN_LABELS)
    train_images = generate_data(list(train_img2label.keys()))
    train_labels = list(train_img2label.values())
    train_dataset = TextLoader(train_images, train_labels, TRAIN_TRANSFORMS, char2idx, idx2char)
    train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True,
                                               batch_size=BATCH_SIZE, pin_memory=True,
                                               drop_last=True, collate_fn=TextCollate())

    print("Загружаем тестовые данные...")
    test_img2label, _, _ = process_data(PATH_TEST_DIR, PATH_TEST_LABELS)
    test_images = generate_data(list(test_img2label.keys()))
    test_labels = list(test_img2label.values())
    test_dataset = TextLoader(test_images, test_labels, TEST_TRANSFORMS, char2idx, idx2char)
    test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=True,
                                              batch_size=BATCH_SIZE, pin_memory=True,
                                              drop_last=True, collate_fn=TextCollate())

    # === Создание модели ===
    model = TransformerModel(
        outtoken=len(ALPHABET),
        hidden=HIDDEN,
        enc_layers=ENC_LAYERS,
        dec_layers=DEC_LAYERS,
        nhead=N_HEADS,
        dropout=DROPOUT
    ).to(DEVICE)

    # Загрузка чекпоинта (если есть)
    checkpoints = [f for f in os.listdir(CHECKPOINT_PATH) if f.startswith('AiVisionBotcheckpoint_') and f.endswith('.pt')]
    start_epoch = 0
    if checkpoints:
        last_checkpoint = max(checkpoints, key=lambda x: int(x.split('_')[1].split('.')[0]))
        checkpoint = torch.load(os.path.join(CHECKPOINT_PATH, last_checkpoint))
        model.load_state_dict(checkpoint)
        start_epoch = int(last_checkpoint.split('_')[1].split('.')[0])
        print(f"Продолжаем обучение с {start_epoch} эпохи")
    else:
        print("Начинаем обучение с нуля.")

    criterion = torch.nn.CrossEntropyLoss(ignore_index=char2idx['PAD'])
    optimizer = torch.optim.__getattribute__(OPTIMIZER_NAME)(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=PATIENCE) if SCHUDULER_ON else None

    print(f'Чекпоинты будут сохраняться каждые {CHECKPOINT_FREQ} эпох')
    for epoch in range(start_epoch, N_EPOCHS, CHECKPOINT_FREQ):
        end_epoch = min(epoch + CHECKPOINT_FREQ, N_EPOCHS)
        fit(model, optimizer, scheduler, criterion, train_loader, test_loader, epoch, end_epoch)
        torch.save(model.state_dict(), os.path.join(CHECKPOINT_PATH, f'checkpoint_{end_epoch}.pt'))
