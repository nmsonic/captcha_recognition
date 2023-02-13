
from .dataset import *
from .utils import *
from .cfg import *

cfg = 

torch.manual_seed(RANDOM_STATE)

mean_images, std_images =  get_statistics(CaptchaDataset(DATA_PATH, transform=transform_compose))


transform_compose =transforms.Compose([
    transforms.PILToTensor(),
    transforms.Lambda(lambda x: x.float()),
    transforms.Normalize(mean=[mean_images], std=[std_images])
])

full_dataset = ImageDataset(DATA_PATH, transform=transform_compose, target_transform=label_to_tensor)

train_size = int(TRAIN_SIZE * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)

print(f'Full shape: {len(full_dataset)}')
print(f'Train shape: {len(train_dataset)}')
print(f'Test shape: {len(test_dataset)}')

model = OCRModel().to(device)
opt = torch.optim.Adam(model.parameters(), lr=0.001)
loss_func = nn.CTCLoss(blank=0, zero_infinity=True, reduction='mean')

N_EPOCH = 70

history_loss = []
model.train()
for i_ep in range(len(history_loss), len(history_loss) + N_EPOCH):
    epoch_loss = []

    t = tqdm(train_loader)
    for data in t:
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        opt.zero_grad()
        outputs = model(inputs)
        loss = compute_ctc_loss(outputs, labels)

        epoch_loss.append(loss.item())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        t.set_description(f'loss={loss.item()}')

    epoch_loss = np.mean(epoch_loss)

    history_loss.append(epoch_loss)

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    clear_output(True)
    ax.plot(history_loss, label='loss')
    ax.legend()
    ax.set_title(f'EPOCH: {i_ep} LOSS: {epoch_loss}')
    plt.show()


model.eval()
# total_cer_test = 0
bad_predict = []
good_predict = []
all_labels = []
all_pred_labels = []
t = tqdm(range(len(test_dataset)))
for i in t:
    # Every data instance is an input + label pair
    inputs, labels = test_dataset[i]
    inputs, labels = inputs.to(device), labels.to(device)
    inputs, labels = inputs.unsqueeze(0), labels.unsqueeze(0)
    with torch.no_grad():
        outputs = model(inputs)

    label, pred_label, cer = get_char_error_rate(outputs, labels)
    print(f'{label=} {pred_label=} {cer}')
    all_labels.append(label)
    all_pred_labels.append(pred_label)

    t.set_description(f'cer={cer.item()}')
    # total_cer_test += cer
    if cer > 0.1:
        bad_predict.append((cer, i, pred_label))
    else:
        good_predict.append((cer, i, pred_label))

# total_cer_test /= len(test_dataset) * (LABEL_LENGTH - 1)
total_cer_test = char_error_rate(all_pred_labels, all_labels)
print('Test CER:', total_cer_test.item())