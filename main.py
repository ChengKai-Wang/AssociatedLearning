from tqdm import tqdm
from torch import optim
import torch
import numpy as np
from models import MLPClassifier, VGG, VGGALClassifier
import configparser
from datasets import CIFAR10

config = configparser.ConfigParser()
config.read("config.ini")
train_batch_size = int(config['data']['train_batch_size'])
test_batch_size = int(config['data']['test_batch_size'])
gen_datasets = {"CIFAR10": CIFAR10(train_batch_size, test_batch_size)}


dataset = config['data']['dataset']
train_loader = gen_datasets[dataset].get_train_loader()
test_loader = gen_datasets[dataset].get_test_loader()
train_size = gen_datasets[dataset].get_dataset_size("train")
test_size = gen_datasets[dataset].get_dataset_size("test")
n_classes = gen_datasets[dataset].get_num_classes()

if config['model']['model'] == 'VGG':
    model = VGG(n_classes)
elif config['model']['model'] == 'VGG_AL':
    model = VGGALClassifier(n_classes)

epochs = int(config['model']['epochs'])

ori_lr = float(config['model']['lr'])

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

model = model.to(device)

#optimizer = torch.optim.Adam(model.parameters(), lr=ori_lr)#, weight_decay=5e-4)
optimizer = optim.SGD(model.parameters(), lr=ori_lr, momentum=0.9)#, weight_decay=5e-4)

steps_per_epoch = int(train_size/train_batch_size)
def lr_scheduling(global_step, learning_rate):
    if ((global_step == int(steps_per_epoch * epochs * 0.4)) |
       (global_step == int(steps_per_epoch * epochs * 0.6)) |
       (global_step == int(steps_per_epoch * epochs * 0.8))):
        return learning_rate*0.1

    elif global_step == int(steps_per_epoch * epochs * 0.9):
        return learning_rate*0.5
    else:
        return learning_rate
    
def set_lr(lr):
    for g in optimizer.param_groups:
        g['lr'] = lr

best_acc = 0
current_lr = ori_lr
losses = []
accs = []

for epoch in tqdm(range(epochs), ncols= 80):
    if config['model']['model'] == 'VGG_AL':
        total_loss = {"b":[0, 0, 0, 0], "ae":[0, 0, 0, 0]}
    else:
        total_loss = 0
    model.train()
    for step, (X, Y) in enumerate(train_loader):
        global_steps = (epoch+1) * (step+1)
        current_lr = lr_scheduling(global_steps, current_lr)
        set_lr(current_lr)
        loss = model(X.to(device), Y.to(device))
        if config['model']['model'] == 'VGG_AL':
            local_loss = sum(loss["b"]) + sum(loss["ae"])
        else:
            local_loss = loss
        optimizer.zero_grad()
        local_loss.backward()
        optimizer.step()
        if config['model']['model'] == 'VGG_AL':
            total_loss["b"] = [a+b for a, b in zip(loss["b"], total_loss["b"])]
            total_loss["ae"] = [a+b for a, b in zip(loss["ae"], total_loss["ae"])]
        else:
            total_loss += loss
    test_correct_labels = 0
    model.eval()
    with torch.no_grad():
        for step, (X, Y) in enumerate(test_loader):
            predict = model(X.to(device), Y.to(device))
            _, predict = torch.max(predict, 1)
            num_correct = torch.sum(predict == Y.to(device))
            test_correct_labels += num_correct
    if (100*test_correct_labels/test_size) > best_acc:
        best_acc = 100*test_correct_labels/test_size
    if config['model']['model'] == 'VGG_AL':
        losses.append(float(sum(total_loss["b"] + total_loss["ae"])))
    else:
        losses.append(float(total_loss))
    accs.append(float(100*test_correct_labels/test_size))
    print("\nlearning rate: {:.6f}".format(current_lr))
    torch.save(model.state_dict(), "./savemodels/model_"+str(epoch)+".pt")
    print("Train Loss is:", total_loss)
    print("Test Accuracy is:{:.4f}%".format(100*test_correct_labels/test_size))
print("Best Accuracy: {:.4f}%".format(best_acc))

np.save("./losses", np.array(losses))
np.save("./acc", np.array(accs))
