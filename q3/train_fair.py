import torch
import torch.nn as nn

# Dummy model
model = nn.Linear(40, 10)

criterion = nn.CrossEntropyLoss()

def fairness_loss(outputs, labels, group_labels):
    loss = 0.0
    groups = set(group_labels.tolist())

    for g in groups:
        idx = (group_labels == g)
        if idx.sum() > 0:
            loss += criterion(outputs[idx], labels[idx])

    return loss / len(groups)

# Dummy batch
outputs = torch.randn(32, 10)
labels = torch.randint(0, 10, (32,))
group_labels = torch.randint(0, 2, (32,))  # gender groups

lambda_fair = 0.1

loss_main = criterion(outputs, labels)
loss_fair = fairness_loss(outputs, labels, group_labels)

loss = loss_main + lambda_fair * loss_fair

print("Main Loss:", loss_main.item())
print("Fairness Loss:", loss_fair.item())
print("Total Loss:", loss.item())