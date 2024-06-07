import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

def starting_train(train_dataset, val_dataset, model, hyperparameters, n_eval):
    batch_size, epochs = hyperparameters["batch_size"], hyperparameters["epochs"]

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adjust learning rate if necessary
    loss_fn = nn.CrossEntropyLoss(ignore_index=-1)

    step = 0
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1} of {epochs}")

        model.train()  
        for input_data, label_data in tqdm(train_loader):
            # optimizer.zero_grad()
            pred = model(input_data)
            with open('step_90.txt', 'a') as file:
                file.write(f"prediction: {pred}\n")
                file.write(f"label data: {label_data}\n")


            loss = loss_fn(pred, label_data)
            loss.backward()
            optimizer.step()

            pred = pred.argmax(axis=1)

            print(f"\n    Train Loss: {loss.item()}")

            
            if step % n_eval == 0:
                train_accuracy = compute_accuracy(pred, label_data)
                print(f"    Train Accu: {train_accuracy}")

                valid_loss, valid_accuracy = evaluate(val_loader, model, loss_fn)
                print(f"    Valid Loss: {valid_loss}")
                print(f"    Valid Accu: {valid_accuracy}")

            step += 1

        print()

    torch.save(model.state_dict(), '/Users/zhangjinyuan/Desktop/biomarkers/aging-biomarkers/trained.pth')

def compute_accuracy(outputs, labels):
    n_correct = (outputs == labels).sum().item()
    n_total = len(outputs)
    return n_correct / n_total

def evaluate(val_loader, model, loss_fn):
    model.eval()

    total_loss, total_correct, total_count = 0, 0, 0
    with torch.no_grad():
        for input_data, label_data in tqdm(val_loader):
            logits = model(input_data)

            total_loss += loss_fn(logits, label_data).mean().item()
            total_correct += (torch.argmax(logits, dim=1) == label_data).sum().item()
            total_count += len(label_data)

    validation_accuracy = total_correct / total_count

    return total_loss / len(val_loader), validation_accuracy

