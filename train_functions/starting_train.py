import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm

def starting_train(train_dataset, val_dataset, model, hyperparameters, n_eval):
    batch_size, epochs = hyperparameters["batch_size"], hyperparameters["epochs"]

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)  # Adjust learning rate if necessary
    # loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
    loss_fn = nn.L1Loss()

    step = 0
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1} of {epochs}")

        model.train()  
        # for input_data, label_data in tqdm(train_loader):
        for i, batch in enumerate(train_loader):
            input_data, label_data = batch
            # optimizer.zero_grad()
            pred = model(input_data)
            with open('step_90.txt', 'a') as file:
                # file.write(f"prediction length: {len(pred)}\n")
                # file.write(f"each prediction length: {len(pred[0])}\n")
                # file.write(f"input_data: {input_data}\n")
                # file.write(f"input data size: {len(input_data)}, {len(input_data[0])}\n")
                file.write(f"prediction: {pred}\n")
                # file.write(f"input data: {input_data}")
                file.write(f"label data: {label_data}\n")
            # print(f"\n    Train Loss: {loss.item()}")

            # pred = torch.round(pred)
            #round_pred = torch.round(pred)
            loss = loss_fn(pred, label_data)
            
            print(f"\n    Train Loss: {loss.item()}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(loss, max_norm=1.0, norm_type=2)

            optimizer.step()

            optimizer.zero_grad()
            # pred = pred.argmax(axis=1)

            # with open('step_90.txt', 'a') as file:
            #     file.write(f"prediction: {pred}\n")
            #     file.write(f"label data: {label_data}\n")
            

            
            if step % n_eval == 0:
                train_accuracy = compute_accuracy(pred, label_data)
                print(f"    Train Accu: {train_accuracy}")

                valid_loss, valid_accuracy = evaluate(val_loader, model, loss_fn)
                print(f"    Valid Loss: {valid_loss}")
                print(f"    Valid Accu: {valid_accuracy}")
                with open('val_loss.txt', 'a') as f:
                    f.write(f"train accu: {train_accuracy}\n")
                    f.write(f"valid loss: {valid_loss}\n")
                    f.write(f"valid accu: {valid_accuracy}\n\n")
            model.train()
                

            step += 1

        print()

    # torch.save(model.state_dict(), '/Users/zhangjinyuan/Desktop/biomarkers/aging-biomarkers/trained.pth')

def compute_accuracy(outputs, labels):
    pred = outputs
    # print("pred is: " + pred)
    temp = 0
    for i in range(len(labels)):
        temp = temp + abs(labels[i].item() - pred[i].item())
    return temp / len(labels)

def evaluate(val_loader, model, loss_fn):
    model.eval()

    total_loss = 0
    total_correct = 0
    total_count = 0
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            input_data, label_data = batch
            logits = model(input_data)

            # Compute loss
            loss = loss_fn(logits, label_data).mean().item()
            total_loss += loss * len(label_data)

            # Round predictions to nearest integer
            rounded_preds = torch.round(logits)

            # Calculate number of correct predictions
            total_correct += (rounded_preds == label_data).sum().item()
            total_count += len(label_data)

    average_loss = total_loss / total_count
    accuracy = total_correct / total_count

    return average_loss, accuracy