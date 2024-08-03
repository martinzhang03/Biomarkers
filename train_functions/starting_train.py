import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

def starting_train(train_dataset, val_dataset, model, hyperparameters, n_eval):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    batch_size, epochs = hyperparameters["batch_size"], hyperparameters["epochs"]

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)  # Adjust learning rate if necessary
    # loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
    loss_fn = nn.L1Loss()

    step = 0
    # for epoch in range(epochs):
    #     print(f"Epoch {epoch + 1} of {epochs}")

    #     model.train()  
    #     # for input_data, label_data in tqdm(train_loader):
    #     for i, batch in enumerate(train_loader):
    #         input_data, label_data = batch
    #         g1 = [] # 25-
    #         g1_label = []
    #         g2 = [] # 25 - 40
    #         g2_label = []
    #         g3 = [] # 40 - 54
    #         g3_label = []
    #         g4 = [] # 54 - 65
    #         g4_label = []
    #         g5 = [] # 65+
    #         g5_label = []

    #         for i in range(len(input_data)):
    #             if int(label_data[i]) <= 25:
    #                 g1.append(input_data[i])
    #                 g1_label.append(label_data[i])
    #             elif int(label_data[i]) > 25 and int(label_data[i]) <= 40:
    #                 g2.append(input_data[i])
    #                 g2_label.append(label_data[i])
    #             elif int(label_data[i]) > 40 and int(label_data[i]) <= 54:
    #                 g3.append(input_data[i])
    #                 g3_label.append(label_data[i])
    #             elif int(label_data[i]) > 54 and int(label_data[i]) <= 65:
    #                 g4.append(input_data[i])
    #                 g4_label.append(label_data[i])
    #             elif int(label_data[i]) > 65:
    #                 g5.append(input_data[i])
    #                 g5_label.append(label_data[i])

    #         # optimizer.zero_grad()
    #         pred = model(input_data)
    #         with open('step_90.txt', 'a') as file:
    #             # file.write(f"prediction length: {len(pred)}\n")
    #             # file.write(f"each prediction length: {len(pred[0])}\n")
    #             # file.write(f"input_data: {input_data}\n")
    #             # file.write(f"input data size: {len(input_data)}, {len(input_data[0])}\n")
    #             file.write(f"prediction: {pred}\n")
    #             # file.write(f"input data: {input_data}")
    #             file.write(f"label data: {label_data}\n")
    #         # print(f"\n    Train Loss: {loss.item()}")

    #         # pred = torch.round(pred)
    #         #round_pred = torch.round(pred)
    #         loss = loss_fn(pred, label_data)
            
    #         print(f"\n    Train Loss: {loss.item()}")
    #         loss.backward()
    #         torch.nn.utils.clip_grad_norm_(loss, max_norm=1.0, norm_type=2)

    #         optimizer.step()

    #         optimizer.zero_grad()
    #         # pred = pred.argmax(axis=1)

    #         # with open('step_90.txt', 'a') as file:
    #         #     file.write(f"prediction: {pred}\n")
    #         #     file.write(f"label data: {label_data}\n")
            

            
    #         if step % n_eval == 0:
    #             train_accuracy = compute_accuracy(pred, label_data)
    #             print(f"    Train Accu: {train_accuracy}")

    #             valid_loss, valid_accuracy = evaluate(val_loader, model, loss_fn)
    #             print(f"    Valid Loss: {valid_loss}")
    #             print(f"    Valid Accu: {valid_accuracy}")
    #             with open('val_loss.txt', 'a') as f:
    #                 f.write(f"train accu: {train_accuracy}\n")
    #                 f.write(f"valid loss: {valid_loss}\n")
    #                 f.write(f"valid accu: {valid_accuracy}\n\n")
    #         model.train()
                

    #         step += 1

    val_los = []
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1} of {epochs}")

        model.train()  
        for i, batch in enumerate(train_loader):
            input_data, label_data = batch

            g1, g2, g3, g4, g5 = [], [], [], [], []
            category_indices = []

            for i in range(len(input_data)):
                if int(label_data[i]) <= 14:
                    g1.append(input_data[i])
                    category_indices.append(0)
                elif int(label_data[i]) > 14 and int(label_data[i]) <= 26:
                    g2.append(input_data[i])
                    category_indices.append(1)
                elif int(label_data[i]) > 26 and int(label_data[i]) <= 45:
                    g3.append(input_data[i])
                    category_indices.append(2)
                elif int(label_data[i]) > 45 and int(label_data[i]) <= 60:
                    g4.append(input_data[i])
                    category_indices.append(3)
                elif int(label_data[i]) > 60:
                    g5.append(input_data[i])
                    category_indices.append(4)

            category_indices = torch.tensor(category_indices).to(input_data.device)

            optimizer.zero_grad()
            pred = model(input_data, category_indices)

            with open('step_90_MST.txt', 'a') as file:
                # file.write(f"prediction length: {len(pred)}\n")
                # file.write(f"each prediction length: {len(pred[0])}\n")
                # file.write(f"input_data: {input_data}\n")
                # file.write(f"input data size: {len(input_data)}, {len(input_data[0])}\n")
                file.write(f"prediction: {pred}\n")
                # file.write(f"input data: {input_data}")
                file.write(f"label data: {label_data}\n")
            # print(f"\n    Train Loss: {loss.item()}")

            l1_lambda = 0.01
            l1_reg = torch.tensor(0.).to(device)
            for param in model.parameters():
                l1_reg += torch.norm(param, 1)
            loss = loss_fn(pred, label_data) + l1_lambda * l1_reg
            # loss = loss_fn(pred, label_data)
            
            print(f"\n    Train Loss: {loss.item()}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, norm_type=2)

            optimizer.step()

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
                val_los.append(valid_loss)
            model.train()
                
            step += 1

        print()

    # Plotting the data
    plt.plot(val_los, marker='o', linestyle='-', color='b', label='Validation Loss')

    # Adding title and labels
    plt.title('Validation Loss Over Epochs')
    plt.xlabel('Each Eval')
    plt.ylabel('Loss')

    # Adding a legend
    plt.legend()
    plt.show()
    print(val_los)

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

            # Generate category_indices for validation data
            category_indices = torch.zeros(input_data.size(0), dtype=torch.long)
            for j in range(input_data.size(0)):
                if label_data[j] <= 14:
                    category_indices[j] = 0
                elif label_data[j] <= 26:
                    category_indices[j] = 1
                elif label_data[j] <= 45:
                    category_indices[j] = 2
                elif label_data[j] <= 60:
                    category_indices[j] = 3
                else:
                    category_indices[j] = 4

            logits = model(input_data, category_indices)

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