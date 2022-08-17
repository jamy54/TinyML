<<<<<<< HEAD
import argparse
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from help_code_demo import ToTensor, IEGM_DataSET
from models.model_1 import IEGMNet, IEGMNet_2, IEGMNet_3, IEGMNet_4, IEGMNet_5_origin, IEGMNet_2_1, IEGMNet_ref_5
import numpy as np
from pytorch_model_summary import summary
from torchmetrics.functional import precision_recall, f1_score

def F_Beta_Score(precision,recall,beta):
    return (1+beta*beta)*precision*recall/((beta*beta*precision)+recall)

def prec_recall(predicted, target):
    t_p_n_prediction = predicted[predicted == target]
    positive_label = target[target == 1]
    t_f_p_prediction = predicted[predicted == 1]
    t_p_prediction = t_p_n_prediction[t_p_n_prediction == 1]
    precision = t_p_prediction.sum() / t_f_p_prediction.sum()
    recall = t_p_prediction.sum() / positive_label.sum()
    return precision, recall

def prec_recall_internet(prob, label):
    prob = prob.bool()
    label = label.bool()
    epsilon = 1e-7
    TP = (prob & label).sum().float()
    TN = ((~prob) & (~label)).sum().float()
    FP = (prob & (~label)).sum().float()
    FN = ((~prob) & label).sum().float()
    #accuracy = (TP+TN)/(TP+TN+FP+FN)
    precision = torch.mean(TP / (TP + FP + epsilon))
    recall = torch.mean(TP / (TP + FN + epsilon))
    #F2 = 2 * precision * recall / (precision + recall + epsilon)
    return precision, recall, #F2

def get_model(number):
    if number ==1:
        return IEGMNet()
    elif number == 2:
        return IEGMNet_2()
    elif number == 3:
        return IEGMNet_3()
    elif number == 4:
        return IEGMNet_4()
    elif number == 5:
        return IEGMNet_5_origin()
    elif number == 6:
        return  IEGMNet_2_1()
    elif number == 7:
        return IEGMNet_ref_5()
    else:
        return IEGMNet()

def main(model_number):
    # Hyperparameters
    BATCH_SIZE = args.batchsz
    BATCH_SIZE_TEST = args.batchsz
    LR = args.lr
    EPOCH = args.epoch
    SIZE = args.size
    path_data = args.path_data
    path_indices = args.path_indices

    # Instantiating NN
    net = get_model(model_number)
    net.train()
    net = net.float().to(device)


    # Start dataset loading
    trainset = IEGM_DataSET(root_dir=path_data,
                            indice_dir=path_indices,
                            mode='train',
                            size=SIZE,
                            transform=transforms.Compose([ToTensor()]))

    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    testset = IEGM_DataSET(root_dir=path_data,
                           indice_dir=path_indices,
                           mode='test',
                           size=SIZE,
                           transform=transforms.Compose([ToTensor()]))

    testloader = DataLoader(testset, batch_size=BATCH_SIZE_TEST, shuffle=True, num_workers=0)

    print("Training Dataset loading finish.")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=LR)
    #optimizer = torch.optim.SGD(net.parameters(), lr=LR, momentum=0.9)
    epoch_num = EPOCH

    Train_loss = []
    Train_acc = []
    Train_prec = []
    Train_recall = []
    Train_beta_score = []

    Test_loss = []
    Test_acc = []
    Test_prec = []
    Test_recall = []
    Test_beta_score = []


    print("Start training")
    for epoch in range(epoch_num):  # loop over the dataset multiple times (specify the #epoch)

        running_loss = 0.0
        correct = 0.0
        accuracy = 0.0
        i = 0
        precision = [0,0,0]
        recall = [0,0,0]
        for j, data in enumerate(trainloader, 0):
            inputs, labels = data['IEGM_seg'], data['label']
            inputs = inputs.float().to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum()
            accuracy += correct / BATCH_SIZE
            correct = 0.0

            p, r = prec_recall_internet(predicted,labels)
            precision[0] += p
            recall[0] +=r
            #precision[1], recall[1] = prec_recall(predicted, labels)
            #precision[2] = precision_recall(predicted, labels, average=None, num_classes=2)[0][1]
            #recall[2] = precision_recall(predicted, labels, average=None, num_classes=2)[1][1]

            running_loss += loss.item()
            i += 1
            #print('[Epoch, Batches] is [%d, %5d] \nTrain Acc: %.5f Train loss: %.5f Train Precisoin: %.5f Train Recall: %.5f' %
                  #(epoch + 1, i, accuracy / i, running_loss / i, precision[0]/i, recall[0]/i))

        print('[Epoch, Batches] is [%d, %5d] \nTrain Acc: %.5f Train loss: %.5f Train Precision: %.5f Train Recall: %.5f' %
            (epoch + 1, i, accuracy / i, running_loss / i, precision[0] / i, recall[0] / i))

        Train_loss.append(running_loss / i)
        Train_acc.append((accuracy / i).item())
        Train_prec.append(precision[0] / i)
        Train_recall.append(recall[0] / i)
        Train_beta_score.append(F_Beta_Score(precision[0]/ i,recall[0]/ i,2))

        running_loss = 0.0
        accuracy = 0.0

        correct = 0.0
        total = 0.0
        i = 0.0
        running_loss_test = 0.0
        precision[0] = recall[0] = 0.0

        for data_test in testloader:
            net.eval()
            IEGM_test, labels_test = data_test['IEGM_seg'], data_test['label']
            IEGM_test = IEGM_test.float().to(device)
            labels_test = labels_test.to(device)
            outputs_test = net(IEGM_test)
            _, predicted_test = torch.max(outputs_test.data, 1)
            total += labels_test.size(0)
            correct += (predicted_test == labels_test).sum()
            p, r = prec_recall_internet(predicted_test, labels_test)
            precision[0] += p
            recall[0] += r

            loss_test = criterion(outputs_test, labels_test)
            running_loss_test += loss_test.item()
            i += 1

        print('Test Acc: %.5f Test Loss: %.5f Test Precisoin: %.5f Test Recall: %.5f' % (correct / total, running_loss_test / i,precision[0] / i, recall[0] / i))

        Test_loss.append(running_loss_test / i)
        Test_acc.append((correct / total).item())
        Test_prec.append(precision[0] / i)
        Test_recall.append(recall[0] / i)
        Test_beta_score.append(F_Beta_Score(precision[0]/ i, recall[0]/ i, 2))

    torch.save(net, './saved_models/IEGM_net.pkl')
    torch.save(net.state_dict(), './saved_models/IEGM_net_state_dict.pkl')

    n = np.zeros(shape=(1, 1, 1250, 1), dtype=np.float32)
    sumry = summary(net, torch.tensor(n).to(device))
    print(sumry)

    file = open('./saved_models/model_{0}_info.txt'.format(model_number), 'a')
    file.write("Train_loss\n")
    file.write(str(Train_loss))
    file.write('\n\n')
    file.write("Train_acc\n")
    file.write(str(Train_acc))
    file.write('\n\n')
    file.write("Train_Precision\n")
    file.write(str(torch.Tensor(Train_prec).tolist()))
    file.write('\n\n')
    file.write("Train_Recall\n")
    file.write(str(torch.Tensor(Train_recall).tolist()))
    file.write('\n\n')
    file.write("Train Beta Score \n")
    file.write(str(torch.Tensor(Train_beta_score).tolist()))
    file.write('\n\n')
    file.write("Test_loss\n")
    file.write(str(Test_loss))
    file.write('\n\n')
    file.write("Test_acc\n")
    file.write(str(Test_acc))
    file.write('\n\n')
    file.write("Test_Precision\n")
    file.write(str(torch.Tensor(Test_prec).tolist()))
    file.write('\n\n')
    file.write("Test_Recall\n")
    file.write(str(torch.Tensor(Test_recall).tolist()))
    file.write('\n\n')
    file.write("Test Beta Score \n")
    file.write(str(torch.Tensor(Test_beta_score).tolist()))
    file.write('\n\n')
    file.write(sumry)
    file.write('\n\n')
    file.close()

    print('Finish training')

    return Test_acc[len(Test_acc)-1], Test_beta_score[len(Test_beta_score)-1]


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, help='epoch number', default=15)
    argparser.add_argument('--lr', type=float, help='learning rate', default=0.0001)
    argparser.add_argument('--batchsz', type=int, help='total batch sz for traindb', default=15)
    argparser.add_argument('--cuda', type=int, default=0)
    argparser.add_argument('--size', type=int, default=1250)
    argparser.add_argument('--path_data', type=str, default='C:/Users/kisho/OneDrive/Desktop/autoformer/TinyML-Contest/tinyml_contest2022_demo_example/tinyml_contest_data_training/')
    argparser.add_argument('--path_indices', type=str, default='./data_indices')

    args = argparser.parse_args()

    device = torch.device("cuda:" + str(args.cuda))

    print("device is --------------", device)

    for model_number in [1,2]:
        for ep in [25, 30, 50]:
            args.epoch = ep
            score = []
            for batch_size in [25]:
                args.batchsz = batch_size
                score.append(main(model_number))
                print("Batch size: "+str(batch_size))
                print("Epoch number is: " + str(ep))
                print("Final Score is: "+str(score))
                print("Learning rate is: " + str(args.lr))

        file = open('./saved_models/model_{0}_info.txt'.format(model_number), 'a')
        file.write(str(torch.Tensor(score).tolist()))
        file.write('\n\n')
        file.close()
=======
import argparse
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from help_code_demo import ToTensor, IEGM_DataSET
from models.model_1 import IEGMNet, IEGMNet_2, IEGMNet_3, IEGMNet_4, IEGMNet_5_origin, IEGMNet_2_1, IEGMNet_ref_5
import numpy as np
from pytorch_model_summary import summary
from torchmetrics.functional import precision_recall, f1_score

def F_Beta_Score(precision,recall,beta):
    return (1+beta*beta)*precision*recall/((beta*beta*precision)+recall)

def prec_recall(predicted, target):
    t_p_n_prediction = predicted[predicted == target]
    positive_label = target[target == 1]
    t_f_p_prediction = predicted[predicted == 1]
    t_p_prediction = t_p_n_prediction[t_p_n_prediction == 1]
    precision = t_p_prediction.sum() / t_f_p_prediction.sum()
    recall = t_p_prediction.sum() / positive_label.sum()
    return precision, recall

def prec_recall_internet(prob, label):
    prob = prob.bool()
    label = label.bool()
    epsilon = 1e-7
    TP = (prob & label).sum().float()
    TN = ((~prob) & (~label)).sum().float()
    FP = (prob & (~label)).sum().float()
    FN = ((~prob) & label).sum().float()
    #accuracy = (TP+TN)/(TP+TN+FP+FN)
    precision = torch.mean(TP / (TP + FP + epsilon))
    recall = torch.mean(TP / (TP + FN + epsilon))
    #F2 = 2 * precision * recall / (precision + recall + epsilon)
    return precision, recall, #F2

def get_model(number):
    if number ==1:
        return IEGMNet()
    elif number == 2:
        return IEGMNet_2()
    elif number == 3:
        return IEGMNet_3()
    elif number == 4:
        return IEGMNet_4()
    elif number == 5:
        return IEGMNet_5_origin()
    elif number == 6:
        return  IEGMNet_2_1()
    elif number == 7:
        return IEGMNet_ref_5()
    else:
        return IEGMNet()

def main(model_number):
    # Hyperparameters
    BATCH_SIZE = args.batchsz
    BATCH_SIZE_TEST = args.batchsz
    LR = args.lr
    EPOCH = args.epoch
    SIZE = args.size
    path_data = args.path_data
    path_indices = args.path_indices

    # Instantiating NN
    net = get_model(model_number)
    net.train()
    net = net.float().to(device)


    # Start dataset loading
    trainset = IEGM_DataSET(root_dir=path_data,
                            indice_dir=path_indices,
                            mode='train',
                            size=SIZE,
                            transform=transforms.Compose([ToTensor()]))

    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    testset = IEGM_DataSET(root_dir=path_data,
                           indice_dir=path_indices,
                           mode='test',
                           size=SIZE,
                           transform=transforms.Compose([ToTensor()]))

    testloader = DataLoader(testset, batch_size=BATCH_SIZE_TEST, shuffle=True, num_workers=0)

    print("Training Dataset loading finish.")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=LR)
    #optimizer = torch.optim.SGD(net.parameters(), lr=LR, momentum=0.9)
    epoch_num = EPOCH

    Train_loss = []
    Train_acc = []
    Train_prec = []
    Train_recall = []
    Train_beta_score = []

    Test_loss = []
    Test_acc = []
    Test_prec = []
    Test_recall = []
    Test_beta_score = []


    print("Start training")
    for epoch in range(epoch_num):  # loop over the dataset multiple times (specify the #epoch)

        running_loss = 0.0
        correct = 0.0
        accuracy = 0.0
        i = 0
        precision = [0,0,0]
        recall = [0,0,0]
        for j, data in enumerate(trainloader, 0):
            inputs, labels = data['IEGM_seg'], data['label']
            inputs = inputs.float().to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum()
            accuracy += correct / BATCH_SIZE
            correct = 0.0

            p, r = prec_recall_internet(predicted,labels)
            precision[0] += p
            recall[0] +=r
            #precision[1], recall[1] = prec_recall(predicted, labels)
            #precision[2] = precision_recall(predicted, labels, average=None, num_classes=2)[0][1]
            #recall[2] = precision_recall(predicted, labels, average=None, num_classes=2)[1][1]

            running_loss += loss.item()
            i += 1
            #print('[Epoch, Batches] is [%d, %5d] \nTrain Acc: %.5f Train loss: %.5f Train Precisoin: %.5f Train Recall: %.5f' %
                  #(epoch + 1, i, accuracy / i, running_loss / i, precision[0]/i, recall[0]/i))

        print('[Epoch, Batches] is [%d, %5d] \nTrain Acc: %.5f Train loss: %.5f Train Precision: %.5f Train Recall: %.5f' %
            (epoch + 1, i, accuracy / i, running_loss / i, precision[0] / i, recall[0] / i))

        Train_loss.append(running_loss / i)
        Train_acc.append((accuracy / i).item())
        Train_prec.append(precision[0] / i)
        Train_recall.append(recall[0] / i)
        Train_beta_score.append(F_Beta_Score(precision[0]/ i,recall[0]/ i,2))

        running_loss = 0.0
        accuracy = 0.0

        correct = 0.0
        total = 0.0
        i = 0.0
        running_loss_test = 0.0
        precision[0] = recall[0] = 0.0

        for data_test in testloader:
            net.eval()
            IEGM_test, labels_test = data_test['IEGM_seg'], data_test['label']
            IEGM_test = IEGM_test.float().to(device)
            labels_test = labels_test.to(device)
            outputs_test = net(IEGM_test)
            _, predicted_test = torch.max(outputs_test.data, 1)
            total += labels_test.size(0)
            correct += (predicted_test == labels_test).sum()
            p, r = prec_recall_internet(predicted_test, labels_test)
            precision[0] += p
            recall[0] += r

            loss_test = criterion(outputs_test, labels_test)
            running_loss_test += loss_test.item()
            i += 1

        print('Test Acc: %.5f Test Loss: %.5f Test Precisoin: %.5f Test Recall: %.5f' % (correct / total, running_loss_test / i,precision[0] / i, recall[0] / i))

        Test_loss.append(running_loss_test / i)
        Test_acc.append((correct / total).item())
        Test_prec.append(precision[0] / i)
        Test_recall.append(recall[0] / i)
        Test_beta_score.append(F_Beta_Score(precision[0]/ i, recall[0]/ i, 2))

    torch.save(net, './saved_models/IEGM_net.pkl')
    torch.save(net.state_dict(), './saved_models/IEGM_net_state_dict.pkl')

    n = np.zeros(shape=(1, 1, 1250, 1), dtype=np.float32)
    sumry = summary(net, torch.tensor(n).to(device))
    print(sumry)

    file = open('./saved_models/model_{0}_info.txt'.format(model_number), 'a')
    file.write("Train_loss\n")
    file.write(str(Train_loss))
    file.write('\n\n')
    file.write("Train_acc\n")
    file.write(str(Train_acc))
    file.write('\n\n')
    file.write("Train_Precision\n")
    file.write(str(torch.Tensor(Train_prec).tolist()))
    file.write('\n\n')
    file.write("Train_Recall\n")
    file.write(str(torch.Tensor(Train_recall).tolist()))
    file.write('\n\n')
    file.write("Train Beta Score \n")
    file.write(str(torch.Tensor(Train_beta_score).tolist()))
    file.write('\n\n')
    file.write("Test_loss\n")
    file.write(str(Test_loss))
    file.write('\n\n')
    file.write("Test_acc\n")
    file.write(str(Test_acc))
    file.write('\n\n')
    file.write("Test_Precision\n")
    file.write(str(torch.Tensor(Test_prec).tolist()))
    file.write('\n\n')
    file.write("Test_Recall\n")
    file.write(str(torch.Tensor(Test_recall).tolist()))
    file.write('\n\n')
    file.write("Test Beta Score \n")
    file.write(str(torch.Tensor(Test_beta_score).tolist()))
    file.write('\n\n')
    file.write(sumry)
    file.write('\n\n')
    file.close()

    print('Finish training')

    return Test_acc[len(Test_acc)-1], Test_beta_score[len(Test_beta_score)-1]


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, help='epoch number', default=15)
    argparser.add_argument('--lr', type=float, help='learning rate', default=0.0001)
    argparser.add_argument('--batchsz', type=int, help='total batch sz for traindb', default=15)
    argparser.add_argument('--cuda', type=int, default=0)
    argparser.add_argument('--size', type=int, default=1250)
    argparser.add_argument('--path_data', type=str, default='C:/Users/kisho/OneDrive/Desktop/autoformer/TinyML-Contest/tinyml_contest2022_demo_example/tinyml_contest_data_training/')
    argparser.add_argument('--path_indices', type=str, default='./data_indices')

    args = argparser.parse_args()

    device = torch.device("cuda:" + str(args.cuda))

    print("device is --------------", device)

    for model_number in [1,2]:
        for ep in [25, 30, 50]:
            args.epoch = ep
            score = []
            for batch_size in [25]:
                args.batchsz = batch_size
                score.append(main(model_number))
                print("Batch size: "+str(batch_size))
                print("Epoch number is: " + str(ep))
                print("Final Score is: "+str(score))
                print("Learning rate is: " + str(args.lr))

        file = open('./saved_models/model_{0}_info.txt'.format(model_number), 'a')
        file.write(str(torch.Tensor(score).tolist()))
        file.write('\n\n')
        file.close()
>>>>>>> 21cdb4e90d75aeb04d53781e1c15b1eefa613a24
