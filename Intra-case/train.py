import pickle
import torch
import numpy as np
from sklearn.metrics import roc_curve, auc, roc_auc_score, f1_score
from torch.nn import Parameter
from torch.nn.init import xavier_uniform_, zeros_
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Intracase_Process(object):
    def __init__(
        self,
        eventVector: torch.tensor,
        eventActivityNumberList: list,
        processAttributeVector: torch.tensor,
        target: 'int8'
    ):
        self.eventVector = eventVector
        self.eventActivityNumberList = eventActivityNumberList
        self.processAttributeVector = processAttributeVector
        self.target = target

class Preprocessed_Dataset(object):
    def __init__(
        self,
        activityNames: list,
        eventAttributeNames: list,
        processAttributeNames: list,
        processes: list,
    ):
        self.activityNames = activityNames
        self.numActivities = len(activityNames)
        self.eventAttributeNames = eventAttributeNames
        self.numEventAttributes = len(eventAttributeNames)
        self.processAttributeNames = processAttributeNames
        self.numProcessAttributes = len(processAttributeNames)
        self.processes = processes
        self.numProcesses = len(processes)

train_dataset= []
val_dataset = []
test_dataset = []

with open("Preprocessed Logs for Intra-case/BPIC 2015/BPIC 2015 1/2015_1_temporal_train_with_all_seed=0.txt", "rb") as fp:   # Unpickling
  train_dataset = pickle.load(fp)

with open("Preprocessed Logs for Intra-case/BPIC 2015/BPIC 2015 1/2015_1_temporal_validation_with_all_seed=0.txt", "rb") as fp:   # Unpickling
  val_dataset = pickle.load(fp)

with open("Preprocessed Logs for Intra-case/BPIC 2015/BPIC 2015 1/2015_1_temporal_test_with_all_seed=0.txt", "rb") as fp:   # Unpickling
  test_dataset = pickle.load(fp)

print("Length of training dataset: {}".format(len(train_dataset.processes)))
print("Length of validation dataset: {}".format(len(val_dataset.processes)))
print("Length of test dataset: {}".format(len(test_dataset.processes)))
print("")

class TempGCN(torch.nn.Module):

    def __init__(
        self,
        N: int,
        in_channels: int,
        out_channels: int,
    ):
        super(TempGCN, self).__init__()

        self.N = N
        self.in_channels = in_channels
        self.out_channels = out_channels
        self._create_states_and_parameters()
        self._set_parameters()


    def _create_update_gate_parameters(self):

        self.W_x_z = Parameter(torch.Tensor(self.in_channels, self.out_channels).to(device))
        self.W_h_z = Parameter(torch.Tensor(self.out_channels, self.out_channels).to(device))
        self.conv_z = Parameter(torch.Tensor(self.out_channels, self.out_channels).to(device))
        self.b_z = Parameter(torch.Tensor(1, self.out_channels).to(device))

    def _create_reset_gate_parameters(self):

        self.W_x_r = Parameter(torch.Tensor(self.in_channels, self.out_channels).to(device))
        self.W_h_r = Parameter(torch.Tensor(self.out_channels, self.out_channels).to(device))
        self.conv_r = Parameter(torch.Tensor(self.out_channels, self.out_channels).to(device))
        self.b_r = Parameter(torch.Tensor(1, self.out_channels).to(device))

    def _create_candidate_state_parameters(self):

        self.W_x_h = Parameter(torch.Tensor(self.in_channels, self.out_channels).to(device))
        self.W_h_h = Parameter(torch.Tensor(self.out_channels, self.out_channels).to(device))
        self.b_h = Parameter(torch.Tensor(1, self.out_channels).to(device))

    def _create_states_and_parameters(self):
        self._create_update_gate_parameters()
        self._create_reset_gate_parameters()
        self._create_candidate_state_parameters()

    def _create_edge_weights_matrix(self):
        return torch.zeros([self.N, self.N]).to(device)

    def _create_hidden_state_matrix(self):
        return torch.zeros([self.N, self.out_channels]).to(device)

    def _set_parameters(self):
        xavier_uniform_(self.W_x_z)
        xavier_uniform_(self.W_h_z)
        xavier_uniform_(self.conv_z)
        xavier_uniform_(self.W_x_r)
        xavier_uniform_(self.W_h_r)
        xavier_uniform_(self.conv_r)
        xavier_uniform_(self.W_x_h)
        xavier_uniform_(self.W_h_h)
        zeros_(self.b_z)
        zeros_(self.b_r)
        zeros_(self.b_h)

    def _update_edge_weights(self, head, tail, E):
        if tail is None:
            return E
        edge_update_matrix = 0.5*torch.eye(self.N).to(device)
        edge_update_matrix[head][tail] = 0.5
        E_new = E.clone()
        E_new[head][tail] = E_new[head][tail] + 1
        E_new = torch.matmul(edge_update_matrix, E_new)
        return E_new

    def _calculate_update_gate(self, X, head, H, E):
        Z = torch.matmul(X, self.W_x_z)
        Z = Z + torch.matmul(H[head], self.W_h_z)
        Z = Z + torch.matmul(torch.matmul(E[head], H), self.conv_z)
        Z = Z + self.b_z
        Z = torch.sigmoid(Z)
        return Z

    def _calculate_reset_gate(self, X, head, H, E):
        R = torch.matmul(X, self.W_x_r)
        R = R + torch.matmul(H[head], self.W_h_r)
        R = R + torch.matmul(torch.matmul(E[head], H), self.conv_r)
        R = R + self.b_r
        R = torch.sigmoid(R)
        return R

    def _calculate_candidate_state(self, X, head, H, R):
        H_tilde = torch.matmul(X, self.W_x_h)
        H_tilde = H_tilde + torch.matmul(H[head] * R, self.W_h_h)
        H_tilde = H_tilde + self.b_h
        H_tilde = torch.tanh(H_tilde)
        return H_tilde

    def _calculate_hidden_state(self, head, H, H_tilde, Z):
        H_new = H.clone()
        H_new[head] = Z * H[head] + (1 - Z) * H_tilde
        return H_new

    def forward(
        self,
        X: torch.FloatTensor,
        head: int,
        tail: int = None,
        E: torch.FloatTensor = None,
        H: torch.FloatTensor = None,
    ) -> torch.FloatTensor:

        if E is None:
            E = self._create_edge_weights_matrix()
        if H is None:
            H = self._create_hidden_state_matrix()

        E_new = self._update_edge_weights(head, tail, E)
        Z = self._calculate_update_gate(X, head, H, E_new)
        R = self._calculate_reset_gate(X, head, H, E_new)
        H_tilde = self._calculate_candidate_state(X, head, H, R)
        H_new = self._calculate_hidden_state(head, H, H_tilde, Z)
        return E_new, H_new

class TemporalGNN(torch.nn.Module):
    def __init__(self, out_channels_GNN, hidden_layer_event_attribute_channels, num_activities, numEventAttributes, numProcessAttributes):

        super(TemporalGNN,self).__init__()
        self.out_channels_GNN = out_channels_GNN
        self.num_activities = num_activities
        self.numEventAttributes = numEventAttributes
        self.gnn = TempGCN(N = num_activities, in_channels = numEventAttributes, out_channels = out_channels_GNN)
        self.fc1 = torch.nn.Linear(out_channels_GNN*num_activities, hidden_layer_event_attribute_channels)
        self.dropout1 = torch.nn.Dropout(p=0.5)
        self.fc2 = torch.nn.Linear(hidden_layer_event_attribute_channels + numProcessAttributes, 2)
        self.dropout2 = torch.nn.Dropout(p=0.3)
        self.lrelu = torch.nn.LeakyReLU()
        self.lsoftmax = torch.nn.LogSoftmax(dim=0)

    def forward(self, eventVector, processAttributeVector, eventActivityNumber, prevEventActivityNumber = -1, E = None, H = None):

        if E is None:
            E, H = self.gnn(X = eventVector, head = eventActivityNumber)
        else:
            E, H = self.gnn(X = eventVector, head = eventActivityNumber, tail = prevEventActivityNumber, E = E, H = H)

        out = torch.flatten(H)
        out = self.dropout1(out)
        out = self.fc1(out)
        out = self.lrelu(out)
        out = torch.cat((out, processAttributeVector))
        out = self.dropout2(out)
        out = self.fc2(out)
        out = self.lsoftmax(out)

        return out, E, H

model = TemporalGNN(
        out_channels_GNN = train_dataset.numEventAttributes//2,
        hidden_layer_event_attribute_channels = train_dataset.numEventAttributes//10,
        num_activities = train_dataset.numActivities,
        numEventAttributes = train_dataset.numEventAttributes,
        numProcessAttributes = train_dataset.numProcessAttributes
    )

print("Number of activities: {}".format(train_dataset.numActivities))
print("Number of event attributes: {}".format(train_dataset.numEventAttributes))
print("Number of process attributes: {}".format(train_dataset.numProcessAttributes))
print("")

print("Number of trainable parameters: {}".format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

optimizer = torch.optim.Adam(model.parameters(), lr=0.00001, weight_decay=1e-5)
loss_func = torch.nn.NLLLoss()

model = model.to(device)

#If prefixLength is None, combined results for all prefix lengths from 1 to maxPrefixLength are computed

@torch.no_grad()
def evaluate(model, dataset, loss_func, maxPrefixLength, prefixLength = None, threshold = None):

    model.eval()

    total_0 = 0
    total_1 = 0
    correct_0 = 0
    correct_1 = 0
    loss = 0

    y_true = []
    y_pred = []

    for process in dataset.processes:

        y = torch.unsqueeze(torch.tensor(process.target),0).type(torch.LongTensor)
        y = y.to(device)

        for i in range(min(maxPrefixLength, len(process.eventActivityNumberList))):

            if i == 0:
                x, E, H = model(
                    process.eventVector[i].to(device),
                    torch.tensor(process.processAttributeVector).to(device),
                    torch.tensor(process.eventActivityNumberList[i]).to(device)
                )
            else:
                x, E, H = model(
                    process.eventVector[i].to(device),
                   torch.tensor(process.processAttributeVector).to(device),
                   torch.tensor(process.eventActivityNumberList[i]).to(device),
                   torch.tensor(process.eventActivityNumberList[i-1]).to(device),
                   E, H
                )

            if prefixLength is None or i == prefixLength - 1:
                x = torch.unsqueeze(x, 0)
                loss += loss_func(x,y)
                y_true.append(y.item())
                y_pred.append(x[0][1].item())

                if prefixLength is not None:
                    break


    print(len(y_true), len(y_pred))
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    AUC = auc(fpr, tpr)

    optimal_threshold = threshold
    if threshold is None:
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]

    y_classifcation = [1 if val >= optimal_threshold else 0 for val in y_pred]
    f1 = f1_score(y_true, y_classifcation)

    for i in range(len(y_true)):
        if y_true[i] == 0:
            total_0 += 1
            if y_pred[i] < optimal_threshold:
                correct_0 += 1

        else:
            total_1 += 1
            if y_pred[i] >= optimal_threshold:
                correct_1 += 1

    print('Loss:', loss)
    print('Accuracy of 0 (TNR):', correct_0/total_0)
    print('Accuracy of 1 (TPR):', correct_1/total_1)
    print('Overall accuracy:', (correct_0 + correct_1)/(total_0 + total_1))
    print('AUC:', AUC)
    print('F1 Score:', f1)
    print('Threshold:', optimal_threshold)
    return loss, AUC

AUC_train = []
AUC_val = []
loss_train = []
loss_val = []

def train_model(model, optimizer, loss_func, epochs, maxPrefixLength, weights_save_path, train_dataset, val_dataset):

    model.train()
    min_val_loss = 1e9
    train_data_length = train_dataset.numProcesses

    for epoch in range(epochs):

        print("\n--------------------------\n")
        print("Epoch : ",epoch+1)

        cnt = 0

        while(cnt < train_data_length):

            process = train_dataset.processes[cnt]
            y = torch.unsqueeze(torch.tensor(process.target),0).type(torch.LongTensor)
            y = y.to(device)
            range_limit = min(maxPrefixLength, len(process.eventActivityNumberList))

            for i in range(range_limit):

                if i == 0:
                    x, E, H = model(
                        process.eventVector[i].to(device),
                        torch.tensor(process.processAttributeVector).to(device),
                        torch.tensor(process.eventActivityNumberList[i]).to(device)
                    )
                else:
                    x, E, H = model(
                        process.eventVector[i].to(device),
                    torch.tensor(process.processAttributeVector).to(device),
                    torch.tensor(process.eventActivityNumberList[i]).to(device),
                    torch.tensor(process.eventActivityNumberList[i-1]).to(device),
                    E, H
                    )

                x = torch.unsqueeze(x, 0)
                loss = loss_func(x,y)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                H = H.detach()

            cnt += 1


        print("\nTraining Results\n")
        train_loss, train_eval = evaluate(model, train_dataset, loss_func, maxPrefixLength)
        loss_train.append(train_loss.item())
        AUC_train.append(train_eval)

        print("\nValidation Results\n")
        val_loss, val_eval = evaluate(model, val_dataset, loss_func, maxPrefixLength)
        loss_val.append(val_loss.item())
        AUC_val.append(val_eval)

        if val_loss<min_val_loss:
            min_val_loss = val_loss
            torch.save(model.state_dict(), weights_save_path)


train_model(model, optimizer, loss_func, 200, 36, "2015_1.pt", train_dataset, val_dataset)

def plot_AUC(AUC_train, AUC_val):
    plt.plot(AUC_train, '-rx')
    plt.plot(AUC_val, '-bx')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.title('AUC vs. No. of epochs')
    plt.legend(['Training', 'Validation'])

plot_AUC(AUC_train, AUC_val)

def plot_loss(loss_train, loss_val):
    plt.plot(loss_train, '-rx')
    plt.plot(loss_val, '-bx')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss vs. No. of epochs')
    plt.legend(['Training', 'Validation'])

plot_loss(loss_train, loss_val)
