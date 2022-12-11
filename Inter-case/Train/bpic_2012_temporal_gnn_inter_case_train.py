import pickle
import torch
import numpy as np
import pandas as pd
import random
from sklearn.metrics import roc_curve, auc, roc_auc_score, f1_score

device = 'cuda' if torch.cuda.is_available() else 'cpu'

#Read dataframe
df = pd.read_csv("/content/drive/My Drive/Process Modelling Datasets/bpic2012_O_ACCEPTED-COMPLETE.csv", sep=';')

#Assert that there are no NaNs for any activity
assert(df['Activity'].isnull().values.any() == False)

df['Activity'] = pd.Categorical(df['Activity'])
activityNames = list(df['Activity'].cat.categories)
numActivities = len(activityNames)
df['Activity'] = df['Activity'].cat.codes

df.loc[df['label'] == "deviant", 'label'] = 0
df.loc[df['label'] == "regular", 'label'] = 1

for val in df['label'].unique():
    assert(val == 0 or val == 1)

#Drop some columns
df = df.drop('hour', axis=1)

processAttributeNames = ['AMOUNT_REQ']

one_hot = pd.get_dummies(df['Resource'])

df = df.drop('Resource',axis = 1)
one_hot = one_hot.add_suffix('_r1')
df = df.join(one_hot)

one_hot = pd.get_dummies(df['lifecycle:transition'])

df = df.drop('lifecycle:transition',axis = 1)
one_hot = one_hot.add_suffix('_r2')
df = df.join(one_hot)

one_hot = pd.get_dummies(df['month'])

df = df.drop('month',axis = 1)
one_hot = one_hot.add_suffix('_r3')
df = df.join(one_hot)

one_hot = pd.get_dummies(df['weekday'])

df = df.drop('weekday',axis = 1)
one_hot = one_hot.add_suffix('_r4')
df = df.join(one_hot)

df = df.sort_values('Complete Timestamp').reset_index(drop=True)
df.drop('Complete Timestamp', axis=1, inplace=True)

final_occurence_index = {}

for i in range(len(df)):
    final_occurence_index[df.at[i, 'Case ID']] = i

event_nr_df = df['event_nr'].copy()

#Normalize some attributes between 0 and 1

for column in df:
    if column == 'timesincelastevent' or column == 'timesincecasestart' or column == 'timesincemidnight' or column == 'event_nr' or column == 'open_cases' or column == 'AMOUNT_REQ':
      if df[column].nunique() == 1:
          df[column]/=df.at[0, column]
      else:
          mex = df[column].max()
          mine = df[column].min()
          df[column] -= mine
          df[column]/=(mex-mine)

#Standardization of 'timesincelastevent' and 'timesincecasestart'

df['timesincelastevent'] = (df['timesincelastevent'] - df['timesincelastevent'].mean())/df['timesincelastevent'].std()
df['timesincecasestart'] = (df['timesincecasestart'] - df['timesincecasestart'].mean())/df['timesincecasestart'].std()

#Assert that there are no NaNs
assert(df.isna().sum().sum() == 0)

random.seed(0)

train_case_ids = set()
validation_case_ids = set()
test_case_ids = set()

case_ids_list = list(df['Case ID'].unique())
train_validation_case_ids_list = case_ids_list[:int(0.8*(len(case_ids_list)))]
[test_case_ids.add(case_id) for case_id in case_ids_list[int(0.8*len(case_ids_list)):]]
random.shuffle(train_validation_case_ids_list)
[train_case_ids.add(case_id) for case_id in train_validation_case_ids_list[:int(0.9*len(train_validation_case_ids_list))]]
[validation_case_ids.add(case_id) for case_id in train_validation_case_ids_list[int(0.9*len(train_validation_case_ids_list)):]]

assert(len(train_case_ids) + len(validation_case_ids) + len(test_case_ids) == len(case_ids_list))
print("Length of training dataset: {}".format(len(train_case_ids)))
print("Length of validation dataset: {}".format(len(validation_case_ids)))
print("Length of test dataset: {}".format(len(test_case_ids)))
print("")

max_occurence_value = 0

occurences = {}
df['occurence'] = 0.0

for i in range(len(df)):

    string = str(df.at[i, 'Case ID']) + " " + str(df.at[i, 'Activity'])
    if string not in occurences:
        occurences[string] = 1
    else:
        occurences[string] += 1

    max_occurence_value = max(max_occurence_value, occurences[string])

    df.at[i, 'occurence'] = occurences[string]

assert(max_occurence_value > 1)

df['occurence'] = (df['occurence'] - 1)/(max_occurence_value - 1)

del occurences

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

eventAttributeNames = [attributeName for attributeName in df.columns if attributeName not in processAttributeNames + ['Case ID', 'label', 'Activity']]
numEventAttributes = len(eventAttributeNames)
numProcessAttributes = len(processAttributeNames)


class TemporalGNN(torch.nn.Module):
    def __init__(self, out_channels_GNN, hidden_layer_event_attribute_channels, num_activities, eventAttributeNames, numEventAttributes, processAttributeNames, numProcessAttributes):

        super(TemporalGNN,self).__init__()
        self.out_channels_GNN = out_channels_GNN
        self.num_activities = num_activities
        self.eventAttributeNames = eventAttributeNames
        self.numEventAttributes = numEventAttributes
        self.processAttributeNames = processAttributeNames
        self.hidden_layer_event_attribute_channels = hidden_layer_event_attribute_channels
        self.gnn = TempGCN(N = num_activities, in_channels = numEventAttributes, out_channels = out_channels_GNN)
        self.fc1 = torch.nn.Linear(out_channels_GNN*num_activities, self.hidden_layer_event_attribute_channels//2)
        self.dropout1 = torch.nn.Dropout(p=0.5)
        self.fc2 = torch.nn.Linear(out_channels_GNN*num_activities, self.hidden_layer_event_attribute_channels//2)
        self.dropout2 = torch.nn.Dropout(p=0.5)
        self.fc3 = torch.nn.Linear((hidden_layer_event_attribute_channels//2)*2 + numProcessAttributes, 2)
        self.dropout3 = torch.nn.Dropout(p=0.3)
        self.lrelu = torch.nn.LeakyReLU()
        self.lsoftmax = torch.nn.LogSoftmax(dim=0)

    def createEventVector(self, df, index):
        return torch.from_numpy(df.loc[index, self.eventAttributeNames].to_numpy().astype(np.float32)).to(device)

    def createProcessVector(self, df, index):
        return torch.from_numpy(df.loc[index, self.processAttributeNames].to_numpy().astype(np.float32)).to(device)

    def forward(self, df, index, prevEventActivityNumbers, E, H):

        case_id = df.at[index, 'Case ID']
        activityNumber = df.at[index, 'Activity']
        eventVector = self.createEventVector(df, index)
        processAttributeVector = self.createProcessVector(df, index)

        if case_id not in E:
            E[case_id], H[case_id] = self.gnn(X = eventVector, head = activityNumber)
        else:
            E[case_id], H[case_id] = self.gnn(X = eventVector, head = activityNumber, tail = prevEventActivityNumbers[case_id], E = E[case_id], H = H[case_id])

        prevEventActivityNumbers[case_id] = activityNumber

        other_case_features = torch.zeros([self.hidden_layer_event_attribute_channels//2]).to(device)
        for key, value in H.items():
            if key == case_id:
                case_features = torch.flatten(value).to(device)
                case_features = self.dropout1(case_features)
                case_features = self.fc1(case_features)
                case_features = self.lrelu(case_features)
            else:
                out = torch.flatten(value).to(device)
                out = self.dropout2(out)
                out = self.fc2(out)
                other_case_features += self.lrelu(out)

        out = torch.cat((case_features, other_case_features, processAttributeVector))
        out = self.dropout3(out)
        out = self.fc3(out)
        out = self.lsoftmax(out)

        return out

model = TemporalGNN(
        out_channels_GNN = numEventAttributes//2,
        hidden_layer_event_attribute_channels = numEventAttributes//10,
        num_activities = numActivities,
        eventAttributeNames = eventAttributeNames,
        numEventAttributes = numEventAttributes,
        processAttributeNames = processAttributeNames,
        numProcessAttributes = numProcessAttributes
    ).to(device)

print("Number of activities: {}".format(numActivities))
print("Number of event attributes: {}".format(numEventAttributes))
print("Number of process attributes: {}".format(numProcessAttributes))
print("")

print("Number of trainable parameters: {}".format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
print("")

optimizer = torch.optim.Adam(model.parameters(), lr=0.00005, weight_decay=1e-5)
loss_func = torch.nn.NLLLoss()

@torch.no_grad()
def evaluate(model, case_ids, loss_func, maxPrefixLength, prefixLength = None, threshold = None):

    model.eval()

    total_0 = 0
    total_1 = 0
    correct_0 = 0
    correct_1 = 0
    loss = 0

    y_true = []
    y_pred = []

    prevEventActivityNumbers = {}
    E = {}
    H = {}

    for index in range(len(df)):

        x = model(df, index, prevEventActivityNumbers, E, H)
        case_id = df.at[index, 'Case ID']

        if((event_nr_df[index] == prefixLength or (prefixLength is None and event_nr_df[index] <= maxPrefixLength)) and case_id in case_ids):

            x = torch.unsqueeze(x, 0)
            y = torch.unsqueeze(torch.tensor(df.at[index, 'label']),0).type(torch.LongTensor).to(device)
            loss += loss_func(x,y)
            y_true.append(y.item())
            y_pred.append(x[0][1].item())

        if index == final_occurence_index[case_id]:
            prevEventActivityNumbers.pop(case_id)
            E.pop(case_id)
            H.pop(case_id)

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

@torch.no_grad()
def AUC_all_prefix_lengths(model,case_ids,loss_func, maxPrefixLength, threshold = None):

    model.eval()

    total_0 = 0
    total_1 = 0
    correct_0 = 0
    correct_1 = 0

    true_dict = {}
    pred_dict = {}

    y_true = []
    y_pred = []

    prevEventActivityNumbers = {}
    E = {}
    H = {}

    for index in range(len(df)):

        x = model(df, index, prevEventActivityNumbers, E, H)
        case_id = df.at[index, 'Case ID']

        if(event_nr_df[index] <= maxPrefixLength and case_id in case_ids):

            x = torch.unsqueeze(x, 0)
            y = torch.unsqueeze(torch.tensor(df.at[index, 'label']),0).type(torch.LongTensor).to(device)
            if event_nr_df[index] not in true_dict:
                true_dict[event_nr_df[index]] = []
                pred_dict[event_nr_df[index]] = []
            true_dict[event_nr_df[index]].append(y.item())
            pred_dict[event_nr_df[index]].append(x[0][1].item())

        if index == final_occurence_index[case_id]:
            prevEventActivityNumbers.pop(case_id)
            E.pop(case_id)
            H.pop(case_id)


    return true_dict, pred_dict

def train_model(model, optimizer, loss_func, epochs, maxPrefixLength, weights_save_path, train_case_ids, validation_case_ids):

    model.train()
    min_val_loss = 1e9

    for epoch in range(epochs):

        print("\n--------------------------\n")
        print("Epoch : ",epoch+1)

        prevEventActivityNumbers = {}
        E = {}
        H = {}

        for index in range(len(df)):

            x = model(df, index, prevEventActivityNumbers, E, H)
            case_id = df.at[index, 'Case ID']

            if(event_nr_df[index] <= maxPrefixLength and case_id in train_case_ids):

                x = torch.unsqueeze(x, 0)
                y = torch.unsqueeze(torch.tensor(df.at[index, 'label']),0).type(torch.LongTensor).to(device)
                loss = loss_func(x,y)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            H[case_id] = H[case_id].detach()

            if index == final_occurence_index[case_id]:
                prevEventActivityNumbers.pop(case_id)
                E.pop(case_id)
                H.pop(case_id)

        if epoch == 0:
            torch.save(model.state_dict(), weights_save_path)

        print("\nValidation Results\n")
        val_loss, val_eval = evaluate(model,validation_case_ids,loss_func, maxPrefixLength)

        if val_loss<min_val_loss:
            min_val_loss = val_loss
            torch.save(model.state_dict(), weights_save_path)

maxPrefixLength = 40

train_model(model, optimizer, loss_func, 200, maxPrefixLength, "/content/drive/MyDrive/Temporal GNN/BPIC 2015/Ordered/2011_1_with_all_seed=0_summed_loss_inter_case.pt", train_case_ids, validation_case_ids)

model.load_state_dict(torch.load("/content/drive/MyDrive/Temporal GNN/BPIC 2015/Ordered/2015_1_with_all_seed=0_summed_loss_inter_case.pt"))

print("Training set results\n")
evaluate(model, train_case_ids, loss_func, maxPrefixLength = maxPrefixLength)
print("---------------------")
print("Validation set results\n")
evaluate(model, validation_case_ids, loss_func, maxPrefixLength = maxPrefixLength)
print("---------------------")
print("Test set results\n")
evaluate(model, test_case_ids, loss_func, maxPrefixLength = maxPrefixLength)
print("---------------------")

true_dict, pred_dict = AUC_all_prefix_lengths(model,test_case_ids,loss_func, maxPrefixLength=maxPrefixLength)

AUC_list = []
prefixLengths = []

for prefixLength in range(1, maxPrefixLength+1):
    prefixLengths.append(prefixLength)
    fpr, tpr, thresholds = roc_curve(true_dict[prefixLength], pred_dict[prefixLength])
    AUC = auc(fpr, tpr)
    print(prefixLength, len(true_dict[prefixLength]), AUC)
    AUC_list.append(AUC)

import matplotlib.pyplot as plt

# plotting the points
plt.plot(prefixLengths, AUC_list)

# naming the x axis
plt.xlabel('Prefix length')
# naming the y axis
plt.ylabel('AUC')
plt.grid()
# function to show the plot
plt.show()
