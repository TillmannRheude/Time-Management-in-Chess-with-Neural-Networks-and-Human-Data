""" File Import """
import CTM_Net

""" Set GPU """
import os 
# set visible devices
os.environ["CUDA_VISIBLE_DEVICES"] = "5,6,7"

""" Package Import """
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import datetime
import zarr
import sys
import joblib

from rtpt import RTPT
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from torch.optim.lr_scheduler import LambdaLR, StepLR, MultiStepLR, ExponentialLR, ReduceLROnPlateau


print("-" * 30)
print("Starting main.py")
print("-" * 30)

# Matplotlib (SciencePlots)
# plt.style.use(['science','grid'])

# set hyperparameters for saving the model, loss visualizations, etc.
model_name = "CTM_"
version = "2.1"
tc = 60

### Set directories
year = "2020"
month = "02"
zarr_dir = "datasets_zarr/" + year + "-" + month + "/Lichess_" + year + "-" + month + " (" + str(tc) + "+0)"
month = "03"
zarr_dir2 = "datasets_zarr/" + year + "-" + month + "/Lichess_" + year + "-" + month + " (" + str(tc) + "+0)"
month = "04"
zarr_dir3 = "datasets_zarr/" + year + "-" + month + "/Lichess_" + year + "-" + month + " (" + str(tc) + "+0)"
# Additional data: 
#month = "08"
#zarr_dir4 = "datasets_zarr/" + year + "-" + month + "/Lichess_" + year + "-" + month + " (" + str(tc) + "+0)"
#month = "07"
#zarr_dir5 = "datasets_zarr/" + year + "-" + month + "/Lichess_" + year + "-" + month + " (" + str(tc) + "+0)"
# directories to train data
train_input_variables_path = [
        zarr_dir + "_inputvar.zarr",
        zarr_dir2 + "_inputvar.zarr",
        zarr_dir3 + "_inputvar.zarr"
    ]
train_boardmaps_path = [
        zarr_dir + "_boardmaps.zarr",
        zarr_dir2 + "_boardmaps.zarr",
        zarr_dir3 + "_boardmaps.zarr"
    ]
train_prediction_variable_path = [
        zarr_dir + "_predvar.zarr",
        zarr_dir2 + "_predvar.zarr",
        zarr_dir3 + "_predvar.zarr"
    ]

year = "2020"
month = "01"
zarr_dir = "datasets_zarr/" + year + "-" + month + "/Lichess_" + year + "-" + month + " (" + str(tc) + "+0)"
# directory to test data (should be one dataset for all training processes due to better comparison)
test_input_variables_path = [
        zarr_dir + "_inputvar.zarr"
    ]
test_boardmaps_path = [
        zarr_dir + "_boardmaps.zarr"
    ]
test_prediction_variable_path = [
        zarr_dir + "_predvar.zarr"
    ]

# Load Training data
train_input_variables_list = []
for i, path in enumerate(train_input_variables_path):
    train_input_variables_tmp = zarr.load(path)
    train_input_variables_list.append(train_input_variables_tmp)
for i, tmp in enumerate(train_input_variables_list):
    if i == 0: 
        train_input_variables = train_input_variables_list[i]
    else: 
        train_input_variables = np.concatenate((train_input_variables, train_input_variables_list[i]))

train_boardmaps_list = []
for i, path in enumerate(train_boardmaps_path):
    train_boardmaps_tmp = zarr.load(path)
    train_boardmaps_list.append(train_boardmaps_tmp)
for i, tmp in enumerate(train_boardmaps_list):
    if i == 0: 
        train_boardmaps = train_boardmaps_list[i]
    else: 
        train_boardmaps = np.concatenate((train_boardmaps, train_boardmaps_list[i]))

train_prediction_variable_list = []
for i, path in enumerate(train_prediction_variable_path):
    train_prediction_variable_tmp = zarr.load(path)
    train_prediction_variable_list.append(train_prediction_variable_tmp)
for i, tmp in enumerate(train_prediction_variable_list):
    if i == 0: 
        train_prediction_variable = train_prediction_variable_list[i]
    else: 
        train_prediction_variable = np.concatenate((train_prediction_variable, train_prediction_variable_list[i]))

# Load Test data
for i, path in enumerate(test_input_variables_path):
    test_input_variables = zarr.load(path)
for i, path in enumerate(test_boardmaps_path):
    test_boardmaps = zarr.load(path)
for i, path in enumerate(test_prediction_variable_path):
    test_prediction_variable = zarr.load(path)

# Sanity check for nans
train_prediction_variable = np.nan_to_num(train_prediction_variable)
test_prediction_variable = np.nan_to_num(test_prediction_variable)




print("Data loaded")
print("-" * 30)




# Get scalar value from input variables (in this case: material p1)
train_input_variables = train_input_variables[:, 0].reshape(-1,1)
test_input_variables = test_input_variables[:, 0].reshape(-1,1)

def make_scalar_planar(scalar_value):
    plane = np.zeros((1,8,8))
    plane.fill(scalar_value)
    return plane

def make_array_planar(array):
    input_variables = array

    input_variables_planar = np.zeros((input_variables.shape[0], 1, 8, 8))

    for i in range(input_variables.shape[0]):
        input_variables_planar[i] = make_scalar_planar(input_variables[i][0])

    return input_variables_planar

# choose scaler
scaler = StandardScaler() # MinMaxScaler()

scaler.fit(train_input_variables)
# Normalize Training data 
train_input_variables = scaler.transform(train_input_variables)
# Normalize Test data (with scales from training data information)
test_input_variables = scaler.transform(test_input_variables)
# save scaler for later prediction
joblib.dump(scaler, 'scalers/Scaler(CTM__' + version + '__tc60__input).save')

# include scalar values as additional plane
train_input_variables = make_array_planar(train_input_variables)
test_input_variables = make_array_planar(test_input_variables)
# stack planes to get whole input features
train_boardmaps = np.hstack((train_boardmaps, train_input_variables))
test_boardmaps = np.hstack((test_boardmaps, test_input_variables))

scaler.fit(train_prediction_variable)
# Normalize Training data 
train_prediction_variable = scaler.transform(train_prediction_variable)
# Normalize Test data (with training data information)
test_prediction_variable = scaler.transform(test_prediction_variable)
# save min, max for later prediction
joblib.dump(scaler, 'scalers/Scaler(CTM__' + version + '__tc60__prediction).save')




print("Data normalized")
print("-" * 30)




# Convert Training data to tensors
x = torch.tensor(train_boardmaps, dtype = torch.float)
y = torch.tensor(train_prediction_variable, dtype = torch.float)

# Convert Test data to tensors
x_val = torch.tensor(test_boardmaps, dtype = torch.float)
y_val = torch.tensor(test_prediction_variable, dtype = torch.float)

### Set hyperparameter
batchsize = 2048
epochs = 20
learning_rate = 3e-4

# Create RTPT object 
#rtpt = RTPT(name_initials = "TR", experiment_name = "CTM_Net_2.1", max_iterations = epochs)
#rtpt.start()

# create DataLoader for Train data
dataset_train = TensorDataset(x, y)
dataset_loader = DataLoader(dataset_train, batch_size = batchsize, shuffle = True, num_workers = 2)
# create DataLoader for Test data
dataset_test = TensorDataset(x_val, y_val)
dataset_loader_test = DataLoader(dataset_test, batch_size = batchsize, shuffle = True, num_workers = 2)

# select model 
model = CTM_Net.CTM_Net_v2_1() # CTM Net v2 or v2.1

# send model to device (GPU)
use_cuda = torch.cuda.is_available()
if use_cuda:
    # Choose device
    device = torch.device('cuda' )
    model = nn.DataParallel(model)
    model.to(device)
else:
    device = torch.device('cpu')
    model.to(device)

class LogCoshLoss(torch.nn.Module):
    # source: https://github.com/tuantle/regression-losses-pytorch
    def __init__(self):
        super().__init__()

    def forward(self, y_t, y_prime_t):
        ey_t = y_t - y_prime_t
        return torch.mean(torch.log(torch.cosh(ey_t + 1e-12)))

class XSigmoidLoss(torch.nn.Module):
    # source: https://github.com/tuantle/regression-losses-pytorch
    def __init__(self):
        super().__init__()

    def forward(self, y_t, y_prime_t):
        ey_t = y_t - y_prime_t
        return torch.mean(2 * ey_t / (1 + torch.exp(-ey_t)) - ey_t)

# use MSE Loss and try out different loss criterions
criterion = nn.MSELoss()
#criterion = nn.L1Loss()
#criterion = nn.SmoothL1Loss(beta = 1.0)
#criterion = LogCoshLoss()
#criterion = XSigmoidLoss()

# use ADAM optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# use SGD optimizer
# optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate, momentum = 0.95)

# set a scheduler for adaptive learning rate in case of SGD
scheduler = StepLR(optimizer, step_size = 10, gamma = 0.1)

# create arrays for train loss, test loss
epoch_loss = []
epoch_loss_val = []
epoch_r2 = []
epoch_r2_val = []

# calculate number of batches needed for loss values in training process
number_batches_train = np.ceil(x.shape[0] / batchsize)
number_batches_test = np.ceil(x_val.shape[0] / batchsize)




print("Start training ...")
print("-" * 30)




# Start training process
for epoch in range(epochs):
    # loss and r2 score for training epochs
    loss_train = 0
    r2_train = 0
    # set model in train mode
    model.train()
    for i, (inputs, labels) in enumerate(dataset_loader):
        # send train and target data to device
        inputs, labels = inputs.to(device), labels.to(device)

        # prediction via train data and model
        y_pred = model(inputs)

        # compute loss
        loss = criterion(y_pred, labels)

        # sum up loss
        loss_train += loss.item()
        r2_train += r2_score(labels.cpu().detach().numpy(), y_pred.cpu().detach().numpy())

        # zero gradient for optimizer
        optimizer.zero_grad()
        # backward pass
        loss.backward()
        optimizer.step()
        
    # do a step with the scheduler
    scheduler.step()

    # append the loss relative to the number of batches
    epoch_loss.append( loss_train / number_batches_train )
    epoch_r2.append( r2_train / number_batches_train )
    
    # difference to before: set model in evaluation mode
    model.eval()
    # loss for test process
    loss_val = 0
    r2_test = 0
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataset_loader_test):
            # send train and target data to device
            inputs, labels = inputs.to(device), labels.to(device)

            # prediction via validation data and model
            y_pred = model(inputs)

            # compute loss
            loss = criterion(y_pred, labels)

            # sum up loss
            loss_val += loss.item()
            r2_test += r2_score(labels.cpu().detach().numpy(), y_pred.cpu().detach().numpy())
    # append the loss relative to the number of batches
    epoch_loss_val.append( loss_val / number_batches_test )
    epoch_r2_val.append( r2_test / number_batches_test )
    
    # Print train and validation loss over epochs
    print('Epoch: {}'.format(epoch+1))
    print(f"Training Loss: {(loss_train / number_batches_train):6.6f}, Train R2: {(r2_train / number_batches_train):7.6f}")
    print(f"Test Loss: {(loss_val / number_batches_test):12.6f}, Test R2: {(r2_test / number_batches_test):9.6f}")
    print("-" * 30)
    
    # Update RTPT, optional: subtitle=f"loss={epoch_loss[-1]:2.2f}"
    # rtpt.step()


# Save model
torch.save(model, "models/" + model_name + "_v" + version + "_tc" + str(tc) + "_epochs" + str(epochs))

# Plot loss over epochs
plt.plot(range(len(epoch_loss)), epoch_loss[0:], label = "Train Loss")
plt.plot(range(len(epoch_loss)), epoch_loss_val[0:], label = "Validation Loss")
plt.title("Loss values (Train and Validation)")
plt.xlabel("Epoch")
plt.ylabel("Loss (MSE)")
plt.legend()
evaluation_file = "Loss_" + model_name + "_v" + version + "_tc" + str(tc) + "_epochs" + str(epochs)
plt.savefig("evaluations/v" + version + "/" + evaluation_file, dpi=1200)

plt.clf()
# Plot R2 over epochs
plt.plot(range(len(epoch_loss)), epoch_r2[0:], label = "Train R2")
plt.plot(range(len(epoch_loss)), epoch_r2_val[0:], label = "Validation R2")
plt.rcParams["font.family"] = "serif"
plt.title("R2-scores (Train and Validation)")
plt.xlabel("Epoch")
plt.ylabel("R2")
plt.legend()
evaluation_file = "R2_" + model_name + "_v" + version + "_tc" + str(tc) + "_epochs" + str(epochs)
plt.savefig("evaluations/v" + version + "/" + evaluation_file, dpi=1200)