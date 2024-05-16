import torch
from torch import optim
from torch.utils.data import DataLoader
import functions as func
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
# Set font to Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'

## PINN -> same setup as Baseline in terms of architecture  
settings_RUL_CaseA = dict()
settings_RUL_CaseA['num_rounds'] = 10
settings_RUL_CaseA['batch_size'] = 100
settings_RUL_CaseA['num_epoch'] = 200
settings_RUL_CaseA['num_layers'] = [2]
settings_RUL_CaseA['num_neurons'] = [10]
settings_RUL_CaseA['lr'] = 1e-5
settings_RUL_CaseA['step_size'] = 50
settings_RUL_CaseA['gamma'] = 0.1
settings_RUL_CaseA['inputs_lib_dynamical'] = ['s_norm, t_norm, U_norm']
settings_RUL_CaseA['inputs_dim_lib_dynamical'] = ['inputs_dim + 1']
torch.save(settings_RUL_CaseA, 'Settings\\settings_RUL_CaseA.pth')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

settings = torch.load('Settings\\settings_RUL_CaseA.pth')
seq_len = 1
perc_val = 0.5
num_rounds = settings['num_rounds']
batch_size = settings['batch_size']
num_epoch = settings['num_epoch']
num_layers = settings['num_layers']
num_neurons = settings['num_neurons']

inputs_lib_dynamical = [
    's_norm',
    't_norm',
    'U_norm',
    'U_s',
    's_norm, t_norm',
    's_norm, U_norm',
    's_norm, U_s',
    't_norm, U_norm',
    't_norm, U_s',
    'U_norm, U_s',
    's_norm, t_norm, U_norm',
    's_norm, t_norm, U_s',
    's_norm, U_norm, U_s',
    't_norm, U_norm, U_s',
    's_norm, t_norm, U_norm, U_s'
]

inputs_dim_lib_dynamical = [
    'inputs_dim - 1',
    '1',
    '1',
    'inputs_dim - 1',
    'inputs_dim',
    'inputs_dim',
    '2 * (inputs_dim - 1)',
    '2',
    'inputs_dim',
    'inputs_dim',
    'inputs_dim + 1',
    '2 * (inputs_dim - 1) + 1',
    '2 * (inputs_dim - 1) + 1',
    'inputs_dim + 1',
    '2 * inputs_dim'
]


metric_mean = dict()
metric_std = dict()
metric_mean['train'] = np.zeros((len(inputs_lib_dynamical), 1))
metric_mean['val'] = np.zeros((len(inputs_lib_dynamical), 1))
metric_mean['test'] = np.zeros((len(inputs_lib_dynamical), 1))
metric_std['train'] = np.zeros((len(inputs_lib_dynamical), 1))
metric_std['val'] = np.zeros((len(inputs_lib_dynamical), 1))
metric_std['test'] = np.zeros((len(inputs_lib_dynamical), 1))

best_round = None
best_rmse = float('inf') 


settings = torch.load('Settings\\settings_RUL_CaseA.pth')
seq_len = 1
perc_val = 0.5
num_rounds = settings['num_rounds']
batch_size = settings['batch_size']
num_epoch = settings['num_epoch']
num_layers = settings['num_layers']
num_neurons = settings['num_neurons']

for l in range(len(inputs_lib_dynamical)):
    inputs_dynamical, inputs_dim_dynamical = inputs_lib_dynamical[l], inputs_dim_lib_dynamical[l]
    layers = num_layers[0] * [num_neurons[0]]
    np.random.seed(1234)
    torch.manual_seed(1234)
    metric_rounds = dict()
    metric_rounds['train'] = np.zeros(num_rounds)
    metric_rounds['val'] = np.zeros(num_rounds)
    metric_rounds['test'] = np.zeros(num_rounds)
    weights_rounds = [[]] * num_rounds
    for round in range(num_rounds):
        inputs_dim = 60
        outputs_dim = 1

        train_set = func.TensorDataset(inputs_train, targets_train)  
        train_loader = DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=True
        )

        _, mean_inputs_train, std_inputs_train = func.standardize_tensor(inputs_train, mode='fit')
        _, mean_targets_train, std_targets_train = func.standardize_tensor(targets_train, mode='fit')

        train_set = func.TensorDataset(inputs_train, targets_train) 
        train_loader = DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=True
        )

        model = func.DeepHPMNN(
            seq_len=seq_len,
            inputs_dim=inputs_dim,
            outputs_dim=outputs_dim,
            layers=layers,
            scaler_inputs=(mean_inputs_train, std_inputs_train),
            scaler_targets=(mean_targets_train, std_targets_train),
            inputs_dynamical=inputs_dynamical,
            inputs_dim_dynamical=inputs_dim_dynamical
        ).to(device)

        log_sigma_u = torch.randn((), requires_grad=True)
        log_sigma_f = torch.randn((), requires_grad=True)
        log_sigma_f_t = torch.randn((), requires_grad=True)

        criterion = func.My_loss(mode='AdpBal')

        params = ([p for p in model.parameters()] + [log_sigma_u] + [log_sigma_f] + [log_sigma_f_t])
        optimizer = optim.Adam(params, lr=settings['lr'])
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=settings['step_size'], gamma=settings['gamma'])
        model, results_epoch = func.train(
            num_epoch=num_epoch,
            batch_size=batch_size,
            train_loader=train_loader,
            num_slices_train=inputs_train.shape[0],
            inputs_val=inputs_val,
            targets_val=targets_val,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            log_sigma_u=log_sigma_u,
            log_sigma_f=log_sigma_f,
            log_sigma_f_t=log_sigma_f_t
        )

        model.eval()

        U_pred_train, F_pred_train, _ = model(inputs=inputs_train)
        RMSE_train = torch.sqrt(torch.mean(((U_pred_train - targets_train)) ** 2))

        U_pred_val, F_pred_val, _ = model(inputs=inputs_val)
        RMSE_val = torch.sqrt(torch.mean(((U_pred_val - targets_val)) ** 2))

        U_pred_test, F_pred_test, _ = model(inputs=inputs_test)
        RMSE_test = torch.sqrt(torch.mean(((U_pred_test - targets_test)) ** 2))

        metric_rounds['train'][round] = RMSE_train.detach().cpu().numpy()
        metric_rounds['val'][round] = RMSE_val.detach().cpu().numpy()
        metric_rounds['test'][round] = RMSE_test.detach().cpu().numpy()
        weights_rounds[round] = dict()
        weights_rounds[round]['lambda_U'] = results_epoch['var_U']
        weights_rounds[round]['lambda_F'] = results_epoch['var_F']
        weights_rounds[round]['lambda_F_t'] = results_epoch['var_F_t']

        # Check if the current round has the lowest RMSE
        if RMSE_val < best_rmse:
            best_rmse = RMSE_val
            best_round = round
            best_U_pred_test = U_pred_test.detach().cpu().numpy()



pass


# Flatten the arrays if they have more than one dimension
targets_test_flat = targets_test_np.flatten()
U_pred_test_flat = best_U_pred_test.flatten()

# Calculate RMSE
# rmse = metric_mean['test'].min()

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(targets_test_flat, U_pred_test_flat))

# Create the plot
plt.figure(figsize=(6, 5))

# Scatter plot of actual vs predicted RUL with adjusted marker size and transparency
plt.scatter(targets_test_flat, U_pred_test_flat, color='crimson', s=100, alpha=0.6)

# Plot the line of perfect prediction
min_val = min(targets_test_flat.min(), U_pred_test_flat.min())
max_val = max(targets_test_flat.max(), U_pred_test_flat.max())
plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='Perfect Prediction')


# Add grid
plt.grid(True)

# Increase font size of label ticks
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)



# Concatenate metrics into a single string
metrics_text = f'RMSE = {rmse:.4f}'

# Add metrics as text in one box
plt.text(0.95, 0.05, metrics_text, ha='right', va='bottom', transform=plt.gca().transAxes, fontsize=25, color='black', bbox=dict(facecolor='white', alpha=0.5))


# plt.savefig("Exported Figs/Cap_forecast_temperature_removed_PINN.svg", format='svg', dpi=600)

# Show plot
plt.show()
