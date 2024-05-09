
import torch
from torch import optim
from torch.utils.data import DataLoader
import functions as func
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

## DATA-DRIVEN BASELINE
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

for l, num_l in enumerate(num_layers):
    for n, num_n in enumerate(num_neurons):
        layers = num_l * [num_n]
        np.random.seed(1234)
        torch.manual_seed(1234)
        metric_rounds = dict()
        metric_rounds['train'] = np.zeros(num_rounds)
        metric_rounds['val'] = np.zeros(num_rounds)
        metric_rounds['test'] = np.zeros(num_rounds)
        for round in range(num_rounds):

            inputs_dim = len(features)
            outputs_dim = 1


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
            model = func.DataDrivenNN(
                seq_len=seq_len,
                inputs_dim=inputs_dim,
                outputs_dim=outputs_dim,
                layers=layers,
                scaler_inputs=(mean_inputs_train, std_inputs_train),
                scaler_targets=(mean_targets_train, std_targets_train),
            ).to(device)

            log_sigma_u = torch.zeros(())
            log_sigma_f = torch.zeros(())
            log_sigma_f_t = torch.zeros(())

            criterion = func.My_loss(mode='Baseline')

            params = ([p for p in model.parameters()])
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

            metric_mean['train'][l, n] = np.mean(metric_rounds['train'])
            metric_mean['val'][l, n] = np.mean(metric_rounds['val'])
            metric_mean['test'][l, n] = np.mean(metric_rounds['test'])
            metric_std['train'][l, n] = np.std(metric_rounds['train'])
            metric_std['val'][l, n] = np.std(metric_rounds['val'])
            metric_std['test'][l, n] = np.std(metric_rounds['test'])
            # torch.save(metric_mean, 'test results/metric_mean_RUL_AllExceptOneBatchFiltered_Baseline__.pth')
            # torch.save(metric_std, 'test results/metric_std_RUL_AllExceptOneBatchFiltered_Baseline__.pth')

pass



# Set font to Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'


U_pred_test_np = U_pred_test.detach().cpu().numpy()
targets_test_np = targets_test.detach().cpu().numpy()
# Flatten the arrays if they have more than one dimension
targets_test_flat = targets_test_np.flatten()
U_pred_test_flat = U_pred_test_np.flatten()

# Calculate RMSE
# rmse = metric_mean['test'].min()

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(targets_test_flat, U_pred_test_flat))

# # Calculate R^2
# r_squared = r2_score(targets_test_flat, U_pred_test_flat)

# Create the plot
plt.figure(figsize=(6, 5))

# Scatter plot of actual vs predicted RUL with adjusted marker size and transparency
plt.scatter(targets_test_flat, U_pred_test_flat, color='blue', s=100, alpha=0.6)

# Plot the line of perfect prediction
min_val = min(targets_test_flat.min(), U_pred_test_flat.min())
max_val = max(targets_test_flat.max(), U_pred_test_flat.max())
plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='Perfect Prediction')

# Add legend
# plt.legend(fontsize=20)

# Add grid
plt.grid(True)

# Increase font size of label ticks
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)



# Concatenate metrics into a single string
metrics_text = f'RMSE = {rmse:.4f}'

# Add metrics as text in one box
plt.text(0.95, 0.05, metrics_text, ha='right', va='bottom', transform=plt.gca().transAxes, fontsize=25, color='black', bbox=dict(facecolor='white', alpha=0.5))


# plt.savefig("Exported Figs/Cap_forecast_temperature_removed_baseline.svg", format='svg', dpi=600)

# Show plot
plt.show()
