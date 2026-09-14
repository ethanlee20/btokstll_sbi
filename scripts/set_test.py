

import numpy
import torch
import matplotlib
import pandas
import sklearn

from helpers.plot import turn_on_hq_plots, turn_on_dark_plots, plot_loss
from helpers.train import train_val
from helpers.dataloader import SetDataLoader
from helpers.set_model import MLP


turn_on_hq_plots()
turn_on_dark_plots()

model_path = "models/model.pt"

feature_cols = ["q_sq", "cos_theta_lepton", "cos_theta_k", "chi", "dC_9"]

df_sm = pandas.read_parquet("data/train_sm/combo.parquet")[feature_cols]
df_vary = pandas.read_parquet("data/train_vary/combo.parquet")[feature_cols]

df_vary_grouped = df_vary.groupby(level="trial_num")
vary_sets = numpy.stack([trial_df.values for _, trial_df in df_vary_grouped])


sm_sets = df_sm.values.reshape(200, 25000, 5)

sm_sets[:, :, 4] = vary_sets[:, :, 4]


sm_set_labels = numpy.zeros(len(sm_sets))
vary_set_labels = numpy.ones(len(vary_sets))

sets = numpy.vstack([sm_sets, vary_sets])
set_labels = numpy.append(sm_set_labels, vary_set_labels)

sets = sets.astype(numpy.float32)
set_labels = set_labels.astype(numpy.float32)

splitter = sklearn.model_selection.StratifiedShuffleSplit(n_splits=1, test_size=0.2)
train_indices, val_indices = next(splitter.split(sets, set_labels))

sets_train = sets[train_indices]
set_labels_train = set_labels[train_indices]
sets_val = sets[val_indices]
set_labels_val = set_labels[val_indices]

sets_train = torch.from_numpy(sets_train)
set_labels_train = torch.from_numpy(set_labels_train)
sets_val = torch.from_numpy(sets_val)
set_labels_val = torch.from_numpy(set_labels_val)

train_dataloader = SetDataLoader(sets_train, set_labels_train, batch_size=10)
val_dataloader = SetDataLoader(sets_val, set_labels_val, batch_size=10)

train_mean = sets_train.mean(axis=(0,1))
train_std = sets_train.std(axis=(0,1))

model = MLP()
model.set_std_scale(train_mean, train_std)
model = model.to('cuda')
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.95, 0.95))
loss_fn = torch.nn.BCEWithLogitsLoss()
num_epochs = 1000

losses = train_val(
    train_data_loader=train_dataloader,
    model=model,
    loss_fn=loss_fn,
    optimizer=optimizer,
    num_epochs=num_epochs,
    val_data_loader=val_dataloader,
    lr_scheduler=torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
)

fig, ax = matplotlib.pyplot.subplots()
plot_loss(
    ax, 
    train_losses=losses["train"],
    val_losses=losses["val"],
    compute_log=False
)
ax.set_ylim(-1, 3)
fig.savefig("plots/loss.png", bbox_inches="tight")
matplotlib.pyplot.close()


torch.save(model.state_dict(), model_path)




model = MLP()
model.load_state_dict(torch.load("models/model.pt", weights_only=True))
model = model.to("cuda")
model.eval()


def scan_likelihood_ratio(model, data, min, max, num):
    points = numpy.linspace(min, max, num)
    ratios = []
    for p in points:
        x = data.astype("float32").values.copy()
        x = numpy.expand_dims(x, axis=0)
        x[:, :, 4] = p
        x = torch.from_numpy(x).to("cuda")
        log_ratio_sum = model(x)
        ratios.append(log_ratio_sum.detach())
    ratios = torch.tensor(ratios)
    return points, ratios


def get_mle(model, data, min, max, num):
    points, ratios = scan_likelihood_ratio(model, data, min, max, num)
    out = points[ratios.argmax()]
    return out



df_val_sets = pandas.read_parquet("data/val_sets/combo.parquet")


labels = []
mles = []
for trial_num in range(20):
    df_val_set = df_val_sets.xs(trial_num, level="trial_num")
    label = df_val_set["dC_9"].unique()
    features = df_val_set[feature_cols]
    mle = get_mle(model, features, -2, 1, 100)
    labels.append(label)
    mles.append(mle)
    if trial_num == 0:
        points, ratios = scan_likelihood_ratio(model, features, -2, 1, 100)
        label_to_plot = label
labels = numpy.array(labels)
mles = numpy.array(mles)

fig, ax = matplotlib.pyplot.subplots()
ax.plot([-2, 1], [-2, 1], color="#377eb8", label="Ideal", zorder=-100)
ax.scatter(labels, mles, color="#ff7f00")
ax.set_ylabel(r"Predicted $\delta C_9$", fontsize=16)
ax.set_xlabel(r"True $\delta C_9$", fontsize=16)
ax.legend()
fig.savefig("plots/lin.png", bbox_inches="tight")
matplotlib.pyplot.close()


def plot_log_likelihood_ratio_scan(
    ax: Axes,
    parameters: ArrayLike,
    log_probabilities: ArrayLike,
    true_value: float,
    color: str,
):
    ax.plot(parameters, log_probabilities, color=color)
    ax.axvline(true_value, color=color, linestyle="--", zorder=-100)
    mle_y = max(log_probabilities)
    mle_x = parameters[numpy.argmax(log_probabilities)]
    ax.scatter(mle_x, mle_y, color=color, s=25, zorder=100)

    # conf = parameters[nonzero(log_probabilities > mle_y - 0.5)]
    # ax.hlines(mle_y, conf[0], conf[-1], color=color, linestyle="--", zorder=-101)

    ax.set_ylabel(r"$\log p(\textrm{data} \;|\; \delta C_9) + c$", fontsize=15)
    ax.set_xlabel(r"$\delta C_9$", fontsize=15)


fig, ax = matplotlib.pyplot.subplots()
plot_log_likelihood_ratio_scan(
    ax=ax,
    parameters=points,
    log_probabilities=ratios,
    true_value=label_to_plot,
    color="#ff7f00"
)
fig.savefig("plots/preds.png", bbox_inches="tight")
matplotlib.pyplot.close()
