
import pickle

from sklearn.base import BaseEstimator, TransformerMixin

from numpy.random import shuffle
from numpy import expand_dims, linspace, log, array, max, argmax, stack, vstack, ones, zeros, float32, append
from matplotlib.pyplot import subplots, close
from pandas import read_parquet, concat
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from skorch import NeuralNetBinaryClassifier
from skorch.dataset import ValidSplit
from skorch.callbacks import LRScheduler
from torch.optim import AdamW, Adam, Adamax, SGD
from torch.optim.lr_scheduler import ExponentialLR
from scipy.stats import uniform, randint

from helpers.plot import plot_loss, turn_on_hq_plots, turn_on_dark_plots
from helpers.dataloader_skorch import DataLoader
from helpers.model_skorch import MLP


turn_on_hq_plots()
turn_on_dark_plots()

feature_cols = ["q_sq", "cos_theta_lepton", "cos_theta_k", "chi", "dC_9"]

df_sm = read_parquet("data/train_sm/combo.parquet")[feature_cols]
df_vary = read_parquet("data/train_vary/combo.parquet")[feature_cols]

breakpoint()

df_vary_grouped = df_vary.groupby(level="trial_num")
vary_sets = stack([trial_df.values for _, trial_df in df_vary_grouped])
shuffle(vary_sets)


sm_sets = df_sm.values.reshape(1000, 5000, 5)

sm_sets[:, :, 4] = vary_sets[:, :, 4]


sm_set_labels = zeros(len(sm_sets))
vary_set_labels = ones(len(vary_sets))

sets = vstack([sm_sets, vary_sets])
set_labels = append(sm_set_labels, vary_set_labels)

sets = sets.astype(float32)
set_labels = set_labels.astype(float32)

breakpoint()

# df_sm["dC_9"] = df_vary["dC_9"].values

# df_sm = df_sm.assign(label=1)
# df_vary = df_vary.assign(label=0)

# df = concat([df_sm, df_vary])
# df = df.sample(frac=1)

# X = df[feature_cols].astype("float32").values
# y = df["label"].astype("float32").values



net = NeuralNetBinaryClassifier(
    MLP,
    optimizer=AdamW,
    optimizer__weight_decay=0.03,
    predict_nonlinearity=None,
    max_epochs=300,
    lr=3e-4,
    train_split=ValidSplit(stratified=True),
    iterator_train=DataLoader,
    iterator_valid=DataLoader,
    iterator_train__batch_size=32,
    iterator_valid__batch_size=-1,
    iterator_train__shuffle=True,
    device="cuda",
    #callbacks=[
    #    ('lr_scheduler', LRScheduler(policy=ExponentialLR, gamma=0.95))
    #]
)

net.set_params(callbacks__valid_acc=None)

class MyStdScaler(TransformerMixin, BaseEstimator):
    def __init__(self):
        pass
    def fit(self, X, y=None):
        self.std_ = X.std(axis=(0,1))
        self.mean_ = X.mean(axis=(0,1))
        return self
    def transform(self, X):
        return (X - self.mean_) / self.std_


pipe = Pipeline(
    [
        ('scale', MyStdScaler()),    
        ('net', net)
    ]
)

"""
net.set_params(train_split=False)
params = {
    'net__lr': uniform(1e-5, 1e-3 - 1e-5),
    'net__max_epochs': randint(1, 100),
    'net__iterator_train__batch_size': randint(1_000, 100_000),
    'net__optimizer__weight_decay': uniform(0.01, 0.1 - 0.01),
}

search = RandomizedSearchCV(pipe, params, refit=False, cv=2, scoring='neg_log_loss')
search.fit(X, y)
print(search.best_score_, search.best_params_)
"""

pipe.fit(sets, set_labels)

train_losses = net.history[:, "train_loss"]
val_losses = net.history[:, "valid_loss"]
fig, ax = subplots()
plot_loss(
    ax, 
    train_losses=train_losses,
    val_losses=val_losses,
    compute_log=True
)
fig.savefig("plots/loss.png", bbox_inches="tight")
close()


with open('models/model.pkl', 'wb') as f:
    pickle.dump(pipe, f)



breakpoint()

def scan_likelihood_ratio(model, data, min, max, num):
    points = linspace(min, max, num)
    ratios = []
    for p in points:
        # data["dC_9"] = p
        x = data.astype("float32").values.copy()
        x = expand_dims(x, axis=0)
        x[:, :, 4] = p
        # probs = model.predict_proba(x)[:, 1]
        # log_ratio = log(1/probs - 1)
        log_ratio_sum = model.predict_proba(x)
        ratios.append(log_ratio_sum)
    ratios = array(ratios)
    return points, ratios


def get_mle(model, data, min, max, num):
    points, ratios = scan_likelihood_ratio(model, data, min, max, num)
    out = points[ratios.argmax()]
    return out



with open('models/model.pkl', 'rb') as f:
    pipe = pickle.load(f)

df_val_sets = read_parquet("data/val_sets/combo.parquet")


labels = []
mles = []
for trial_num in range(20):
    df_val_set = df_val_sets.xs(trial_num, level="trial_num")
    label = df_val_set["dC_9"].unique()
    features = df_val_set[feature_cols]
    mle = get_mle(pipe, features, -2, 1, 100)
    labels.append(label)
    mles.append(mle)
    if trial_num == 0:
        points, ratios = scan_likelihood_ratio(pipe, features, -2, 1, 100)
        label_to_plot = label
labels = array(labels)
mles = array(mles)

fig, ax = subplots()
ax.plot([-2, 1], [-2, 1], color="#377eb8", label="Ideal", zorder=-100)
ax.scatter(labels, mles, color="#ff7f00")
ax.set_ylabel(r"Predicted $\delta C_9$", fontsize=16)
ax.set_xlabel(r"True $\delta C_9$", fontsize=16)
ax.legend()
fig.savefig("plots/lin.png", bbox_inches="tight")
close()


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
    mle_x = parameters[argmax(log_probabilities)]
    ax.scatter(mle_x, mle_y, color=color, s=25, zorder=100)

    # conf = parameters[nonzero(log_probabilities > mle_y - 0.5)]
    # ax.hlines(mle_y, conf[0], conf[-1], color=color, linestyle="--", zorder=-101)

    ax.set_ylabel(r"$\log p(\textrm{data} \;|\; \delta C_9) + c$", fontsize=15)
    ax.set_xlabel(r"$\delta C_9$", fontsize=15)


fig, ax = subplots()
plot_log_likelihood_ratio_scan(
    ax=ax,
    parameters=points,
    log_probabilities=ratios,
    true_value=label_to_plot,
    color="#ff7f00"
)
fig.savefig("plots/preds.png", bbox_inches="tight")
close()
