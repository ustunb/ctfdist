from ctfdist.paths import *
from ctfdist.descent import *
from ctfdist.helper_functions import *
from sklearn.linear_model import LogisticRegression
from scripts.toy_helper import *
from scripts.plot_helper import *

# setup
random_seed = 2338
random_state = np.random.RandomState(random_seed)
outcome_column_name = 'y'
group_column_name = 's'
minority_group = 0
majority_group = 1

train_size = 0.30
test_size = 0.20
audit_size = 1.0 - train_size - test_size

# generate full dataset
ps = 0.5
coefs = np.array([5.0, -2.5, -2.5])

param_min = {
    'coefs': coefs,
    'group_label': minority_group,
    'px': [0.9, 0.2, 0.2],
    'n_samples': 50000,
    }

param_maj = {
    'coefs': coefs,
    'group_label': majority_group,
    'px': [0.1, 0.5, 0.5],
    'n_samples': 50000,
    }

df_min, h_min = generate_toy_dataset(**param_min)
df_maj, h_maj = generate_toy_dataset(**param_maj)
df = df_min.append(df_maj)

#### Data Splitting ####

df_train, df_audit, df_test = train_test_audit_split(df = df,
                                                     train_size = train_size,
                                                     test_size = test_size,
                                                     audit_size = audit_size,
                                                     group_column_name = group_column_name,
                                                     outcome_column_name = outcome_column_name,
                                                     random_state = random_state)

#### Models ####

#handle for p(S = 1 | X = x) -- model to get true outcomes for minority group
hs_min = h_min['get_px']
hs_maj = h_maj['get_px']
predict_group = lambda x: np.multiply(ps, hs_maj(x)) / (np.multiply(ps, hs_maj(x)) + np.multiply(1.0 - ps, hs_min(x)))


# handle for p(Y = 1 | X = x) -- model to get true outcomes for minority group:
get_py_min = h_min['get_py']
get_py_maj = h_maj['get_py']
predict_ytrue = lambda x: get_py_min(x).flatten()
predict_ytrue_maj = lambda x: get_py_maj(x).flatten()

# train predictive model
X, Y, S = df_to_XYS(df_train, minority_group = minority_group)
f_hat = LogisticRegression(penalty = 'l2',  C = 100, fit_intercept = False)
f_hat.fit(X = X, y = Y)
pos_class_idx = np.flatnonzero(f_hat.classes_ > 0)
predict_prob = lambda x: f_hat.predict_proba(x)[:, pos_class_idx].flatten()
predict_yhat = lambda x: 2.0 * np.greater(predict_prob(x), 0.5) - 1.0


#### Distribution Descent Experiment ####

script_metric_weights = {
    'cal': {'cal': 1.0},
    'fpr': {'fpr': 1.0},
    'fnr': {'fnr': 1.0},
    'fdr': {'fdr': 1.0},
    'da_out': {'da_out': 1.0},
    'da_in': {'da_in': 1.0},
    'da_mix': {'da_out': 0.8, 'da_in': 0.2},
    }

data_name = 'toy'
metric_name = 'fpr'
metric_weights = script_metric_weights[metric_name]
figure_name = '%s%s_descent_%s.pdf' % (results_dir, data_name, metric_name)

### SANITY CHECK PERFORMANCE
np.mean(S == np.greater(predict_group(X), 0.5))


train_stats_df = get_performance_metrics(df = df_train,
                                         data_type = 'train',
                                         predict_yhat = predict_yhat,
                                         predict_prob = predict_prob)


audit_stats_df = get_performance_metrics(df = df_audit,
                                         data_type = 'audit',
                                         predict_yhat = predict_yhat,
                                         predict_prob = predict_prob)

test_stats_df = get_performance_metrics(df = df_test,
                                        data_type = 'test',
                                        predict_yhat = predict_yhat,
                                        predict_prob = predict_prob)

train_stats_df[train_stats_df['s'] >= 0][['s', metric_name]]



# Fit Counterfactual Distribution
ctf_args = {
    'df': df_audit,
    'metric_weights': metric_weights,
    #
    'predict_yhat': predict_yhat,
    'predict_group':predict_group,
    'predict_ytrue': predict_ytrue,
    'predict_ytrue_maj': predict_ytrue_maj,
    #
    'outcome_column_name': outcome_column_name,
    'group_column_name':group_column_name,
    'minority_group':minority_group,
    #
    'random_state':random_state,
    }

ctf_dist = CounterfactualDistribution(**ctf_args)
df_audit_min, df_audit_maj = split_by_group(df_audit)
results = ctf_dist.fit(df_min_test = df_audit_min, df_maj_test = df_audit_maj, max_iterations = 1e2, step_size = 0.1)
f, _ = plot_descent_profile(results)
f.savefig(figure_name)

# Build Preprocessor
pp = ctf_dist.build_preprocessor(df_audit)
stats_df = get_performance_metrics(df = df_test,
                                   data_type = 'test',
                                   pp = pp,
                                   predict_yhat = predict_yhat,
                                   predict_prob = predict_prob,
                                   minority_group = minority_group)

stats_df[stats_df['s'] >= 0][['s', metric_name, 'dist_type']]


