from ctfdist.paths import *
from ctfdist.descent import *
from ctfdist.helper_functions import *
from scripts.plot_helper import *
from sklearn.linear_model import LogisticRegression

# dashboard variables
data_name = 'adult'
random_seed = 2338
train_size = 0.20
test_size = 0.20
audit_size = 0.60

script_metric_weights = {
    'fpr': {'fpr': 1.0},
    'fnr': {'fnr': 1.0},
    #'fdr': {'fdr': 1.0},
    'da_out': {'da_out': 1.0},
    #'da_mix': {'da_out': 0.9, 'da_in': 0.1},
    # 'da_in': {'da_in': 1.0},
    }

# setup file names
data_file_name = '%s/%s_proxy_data.csv' % (data_dir, data_name)

results_file_names = {
    'dataset_summary_table': '%s_data_summary.tex',
    'ctf_distribution_table': '%s_ctf_distribution_table.tex',
    'model_performance_table': '%s_model_performance_table.tex',
    'repair_performance_stats': '%s_postprocessing.pkl',
    }

results_header = '%s/%s' % (results_dir, data_name)
results_files = {k: v % results_header for k, v in results_file_names.items()}
results_files['header'] = results_header

##### LOAD DATASET ####
if 'adult' in data_name:
    outcome_column_name = 'IncomeOver50K'
    group_column_name = 'Sex'
    default_minority_group = 'Male'
    default_majority_group = 'Female'

if 'compas' in data_name:
    outcome_column_name = 'arrest'
    group_column_name = 'race'
    default_minority_group = 'non-white'
    default_majority_group = 'white'

random_state = np.random.RandomState(random_seed)

# load dataset
raw_df = pd.read_csv(data_file_name)
raw_df[outcome_column_name] = raw_df[outcome_column_name] + 0.0
if np.any(raw_df[outcome_column_name] == -1):
    raw_df[outcome_column_name] = (raw_df[outcome_column_name] + 1.0) / 2.0

df_train, df_audit, df_test = train_test_audit_split(df = raw_df,
                                                     train_size = train_size,
                                                     test_size = test_size,
                                                     audit_size = audit_size,
                                                     group_column_name = group_column_name,
                                                     outcome_column_name = outcome_column_name,
                                                     random_state = random_state)

X_train, Y_train, S_train = df_to_XYS(df_train,
                                      outcome_column_name = outcome_column_name,
                                      group_column_name = group_column_name,
                                      minority_group = default_minority_group)

##### TRAIN MODEL TO REPAIR #####
f_hat = LogisticRegression(penalty = 'l2', C = 100)
f_hat.fit(X = X_train, y = Y_train)
class_idx_yhat = np.flatnonzero(f_hat.classes_ > 0)
predict_prob = lambda x: f_hat.predict_proba(x)[:, class_idx_yhat]
predict_yhat = lambda x: np.greater(predict_prob(x), 0.5)

# check performance
performance_args = {
    'predict_yhat': predict_yhat,
    'predict_prob': predict_prob,
    "outcome_column_name": outcome_column_name,
    "group_column_name": group_column_name,
    "minority_group": default_minority_group,
    }

all_stats_df = get_all_performance_metrics(df_dict = {'train': df_train, 'audit': df_audit, 'test': df_test},
                                           **performance_args)

cv_stats_df, fold_stats_df = get_cv_based_model_stats(predict_yhat = predict_yhat,
                                                      predict_ytrue = predict_prob,
                                                      x = X_train,
                                                      y = Y_train,
                                                      s = S_train,
                                                      group_labels = [default_minority_group, default_majority_group],
                                                      n_folds = 10)

cv_stats_df.to_csv(results_files['model_performance_table'])

##### RUN REPAIR EXPERIMENTS #####

# determine minority for each metric
gap_df = all_stats_df[(all_stats_df['data_type'] == 'test') & (all_stats_df['s'] >= 0) & (all_stats_df['dist_type'] == 'obs')]
gap_df = gap_df.sort_values(by = ['s'], ascending = [False])

all_results = {}
minority_choice = {}
min_idx = gap_df['s'] == 0
maj_idx = gap_df['s'] == 1

for metric_name in script_metric_weights.keys():

    if metric_name in ['da_in', 'da_out', 'da_mix']:
        minority_choice[metric_name] = default_minority_group
    else:
        maj_value = float(gap_df[maj_idx][metric_name].values)
        min_value = float(gap_df[min_idx][metric_name].values)
        gap_value = min_value - maj_value
        if gap_value > 0:
            minority_choice[metric_name] = default_minority_group
        else:
            minority_choice[metric_name] = default_majority_group


def pp_table_row(stats_df, metric_name, minority_group, majority_group):
    obs_idx = (stats_df['dist_type'] == 'obs')
    ctf_idx = (stats_df['dist_type'] == 'ctf')
    all_idx = (stats_df['s'] == -1)
    min_idx = (stats_df['s'] == 0)
    maj_idx = (stats_df['s'] == 1)

    tmp_all = stats_df[all_idx & obs_idx].reset_index(drop = True)
    tmp_all_ctf = stats_df[all_idx & ctf_idx].reset_index(drop = True)
    tmp_maj = stats_df[maj_idx & obs_idx].reset_index(drop = True)
    tmp_min = stats_df[min_idx & obs_idx].reset_index(drop = True)
    tmp_min_ctf = stats_df[min_idx & ctf_idx].reset_index(drop = True)

    row_df = pd.DataFrame({
                              'metric_name': [metric_name],
                              'minority_group': [minority_group],
                              'majority_group': [majority_group]})

    row_df['obs_auc_min'] = tmp_min['auc']
    row_df['ctf_auc_min'] = tmp_min_ctf['auc']

    if metric_name not in ['da_in', 'da_out', 'da_mix']:

        row_df['obs_value_maj'] = tmp_maj[metric_name]
        row_df['ctf_value_maj'] = tmp_maj[metric_name]

        row_df['obs_value_min'] = tmp_min[metric_name]
        row_df['ctf_value_min'] = tmp_min_ctf[metric_name]

        row_df['obs_gap_min'] = row_df['obs_value_min'] - row_df['obs_value_maj']
        row_df['ctf_gap_min'] = row_df['ctf_value_min'] - row_df['obs_value_maj']

    else:

        row_df['obs_value_maj'] = np.nan
        row_df['ctf_value_maj'] = np.nan
        row_df['obs_value_min'] = np.nan
        row_df['ctf_value_min'] = np.nan
        row_df['obs_gap_min'] = np.nan
        row_df['ctf_gap_min'] = np.nan

    row_df = row_df[['metric_name', 'minority_group', 'majority_group',
                     'obs_value_maj', 'obs_value_min', 'obs_gap_min', 'ctf_value_min', 'ctf_gap_min',
                     'obs_auc_min', 'ctf_auc_min']]

    return row_df


for metric_name, metric_weights in script_metric_weights.items():

    # pick correct minority
    if minority_choice[metric_name] == default_minority_group:
        minority_group = default_minority_group
        majority_group = default_majority_group
    else:
        minority_group = default_majority_group
        majority_group = default_minority_group

    # build true model
    X_audit, Y_audit, S_audit = df_to_XYS(df_audit, minority_group = minority_group)
    f_true = LogisticRegression(penalty = 'l2', C = 100)
    f_true.fit(X = X_audit[S_audit == 0, :], y = Y_audit[S_audit == 0])
    class_idx_ytrue = np.flatnonzero(f_true.classes_ > 0)
    predict_ytrue = lambda x: f_true.predict_proba(x)[:, class_idx_ytrue]

    # build model to get membership
    f_group = LogisticRegression(penalty = 'l2', C = 100)
    f_group.fit(X = X_audit, y = S_audit)
    pos_class_idx = np.flatnonzero(f_group.classes_ > 0)
    class_idx_group = np.flatnonzero(f_group.classes_ > 0)
    predict_group = lambda x: f_group.predict_proba(x)[:, class_idx_group]

    # evaluate performance of true / group model
    true_model_stats_df, group_model_stats_df = evaluate_repair_models(predict_ytrue,
                                                                       predict_group,
                                                                       minority_group,
                                                                       majority_group,
                                                                       X_audit,
                                                                       Y_audit,
                                                                       S_audit)

    # setup counterfactual distribution
    ctf_args = {
        'df': df_audit,
        'metric_weights': metric_weights,
        'outcome_column_name': outcome_column_name,
        'group_column_name': group_column_name,
        'minority_group': minority_group,
        'predict_yhat': predict_yhat,
        'predict_group': predict_group,
        'predict_ytrue': predict_ytrue,
        'predict_ytrue_maj': predict_ytrue,
        'random_state': random_state,
        }

    # fit counterfactual distribution
    ctf_dist = CounterfactualDistribution(**ctf_args)
    df_audit_min, df_audit_maj = split_by_group(df_audit, minority_group = minority_group)
    descent_results_df = ctf_dist.fit(df_min_test = df_audit_min, df_maj_test = df_audit_maj, max_iterations = 1e2, step_size = 0.1)

    # build preprocessor
    pp = ctf_dist.build_preprocessor(df_audit)

    # store metrics for table
    stats_df = get_performance_metrics(df = df_test,
                                       data_type = 'test',
                                       pp = pp,
                                       predict_yhat = predict_yhat,
                                       predict_prob = predict_prob,
                                       outcome_column_name = outcome_column_name,
                                       group_column_name = group_column_name,
                                       minority_group = minority_group)

    # pp table row
    pp_row = pp_table_row(stats_df, metric_name, minority_group, majority_group)
    if metric_name in ['da_in', 'da_out', 'da_mix']:
        tmp = pd.DataFrame(descent_results_df)
        pp_row['obs_gap_min'] = np.max(tmp[tmp['type'] == 'descent']['gap'].values)
        pp_row['ctf_gap_min'] = np.min(tmp[tmp['type'] == 'descent']['gap'].values)

    all_results[metric_name] = {'ctf_dist': ctf_dist,
                                'pp': pp,
                                'pp_row': pp_row,
                                'descent_results': descent_results_df,
                                'true_model_stats_df': true_model_stats_df,
                                'group_model_stats_df': group_model_stats_df,
                                'metric_name': metric_name,
                                'minority_group': minority_group,
                                'majority_group': majority_group,
                                'stats_df': stats_df,
                                'dist_table_df': ctf_dist.marginal_distributions(df = df_test)}

#### Save Output

# save table
pp_table = pd.concat([v['pp_row'] for k, v in all_results.items()])
pp_table = pp_table.reset_index(drop = True)
pp_table.insert(loc = 0, column = 'data_name', value = data_name)
pp_table[['metric_name', 'obs_value_min', 'ctf_value_min', 'obs_gap_min', 'ctf_gap_min', 'obs_auc_min', 'ctf_auc_min']]
pp_table.to_pickle(results_files['repair_performance_stats'])

# save counterfactual distribution in table
all_ctf_dists = [v['dist_table_df'] for k, v in all_results.items()]
ctf_dist_metrics = list(script_metric_weights.keys())

obs_dist_table = all_results[ctf_dist_metrics[1]]['dist_table_df'].iloc[:, 0:2]
ctf_dist_table = pd.concat([all_results[m]['dist_table_df'].iloc[:, 2] for m in ctf_dist_metrics], axis = 1)
ctf_df = pd.concat([obs_dist_table, ctf_dist_table], axis = 1)
ctf_df.to_latex(results_files['ctf_distribution_table'])

# save descent plots for each metric
for metric_name, metric_results in all_results.items():
    f, _ = plot_descent_profile(metric_results['descent_results'])
    plot_file_name = '%s_%s_descent.pdf' % (results_files['header'], metric_name)
    f.savefig(plot_file_name)
