import numpy as np
import pandas as pd
import itertools
import warnings

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import confusion_matrix, roc_auc_score as get_auc, brier_score_loss as get_brier
from ctfdist.preprocessor import RandomizedPreprocessor
from ctfdist.debug import ipsh

warnings.filterwarnings("ignore", category = FutureWarning)


#### Data Frame Management

def compress_df(df, include_count = True):
    if include_count:
        mini_df = df.groupby(df.columns.tolist()).size().reset_index().rename(columns = {0: 'n'})
    else:
        mini_df = df.groupby(df.columns.tolist()).size().reset_index().drop(columns = [0], axis = 1)
    return mini_df


def expand_df(df, count_name = 'n', drop_count = True):
    assert count_name in df.columns
    full_df = df.loc[np.repeat(df.index.values, df[count_name])]
    if drop_count:
        full_df = full_df.drop(columns = [count_name], axis = 1)
    full_df = full_df.reset_index
    return full_df


def train_test_audit_split(df, train_size, test_size, audit_size, outcome_column_name, group_column_name, random_state):

    assert isinstance(df, pd.DataFrame)
    assert isinstance(train_size, float)
    assert isinstance(test_size, float)
    assert isinstance(audit_size, float)

    assert 0.0 <= train_size <= 1.0
    assert 0.0 <= test_size <= 1.0
    assert 0.0 <= audit_size <= 1.0
    assert np.isclose(train_size + test_size + audit_size, 1.0)

    # split into audit and other
    df_audit, df_other = train_test_split(df,
                                          train_size = audit_size,
                                          stratify = df[[group_column_name, outcome_column_name]],
                                          random_state = random_state)

    # split other into train and test
    other_size = train_size + test_size
    train_size = train_size / other_size
    test_size = test_size / other_size
    df_train, df_test = train_test_split(df_other,
                                         train_size = train_size,
                                         stratify = df_other[[group_column_name, outcome_column_name]],
                                         random_state = random_state)

    # cols = [group_column_name, outcome_column_name]
    # chk_df = compress_df(df[cols])
    # chk_other = compress_df(df_other[cols])
    # chk_train = compress_df(df_train[cols])
    # chk_test = compress_df(df_test[cols])
    # chk_audit = compress_df(df_audit[cols])
    assert len(df) == len(df_audit) + len(df_train) + len(df_test)
    return df_train, df_audit, df_test


def df_to_XYS(df, outcome_column_name = None, group_column_name = None, minority_group = 0):

    if outcome_column_name is not None:
        Y = df[outcome_column_name].values
    else:
        Y = df.iloc[:, 0].values

    if group_column_name is not None:
        G = df[group_column_name]
    else:
        G = df.iloc[:, 1]

    S = np.isin(G.values, minority_group, invert = True)
    X = np.array(df.iloc[:, 2:].values, dtype = np.float)
    Y = np.array(Y, dtype = np.float)
    return X, Y, S


#### Performance

def evaluate_repair_models(predict_ytrue, predict_group, minority_group, majority_group, X, Y, S):

    s = S.flatten()
    labels, indices = np.unique(s, return_inverse = True)
    group_indices = [np.ones_like(s, dtype = 'bool')]
    group_labels = ['All', minority_group, majority_group]
    for label in labels:
        idx = np.isin(indices, label)
        group_indices.append(idx)

    true_model_stats = []
    for label, idx in zip(group_labels, group_indices):
        stats = get_cv_stats(handle = predict_ytrue, x = X[idx, :], y = Y[idx], group_label = label)
        true_model_stats += [stats]

    group_model_stats = []
    for label, idx in zip(group_labels, group_indices):
        stats = get_cv_stats(handle = predict_group, x = X[idx, :], y = S[idx], group_label = label)
        group_model_stats += [stats]

    true_model_stats_df = pd.concat(true_model_stats)
    group_model_stats_df = pd.concat(group_model_stats)
    return true_model_stats_df, group_model_stats_df


def get_all_performance_metrics(df_dict, **kwargs):
    all_stats_df = [get_performance_metrics(df = v, data_type = k, **kwargs) for k, v in df_dict.items()]
    return pd.concat(all_stats_df)


def get_performance_metrics(df, predict_yhat, predict_prob, pp = None, outcome_column_name = None, group_column_name = None, minority_group = None, data_type = 'test'):

    assert isinstance(df, pd.DataFrame)
    assert callable(predict_yhat)
    assert callable(predict_prob)

    X, Y, S = df_to_XYS(df, outcome_column_name = outcome_column_name, group_column_name = group_column_name, minority_group = minority_group)
    X_ctf = np.array(X)
    min_idx = S == 0
    maj_idx = S == 1

    dist_types = ['obs']

    if pp is not None:
        assert isinstance(pp, RandomizedPreprocessor)
        X_ctf[min_idx] = pp.adjust(X[min_idx, :])
        dist_types += ['ctf']

    stats = []
    for t in dist_types:
        for s in [-1, 0, 1]:
            x = X_ctf if t == 'ctf' else X
            y = Y

            if s == 0:
                x = x[min_idx,]
                y = Y[min_idx]
            elif s == 1:
                x = x[maj_idx,]
                y = Y[maj_idx]

            # compute key metrics
            prob = predict_prob(x)
            h = predict_yhat(x)
            try:
                tn, fp, fn, tp = confusion_matrix(y_true = y, y_pred = h).ravel()
            except Exception:
                ipsh()
            n_pos, n_neg = np.sum(y == 1), np.sum(y < 1)

            stats.append({'s': s,
                          'dist_type': t,
                          'data_type': data_type,
                          #
                          'n_pos': n_pos,
                          'n_neg': n_neg,
                          'h_pos': np.sum(h == 1),
                          'h_neg': np.sum(h < 0),
                          #
                          'tpr': tp / n_pos,
                          'fnr': fn / n_pos,
                          'fpr': fp / n_neg,
                          'tnr': tn / n_neg,
                          #
                          'auc': get_auc(y_score = prob, y_true = y),
                          'cal': get_brier(y_prob = prob, y_true = y),
                          'mse': np.mean(np.not_equal(h, y))})


    return pd.DataFrame(stats).sort_values(by = ['dist_type', 's'], ascending = [False, True])


def split_by_group(df, outcome_column_name = None, group_column_name = None, minority_group = 0):

    assert isinstance(df, pd.DataFrame)
    assert df.ndim == 2
    assert df.shape[0] >= 4
    assert df.shape[1] >= 3
    df = df.copy()

    if outcome_column_name is None:
        outcome_column_name = df.columns[0]

    if group_column_name is None:
        group_column_name = df.columns[1]

    assert outcome_column_name in df.columns
    assert group_column_name in df.columns
    assert len(df[outcome_column_name].unique()) == 2
    assert len(df[group_column_name].unique()) >= 2

    # infer group labels
    group_counts = df[group_column_name].value_counts()
    n_groups = group_counts.size
    group_labels = group_counts.index.tolist()

    assert minority_group in df[group_column_name].values
    group_labels.remove(minority_group)
    if n_groups == 2:
        majority_group = group_labels[0]
    else:
        majority_group = 'Not%s' % minority_group

    # split data frame based on group membership
    minority_idx = np.isin(df[group_column_name], minority_group)
    df = df.drop([outcome_column_name, group_column_name], axis = 1)
    df_min = df[minority_idx].copy()
    df_maj = df[~minority_idx].copy()
    return df_min, df_maj


def split_by_group_with_SY(df, outcome_column_name = None, group_column_name = None, minority_group = None):

    assert isinstance(df, pd.DataFrame)
    assert df.ndim == 2
    assert df.shape[0] >= 4
    assert df.shape[1] >= 3
    df = df.copy()

    if outcome_column_name is None:
        outcome_column_name = df.columns[0]

    if group_column_name is None:
        group_column_name = df.columns[1]

    assert outcome_column_name in df.columns
    assert group_column_name in df.columns
    assert len(df[outcome_column_name].unique()) == 2
    assert len(df[group_column_name].unique()) >= 2

    # infer group labels
    group_counts = df[group_column_name].value_counts()
    n_groups = group_counts.size

    if minority_group is None:
        minority_group = group_counts.tail(1).index[0]
    else:
        assert minority_group in df[group_column_name].values

    if n_groups == 2:
        majority_group = group_counts.index[0]
    else:
        majority_group = 'Not%s' % minority_group

    # split data frame based on group membership
    minority_idx = np.isin(df[group_column_name], minority_group)

    df_min = df[minority_idx].copy()
    df_maj = df[~minority_idx].copy()
    return df_min, df_maj


def binary_features_df(n_dim, variable_names = None):
    assert isinstance(n_dim, int)
    assert n_dim >= 1
    X = np.stack([np.array(row) for row in itertools.product([0.0, 1.0], repeat = n_dim)])
    if variable_names is None:
        variable_names = ['x%d' % j for j in range(n_dim)]
    else:
        assert len(variable_names) == n_dim
    df = pd.DataFrame(X, columns = variable_names).sort_values(variable_names)
    return df


def get_empirical_pdf(df):

    """
    :param df: data frame of observations
    :return:  pdf_handle function handle to obtain empirical pdf using a row in x
              pdf        data frame containing all rows and observed empirical probabilities
    """

    variable_names = df.columns.tolist()
    pdf = df.groupby(variable_names, as_index = False).size().reset_index().rename(columns = {0: 'count'})

    # fill missing values
    # n_dim = len(variable_names)
    # full_df = binary_features_df(n_dim, variable_names)
    # pdf = pdf.merge(full_df.get(variable_names), on = variable_names, how = 'right')
    # pdf = pdf.fillna({'count': 0}).sort_values(by = variable_names).reset_index()

    # compute empirical probability
    assert 'p' not in pdf.columns
    pdf['p'] = pdf['count'] / pdf['count'].sum()
    P = np.array(pdf['p'].values)

    assert np.all(np.isfinite(P))
    assert np.all(P >= 0.0)
    assert np.isclose(np.sum(P), 1.0)

    lookup_table = np.array(pdf[variable_names].values)
    n_variables = len(variable_names)

    # todo change this so that it will return rows for matrix / handle different datatypes
    def pdf_handle(vals):

        assert isinstance(vals, np.ndarray)

        if vals.ndim == 1:

            assert len(vals) == n_variables
            idx = np.flatnonzero((lookup_table == vals).all(axis = 1))

            if len(idx) == 1:
                return P[idx]
            else:
                return np.array([0.0])

        else:

            assert vals.ndim == 2
            n, m = vals.shape
            assert n >= 1
            assert m == n_variables

        p = np.zeros(n, dtype = np.float)
        for k, row in enumerate(vals):
            idx = np.flatnonzero((lookup_table == row).all(axis = 1))
            if len(idx) == 1:
                p[k] = P[idx]

        return p

    return pdf_handle


def get_empirical_counts(df):

    """
    :param df: data frame of observations
    :return:  empirical distribution
    """

    variable_names = df.columns.tolist()
    pdf = df.groupby(variable_names, as_index = False).size().reset_index().rename(columns = {0: 'count'})

    # fill missing values
    n_dim = len(variable_names)
    full_df = binary_features_df(n_dim, variable_names)
    pdf = pdf.merge(full_df.get(variable_names), on = variable_names, how = 'right')
    pdf = pdf.fillna({'count': 0}).sort_values(by = variable_names).reset_index()
    
    #sort
    pdf = pdf.sort_values(by=variable_names)
    
    # compute empirical probability
    assert 'p' not in pdf.columns
    pdf['p'] = pdf['count'] / pdf['count'].sum()
    P = np.array(pdf['p'].values)

    assert np.all(np.isfinite(P))
    assert np.all(P >= 0.0)
    assert np.isclose(np.sum(P), 1.0)

    return pdf, P


def get_group_stats(df, predict_handle, outcome_name, group_name, minority_group, group_label, true_handle = None):

    assert callable(predict_handle)
    assert callable(true_handle)
    assert isinstance(minority_group, (int, str))
    assert isinstance(group_label, (int, str))
    TRUE_STATS = ['cal', 'fpr', 'fnr', 'fdr']

    # split data
    X, Y, S = df_to_XYS(df, outcome_column_name = outcome_name, group_column_name = group_name, minority_group = minority_group)

    if group_label == minority_group:
        group_idx = S == 0
    else:
        group_idx = S == 1

    # get model predictions
    X = X[group_idx, :]
    Y = Y[group_idx, :]

    hp = predict_handle(X)

    if callable(true_handle):
        ht = true_handle(X)
    else:
        ht = np.repeat(float('nan'), X.shape[0])

    mean_ht = np.mean(ht)

    group_stats = {
        'group': group_name,
        'label': group_label,
        'n': np.sum(group_idx),
        'n_pos': np.sum(Y > 0),
        'n_neg': np.sum(Y <= 0),
        #
        'mean_predicted_outcome': np.mean(hp),
        'mean_true_outcome': np.mean(ht),
        'mean_cal': np.mean(np.abs(hp - ht)),
        'mean_fpr': np.mean(np.multiply(hp, 1.0 - ht)) / (1.0 - mean_ht),
        'mean_fnr': np.mean(np.multiply(1.0 - hp, 1.0 - ht)) / mean_ht,
        'mean_fdr': np.mean(np.multiply(hp, 1.0 - ht)) / mean_ht
        }

    return group_stats


def get_cv_based_model_stats(predict_yhat, predict_ytrue, x, y, s, group_labels, n_folds = 10, random_state = None):

    agg_handles = [np.mean, np.std]

    cv_splitter = StratifiedKFold(n_splits = n_folds, random_state = random_state, shuffle = False)
    _, cv_strata = np.unique(np.column_stack((s, y)), axis = 0, return_inverse = True)

    fold_id = 0
    fold_stats = []
    for train_idx, test_idx in cv_splitter.split(X = x, y = cv_strata):

        x_test = x[test_idx, :]
        y_test = y[test_idx]
        s_test = s[test_idx]
        fold_id += 1

        h_test = predict_yhat(x_test).flatten()
        tn, fp, fn, tp = confusion_matrix(y_true = y_test, y_pred = h_test).ravel()
        n_pos, n_neg = np.sum(y_test >0), np.sum(y_test <= 0)

        stats = {
            'group': 'All',
            'fold_id': fold_id,
            'n': len(test_idx),
            'n_pos': int(n_pos),
            'n_neg': int(n_neg),
            'auc': get_auc(y_score = h_test, y_true = y_test),
            'cal': get_brier(y_prob = h_test, y_true = y_test),
            'mse': np.mean(np.not_equal(h_test, y_test)),
            'tpr': tp / n_pos,
            'fnr': fn / n_pos,
            'fpr': fp / n_neg,
            'tnr': tn / n_neg,
            }

        fold_stats.append(stats)

        for g, label in enumerate(group_labels):
            idx = s_test == g
            tn, fp, fn, tp = confusion_matrix(y_true = y_test[idx], y_pred = h_test[idx]).ravel()
            n_pos, n_neg = np.sum(y_test[idx] > 0), np.sum(y_test[idx] <= 0)

            stats = {
                'group': label,
                'fold_id': fold_id,
                'n': np.sum(test_idx),
                'n_pos': int(np.sum(y_test[idx] > 0)),
                'n_neg': int(np.sum(y_test[idx] <= 0)),
                'auc': get_auc(y_score = h_test[idx], y_true = y_test[idx]),
                'cal': get_brier(y_prob = h_test[idx], y_true = y_test[idx]),
                'mse': np.mean(np.not_equal(h_test[idx], y_test[idx])),
                'tpr': tp / n_pos,
                'fnr': fn / n_pos,
                'fpr': fp / n_neg,
                'tnr': tn / n_neg,
                }
            fold_stats.append(stats)

    fold_stats_df = pd.DataFrame(fold_stats).rename(columns = {'auc': 'AUC',
                                                               'cal': 'CAL',
                                                               'mse': 'MSE',
                                                               'tpr': 'TPR',
                                                               'fpr': 'FPR',
                                                               'fnr': 'FNR',
                                                               'tnr': 'TNR'})

    grouped = fold_stats_df.drop(columns = 'fold_id').groupby('group', as_index = False)
    cv_stats_df = grouped.agg(agg_handles)
    cv_stats_df = cv_stats_df.transpose()
    return cv_stats_df, fold_stats_df


def get_cv_stats(handle, x, y, group_label = 'All', s = None, n_folds = 10, random_state = None):

    agg_handles = [np.mean, np.std, np.min, np.max]

    cv_splitter = StratifiedKFold(n_splits = n_folds, random_state = random_state, shuffle = False)
    if s is None:
        cv_strata = y
    else:
        _, cv_strata = np.unique(np.column_stack((s, y)), axis = 0, return_inverse = True)

    fold_id = 0
    fold_stats = []
    for train_idx, test_idx in cv_splitter.split(X = x, y = cv_strata):

        x_test = x[test_idx, :]
        y_test = y[test_idx]
        fold_id += 1
        h_test = handle(x_test).flatten()

        stats = {
            'group': group_label,
            'fold_id': fold_id,
            'n': len(test_idx),
            'n_pos': int(np.sum(y_test > 0)),
            'n_neg': int(np.sum(y_test <= 0)),
            'auc': float('nan'),
            'cal': get_brier(y_prob = h_test, y_true = y_test)
            }

        try:
            stats['auc'] = get_auc(y_score = h_test, y_true = y_test)
        except ValueError:
            pass

        fold_stats.append(stats)

    fold_stats_df = pd.DataFrame(fold_stats)
    fold_stats_df = fold_stats_df.rename(columns = {'auc': 'AUC', 'cal': 'CAL'})
    grouped = fold_stats_df.drop(columns = 'fold_id').groupby('group', as_index=False)
    cv_stats_df = grouped.agg(agg_handles)
    return cv_stats_df


def get_model_stats(df, predict_handle, outcome_name = None, group_name = None, minority_group = None, group_handle = None, true_handle_min = None, true_handle_maj = None):

    assert callable(predict_handle)
    assert isinstance(minority_group, (int, str))
    assert callable(group_handle) or group_handle is None
    assert callable(true_handle_min) or true_handle_min is None
    assert callable(true_handle_maj) or true_handle_maj is None

    TRUE_STATS = ['cal', 'fpr', 'fnr', 'fdr']

    # split data
    X, Y, S = df_to_XYS(df, outcome_column_name = outcome_name, group_column_name = group_name, minority_group = minority_group)

    minority_idx = S == 0
    majority_idx = S != 0

    # split groups
    labels, indices = np.unique(S, return_inverse = True)
    n_groups = len(labels)


    # determine labels if they were not provided
    if minority_group is None:
        minority_group = df[group_name][S == 0].iloc[0]

    if n_groups == 2:
        majority_group = df[group_name][S != 0].iloc[0]
    else:
        majority_group = 'Not%s' % minority_group

    # get model predictions
    H = predict_handle(X)

    # get predictions from true model if handles are provided
    has_true_handles = callable(true_handle_min) and callable(true_handle_maj)
    H_true = np.repeat(float('nan'), X.shape[0])
    if has_true_handles:
        H_true[minority_idx] = true_handle_min(X[minority_idx]).flatten()
        H_true[majority_idx] = true_handle_maj(X[majority_idx]).flatten()

    # get group statistics
    group_stats = {}
    for s in labels:

        idx = s == indices
        name = minority_group if s == 0 else majority_group
        hp = H[idx]

        stats = {
            'group': name,
            'label': s,
            'n': np.sum(idx),
            'n_pos': np.sum(Y[idx] > 0),
            'n_neg': np.sum(Y[idx] <= 0),
            #
            'mean_dist_out': float('nan'),
            'mean_predicted_outcome': np.mean(hp),
            'mean_true_outcome': float('nan'),
            'mean_cal': float('nan'),
            'mean_fpr': float('nan'),
            'mean_fnr': float('nan'),
            'mean_fdr': float('nan'),
            }

        if has_true_handles:

            ht = H_true[idx]
            mean_ht = np.mean(ht)

            stats.update({
                'mean_true_outcome': np.mean(ht),
                'mean_cal': np.mean(np.abs(hp - ht)),
                'mean_fpr': np.mean(np.multiply(hp, 1.0 - ht)) / (1.0 - mean_ht),
                'mean_fnr': np.mean(np.multiply(1.0 - hp, 1.0 - ht)) / mean_ht,
                'mean_fdr': np.mean(np.multiply(hp, 1.0 - ht)) / mean_ht
                })

        group_stats[name] = stats

    # cross-group statistics
    model_stats = {
        'group': 'Gap',
        'label': float('nan'),
        'n': X.shape[0],
        'n_pos': np.sum(Y  > 0),
        'n_neg': np.sum(Y <= 0),
        #
        'mean_dist_out': float('nan'),
        'mean_predicted_outcome': np.mean(hp),
        'mean_true_outcome': float('nan'),
        'mean_cal': float('nan'),
        'mean_fpr': float('nan'),
        'mean_fnr': float('nan'),
        'mean_fdr': float('nan'),
        }

    for name in TRUE_STATS:
        model_stats['mean_%s' % name] = float('nan')


    # output distributions
    h_min = np.mean(H[minority_idx])
    h_maj = np.mean(H[majority_idx])
    model_stats['mean_dist_out'] = np.multiply(h_min, np.log(h_min) - np.log(h_maj)) + np.multiply(1.0 - h_min, np.log(1.0 - h_min) - np.log(h_maj))

    # statistics based on true outcomes
    if has_true_handles:
        get_gap = lambda name: group_stats[minority_group]['mean_%s' % name] - group_stats[majority_group]['mean_%s' % name]
        for name in TRUE_STATS:
            model_stats['mean_%s' % name] = get_gap(name)

    audit_stats_df = [pd.DataFrame.from_dict(group_stats[minority_group], orient = 'index')]
    audit_stats_df.append(pd.DataFrame.from_dict(group_stats[majority_group], orient = 'index'))
    audit_stats_df.append(pd.DataFrame.from_dict(model_stats, orient = 'index'))
    return pd.concat(audit_stats_df, axis = 1)


##### DESCENT #####

def get_fpr(x, get_pyhat, get_pytrue, get_px = None):
    hp = get_pyhat(x)
    if callable(get_pytrue):
        ht = get_pytrue(x).flatten()
    elif isinstance(get_pytrue, np.ndarray):
        ht = get_pytrue.flatten()
        assert x.shape[0] == len(ht)
    return np.mean(np.multiply(hp, 1.0 - ht)) / np.mean(1.0 - ht)


def get_fnr(x, get_pyhat, get_pytrue, get_px = None):
    hp = get_pyhat(x)
    if callable(get_pytrue):
        ht = get_pytrue(x).flatten()
    elif isinstance(get_pytrue, np.ndarray):
        ht = get_pytrue.flatten()
        assert x.shape[0] == len(ht)
    return np.mean(np.multiply(1.0 - hp, ht)) / np.mean(ht)


def get_fdr(x, get_pyhat, get_pytrue, get_px = None):
    hp = get_pyhat(x)
    if callable(get_pytrue):
        ht = get_pytrue(x).flatten()
    elif isinstance(get_pytrue, np.ndarray):
        ht = get_pytrue.flatten()
        assert x.shape[0] == len(ht)
    return np.mean(np.multiply(hp, 1.0 - ht)) / np.mean(hp)


def get_cal(x, get_pyhat, get_pytrue, get_px = None):
    hp = get_pyhat(x)
    if callable(get_pytrue):
        ht = get_pytrue(x).flatten()
    elif isinstance(get_pytrue, np.ndarray):
        ht = get_pytrue.flatten()
        assert x.shape[0] == len(ht)
    return np.mean(np.abs(hp - ht))


def get_nan(x, get_pyhat, get_pytrue, get_px = None):
    return float('nan')


def get_accu(x, get_pyhat, get_pytrue, get_px = None):
    hp = get_pyhat(x)
    if callable(get_pytrue):
        ht = get_pytrue(x).flatten()
    elif isinstance(get_pytrue, np.ndarray):
        ht = get_pytrue.flatten()
        assert x.shape[0] == len(ht)
    return np.mean(np.multiply(hp, 1.0 - ht)) + np.mean(np.multiply(1.0 - hp, ht))





















