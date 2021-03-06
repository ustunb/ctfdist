from ctfdist.preprocessor import *
from ctfdist.helper_functions import get_empirical_pdf, compress_df, split_by_group


class RepairData(object):

    """
    helper class to store, access, and update classification data with subgroups
    """


    _OUTCOME_COL_IDX = 0
    _GROUP_COL_IDX = 1


    #### Initialization ####
    def __init__(self, df, **kwargs):

        """
        :param df: pandas.DataFrame with outcome in column 0, group in column 1, must have at least 3 columns and 4 rows (e.g. 1 pos/neg point for each group)
        :param outcome_column_name: column name of outcome variable in df. set as df.columns[0] if None
        :param group_column_name:  column name of the group variable df. set as df.columns[1] if None
        :param minority_group: value of the minority group in group_column; if None, chosen as the least common group
        """

        assert isinstance(df, pd.DataFrame)
        assert df.ndim == 2
        assert df.shape[0] >= 4
        assert df.shape[1] >= 3

        df = df.copy()
        column_names = df.columns.tolist()

        outcome_column_name = kwargs.get('outcome_column_name', column_names[self._OUTCOME_COL_IDX])
        group_column_name = kwargs.get('group_column_name', column_names[self._GROUP_COL_IDX])
        assert outcome_column_name in df.columns
        assert group_column_name in df.columns
        assert len(df[outcome_column_name].unique()) == 2
        assert len(df[group_column_name].unique()) >= 2

        # infer group labels
        group_names = df[group_column_name].value_counts().index.tolist()
        n_groups = len(group_names)
        minority_group = kwargs.get('minority_group', str(group_names[0]))
        assert minority_group in df[group_column_name].values and n_groups >= 1

        if n_groups == 2:
            group_names.remove(minority_group)
            majority_group = str(group_names[0])
        else:
            majority_group = 'Not%s' % minority_group

        # store names
        self._outcome_name = str(outcome_column_name)
        self._group_name = str(group_column_name)
        self._minority_group = str(minority_group)
        self._majority_group = str(majority_group)
        self._n_groups = int(n_groups)

        # identify minority rows
        minority_idx = np.isin(df[group_column_name], minority_group)

        # store outcome values
        self._y_min = np.array(df[outcome_column_name][minority_idx].values, dtype = np.float)
        self._y_maj = np.array(df[outcome_column_name][~minority_idx].values, dtype = np.float)

        # drop old fields
        df = df.drop([outcome_column_name, group_column_name], axis = 1)
        self._variable_names = df.columns.tolist()

        # split data frame based on group membership
        self.df_min = df[minority_idx]
        self.df_maj = df[~minority_idx]

        # uninitialized
        self._get_px = None


    @property
    def variable_names(self):
        """
        :return: names of the variables
        """
        return self._variable_names


    @property
    def outcome_name(self):
        """
        :return: name of the outcome variable
        """
        return str(self._outcome_name)


    @property
    def group_name(self):
        """
        :return: name of the sensitive attribute
        """
        return str(self._group_name)


    @property
    def minority_group(self):
        """
        :return: name of the minority group as stored in the data
        """
        return str(self._minority_group)


    @property
    def majority_group(self):
        """
        :return: name of the majority group as stored in the data. set to "Not[MinorityGroup]" if n_groups > 2
        """
        return str(self._majority_group)


    @property
    def n_groups(self):
        """
        :return: number of distinct groups based on the sensitive attribute (must be >= 2)
        """
        return int(self._n_groups)


    @property
    def x_min(self):
        """
        :return: outcome values for minority group
        """
        return self._df_min.values


    @property
    def n_min(self):
        """
        :return: number of samples for the minority group
        """
        return self._df_min.shape[0]


    @property
    def x_maj(self):
        """
        :return: outcome values for majority group
        """
        return self.df_maj.values


    @property
    def n_maj(self):
        """
        :return: number of samples for the majority group
        """
        return self.df_maj.shape[0]


    @property
    def px_min(self):
        if self._get_px is None:
            self.px_min = get_empirical_pdf(self._df_min)
            assert callable(self._get_px)

        return self._get_px


    @px_min.setter
    def px_min(self, get_px):
        assert callable(get_px)
        self._get_px = get_px


    @property
    def y_min(self):
        """
        :return: outcome values for minority group
        """
        return np.array(self._y_min)


    @property
    def y_maj(self):
        """
        :return: outcome values for majority group
        """
        return np.array(self._y_maj)


    def __repr__(self):

        rule_width = 20
        rule_bold = '=' * rule_width

        s = [
            rule_bold,
            'outcome: %s' % self._outcome_name,
            'n_variables: %d' % len(self.variable_names),
            'group variable: %s (n = %d subgroups)' % (self._group_name, self._n_groups),
            'minority group: %r (n = %d)' % (self._minority_group, self._n_min),
            'majority group: %r (n = %d)' % (self._majority_group, self._n_maj),
            ]

        s += [rule_bold]
        return '\n'.join(s)


    @property
    def df_min(self):
        return self._df_min


    @df_min.setter
    def df_min(self, df):
        assert isinstance(df, pd.DataFrame)
        assert df.ndim == 2
        assert df.shape[0] >= 4
        assert df.shape[1] >= 3
        assert df.columns.tolist() == self._variable_names
        self._df_min = df.copy()
        self._uf_min = self._df_min.drop_duplicates()


    @property
    def uf_min(self):
        return self._uf_min


class InfluenceFunction:

    SUPPORTED_METRICS = {'da_in', 'da_out', 'cal', 'fnr', 'fpr', 'fdr'}
    NEED_YTRUE_METRICS = {'cal', 'fnr', 'fpr', 'fdr'}
    NEED_GROUP_METRICS = {'da_in'}

    #### Initialization ####
    def __init__(self, df, predict_yhat, metric_weights, **kwargs):
        """
        :param df: pandas.DataFrame with outcome in column 0, group in column 1, must have at least 3 columns and 4 rows (e.g. 1 pos/neg point for each group)
        :param predict_yhat: Pr[yhat = 1|X]
        :param predict_group: Pr[S = 1 | X]
        :param predict_ytrue: Pr[y = 1 | X, S = 0]
        :param outcome_column_name: column name of outcome variable in df. set as df.columns[0] if None
        :param group_column_name:  column name of the group variable df. set as df.columns[1] if None
        :param minority_group: value of the minority group in group_column; if None, chosen as the least common group
        """

        # split dataframe by group
        self.data = RepairData(df, **kwargs)
        self._init_metric_weights(metric_weights)

        # attach function handles
        self.predict_yhat = predict_yhat
        self.predict_group = kwargs.get('predict_group', None)
        self.predict_ytrue = kwargs.get('predict_ytrue', None)
        self.predict_ytrue_maj = kwargs.get('predict_ytrue_maj', None)

        if self.need_predict_group:
            assert callable(self.predict_group)

        if self.need_predict_ytrue:
            assert callable(self.predict_ytrue)
            assert callable(self.predict_ytrue_maj)

        self.random_state = kwargs.get('random_state', None)

        # initial other empty fields
        self._fitted = False
        self._values = None


    def _init_metric_weights(self, weights):

        """
        validates weight for different metrics passed to cf.fit
        :param weights:
        :return:
        """
        assert isinstance(weights, dict)
        assert self.SUPPORTED_METRICS.issuperset(weights.keys())

        weight_values = np.array(list(weights.values()), dtype = np.float)
        assert np.isfinite(weight_values).all() and np.greater_equal(weight_values, 0.0).all()

        total_weight = np.sum(weight_values)
        assert np.greater(total_weight, 0.0)

        processed_weights = {k: float(w)/total_weight for k, w in weights.items() if w > 0.0}
        assert len(processed_weights) >= 1

        self._weights = {metric: np.float(weight) for metric, weight in processed_weights.items()}


    @property
    def need_predict_ytrue(self):
        return any([metric_name in self.NEED_YTRUE_METRICS for metric_name in self._weights.keys()])


    @property
    def need_predict_group(self):
        return any([metric_name in self.NEED_GROUP_METRICS for metric_name in self._weights.keys()])


    @property
    def weights(self):
        return self._weights


    @property
    def fitted(self):
        return bool(self._fitted)


    @property
    def values(self):
        """
        :return: values of correction function from last time the correction function was run
        """
        if self._fitted:
            return np.array(self._values)
        else:
            raise ValueError('no values found. values are stored are each fit')


    @property
    def random_state(self):
        return self._random_state


    @random_state.setter
    def random_state(self, seed):
        if isinstance(seed, int):
            self._random_state = np.random.RandomState(seed)
        elif isinstance(seed, np.random.RandomState):
            self._random_state = seed
        elif seed is None:
            self._random_state = np.random.RandomState()


    #### Validation ####
    def _check_rep(self):

        # check data
        assert self.n_groups >= 2
        assert self.n_min == self.df_min.shape[0]
        assert self.n_maj == self.df_maj.shape[0]
        assert self.n_min == self._x_min.shape[0]
        assert self.n_maj == self._x_maj.shape[0]

        # check weights
        assert isinstance(self._weights, dict)
        assert len(self._weights) >= 1
        assert self.SUPPORTED_METRICS.issuperset(list(self._weights.keys()))

        if self.fitted:
            assert self._values is not None
            assert self._check_values(self._values)
            assert self._check_influence_function_handles()
        else:
            assert self._values is None
        return True


    def _check_fit(self):
        if not self._fitted:
            raise ValueError('need to fit first')


    def _check_values(self, values):

        """
        checks values of the correction function
        :param values: vector of correction function values for minority class
        :return: True if vals satisfies output conditions
        """
        assert len(values) == self.data.n_min
        assert np.isfinite(values).all()
        assert np.isclose(np.mean(values), 0.0)
        return True


    def _clear_influence_function_handles(self):
        for name in self.SUPPORTED_METRICS:
            handle_name = '_phi_%s' % name
            self.__setattr__(handle_name, None)


    def _check_influence_function_handles(self):
        for name in self.weights.keys():
            handle_name = '_phi_%s' % name
            assert callable(self.__getattribute__(handle_name))


    #### API Methods ####

    def update_df_min(self, df_min):
        self.data.df_min = df_min.copy()
        self._fitted = False
        self._values = None
        self._clear_influence_function_handles()


    def fit(self):
        """
        :return: fit influence function using data
        """

        # compute parameters used by multiple metrics
        x_min = self.data.x_min
        x_maj = self.data.x_maj
        p_yhat = self.predict_yhat(x_min).flatten()

        if self.need_predict_ytrue:
            p_y = self.predict_ytrue(x_min).flatten()
            self._setup_helper_metrics()

        self._clear_influence_function_handles()

        # compute unnormalized influence function
        values = np.zeros(x_min.shape[0])
        for metric, weight in self.weights.items():

            if metric == 'da_in':

                p_s = self.predict_group(x_min).flatten()
                logodds = np.log((1.0 - p_s) / p_s)
                cons = np.mean(logodds)
                vals = logodds - cons
                self._phi_da_in = lambda p: np.log((1.0 - p)/p) - cons

            elif metric == 'da_out':

                p_yhat_maj = self.predict_yhat(x_maj).flatten()
                mean_pyhat_min = np.mean(p_yhat)
                mean_pyhat_maj = np.mean(p_yhat_maj)

                cons = np.log((mean_pyhat_min * (1.0 - mean_pyhat_maj)) / ((1.0 - mean_pyhat_min) * mean_pyhat_maj))
                vals = cons * (p_yhat - mean_pyhat_min)
                self._phi_da_out = lambda p: cons * (p - mean_pyhat_min)

            elif metric == 'cal':

                abs_err = np.abs(p_yhat - p_y)
                cons = np.mean(abs_err)

                if self._compute_cal() >= 0.0:
                    vals = abs_err - cons
                    get_cal = lambda p_yhat, p_y: np.abs(p_yhat - p_y) - cons
                else:
                    vals = cons - abs_err
                    get_cal = lambda p_yhat, p_y: cons - np.abs(p_yhat - p_y)

                self._phi_cal = get_cal

            elif metric == 'fpr':

                a = np.multiply(p_yhat, 1.0 - p_y)
                b = 1.0 - p_y

                # setup parameters
                var_b = np.square(np.mean(b))

                if self._compute_fpr() >= 0.0:
                    con_a = np.mean(b) / var_b
                    con_b = -np.mean(a) / var_b
                else:
                    con_a = -np.mean(b) / var_b
                    con_b = np.mean(a) / var_b

                vals = con_a * a + con_b * b

                def get_fpr(p_yhat, p_y):
                    a = np.multiply(p_yhat, 1.0 - p_y)
                    b = 1.0 - p_y
                    v = con_a * a + con_b * b
                    return v

                self._phi_fpr = get_fpr

            elif metric == 'fnr':

                a = np.multiply(1.0 - p_yhat, p_y)
                b = p_y

                # setup parameters
                var_b = np.square(np.mean(b))

                if self._compute_fnr() >= 0.0:
                    con_a = np.mean(b) / var_b
                    con_b = -np.mean(a) / var_b
                else:
                    con_a = -np.mean(b) / var_b
                    con_b = np.mean(a) / var_b

                vals = con_a * a + con_b * b

                def get_fnr(p_yhat, p_y):
                    a = np.multiply(1.0 - p_yhat, p_y)
                    b = p_y
                    v = con_a * a + con_b * b
                    return v

                self._phi_fnr = get_fnr

            elif metric == 'fdr':

                a = np.multiply(p_yhat, 1.0 - p_y)
                b = p_yhat

                # setup parameters
                var_b = np.square(np.mean(b))

                if self._compute_fdr() >= 0.0:
                    con_a = np.mean(b) / var_b
                    con_b = -np.mean(a) / var_b
                else:
                    con_a = -np.mean(b) / var_b
                    con_b = np.mean(a) / var_b

                vals = con_a * a + con_b * b

                def get_fdr(p_yhat, p_y):
                    a = np.multiply(p_yhat, 1.0 - p_y)
                    b = p_yhat
                    v = con_a * a + con_b * b
                    return v

                self._phi_fdr = get_fdr

            assert np.isfinite(vals).all()
            values += weight * vals

        # update values
        assert self._check_values(values)
        self._fitted = True
        self._values = np.array(values, dtype = np.float).flatten()
        self._clear_helper_metrics()
        self._check_influence_function_handles()
        return values


    def apply(self, df):
        """
        return values of influence function for rows in data frame
        :param df:
        :param normalize:
        :return:
        """

        self._check_fit()
        self._check_influence_function_handles()

        # compute parameters used by multiple metrics
        x_min = df[self.data.variable_names].values
        p_yhat = self.predict_yhat(x_min).flatten()
        p_s = self.predict_group(x_min).flatten() if self.need_predict_group else float('nan')
        p_y = self.predict_ytrue(x_min).flatten() if self.need_predict_ytrue else float('nan')

        # compute value of influence function
        values = np.zeros(x_min.shape[0])
        for metric, weight in self.weights.items():

            if metric == 'da_in':
                vals = self._phi_da_in(p_s)

            elif metric == 'da_out':
                vals = self._phi_da_out(p_yhat)

            elif metric == 'cal':
                vals = self._phi_cal(p_yhat, p_y)

            elif metric == 'fnr':
                vals = self._phi_fnr(p_yhat, p_y)

            elif metric == 'fpr':
                vals = self._phi_fpr(p_yhat, p_y)

            elif metric == 'fdr':
                vals = self._phi_fdr(p_yhat, p_y)

            assert np.isfinite(vals).all()
            values += weight * vals

        return values


    #### Resampling ####
    def resample(self, df = None, n_samples = None, sampling_weights = None, step_size = 0.001, normalize = False, drop_safe = True):
        """
        :param df:
        :param n_samples:
        :param step_size:
        :param step_type:
        :param drop_safe: if True, the sample includes at least one sample from all points in the original
                          sample. That is, the empirical distribution of the resampled points obeys:
                          q[x] > 0 for all x such that p[x] > 0
        :return:
        """
        self._check_fit()
        if df is None:
            df = self.data.df_min
            uf = self.data.uf_min
        else:
            assert isinstance(df, pd.DataFrame)
            if drop_safe:
                uf = df.drop_duplicates()

        if sampling_weights is None:
            sampling_weights = self.sampling_weights(df = df, step_size = step_size, normalize = normalize)
        else:
            assert len(df) == len(sampling_weights)

        if n_samples is None:
            n_samples = len(sampling_weights)
        else:
            n_samples = int(n_samples)
            assert n_samples > 0

        sampled_df = df.sample(n_samples, weights = sampling_weights, replace = True, random_state = self.random_state)
        if drop_safe:
            missing_df = uf[~uf.index.isin(sampled_df.index)]
            if len(missing_df) > 0:
                sampled_df = sampled_df.append(missing_df)

        return sampled_df


    def sampling_weights(self, df = None, step_size = 0.001, normalize = False):

        self._check_fit()
        try:
            step_size = float(step_size)
        except TypeError:
            return TypeError('step_size must be a float or castable to float')

        if np.less_equal(step_size, 0.0):
            raise ValueError('step_size must be positive')

        if df is None:
            values = self.values
        else:
            assert isinstance(df, pd.DataFrame)
            values = self.apply(df)

        if normalize:
            values = values / np.std(values)

        # adjust step_size
        max_weight = np.max(values)
        if max_weight > 0.0:
            max_step_size = np.abs(1.0 / max_weight)
            if step_size > max_step_size:
                step_size = float(max_step_size) / (1.0 + 1e-4)

        sampling_weights = (1.0 - step_size * values)
        assert np.isfinite(sampling_weights).all() and np.greater(sampling_weights, 0.0).all()
        return sampling_weights


    #### Objective Value Computation ####

    def compute_objective(self, df_min = None, df_maj = None):
        """
        computes the disparity measure of interest
        :param df_min:
        :param df_maj:
        :return:
        """

        self._setup_helper_metrics(df_min, df_maj)
        objval = 0.0
        for metric, weight in self.weights.items():
            if metric == 'da_out':
                val = self._compute_da_out()
            elif metric == 'da_in':
                val = self._compute_da_in()
            elif metric == 'cal':
                val = self._compute_cal()
            elif metric == 'fpr':
                val = self._compute_fpr()
            elif metric == 'fdr':
                val = self._compute_fdr()
            elif metric == 'fnr':
                val = self._compute_fnr()
            objval += weight * val
        self._clear_helper_metrics()
        return float(objval)


    def _setup_helper_metrics(self, df_min = None, df_maj = None):

        # compute using repeated values
        if df_min is None:
            x_min = self.data.x_min
        else:
            assert isinstance(df_min, pd.DataFrame), 'df_min must be None or pd.DataFrame'
            x_min = df_min[self.data.variable_names].values

        if df_maj is None:
            x_maj = self.data.x_maj
        else:
            assert isinstance(df_maj, pd.DataFrame), 'df_maj must be None or pd.DataFrame'
            x_maj = df_maj[self.data.variable_names].values

        self._pyhat_min = self.predict_yhat(x_min).flatten()
        self._pyhat_maj = self.predict_yhat(x_maj).flatten()
        self._mean_pyhat_min = np.mean(self._pyhat_min)
        self._mean_pyhat_maj = np.mean(self._pyhat_maj)

        if self.need_predict_group:
            self._ps_min = self.predict_group(x_min).flatten()

        if self.need_predict_ytrue:
            self._pytrue_min = self.predict_ytrue(x_min).flatten()
            self._pytrue_maj = self.predict_ytrue_maj(x_maj).flatten()
            self._mean_pytrue_min = np.mean(self._pytrue_min)
            self._mean_pytrue_maj = np.mean(self._pytrue_maj)


    def _clear_helper_metrics(self):
        self._pyhat_min = None
        self._pyhat_maj = None
        self._ps_min = None
        self._pytrue_min = None
        self._pytrue_maj = None
        self._mean_pyhat_min = None
        self._mean_pyhat_maj = None
        self._mean_pytrue_min = None
        self._mean_pytrue_maj = None


    def _compute_da_in(self):
        ps = self._ps_min
        a = np.mean(np.log(1.0 - ps) - np.log(ps))
        b = np.log(self.data.n_maj) - np.log(self.data.n_min)
        return a + b


    def _compute_da_out(self):
        a = self._mean_pyhat_min
        b = self._mean_pyhat_maj
        da_out = np.multiply(a, np.log(a) - np.log(b)) + np.multiply(1.0 - a, np.log(1.0 - a) - np.log(1.0 - b))
        return da_out


    def _compute_cal(self):
        cal_min = np.mean(np.abs(self._pyhat_min - self._pytrue_min))
        cal_maj = np.mean(np.abs(self._pyhat_maj - self._pytrue_maj))
        return cal_min - cal_maj


    def _compute_fpr(self):
        fpr_min = np.mean(np.multiply(self._pyhat_min, 1.0 - self._pytrue_min)) / (1.0 - self._mean_pytrue_min)
        fpr_maj = np.mean(np.multiply(self._pyhat_maj, 1.0 - self._pytrue_maj)) / (1.0 - self._mean_pytrue_maj)
        return fpr_min - fpr_maj


    def _compute_fnr(self):
        fnr_min = np.mean(np.multiply(1.0 - self._pyhat_min, self._pytrue_min)) / self._mean_pytrue_min
        fnr_maj = np.mean(np.multiply(1.0 - self._pyhat_maj, self._pytrue_maj)) / self._mean_pytrue_maj
        return fnr_min - fnr_maj


    def _compute_fdr(self):
        fdr_min = np.mean(np.multiply(self._pyhat_min, 1.0 - self._pytrue_min)) / self._mean_pytrue_min
        fdr_maj = np.mean(np.multiply(self._pyhat_maj, 1.0 - self._pytrue_maj)) / self._mean_pytrue_maj
        return fdr_min - fdr_maj


    def __repr__(self):

        rule_width = 20
        rule_bold = '=' * rule_width
        rule_thin = '-' * int(0.8 * rule_width)

        s = [
            rule_bold,
            'outcome: %s' % self._outcome_name,
            'n_variables: %d' % len(self.variable_names),
            #
            rule_thin,
            'group variable: %s (n = %d subgroups)' % (self._group_name, self._n_groups),
            'minority group: %r (n = %d)' % (self._minority_group, self._n_min),
            'majority group: %r (n = %d)' % (self._majority_group, self._n_maj),
            #
            rule_thin,
            'prediction_model: %r' % (self.predict_yhat is not None),
            'group_model: %r' % (self.predict_group is not None),
            'true_model_min: %r' % (self.predict_ytrue is not None),
            'true_model_maj: %r' % (self.predict_ytrue_maj is not None),
            ]

        if self.fitted:
            s += [rule_bold]
            for k, v in self._weights.items():
                s.append('weight for %s: %1.1f' % (k, v))


        s += [rule_bold]
        return '\n'.join(s)


class CounterfactualDistribution:


    def __init__(self, **kwargs):

        self.gf = InfluenceFunction(**kwargs)
        self.data = self.gf.data

        self.outcome_name = self.data.outcome_name
        self.group_name = self.data.group_name
        self.variable_names = self.data.variable_names
        self.minority_group = self.data.minority_group
        self.majority_group = self.data.majority_group

        self.random_state = kwargs.get('random_state')
        self.drop_safe = kwargs.get('drop_safe', True)

        self._step_size = None
        pmf = self.data.uf_min.copy(deep = True)
        pmf.insert(loc = len(self.variable_names), column = 'n', value = 1)
        self.pmf = pmf


    @property
    def metric_name(self):
        metric_names = list(self.gf.weights)
        if len(metric_names) == 1:
            return metric_names[0]
        else:
            return 'mixed_' + '_'.join(metric_names)


    @property
    def metric_label(self):
        metric_names = [s.upper() for s in list(self.gf.weights)]
        if len(metric_names) == 1:
            return metric_names[0]
        else:
            return 'mixed_' + '_'.join(metric_names)


    @property
    def drop_safe(self):
        return bool(self._drop_safe)


    @drop_safe.setter
    def drop_safe(self, flag):
        assert isinstance(flag, bool)
        self._drop_safe = flag


    @property
    def random_state(self):
        return self._random_state


    @random_state.setter
    def random_state(self, seed):
        if isinstance(seed, int):
            self._random_state = np.random.RandomState(seed)
        elif isinstance(seed, np.random.RandomState):
            self._random_state = seed
        elif seed is None:
            self._random_state = np.random.RandomState()


    @property
    def fitted(self):
        return bool(self._fitted)


    @property
    def pmf(self):
        return self._pmf.copy(deep = True)


    @pmf.setter
    def pmf(self, df):
        assert isinstance(df, pd.DataFrame)
        assert len(df) >= 1
        pmf_names = df.columns.tolist()
        assert 'n' in pmf_names and len(pmf_names) >= 2
        df = df.copy(deep = True)
        df = df.sort_values(by = pmf_names, axis = 0).reset_index(drop = True)
        self._pmf = df


    def resample(self, df = None, n_samples = None, drop_safe = True):
        """
        :param df:
        :param n_samples:
        :param step_size:
        :param drop_safe: if True, then the returned sample has the same support as the original sample, i.e.,
                          the empirical pdf of the resampled points obeys: q[x] > 0 for all x such that p[x] > 0
        :return:
        """
        assert self.fitted, 'need to fit first'

        pmf = self.pmf
        if df is None:
            n_samples = int(n_samples)
            assert n_samples > 0
        else:
            n_samples = len(df)
            uf = df[self.variable_names].drop_duplicates().reset_index(drop = True)
            if drop_safe:
                pmf = pmf.merge(uf, on = self.variable_names, validate = 'one_to_one', how = 'outer')
            else:
                pmf = pmf.merge(uf, on = self.variable_names, validate = 'one_to_one', how = 'left')

        sampled_df = pmf.sample(n_samples, weights = 'n', axis = 0, replace = True, random_state = self.random_state)
        sampled_df = sampled_df.drop(columns = ['n']).reset_index(drop = True)

        if drop_safe:
            missing_df = uf.merge(sampled_df, on = self.variable_names, how = 'outer', indicator = True).query('_merge == "left_only"').drop('_merge', 1)
            if len(missing_df) > 0:
                sampled_df = sampled_df.append(missing_df)

        return sampled_df


    def fit(self, df_min_test, df_maj_test, step_size = 0.01, max_iterations = 5e2, stopping_tolerance = 1e-2, n_boot = 5, drop_safe = True, save_progress = True):
        """
        recovers counterfactual distribution using descent procedure

        :param df_min_test:
        :param df_maj_test:
        :param step_size:
        :param max_iterations:
        :param stopping_tolerance:
        :param n_boot:
        :param save_progress:

        :return:
        """

        ### check inputs
        assert isinstance(df_min_test, pd.DataFrame)
        assert isinstance(df_maj_test, pd.DataFrame)
        assert isinstance(max_iterations, (float, int)) and max_iterations >= 1
        assert isinstance(step_size, float) and step_size >= 0.0
        assert isinstance(stopping_tolerance, float) and stopping_tolerance >= 0.0
        assert isinstance(n_boot, (float, int)) and n_boot >= 1
        assert isinstance(drop_safe, bool)
        assert isinstance(save_progress, bool)

        max_iterations = int(max_iterations)
        n_boot = int(n_boot)
        df_min_test = df_min_test.copy(deep = True)
        gf = self.gf

        n = 0
        current_gap = gf.compute_objective(df_min = df_min_test, df_maj = df_maj_test)
        w_aggregate = np.ones(len(df_min_test))

        progress = []
        progress.append({
            'iteration': 0,
            'type': 'descent',
            'gap': current_gap,
            })

        w_best = w_aggregate
        best_gap = current_gap
        best_iteration = 0

        while True:

            n += 1

            # get direction for next
            gf.fit()

            # get samples from next distribution
            next_df = gf.resample(step_size = step_size, drop_safe = drop_safe)

            # this does not reflect a meaningful value since it reflects the disparity using the 'resampled population'
            # objval_gap_train = gf.compute_objective(df_min = next_df)
            # results.append({
            #     'Iteration': n,
            #     'Type': 'audit',
            #     'Disparity Gap': objval_gap_train,
            #     })

            # update aggregate perturbation weights
            w_aggregate *= gf.sampling_weights(df = df_min_test, step_size = step_size)

            gaps = []
            for i in range(n_boot):
                check_df = gf.resample(df = df_min_test, sampling_weights = w_aggregate, step_size = step_size, drop_safe = drop_safe)
                boot_gap = gf.compute_objective(df_min = check_df, df_maj = df_maj_test)
                gaps.append(boot_gap)
                progress.append({
                    'iteration': n,
                    'type': 'descent',
                    'gap': boot_gap,
                    })

            new_gap = np.mean(gaps)
            if np.abs(new_gap - 0.0) < np.abs(best_gap - 0.0):
                best_gap = new_gap
                best_iteration = n
                w_best = w_aggregate

            if n == max_iterations:
                print('stopped after %d iterations (reason: hit max iterations of %d)' % (n, max_iterations))
                break

            if np.isclose(new_gap, 0.0, rtol = stopping_tolerance):
                print('stopped after %d iterations (reason: gap is close to 0.0)' % n)
                break

            if n > 50 and np.isclose(new_gap, current_gap, rtol = stopping_tolerance):
                print('stopped after %d iterations (reason: current gap has stabilized)' % n)
                break

            current_gap = new_gap
            gf.update_df_min(df_min = next_df)

        progress_df = pd.DataFrame(progress)
        progress_df['is_best_iteration'] = progress_df['iteration'] == best_iteration

        ctf_pmf = gf.resample(df = df_min_test, sampling_weights = w_best, step_size = step_size, drop_safe = True)
        self.pmf = compress_df(ctf_pmf)
        self._fitted = True

        return progress_df


    def build_preprocessor(self, df):
        """
        build randomized preprocessor using the fitted counterfactual distribution
        :param df:
        :return:
        """
        assert self.fitted, 'need to fit first'
        assert isinstance(df, pd.DataFrame)
        df_min, _ = split_by_group(df, minority_group = self.minority_group)
        samples_raw = df_min[self.variable_names]
        samples_ctf = self.resample(df = samples_raw)
        U, P, Q = get_preprocessing_space_and_distributions(samples_raw = samples_raw, samples_ctf = samples_ctf)
        T = compute_preprocessing_distribution(U, P, Q)
        pp = RandomizedPreprocessor(U, processor_pdf = T)
        return pp


    def marginal_distributions(self, df, n_samples = None, n_bootstap = 10, drop_safe = False, pretty_entries = True):
        """
        outputs vector containing marginal probabilties for plotting
        :param df:
        :param n_samples:
        :param n_bootstap:
        :param drop_safe:
        :param pretty_entries:
        :return:
        """
        assert self.fitted, 'need to fit first'
        assert isinstance(df, pd.DataFrame)
        df_min, df_maj = split_by_group(df, minority_group = self.minority_group)
        if pretty_entries:
            fmt = lambda s: np.round(100 * s).astype('int')
        else:
            fmt = lambda s: s.values()

        if n_samples is None:
            n_samples = len(df_min)
        else:
            n_samples = int(n_samples)
            assert n_samples > 0

        dist_ctf = [self.resample(n_samples = n_samples, drop_safe = drop_safe).mean() for k in range(n_bootstap)]
        dist_ctf = fmt(pd.concat(dist_ctf, axis = 1).mean(axis = 1))
        dist_min = fmt(df_min.mean())
        dist_maj = fmt(df_maj.mean())
        table_df = pd.concat([dist_maj, dist_min, dist_ctf], axis = 1)
        table_df.columns = ['%s' % self.majority_group, '%s' % self.minority_group, '%s (%s)' % (self.metric_label, self.minority_group)]
        return table_df