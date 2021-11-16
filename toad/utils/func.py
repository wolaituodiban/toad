import re
import json
import string
import numpy as np
import pandas as pd
from typing import List, Tuple, Callable, Union
from functools import wraps

from multiprocessing import Pool, current_process, cpu_count


CONTINUOUS_NUM = 10
FEATURE_THRESHOLD = 1e-7
NAN_REPLACEMENT = -2e10


NAN_LIST = [
    'nan',
    'Nan',
    'null',
    'None',
    None,
    np.nan,
]


class Parallel:
    def __init__(self):
        self.ismain = False
        self.results = []
        self.pro = current_process()

        if self.pro.name == 'MainProcess':
            self.ismain = True
            self.pool = Pool(cpu_count())


    def apply(self, func, args = (), kwargs = {}):
        if not self.ismain:
            r = func(*args, **kwargs)
        else:
            r = self.pool.apply_async(func, args = args, kwds = kwargs)

        self.results.append(r)

    def join(self):
        if not self.ismain:
            return self.results

        self.pool.close()
        self.pool.join()

        return [r.get() for r in self.results]



def np_count(arr, value, default = None):
    c = (arr == value).sum()

    if default is not None and c == 0:
        return default

    return c


def _replace_nan(arr):
    a = np.copy(arr)
    a[a == NAN_REPLACEMENT] = np.nan
    return a


def has_nan(arr):
    return np.any(pd.isna(arr))


def np_unique(arr, **kwargs):
    arr = to_ndarray(arr)

    if not has_nan(arr):
        return np.unique(arr, **kwargs)

    arr[np.isnan(arr)] = NAN_REPLACEMENT

    res = np.unique(arr, **kwargs)

    if isinstance(res, tuple):
        u = _replace_nan(res[0])
        return (u, *res[1:])

    return _replace_nan(res)


def to_ndarray(s, dtype = None):
    """
    """
    if isinstance(s, np.ndarray):
        arr = np.copy(s)
    elif isinstance(s, pd.core.base.PandasObject):
        arr = np.copy(s.values)
    elif s is None:
        arr = np.array(np.nan)
    else:
        arr = np.array(s)


    if dtype is not None:
        arr = arr.astype(dtype)
    # covert object type to str
    elif arr.dtype.type is np.object_:
        arr = arr.astype(np.str)

    return arr


def fillna(feature, by = -1):
    # copy array
    copied = np.copy(feature)

    mask = pd.isna(copied)

    copied[mask] = by

    return copied

def bin_by_splits(feature, splits):
    """Bin feature by split points
    """
    feature = fillna(feature)

    if not isinstance(splits, (list, np.ndarray)):
        splits = [splits]
    
    return np.digitize(feature, splits)


def feature_splits(feature, target):
    """find posibility spilt points
    """
    feature = to_ndarray(feature)
    target = to_ndarray(target)

    matrix = np.vstack([feature, target])
    matrix = matrix[:, matrix[0,:].argsort()]

    splits_values = []
    for i in range(1, len(matrix[0])):
        # if feature value is almost same, then skip
        if matrix[0,i] <= matrix[0, i-1] + FEATURE_THRESHOLD:
            continue

        # if target value is not same, calculate split
        if matrix[1, i] != matrix[1, i-1]:
            v = (matrix[0, i] + matrix[0, i-1]) / 2.0
            splits_values.append(v)

    return np.unique(splits_values)


def iter_df(dataframe, feature, target, splits):
    """iterate dataframe by split points

    Returns:
        iterator (df, splitter)
    """
    splits.sort()
    df = pd.DataFrame()
    df['source'] = dataframe[feature]
    df[target] = dataframe[target]
    df[feature] = 0

    for v in splits:
        df.loc[df['source'] < v, feature] = 1
        yield df, v


def inter_feature(feature, splits):
    splits.sort()
    bin = np.zeros(len(feature))

    for v in splits:
        bin[feature < v] = 1
        yield bin


def is_continuous(series):
    series = to_ndarray(series)
    if not np.issubdtype(series.dtype, np.number):
        return False

    n = len(np.unique(series))
    return n > CONTINUOUS_NUM or n / series.size > 0.5
    # return n / series.size > 0.5


def split_target(frame, target):
    """
    """
    if isinstance(target, str):
        cols = frame.columns.copy().drop(target)
        f = frame[cols]
        t = frame[target]
    else:
        f = frame.copy(deep = False)
        t = target

    return f, t


def unpack_tuple(x):
    if len(x) == 1:
        return x[0]
    else:
        return x

ALPHABET = string.ascii_uppercase + string.digits
def generate_str(size = 6, chars = ALPHABET):
    return ''.join(np.random.choice(list(chars), size = size))



def save_json(contents, file, indent = 4):
    """save json file

    Args:
        contents (dict): contents to save
        file (str|IOBase): file to save
    """
    if isinstance(file, str):
        file = open(file, 'w')

    with file as f:
        json.dump(contents, f, ensure_ascii = False, indent = indent)


def read_json(file):
    """read json file
    """
    if isinstance(file, str):
        file = open(file)

    with file as f:
        res = json.load(f)

    return res




def clip(series, value = None, std = None, quantile = None):
    """clip series

    Args:
        series (array-like): series need to be clipped
        value (number | tuple): min/max value of clipping
        std (number | tuple): min/max std of clipping
        quantile (number | tuple): min/max quantile of clipping
    """
    series = to_ndarray(series)

    if value is not None:
        min, max = _get_clip_value(value)

    elif std is not None:
        min, max = _get_clip_value(std)
        s = np.std(series, ddof = 1)
        mean = np.mean(series)
        min = None if min is None else mean - s * min
        max = None if max is None else mean + s * max

    elif quantile is not None:
        if isinstance(quantile, tuple):
            min, max = quantile
        else:
            min = quantile
            max = 1 - quantile

        min = None if min is None else np.quantile(series, min)
        max = None if max is None else np.quantile(series, max)

    else:
        return series

    return np.clip(series, min, max)


def _get_clip_value(params):
    if isinstance(params, tuple):
        return params
    else:
        return params, params


def diff_time(base, target, format = None, time = 'day'):
    # if base is not a datetime list
    if not np.issubdtype(base.dtype, np.datetime64):
        base = pd.to_datetime(base, format = format, cache = True)

    target = pd.to_datetime(target, format = format, cache = True)

    delta = target - base

    if time == 'day':
        return delta.dt.days

    return delta


def diff_time_frame(base, frame, format = None):
    res = pd.DataFrame()

    base = pd.to_datetime(base, format = format, cache = True)

    for col in frame:
        try:
            res[col] = diff_time(base, frame[col], format = format)
        except Exception as e:
            continue

    return res


def flatten_columns(columns, sep = '_'):
    """flatten multiple columns to 1-dim columns joined with '_'
    """
    l = []
    for col in columns:
        if not isinstance(col, str):
            col = sep.join(col)
        
        l.append(col)
    
    return l


def bin_to_number(reg = None):
    """
    Returns:
        function: func(string) -> number
    """
    if reg is None:
        reg = r'\d+'

    def func(x):
        if pd.isnull(x):
            return np.nan

        res = re.findall(reg, x)
        l = len(res)
        res = map(float, res)
        if l == 0:
            return np.nan
        else:
            return sum(res) / l

    return func


def generate_target(size, rate = 0.5, weight = None, reverse = False):
    """generate target for reject inference

    Args:
        size (int): size of target
        rate (float): rate of '1' in target
        weight (array-like): weight of '1' to generate target
        reverse (bool): if need reverse weight

    Returns:
        array
    """
    if weight is not None:
        weight = np.asarray(weight)

        if reverse is True:
            weight = (np.max(weight) + np.min(weight)) - weight

        weight = weight / weight.sum()

    res = np.zeros(size)

    choice_num = int(size * rate)
    ix = np.random.choice(size, choice_num, replace = False, p = weight)
    res[ix] = 1

    return res


def get_dummies(dataframe, exclude = None, binary_drop = False, **kwargs):
    """get dummies
    """
    columns = dataframe.select_dtypes(exclude = 'number').columns

    if len(columns) == 0:
        return dataframe

    if exclude is not None:
        columns = columns.difference(exclude)

    if binary_drop:
        mask = dataframe[columns].nunique(dropna = False) == 2

        if mask.sum() != 0:
            dataframe = pd.get_dummies(
                dataframe,
                columns = columns[mask],
                drop_first = True,
                **kwargs,
            )
            columns = columns[~mask]

    data = pd.get_dummies(dataframe, columns = columns, **kwargs)
    return data


def _metric(func, label_first):
    """
    a wrapper for func, which will make the func compatible with dict and list
    Examples:
        >>> from sklearn.metrics import roc_auc_score
        ... y_hat = ...
        ... y = ...
        ... auc = metric(roc_auc_score, label_first=True)
        ... auc(y, y_hat)
        0.5

        >>> y_hat = {'a': ..., 'b': ...}
        ... y = {'a': ..., 'b': ...}
        ... auc = metric(roc_auc_score, label_first=True)
        ... auc(y, y_hat)
        {'a': 0.5, 'b': 0.5}

        >>> y_hat = [..., ...]
        ... y = [..., ...]
        ... auc = metric(roc_auc_score, label_first=True)
        ... auc(y, y_hat)
        [0.5, 0.5]

    Args:
        func: metric func, list of funcs or dict of funcs
        label_first: whether the metric func receive label as the first argument

    Returns:
        new function
    """

    def label_last_func(pred, label):
        """transform the func to label last form"""
        if label_first:
            return func(label, pred)
        else:
            return func(pred, label)

    def new_func(pred, label):
        if isinstance(pred, np.ndarray):
            return label_last_func(pred=pred, label=label)
        elif isinstance(pred, list):
            return [new_func(pred=_p, label=_l) for _p, _l in zip(pred, label)]
        elif isinstance(pred, dict):
            return {key: new_func(pred=pred[key], label=label[key]) for key in set(pred).intersection(label)}
        raise TypeError('only accept "numpy", "list", "dict"')

    return new_func


def metric(x: Union[Callable, List[Tuple[str, Callable, bool]]], label_first=None):
    """
    given a json format input, several json path and metrics func
    return metric value for certern json path and metric
    Examples:
        >>> from sklearn.metrics import roc_auc_score
        ... y_hat = ...
        ... y = ...
        ... auc = metric(roc_auc_score, label_first=True)
        ... auc(y, y_hat)
        0.5

        >>> y_hat = {'a': ..., 'b': ...}
        ... y = {'a': ..., 'b': ...}
        ... auc = metric(roc_auc_score, label_first=True)
        ... auc(y, y_hat)
        {'a': 0.5, 'b': 0.5}

        >>> y_hat = [..., ...]
        ... y = [..., ...]
        ... auc = metric(roc_auc_score, label_first=True)
        ... auc(y, y_hat)
        [0.5, 0.5]

        >>> from sklearn.metrics import roc_auc_score, mean_squared_error
        ... y_hat = {'a': {'c': ..., 'd': ...}, 'b': [..., ...]}
        ... y = {'a': {'c': ..., 'd': ...}, 'b': [..., ...]}
        ... auc = json_metric(
        ...     [
        ...         ('$.a', roc_auc_score, True),
        ...         ('$.b', mean_squared_error, True)
        ...     ]
        ... )
        ... auc(y, y_hat)
        [{'c': 0.5', 'd': 0.6}, [0.23, 0.45]]
    Args:
        x: Callable or list of tuple(json path (str), function (callable), label_first (bool))
        label_fist: if is not None, func should be a single function

    Returns:
        new function
    """
    if label_first is not None:
        return _metric(x, label_first=label_first)

    from jsonpromax import JsonPathTree

    tree_and_funcs = []
    for path, func, label_first in x:
        tree = JsonPathTree([(path, )])
        new_func = _metric(func, label_first=label_first)
        tree_and_funcs.append((tree, new_func))

    def func_wrapper(pred, label):
        outputs = []
        for t, f in tree_and_funcs:
            _pred = t(pred)
            _label = t(label)
            outputs.append(f(pred=_pred, label=_label))
        return outputs

    return func_wrapper
