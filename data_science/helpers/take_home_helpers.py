import re
import pandas as pd
from functools import wraps
from inspect import getmembers

from abc import ABCMeta, abstractmethod

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import Imputer, StandardScaler
from sklearn.pipeline import Pipeline, FeatureUnion


def pandas_series(func):
    @wraps(func)
    def _wrapped(self, X, index='index', name='', **kwargs):
        yhat = func(self, X, **kwargs)
        index = X[index] if index in X else X.index
        name = name if name else self.name
        return pd.Series(yhat, index=index, name=name)
    return _wrapped


def pandas_df(func):
    @wraps(func)
    def _wrapped(self, X, index='index', columns=[], **kwargs):
        _df = func(self, X, **kwargs)
        index = X[index] if index in X else X.index
        columns = columns if columns else X.columns
        return pd.DataFrame(_df, index=index, columns=columns)
    return _wrapped


class BaseTransformer(BaseEstimator, TransformerMixin):
    __meta__ = ABCMeta

    @abstractmethod
    def fit(self, X, y=None, **fiParams):
        return self


def _base_methods():
    methods = set([])
    for _type in [BaseTransformer, Pipeline, FeatureUnion]:
        methods = methods.union(set([i[0] for i in getmembers(_type)]))
    return methods


base_methods = _base_methods()


class PandasMixin(object):
    """
    Mixin for improving scikit-learn <> pandas interaction
    """
    @property
    def fields(self):
        """
        Incoming column names
        """
        return getattr(self, '_fields', None)

    @fields.setter
    def fields(self, value):
        self._fields = value

    @property
    def features(self):
        """
        Outgoing column names
        """
        return getattr(self, '_features', None)

    @features.setter
    def features(self, value):
        self._features = value

    @property
    def transformedDtypes(self):
        return getattr(self, '_dtypes', {})

    @transformedDtypes.setter
    def transformedDtypes(self, value):
        """
        Transformed data types
        """
        self._dtypes = value

    def get_fields(self):
        return self.fields

    def get_feature_names(self):
        return self.features


def pandas_transformer(cls):
    # Need a classtools wraps
    class Wrapper(PandasMixin):
        def __init__(self, *args, **kwargs):
            self.transformerWrapped = cls(*args, **kwargs)

        def __getattr__(self, name):
            base = getattr(self.transformerWrapped, name)
            if callable(base):
                if name == 'fit':
                    def wrapped(*args, **kwargs):
                        self.fields = args[0].columns.tolist()
                        self.features = None
                        result = base(*args, **kwargs)
                        return result
                    return wrapped

                elif name == 'transform':
                    def wrapped(*args, **kwargs):
                        result = base(*args, **kwargs)
                        if not self.features:
                            features = result.columns.tolist() if hasattr(result, 'columns') else self.fields
                            self.features = features
                        result = result if isinstance(result, pd.DataFrame) else pd.DataFrame(result, columns=self.features)
                        result = result.reindex(columns=self.features)
                        self.transformedDtypes = result.dtypes.to_dict()
                        return result
                    return wrapped

                # Patch for Feature Union - TODO investigate
                elif name == 'fit_transform':
                    def wrapped(*args, **kwargs):
                        self.fit(*args, **kwargs)
                        result = self.transform(*args, **kwargs)
                        return result
                    return wrapped

                # elif name in base_methods:
                else:
                    def wrapped(*args, **kwargs):
                        result = base(*args, **kwargs)
                        return result
                    return wrapped
                # else:
                    # return base

            else:
                return base
    return Wrapper


@pandas_transformer
class Imputer(Imputer):
    pass


@pandas_transformer
class StandardScaler(StandardScaler):
    pass


@pandas_transformer
class Pipeline(Pipeline):
    pass


@pandas_transformer
class Selector(BaseTransformer):
    def __init__(self, selectMethod=None, selectValue=None, reverse=False):
        self.selectMethod = selectMethod
        self.selectValue = selectValue
        self.reverse = reverse

    def data_type(self, df, inclusionExclusionKwargs):
        inclusions = df.select_dtypes(**inclusionExclusionKwargs).columns.tolist()
        return inclusions

    def regex(self, X, patterns):
        inclusions = [i for i in X if any([re.match(j, i) for j in patterns])]
        return inclusions

    def fit(self, X, y=None):
        if self.selectMethod:
            inclusions = getattr(self, self.selectMethod)(X, self.selectValue)
        else:
            inclusions = self.selectValue

        if self.reverse:
            exclusions = set(inclusions)
            inclusions = [i for i in X if i not in exclusions]
        self.inclusions = inclusions

        return self

    def transform(self, X, y=None):
        return X.reindex(columns=self.inclusions)


@pandas_transformer
class CategoricalTransformer(BaseTransformer):
    def __init__(self, dropFirst=False):
        self.dropFirst = dropFirst

    def fit(self, X, y=None):
        self.categories = {field: [] for field in self.fields}
        for field in self.categories:
            self.categories[field] = X.loc[X[field].notnull()][field].unique().tolist()
        return self

    def transform(self, X, y=None):
        for field in self.fields.intersection(X):
            X[field] = pd.Series(X[field], dtype='category').cat.set_categories(self.categories[field])
        transformed = pd.get_dummies(X, columns=self.fields, drop_first=self.dropFirst)

        return transformed


@pandas_transformer
class Cutter(BaseTransformer):
    def __init__(self, payload):
        self.payload = payload

    def transform(self, X, y=None):
        fields = set(self.payload.keys()).intersection(X)
        for field in fields:
            if isinstance(self.payload[field], dict):
                buckets = self.payload[field]['buckets']
                labels = self.payload[field]['labels']
            else:
                buckets = self.payload[field]
                labels = [i for i, _ in enumerate(self.payload[field][:-1])]

            X[field] = pd.cut(X[field], bins=buckets, labels=labels)
        return X


@pandas_transformer
class Existance(BaseTransformer):
    def __init__(self, similarityCheck=1):
        self.similarityCheck = similarityCheck

    def _check_sim(self, X, series):
        for i in X:
            if (X[i].notnull() == series.notnull()).mean() >= self.similarityCheck:
                return False
        return True

    def fit(self, X, y=None):
        self.inclusions = set([])
        # FIX THIS!!!
        for i in X:
            uniques = X[i].unique()
            checked = self.extensions + self.replacements
            if any(pd.isnull(uniques)) and self._check_sim(X[checked], X[i]):
                if len(uniques) > 2:
                    self.inclusions.add(i)
        return self

    def transform(self, X, y=None):
        for column in self.inclusions.intersection(X):
            X["_".join(['has', column])] = X[column].notnull()
        return X


@pandas_transformer
class AttributeTransformer(BaseTransformer):
    def __init__(self, attirbute=None, args=(), kwargs={}):
        self.attribute = attribute
        self.args = args
        self.kwargs = kwargs

    def transform(self, X, **fitParams):
        transformed = getattr(X, self.attribute), *self.args, **self.kwargs)
        return transformed

@pandas_transformer
class CallbackTransformer(BaseTransformer):
    def __init__(self, callback=None, args=(), kwargs={}):
        self.callback
        self.args = args
        self.kwargs = kwargs

    def transform(self, X, **fitParams):
        transformed = self.callback(X, *self.args, **self.kwargs)
        return transformed
