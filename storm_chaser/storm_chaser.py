import numpy as np
import pandas as pd

# import tensorflow as tf

# from tfscripts import layers as tfs


def StormChaser():
    """Analyzer for systematic uncertainties"""

    def __init__(
        self,
        params: list[str],
        params_sys: list[str] = [],
        param_bounds: dict[str, tuple[float, float]] = {},
        param_bin_width_range: dict[str, tuple[float, float]] = {},
        min_events_per_bin: int = 100,
        trafo: dict[str, str] = {},
        param_sys_type: str | dict[str, str] = "uniform",
    ):
        """Initialize the StormChaser

        Parameters
        ----------
        params : list[str]
            The names of the observable parameters.
        params_sys : list[str], optional
            The names of the systematic parameters, by default [].
            These parameters are treated as nuisance parameters.
        param_bounds : dict[str, tuple[float, float]], optional
            A dictionary of parameter bounds, by default {}.
            This dictionary must contain all parameters in `params` and
            `params_sys`. Note that the bin edges must be supplied in the
            non-transformed space.
        param_bin_width_range : dict[str, tuple[float, float]], optional
            The range of bin widths to allow for each parameter. This must
            be a dictionary of the form {param: (min_width, max_width)}, where
            `param` is the name of the parameter and `min_width` and
            `max_width` are the minimum and maximum allowed bin widths,
            respectively. If a parameter is not specified, the default range
            is specified as 1% to 10% of the total value range of the
            parameter. Note that the bin widths must be provided in the
            non-transformed space.
        min_target_per_bin : int, optional
            The target minimum number of events per bin when creating training
            samples for the model. Note that this is not the actual minimum
            number of events per bin,
        trafo : dict[str, str], optional
            An optional data transformation to apply to the data before
            training the model. This must be a dictionary of the form
            {param: trafo}, where `param` is the name of the parameter to
            transform and `trafo` is the name of the transformation to
            perform. Currently, only the `log`, `log+1`, and `cos`
            transformations are supported.
        param_sys_type : str | dict[str, str], optional
            The type of the systematic parameters, by default 'uniform'.
            If provided as a string, all systematic parameters are assumed
            to have the same type. If provided as a dictionary, the type of
            each systematic parameter must be specified individually.

        Raises
        ------
        NotImplementedError
            _description_
        """
        self.params = sorted(params)
        self.params_sys = sorted(params_sys)
        self._params_all = sorted(self.params + self.params_sys)
        self.param_bounds = param_bounds
        self.min_events_per_bin = min_events_per_bin
        self.trafo = trafo
        self.param_sys_type = param_sys_type
        self._model_is_built = False

        if isinstance(self.param_sys_type, str):
            self.param_sys_type = {
                param: param_sys_type for param in self.params_sys
            }

        for sys_type in self.param_sys_type.values():
            if sys_type != "uniform":
                raise NotImplementedError(
                    f"Systematic parameter type {sys_type} not implemented"
                )

        if sorted(self.param_bounds.keys()) != self._params_all:
            raise ValueError(
                "param_bounds must contain all parameters in params "
                "and params_sys."
            )

        # get bin width ranges for each observable parameter
        self.param_bin_width_range = param_bin_width_range.copy()
        for param, bounds in self._param_bounds_trafo.items():
            if param not in self.param_bin_width_range:
                self.param_bin_width_range[param] = (
                    0.01 * (bounds[1] - bounds[0]),
                    0.1 * (bounds[1] - bounds[0]),
                )

        # transform parameter bounds
        self._param_bounds_trafo = self.trafo(self.param_bounds)
        self._param_bin_width_range_trafo = self.trafo(
            self.param_bin_width_range
        )

        # compute width for each systematic parameter
        self._width = {}
        for param in self.params_sys:
            lower, upper = self.param_bounds[param]
            self._width[param] = upper - lower

    def trafo(self, df: dict | pd.DataFrame, invert: bool = False):
        """Apply the data transformation to the data

        Parameters
        ----------
        df : dict | pd.DataFrame
            The data to transform.
        invert : bool, optional
            Whether to invert the transformation, by default False.

        Returns
        -------
        pd.DataFrame
            The transformed data.
        """
        df_out = df[self._params_all].copy()
        for param, trafo in self.trafo.items():
            if trafo == "log+1":
                if invert:
                    df_out[param] = np.exp(df_out[param]) - 1.0
                else:
                    df_out[param] = np.log(df_out[param] + 1.0)
            elif trafo == "log":
                if invert:
                    df_out[param] = np.exp(df_out[param])
                else:
                    df_out[param] = np.log(df_out[param])
            elif trafo == "cos":
                if invert:
                    df_out[param] = np.arccos(df_out[param])
                else:
                    df_out[param] = np.cos(df_out[param])
            else:
                raise NotImplementedError(
                    f"Transformation {trafo} not implemented"
                )
        return df_out

    def build(
        self,
        df_sys: pd.Datafram,
        df_base: pd.Dataframe = None,
        weights_sys: str = None,
        weights_base: str = None,
        seed: int = 42,
    ):
        """Build model

        The model is utilized to predict impact of systematic
        uncertainties on the observable parameters.

        Parameters
        ----------
        df_sys : pd.Dataframe
            The systematic data sample.
        df_base : pd.Dataframe, optional
            The base data sample, by default None. If None, the total
            `df_sys` sample is used as the base sample. Note that this
            expects the systemaatic uncertainties to have symmetric
            priors around the nominal value.
        weights_sys : str, optional
            The name of the column containing the systematic weights,
            by default None. If None, all events are assumed to have
            the same weight.
        weights_base : str, optional
            The name of the column containing the base weights,
            by default None. If None, all events are assumed to have
            the same weight.
        seed : int, optional
            The random seed to use for sampling training samples.

        Raises
        ------
        NotImplementedError
            _description_
        NotImplementedError
            _description_
        """
        # apply data transformation
        df_sys = self.trafo(df_sys)

        if df_base is None:
            df_base = df_sys
        else:
            df_base = self.trafo(df_base)

            # Note: code needs to be added to ensure that
            #       df_base and df_sys have the same distribution
            #       for the base parameters.
            #       Anything else missing?
            raise NotImplementedError(
                "Fitting to a different base sample not implemented"
            )

        # # check possible binning sizes
        # fraction = self.min_events_per_bin / len(df_sys)
        # bin_widths = {}

        # rng = np.random.default_rng(seed)

        self._model_is_built = True

        raise NotImplementedError

    def sample_df(
        self,
        df: pd.DataFrame,
        params: dict[str, float],
        rng: np.random.Generator = None,
    ):
        """Sample bin range for each parameter

        Parameters
        ----------
        df : pd.DataFrame
            The data sample to sample from.
        params : dict[str, float]
            The parameter values to sample around.
            These values must be in the non-transformed space.
        rng : np.random.Generator, optional
            A random number generator.

        Raises
        ------
        NotImplementedError
            _description_
        """
        params = self.trafo(params)

        if rng is None:
            rng = np.random.default_rng()

        # sample ranges for each observable parameter
        ranges = {}
        for param, value in params.items():
            width = (
                rng.uniform(*self._param_bin_width_range_trafo[param]) / 2.0
            )
            ranges[param] = np.clip(
                (value - width, value + width),
                *self._param_bounds_trafo[param],
            )

        # apply observables ranges
        mask = np.ones(len(df), dtype=bool)
        for param, (lower, upper) in ranges.items():
            mask &= (df[param] >= lower) & (df[param] <= upper)

        # # sample number of systematic parameters to vary
        # n_sys = rng.uniform(0, len(self.params_sys))
        # sys_params = rng.choice(self.params_sys, size=n_sys, replace=False)

        # # compute the fraction of eve
        # min_fraction = self.min_events_per_bin / mask.sum()

        # # sample ranges for each systematic parameter
        # for param in sys_params:
        #     # allowed_width = self._width[param] *
        #     wdith = (
        #         rng.uniform(*self._param_bin_width_range_trafo[param]) / 2.0
        #     )
        #     ranges[param] = (
        #         value - self._width[param],
        #         value + self._width[param],
        #     )

        raise NotImplementedError

    def __call__(self):
        if not self._model_is_built:
            raise RuntimeError("Model not built")
        raise NotImplementedError
