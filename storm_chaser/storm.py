from copy import deepcopy
import numpy as np
import pandas as pd
import logging
import tqdm

import tensorflow as tf

from tfscripts.model import DenseNN


class StormChaser(DenseNN):
    """Analyzer for systematic uncertainties"""

    def __init__(
        self,
        params: list[str],
        params_sys: list[str] = [],
        param_bounds: dict[str, tuple[float, float]] = {},
        params_trafo_bin_width_range: dict[str, tuple[float, float]] = {},
        min_target_per_bin: int = 1000,
        min_events_per_bin: int = 100,
        data_trafo: dict[str, str] = {},
        param_sys_type: str | dict[str, str] = "uniform",
        # NN parameters
        fc_sizes=[64, 64, 1],
        use_dropout_list=False,
        activation_list="relu",
        use_batch_normalisation_list=False,
        use_residual_list=True,
        dtype="float32",
    ):
        """Initialize the StormChaser object

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
        params_trafo_bin_width_range : dict[str, tuple[float, float]], optional
            The range of bin widths to allow for each observable parameter in
            `params`. This must be a dictionary of the form
                {param: (min_width, max_width)},
            here `param` is the name of the parameter and `min_width` and
            `max_width` are the minimum and maximum allowed bin widths,
            respectively. If not provided, the bin width range is set to
            1% and 10% of the parameter range, respectively.
            Note that the bin widths must be provided in the
            transformed space!
        min_target_per_bin : int, optional
            The target minimum number of events per bin when creating training
            samples for the model. Note that this is not the actual minimum
            number of events per bin.
        min_events_per_bin : int, optional
            The minimum number of events per bin when creating training
            samples for the model. If the number of events in a bin is lower
            than this value, the bin is not used for training.
        data_trafo : dict[str, str], optional
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

        # NN parameters
        fc_sizes : list of int
            The number of nodes for each layer. The ith int denotes the number
            of nodes for the ith layer. The number of layers is inferred from
            the length of 'fc_sizes'.
        use_dropout_list : bool, optional
            Denotes whether to use dropout in the layers.
            If only a single boolean is provided, it will be used for all
            layers.
        activation_list : str or callable, optional
            The type of activation function to be used in each layer.
            If only one activation is provided, it will be used for all layers.
        use_batch_normalisation_list : bool or list of bool, optional
            Denotes whether to use batch normalisation in the layers.
            If only a single boolean is provided, it will be used for all
            layers.
        use_residual_list : bool or list of bool, optional
            Denotes whether to use residual additions in the layers.
            If only a single boolean is provided, it will be used for all
            layers.
        dtype : str, optional
            The float precision type.

        Raises
        ------
        NotImplementedError
            _description_
        """
        self.logger = logging.getLogger(__name__)

        # Create NN model
        super().__init__(
            input_shape=[-1, len(params) + len(params_sys)],
            fc_sizes=fc_sizes,
            use_dropout_list=use_dropout_list,
            activation_list=activation_list,
            use_batch_normalisation_list=use_batch_normalisation_list,
            use_residual_list=use_residual_list,
            dtype=dtype,
        )

        self.params = sorted(params)
        self.params_sys = sorted(params_sys)
        self._params_all = sorted(self.params + self.params_sys)
        self.param_bounds = param_bounds
        self.min_target_per_bin = min_target_per_bin
        self.min_events_per_bin = min_events_per_bin
        self.data_trafo = data_trafo
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

        # save trafo types for each parameter
        self._data_trafo_idx = [
            (i, self.data_trafo[param])
            for i, param in enumerate(self._params_all)
            if param in self.data_trafo
        ]

        # transform parameter bounds
        self._param_bounds_trafo = self.trafo(self.param_bounds)

        # make sure that the bounds are in the correct order
        for param, bounds in self._param_bounds_trafo.items():
            if bounds[0] > bounds[1]:
                self._param_bounds_trafo[param] = (bounds[1], bounds[0])

        # get bin width ranges for each observable parameter
        self.params_trafo_bin_width_range = params_trafo_bin_width_range.copy()
        for param, bounds in self._param_bounds_trafo.items():
            if param not in self.params_trafo_bin_width_range:
                self.params_trafo_bin_width_range[param] = (
                    0.01 * (bounds[1] - bounds[0]),
                    0.1 * (bounds[1] - bounds[0]),
                )

        # compute width for each systematic parameter
        self._width = {}
        for param in self.params_sys:
            lower, upper = self.param_bounds[param]
            self._width[param] = upper - lower

    def get_config(self):
        """Get Configuration of StormChaser

        Returns
        -------
        dict
            A dictionary with all configuration settings. This can be used
            to serealize and deserealize the model.
        """
        config = super().get_config()
        config.pop("input_shape")
        updates = {
            "params": self.params,
            "params_sys": self.params_sys,
            "param_bounds": self.param_bounds,
            "params_trafo_bin_width_range": self.params_trafo_bin_width_range,
            "min_target_per_bin": self.min_target_per_bin,
            "min_events_per_bin": self.min_events_per_bin,
            "data_trafo": self.data_trafo,
            "param_sys_type": self.param_sys_type,
        }
        config.update(updates)
        return deepcopy(config)

    def trafo_param(
        self, values: list[float], param: str, invert: bool = False
    ):
        """Apply the data transformation to the data

        Parameters
        ----------
        values : list[float]
            The values to transform.
        param : str
            The name of the parameter to transform.
        invert : bool, optional
            Whether to invert the transformation, by default False.

        Returns
        -------
        list[float]
            The transformed values.
        """
        trafo = self.data_trafo[param]

        if trafo == "log+1":
            if invert:
                result = 10**values - 1.0
            else:
                result = np.log10(values + 1.0)
        elif trafo == "log":
            if invert:
                result = 10**values
            else:
                result = np.log10(values)
        elif trafo == "cos":
            if invert:
                result = np.arccos(values)
            else:
                result = np.cos(values)
        else:
            raise NotImplementedError(
                f"Transformation {trafo} not implemented"
            )
        return result

    def trafo_inputs(self, inputs: np.ndarray, invert: bool = False):
        """Transforms the input array

        Parameters
        ----------
        inputs : np.ndarray
            The data to transform. These inputs must be provided
            in the non-transformed space and in the same order as the
            sorted list of observable and systematic parameters in
            `self._params_all`.
        invert : bool, optional
            Whether to invert the transformation, by default False.

        Returns
        -------
        pd.DataFrame
            The transformed data.
        """
        inputs = np.empty_like(inputs)

        for i, trafo in self._data_trafo_idx:
            inputs[:, i] = self.trafo_param(
                values=inputs[:, i], param=self._params_all[i], invert=invert
            )
        return inputs

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
        df_out = {}

        for param in df.keys():
            if param in self.data_trafo:
                df_out[param] = self.trafo_param(
                    values=df[param], param=param, invert=invert
                )
            else:
                df_out[param] = df[param]

        if isinstance(df, pd.DataFrame):
            df_out = pd.DataFrame(df_out)
        return df_out

    def df2input(self, df: pd.DataFrame, return_y: bool = False):
        """Convert data frame to model input

        Parameters
        ----------
        df : pd.DataFrame
            The data frame to convert.
        return_y : bool, optional
            Whether to return the labels, by default False.

        Returns
        -------
        np.ndarray
            The model input: x.
        np.ndarray [optional]
            The labels: y.
        """
        df_trafo = self.trafo(df)
        x = df_trafo[self._params_all].values
        if return_y:
            y = df_trafo["variation"].values
            return x, y
        else:
            return x

    def chase(self, df):
        """Predict the impact of systematic uncertainties on the observable
        parameters

        Parameters
        ----------
        df : pd.DataFrame
            The data sample to predict on.

        Returns
        -------
        np.ndarray
            The predicted impact of systematic uncertainties on the
            observable parameters.
        """
        if not self._model_is_built:
            raise RuntimeError("Model not built")
        return self.concrete_func(self.df2input(df)).numpy()

    def chase_inputs(self, inputs: np.ndarray):
        """Predict the impact of systematic uncertainties on the observable
        parameters

        Parameters
        ----------
        inputs : np.ndarray
            The data sample to predict on. These inputs must be provided
            in the non-transformed space and in the same order as the
            sorted list of observable and systematic parameters in
            `self._params_all`.

        Returns
        -------
        np.ndarray
            The predicted impact of systematic uncertainties on the
            observable parameters.
        """
        if not self._model_is_built:
            raise RuntimeError("Model not built")
        return self.concrete_func(self.trafo_inputs(inputs)).numpy()

    def build_model(
        self,
        df_sys: pd.DataFrame,
        df_base: pd.DataFrame = None,
        weight_key: str = None,
        seed: int = 42,
        max_n_sys: int = None,
        ignore_missing_samples: bool = False,
        epochs: int = 100,
        steps_per_epoch: int = 1000,
        **kwargs,
    ):
        """Build model

        The model is utilized to predict impact of systematic
        uncertainties on the observable parameters.

        Parameters
        ----------
        df_sys : pd.DataFrame
            The systematic data sample.
        df_base : pd.DataFrame, optional
            The base data sample, by default None. If None, the total
            `df_sys` sample is used as the base sample. Note that this
            expects the systemaatic uncertainties to have symmetric
            priors around the nominal value.
        weight_key : str, optional
            The name of the column containing the weights,
            by default None. If None, all events are assumed to have
            the same weight.
        seed : int, optional
            The random seed to use for sampling training samples.
        max_n_sys : int, optional
            The maximum number of systematic parameters to vary in a generated
            sample. If None, up to the total number of systematic parameters
            are varied.
        ignore_missing_samples : bool, optional
            If True, missing samples are ignored, by default False.
            If False, a RuntimeError is raised if a sample cannot be created.
        epochs : int, optional
            The number of epochs to train the model, by default 100.
        steps_per_epoch : int, optional
            The number of steps per epoch, by default 1000.
        kwargs : dict
            Additional arguments to pass to the model training.

        Raises
        ------
        NotImplementedError
            _description_
        NotImplementedError
            _description_
        """

        # create random number generator
        rng = np.random.RandomState(seed=seed)

        if weight_key is None:
            keys = self._params_all
        else:
            keys = self._params_all + [weight_key]

        # only select relevant columns
        print("keys", keys)
        df_sys = df_sys[keys]

        if df_base is None:
            df_base = df_sys
        else:
            df_base = self.trafo(df_base[keys])

            # Note: code needs to be added to ensure that
            #       df_base and df_sys have the same distribution
            #       for the base parameters.
            #       Anything else missing?
            raise NotImplementedError(
                "Fitting to a different base sample not implemented"
            )

        # create trafo model
        trafo_data = []
        for param in self._params_all:
            trafo_data.append(
                rng.uniform(*self._param_bounds_trafo[param], size=1000)
            )
        self.create_trafo_model(np.stack(trafo_data, axis=1))

        # create data generator
        generator = self.get_data_generator(
            df_base=df_base,
            df_sys=df_sys,
            weight_key=weight_key,
            max_n_sys=max_n_sys,
            rng=rng,
            ignore_missing_samples=ignore_missing_samples,
        )

        def input_generator(generator):
            while True:
                yield self.df2input(next(generator), return_y=True)

        train_generator = input_generator(generator)

        # compile model
        self.compile(
            optimizer="adam",
            loss="mse",
            metrics=["mse"],
        )

        # perform training of NN
        self.fit(
            x=train_generator,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            **kwargs,
        )

        # create concrete function of trained model
        input_shape = list(self._input_shape)
        input_shape[0] = None
        concrete_func = tf.function(self).get_concrete_function(
            tf.TensorSpec(input_shape, self.dtype)
        )

        self.concrete_func = concrete_func

        # done
        self._model_is_built = True

    def sample_df(
        self,
        df: pd.DataFrame,
        params: dict[str, float] = None,
        weight_key: str = None,
        max_n_sys: int = None,
        rng: np.random.RandomState = None,
    ):
        """Sample events from data frame for given range of parameters

        Parameters
        ----------
        df : pd.DataFrame
            The data sample to sample from.
        params : dict[str, float]
            If provided, these will be used as parameter values to sample
            around. These values must be in the non-transformed space.
            If None is provided, the parameter values are sampled within the
            provided bounds.
        weight_key : str, optional
            The name of the column containing the weights,
            by default None. If None, all events are assumed to have
            the same weight.
        max_n_sys : int, optional
            The maximum number of systematic parameters to vary in a generated
            sample. If None, up to the total number of systematic parameters
            are varied.
        rng : np.random.RandomState, optional
            A random number generator.

        Returns
        -------
        pd.DataFrame
            The sampled data.
        np.ndarray
            A mask indicating which events were selected.
        dict[str, tuple[float, float]]
            The ranges for each parameter.
        float
            The weight scaling factor.
        """
        if max_n_sys is None:
            max_n_sys = len(self.params_sys)
        else:
            max_n_sys = min(max_n_sys, len(self.params_sys))

        if rng is None:
            rng = np.random.RandomState()

        if params is None:
            params = {}
            for param in self.params:
                # directly sample in trafo space
                params[param] = rng.uniform(*self._param_bounds_trafo[param])
        else:
            params = self.trafo(params)

        # sample ranges for each observable parameter
        ranges = {}
        for param, value in params.items():
            half_width = (
                rng.uniform(*self.params_trafo_bin_width_range[param]) / 2.0
            )
            ranges_i = np.clip(
                (value - half_width, value + half_width),
                *(self._param_bounds_trafo[param]),
            )
            ranges[param] = self.trafo_param(
                ranges_i, param=param, invert=True
            )
            if ranges[param][0] > ranges[param][1]:
                ranges[param] = (ranges[param][1], ranges[param][0])

        # apply observables ranges
        mask = np.ones(len(df), dtype=bool)
        for param, (lower, upper) in ranges.items():
            mask &= (df[param] >= lower) & (df[param] <= upper)

        # sample number of systematic parameters to vary
        n_sys = rng.randint(1, max_n_sys + 1)
        sys_params = rng.choice(self.params_sys, size=n_sys, replace=False)

        # compute how large the range of each systematic parameter can be
        # (while assuming these are distributed uniformly)
        if not np.unique(list(self.param_sys_type.values()))[0] == "uniform":
            raise NotImplementedError
        else:
            n_events = mask.sum()
            if n_events < self.min_events_per_bin:
                raise RuntimeError(
                    "Not enough events in sample to sample parameters!",
                    ranges,
                )
            min_fraction = pow(self.min_target_per_bin / n_events, 1.0 / n_sys)
            weight_scaling = n_events / self.min_target_per_bin

        # sample ranges for each systematic parameter
        for param in sys_params:
            allowed_half_width = self._width[param] * min_fraction / 2.0
            value = rng.uniform(
                self.param_bounds[param][0] + allowed_half_width,
                self.param_bounds[param][1] - allowed_half_width,
            )
            lower = value - allowed_half_width
            upper = value + allowed_half_width
            ranges[param] = (lower, upper)

            # apply observables ranges
            mask &= (df[param] >= lower) & (df[param] <= upper)

        # adjust weights
        df_out = df[mask].copy(deep=True)
        if weight_key is not None:
            df_out[weight_key] *= weight_scaling

        return df_out, mask, ranges, weight_scaling

    def create_training_samples(
        self,
        df_base: pd.DataFrame,
        df_sys: pd.DataFrame,
        size: int,
        weight_key: str = None,
        max_n_sys: int = None,
        rng: np.random.RandomState = None,
        ignore_missing_samples: bool = False,
        verbose: bool = True,
    ):
        """Create a training dataset for the model

        Parameters
        ----------
        df_base : pd.DataFrame
            The baseline data sample.
        df_sys : pd.DataFrame
            The data sample to sample from for systematic variations.
        size : int
            The number of training samples to create.
        weight_key : str, optional
            The name of the column containing the weights,
            by default None. If None, all events are assumed to have
            the same weight.
        max_n_sys : int, optional
            The maximum number of systematic parameters to vary in a generated
            sample. If None, up to the total number of systematic parameters
            are varied.
        rng : np.random.RandomState, optional
            A random number generator.
        ignore_missing_samples : bool, optional
            If True, missing samples are ignored, by default False.
            If False, a RuntimeError is raised if a sample cannot be created.
        verbose : bool, optional
            Whether to show a progress bar, by default True.

        Returns
        -------
        pd.DataFrame
            The training stamples f(var_1, ... var_N, sys_1, ..., sys_M) = m
        """

        # create data generator
        generator = self.get_data_generator(
            df_base=df_base,
            df_sys=df_sys,
            weight_key=weight_key,
            max_n_sys=max_n_sys,
            rng=rng,
            ignore_missing_samples=ignore_missing_samples,
        )

        # create empty data frame
        df_list = []

        # collect data
        for _ in tqdm.trange(size, disable=not verbose):
            df_list.append(next(generator))

        return pd.concat(df_list, axis=0, ignore_index=True)

    def get_data_generator(
        self,
        df_base: pd.DataFrame,
        df_sys: pd.DataFrame,
        size: int = None,
        weight_key: str = None,
        max_n_sys: int = None,
        rng: np.random.RandomState = None,
        ignore_missing_samples: bool = False,
    ):
        """Create a training dataset for the model

        Parameters
        ----------
        df_base : pd.DataFrame
            The baseline data sample.
        df_sys : pd.DataFrame
            The data sample to sample from for systematic variations.
        weight_key : str, optional
            The name of the column containing the weights,
            by default None. If None, all events are assumed to have
            the same weight.
        size : int, optional
            The number of training samples to create.
            If None, an infinite generator is returned.
        max_n_sys : int, optional
            The maximum number of systematic parameters to vary in a generated
            sample. If None, up to the total number of systematic parameters
            are varied.
        rng : np.random.RandomState, optional
            A random number generator.
        ignore_missing_samples : bool, optional
            If True, missing samples are ignored, by default False.
            If False, a RuntimeError is raised if a sample cannot be created.

        Returns
        -------
        pd.DataFrame
            The training stamples f(var_1, ... var_N, sys_1, ..., sys_M) = m
        """

        if weight_key is None:
            keys = self._params_all
        else:
            keys = self._params_all + [weight_key]

        df_sys_r = df_sys[keys].copy()
        df_base_r = df_base[keys].copy()

        counter = 0
        while True:
            try:
                df_sys_i, mask_i, ranges_i, weight_scaling_i = self.sample_df(
                    df_sys_r,
                    rng=rng,
                    weight_key=weight_key,
                    max_n_sys=max_n_sys,
                )
            except RuntimeError:
                if ignore_missing_samples:
                    continue
                    self.logger.warning(
                        "Not enough events in sample to sample parameters"
                    )
                else:
                    raise

            # get average parameter values
            data = {}
            for param in self._params_all:
                data[param] = df_sys_i[param].mean()

            # apply mask to baseline sample
            mask_base = np.ones(len(df_base_r), dtype=bool)
            for param in self.params:
                lower, upper = ranges_i[param]
                mask_base &= (df_base_r[param] >= lower) & (
                    df_base_r[param] <= upper
                )

            df_base_i = df_base_r[mask_base]

            # compute varied fraction wrt baseline
            if weight_key is None:
                variation = (
                    df_sys_i.shape[0] * weight_scaling_i / df_base_i.shape[0]
                )
            else:
                variation = (
                    df_sys_i[weight_key].sum() / df_base_i[weight_key].sum()
                )
            data["variation"] = variation

            yield pd.DataFrame(data, index=[counter])

            counter += 1
            if size is not None and counter >= size:
                break

    # def __call__(self):
    #     if not self._model_is_built:
    #         raise RuntimeError("Model not built")
    #     raise NotImplementedError
