from copy import deepcopy
import numpy as np
import pandas as pd
import logging
import tqdm

import tensorflow as tf

from tfscripts.model import DenseNNGaussian


class StormChaser(DenseNNGaussian):
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
        # NN parameters: main prediction
        fc_sizes=[64, 1],
        use_dropout_list=False,
        activation_list=["relu", None],
        use_batch_normalisation_list=False,
        use_residual_list=False,
        # NN parameters: uncertainty prediction
        fc_sizes_unc=[64, 1],
        use_dropout_list_unc=False,
        activation_list_unc=["relu", None],
        use_batch_normalisation_list_unc=False,
        use_residual_list_unc=False,
        use_nth_fc_layer_as_input=None,
        min_sigma_value=1e-3,
        # NN parameters: general
        dtype="float32",
        **kwargs,
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

        # NN parameters: main prediction
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

        # NN parameters: uncertainty prediction
        fc_sizes_unc : list of int
            The number of nodes for each uncertainty layer. The ith int
            denotes the number of nodes for the ith layer. The number of
            layers is inferred from the length of 'fc_sizes_unc'.
        use_dropout_list_unc : bool, optional
            Denotes whether to use dropout in the uncertainty layers.
            If only a single boolean is provided, it will be used for all
            layers.
        activation_list_unc : str or callable, optional
            The activation function to be used in each uncertainty layer.
            If only one activation is provided, it will be used for all
            layers.
        use_batch_normalisation_list_unc : bool or list of bool, optional
            Denotes whether to use batch normalisation in the uncertainty
            layers. If only a single boolean is provided, it will be used for
            all layers.
        use_residual_list_unc : bool or list of bool, optional
            Denotes whether to use residual additions in the uncertainty
            layers. If only a single boolean is provided, it will be used
            for all layers.
        use_nth_fc_layer_as_input : int, optional
            Denotes which layer to use as input for the uncertainty layers.
            If None, the same input as for the main prediction network
            is used as input for the uncertainty sub-network.
            If negative, the layer is counted from the last layer.
            For example, use_nth_fc_layer_as_input=-2 denotes the second last
            layer.
        min_sigma_value : float
            The lower bound for the uncertainty estimation.
            This is used to ensure robustness of the training.

        # NN parameters: general
        dtype : str, optional
            The float precision type.
        **kwargs : dict
            Additional arguments to pass to the DenseNNGaussian class.

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
            fc_sizes_unc=fc_sizes_unc,
            use_dropout_list_unc=use_dropout_list_unc,
            activation_list_unc=activation_list_unc,
            use_batch_normalisation_list_unc=use_batch_normalisation_list_unc,
            use_residual_list_unc=use_residual_list_unc,
            use_nth_fc_layer_as_input=use_nth_fc_layer_as_input,
            min_sigma_value=min_sigma_value,
            dtype=dtype,
            **kwargs,
        )

        self.params = sorted(params)
        self.params_sys = sorted(params_sys)
        self._params_all = sorted(self.params + self.params_sys)
        self._param_idx = {
            param: i for i, param in enumerate(self._params_all)
        }
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

    def call(self, inputs, training=False, keep_prob=None):
        """Apply model

        Parameters
        ----------
        inputs : tf.Tensor or array_like
            The input data.
            Shape: [n_batch, n_inputs]
        training : None, optional
            Indicates whether currently in training or inference mode.
            Must be provided if batch normalisation is used.
            True: in training mode
            False: inference mode.
        keep_prob : None, optional
            The keep probability to be used for dropout.
            Can either be a float or a scalar float tf.Tensor.

        Returns
        -------
        tf.Tensor
            The predicted sigma.
        """
        y_pred, y_pred_unc = super().call(
            inputs=inputs,
            training=training,
            keep_prob=keep_prob,
        )

        # stack and return
        return tf.stack([y_pred, y_pred_unc], axis=2)

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
        if param not in self.data_trafo:
            return values

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
        inputs = np.array(inputs)

        for i, trafo in self._data_trafo_idx:
            inputs[:, i] = self.trafo_param(
                values=inputs[:, i],
                param=self._params_all[i],
                invert=invert,
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
        bootstrap_mc: bool = False,
        ignore_missing_samples: bool = False,
        epochs: int = 100,
        steps_per_epoch: int = 1000,
        steps_burn_in: int = 0,
        num_trafo_samples=1000,
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        **kwargs,
    ):
        """Build model

        The model is utilized to predict impact of systematic
        uncertainties on the observable parameters.

        Parameters
        ----------
        df_sys : pd.DataFrame
            The systematic data sample in original, non-transformed space.
        df_base : pd.DataFrame, optional
            The base data sample, by default None. If None, the total
            `df_sys` sample is used as the base sample. Note that this
            expects the systemaatic uncertainties to have symmetric
            priors around the nominal value.
            Data must be in original, non-transformed space.
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
        bootstrap_mc : bool, optional
            If True, an additional bootstrap sampling is applied when
            calculating the variation due to systematic uncertainties.
            This sampling is applied for the unweighted events in order to
            assess the limited simulation statistics.
            The bootstrapping is able to cover some of the statistical
            uncertainties in the data sample.
        ignore_missing_samples : bool, optional
            If True, missing samples are ignored, by default False.
            If False, a RuntimeError is raised if a sample cannot be created.
        epochs : int, optional
            The number of epochs to train the model, by default 100.
        steps_per_epoch : int, optional
            The number of steps per epoch, by default 1000.
        steps_burn_in : int, optional
            The number of steps to run with MSE loss before switching to
            Gaussian likelihood, by default 0.
        kwargs : dict
            Additional arguments to pass to the model training.

        Returns
        -------
        dict
            The training history.

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
        df_sys = df_sys[keys]

        if df_base is None:
            df_base = df_sys
        else:
            df_base = df_base[keys]

            # Note: code needs to be added to ensure that
            #       df_base and df_sys have the same distribution
            #       for the base parameters.
            #       Anything else missing?
            raise NotImplementedError(
                "Fitting to a different base sample not implemented"
            )

        # create trafo model
        self.logger.info("Creating trafo model")
        trafo_data = []
        for param in self._params_all:
            trafo_data.append(
                rng.uniform(*self._param_bounds_trafo[param], size=1000)
            )

        def get_y_true_distribution():
            df = self.create_training_samples(
                df_base=df_base,
                df_sys=df_sys,
                size=num_trafo_samples,
                weight_key=weight_key,
                max_n_sys=max_n_sys,
                bootstrap_mc=bootstrap_mc,
                rng=rng,
                ignore_missing_samples=ignore_missing_samples,
            )
            return np.expand_dims(df["variation"].values, axis=-1)

        self.create_trafo_model(
            inputs=np.stack(trafo_data, axis=1),
            y_true=get_y_true_distribution(),
        )

        # create data generator
        generator = self.get_data_generator(
            df_base=df_base,
            df_sys=df_sys,
            weight_key=weight_key,
            max_n_sys=max_n_sys,
            bootstrap_mc=bootstrap_mc,
            rng=rng,
            ignore_missing_samples=ignore_missing_samples,
        )

        def input_generator(generator):
            while True:
                yield self.df2input(next(generator), return_y=True)

        train_generator = input_generator(generator)

        # define loss functions
        class MSE(tf.keras.losses.Loss):
            def call(self, y_true, y_pred):
                y_pred_mean = y_pred[..., 0]
                return tf.reduce_mean(
                    tf.math.square(y_pred_mean - y_true), axis=-1
                )

        class GaussianLikelihood(tf.keras.losses.Loss):
            def call(self, y_true, y_pred):
                y_pred_mean = y_pred[..., 0]
                y_pred_sigma = y_pred[..., 1]
                return tf.reduce_mean(
                    ((y_pred_mean - y_true) / y_pred_sigma) ** 2
                    + 2 * tf.math.log(y_pred_sigma),
                    axis=-1,
                )

        # create placeholder for training history
        history = {}

        # run some iterations on MSE
        if steps_burn_in > 0:
            self.logger.info("Running initial ireations on MSE loss")
            self.compile(
                optimizer=optimizer,
                loss=MSE(),
            )
            history["mse"] = self.fit(
                x=train_generator,
                epochs=1,
                steps_per_epoch=steps_burn_in,
                **kwargs,
            )

        # run some iterations on Gaussian Likelihood
        # compile model
        self.logger.info("Running training on Gaussian Likelihood loss")
        self.compile(
            optimizer=optimizer,
            loss=GaussianLikelihood(),
        )

        # perform training of NN
        history["gaussian_likelihood"] = self.fit(
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

        return history

    def sample_df(
        self,
        df: pd.DataFrame,
        params: dict[str, float] = None,
        weight_key: str = None,
        max_n_sys: int = None,
        bootstrap_mc: bool = False,
        rng: np.random.RandomState = None,
    ):
        """Sample events from data frame for given range of parameters

        Parameters
        ----------
        df : pd.DataFrame
            The data sample to sample from.
            Data must be in original, non-transformed space.
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
        bootstrap_mc : bool, optional
            If True, an additional bootstrap sampling is applied when
            sampling events from the simulation dataset. This sampling
            is applied for the unweighted events in order to assess
            the limited simulation statistics.
            The bootstrapping is able to cover some of the statistical
            uncertainties in the data sample.
        rng : np.random.RandomState, optional
            A random number generator.

        Returns
        -------
        pd.DataFrame
            The sampled data in original, non-transformed space.
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
            # note: faster via numpy than pandas
            values = df[param].values
            mask &= (values >= lower) & (values <= upper)

        # sample number of systematic parameters to vary
        n_sys = rng.randint(1, max_n_sys + 1)
        sys_params = rng.choice(self.params_sys, size=n_sys, replace=False)

        # compute how large the range of each systematic parameter can be
        # (while assuming these are distributed uniformly)
        if not np.unique(list(self.param_sys_type.values()))[0] == "uniform":
            raise NotImplementedError
        else:
            n_events = mask.sum()
            if n_events < self.min_target_per_bin:
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
            # note: faster via numpy than pandas
            values = df[param].values
            mask &= (values >= lower) & (values <= upper)

        if mask.sum() < self.min_events_per_bin:
            raise RuntimeError(
                "Not enough events in sample to sample parameters!",
                mask.sum(),
                ranges,
            )

        # adjust weights
        df_out = df[mask].copy(deep=True)
        if weight_key is not None:
            df_out[weight_key] *= weight_scaling

        # apply bootstrap sampling
        if bootstrap_mc:
            df_out = df_out.sample(len(df_out), replace=True)

        return df_out, ranges, weight_scaling

    def create_training_samples(
        self,
        df_base: pd.DataFrame,
        df_sys: pd.DataFrame,
        size: int,
        weight_key: str = None,
        max_n_sys: int = None,
        bootstrap_mc: bool = False,
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
        bootstrap_mc : bool, optional
            If True, an additional bootstrap sampling is applied when
            calculating the variation due to systematic uncertainties.
            This sampling is applied for the unweighted events in order to
            assess the limited simulation statistics.
            The bootstrapping is able to cover some of the statistical
            uncertainties in the data sample.
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
            bootstrap_mc=bootstrap_mc,
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
        bootstrap_mc: bool = False,
        rng: np.random.RandomState = None,
        ignore_missing_samples: bool = False,
    ):
        """Create a training dataset for the model

        Parameters
        ----------
        df_base : pd.DataFrame
            The baseline data sample in original, non-transformed space.
        df_sys : pd.DataFrame
            The data sample to sample from for systematic variations.
            Data must be in original, non-transformed space.
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
        bootstrap_mc : bool, optional
            If True, an additional bootstrap sampling is applied when
            calculating the variation due to systematic uncertainties.
            This sampling is applied for the unweighted events in order to
            assess the limited simulation statistics.
            The bootstrapping is able to cover some of the statistical
            uncertainties in the data sample.
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
                df_sys_i, ranges_i, weight_scaling_i = self.sample_df(
                    df_sys_r,
                    weight_key=weight_key,
                    max_n_sys=max_n_sys,
                    bootstrap_mc=bootstrap_mc,
                    rng=rng,
                )
            except RuntimeError:
                if ignore_missing_samples:
                    self.logger.info(
                        "Not enough events in sample to sample parameters"
                    )
                    continue
                else:
                    raise

            # get median parameter values
            data = {}
            for param in self._params_all:
                # note: faster via numpy median on array rather than pandas
                data[param] = np.median(df_sys_i[param].values)

            # apply mask to baseline sample
            mask_base = np.ones(len(df_base_r), dtype=bool)
            for param in self.params:
                lower, upper = ranges_i[param]
                # note: faster via numpy than pandas
                values = df_base_r[param].values
                mask_base &= (values >= lower) & (values <= upper)

            df_base_i = df_base_r[mask_base]

            # check stats
            min_num_events = min(len(df_sys_i), len(df_base_i))
            if min_num_events < self.min_events_per_bin:
                if ignore_missing_samples:
                    self.logger.info(
                        f"Too few events in sample: {min_num_events}"
                    )
                    continue
                else:
                    raise RuntimeError(
                        f"Too few events in sample: {min_num_events}"
                    )

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
