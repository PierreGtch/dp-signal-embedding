import threading

import numpy as np
import pylsl
from scipy.signal import resample

from dareplane_utils.general.time import sleep_s
from dareplane_utils.signal_processing.filtering import FilterBank
from dareplane_utils.stream_watcher.lsl_stream_watcher import StreamWatcher

from signal_embedding.utils.logging import logger
from signal_embedding.models.base import ModelGetter


class SignalEmbedder:
    """Class to embed windows of signal using a sklearn transformer.

    Parameters
    ----------
    model_getter: ModelGetter
        Function to get the model, according to the signal parameters.
    input_stream_name: str
        Name of the input stream (containing the signal).
    output_stream_name: str
        Name of the output stream (containing the embeddings).
    input_window_seconds: float
        Temporal size of the window that should be passed to the  model.
    new_sfreq: float | None
        If not None, resample the signal to this frequency before passing it to the model.
    band: tuple[float, float]
        Band to filter the signal before passing it to the model.
    t_sleep: float
        Time to sleep between updates.
    """

    def __init__(
            self,
            model_getter: ModelGetter,
            input_stream_name: str,
            output_stream_name: str,
            input_window_seconds: float,
            new_sfreq: float | None = None,
            band: tuple[float, float] = (0.5, 40),
            t_sleep: float = 0.1,
    ):
        self.model_getter = model_getter
        self.input_stream_name = input_stream_name
        self.output_stream_name = output_stream_name
        self.input_window_seconds = input_window_seconds
        self.new_sfreq = new_sfreq
        self.band = band
        self.t_sleep = t_sleep

        self._running = False
        self.model = None
        self.inlet = None
        self.fb = None
        self.outlet = None

    def get_self(self):
        return self

    @property
    def buffer_size_s(self):
        # we take a larger buffer for margin:
        return 3 * max(self.input_window_seconds, self.t_sleep)

    @property
    def model_sfreq(self):
        return self.signal_sfreq if self.new_sfreq is None else self.new_sfreq

    @property
    def n_outputs(self):
        x = np.zeros((1, len(self.chs_info), int(self.input_window_seconds * self.model_sfreq)),
                     dtype=np.float32)
        y = self.model.transform(x)
        assert y.ndim == 2
        assert y.shape[0] == 1
        return y.shape[1]

    def set_signal_info(self):
        self.signal_sfreq = self.inlet.inlet.info().nominal_srate()
        self.chs_info = [
            dict(ch_name=ch_name, type="EEG")
            for ch_name in self.inlet.channel_names
        ]

    def connect_input_stream(self):
        logger.info(f"Connecting to input stream {self.input_stream_name}")
        self.inlet = StreamWatcher(
            self.input_stream_name, buffer_size_s=self.buffer_size_s, logger=logger)

        self.inlet.connect_to_stream()

        # Grab latest samples
        self.inlet.update()

        # set signal_sfreq and chs_info
        self.set_signal_info()
        return 0

    def create_model(self):
        logger.info("Creating the model and the filter bank.")
        self.fb = FilterBank(
            bands={"band": self.band},
            sfreq=self.signal_sfreq,
            output="signal",
            n_in_channels=len(self.chs_info),
            filter_buffer_s=self.buffer_size_s,
        )

        self.model = self.model_getter(
            sfreq=self.model_sfreq,
            input_window_seconds=self.input_window_seconds,
            chs_info=self.chs_info,
        )
        return 0

    def create_output_stream(self):
        logger.info(f"Creating the output stream {self.output_stream_name}")
        info = pylsl.StreamInfo(
            self.output_stream_name, "MISC", channel_count=self.n_outputs,
            nominal_srate=pylsl.IRREGULAR_RATE, channel_format="float32",
            source_id="signal_embedding",  # TODO should it be more unique?
        )
        self.outlet = pylsl.StreamOutlet(info)
        return 0

    def init_all(self):
        self.connect_input_stream()
        self.create_model()
        self.create_output_stream()
        return 0

    def update(self):
        logger.debug(f"Start update")
        if self.inlet is None or self.outlet is None or self.model is None:
            logger.error("SignalEmbedder not initialized, call init_all first")
            return 1
        # Grab latest samples
        self.inlet.update()
        if self.inlet.n_new == 0:
            logger.debug("Skipping filtering because no new samples")
            return 0

        logger.debug(f"Filtering {self.inlet.n_new} new samples")
        # Filter the data
        self.fb.filter(
            # look back only new data
            self.inlet.unfold_buffer()[-self.inlet.n_new:, :],
            # and this is getting the times
            self.inlet.unfold_buffer_t()[-self.inlet.n_new:],
        )  # after this step, the buffer within fb has the filtered data
        self.inlet.n_new = 0

        # Most recent samples
        logger.debug(f"Loading the latest samples")
        # FilterBank returns (n_times, n_channel, n_bands)
        x = self.fb.get_data()
        n_times = int(self.input_window_seconds * self.signal_sfreq)
        if x.shape[0] < n_times:
            logger.debug(
                f"Skipping embedding because not enough samples in buffer ({x.shape[0]}/{n_times})")
            return 0
        logger.debug(f"Embedding the {n_times} latest samples")
        x = x[-n_times:, :, 0]

        # Resample if necessary
        if self.new_sfreq is not None:
            new_n_times = int(self.input_window_seconds * self.new_sfreq)
            x = resample(x, new_n_times, axis=0)

        # Transpose and add batch dim
        x = x.T[None, :, :]  # (1, n_channel, n_times)

        # Compute the embedding
        y = self.model.transform(x)
        assert y.ndim == 2
        assert y.shape[0] == 1

        # Push the embedding
        self.outlet.push_sample(y[0])
        return 0

    def _run_loop(self, stop_event: threading.Event):
        logger.debug("Starting the run loop")
        if self.inlet is None or self.outlet is None or self.model is None:
            logger.error("SignalEmbedder not initialized, call init_all first")
            return 1
        while not stop_event.is_set():
            t_start = pylsl.local_clock()
            self.update()
            t_end = pylsl.local_clock()

            # reduce sleep by processing time
            sleep_s(self.t_sleep - (t_end - t_start))
        return 0

    def run(self) -> tuple[threading.Thread, threading.Event]:
        stop_event = threading.Event()
        thread = threading.Thread(
            target=self._run_loop,
            kwargs={"stop_event": stop_event},
        )
        thread.start()
        return thread, stop_event
