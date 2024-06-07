from pathlib import Path

from jsonargparse import CLI

from signal_embedding.controller import SignalEmbedder

config_dir = (Path(__file__).parent / "../configs").resolve()
config_file = "jumping_means.yaml"


def init_signal_embedding(
        config_path: Path | str = config_dir / config_file,
) -> SignalEmbedder:
    return CLI(SignalEmbedder, args=[f'--config={config_path}', 'get_self'], as_positional=False)


def mockup_stream(stop_event):
    import pylsl
    import numpy as np
    from dareplane_utils.general.time import sleep_s

    # create mockup input stream:
    info = pylsl.StreamInfo(
        name="AODataStream",
        type="EEG",
        channel_count=2,
        nominal_srate=100,
    )
    outlet = pylsl.StreamOutlet(info)

    while not stop_event.is_set():
        data = np.random.randn(
            10,
            2,
        )
        outlet.push_chunk(data)
        sleep_s(.1)
    return 0


if __name__ == "__main__":
    import threading
    import time
    from signal_embedding.utils.logging import logger

    logger.setLevel("DEBUG")

    mockup_event = threading.Event()
    mockup_thread = threading.Thread(target=mockup_stream, kwargs=dict(stop_event=mockup_event))
    mockup_thread.start()

    embedder = init_signal_embedding()
    embeder_thread, embedder_event = embedder.run()

    time.sleep(5)

    mockup_event.set()
    embedder_event.set()
