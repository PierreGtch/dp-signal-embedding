from pathlib import Path

from jsonargparse import ArgumentParser

from signal_embedding.controller import SignalEmbedder

config_dir = (Path(__file__).parent / "../configs").resolve()
# default_config_file = "skorch_nn_Lee2019_MI.yaml"
default_config_file = "jumping_means.yaml"

if __name__ == "__main__":
    import threading
    import time
    from signal_embedding.utils.logging import logger
    from signal_embedding.utils.mockup import mockup_stream, mockup_marker_stream

    logger.setLevel("DEBUG")

    parser = ArgumentParser(default_config_files=[config_dir / default_config_file])
    parser.add_argument("--signal_embedder", type=SignalEmbedder)
    parser.add_argument("--markers", action="store_true", default=False)
    args = parser.parse_args()
    # args = parser.parse_args(["--signal_embedder.init_args.marker_stream_name=MarkersStream",
    #                        "--markers", ])
    args = parser.instantiate_classes(args)
    embedder = args.signal_embedder
    markers = args.markers

    mockup_event = threading.Event()
    mockup_thread = threading.Thread(target=mockup_stream, kwargs=dict(stop_event=mockup_event))
    mockup_thread.start()

    if markers:
        mockup_marker_event = threading.Event()
        mockup_marker_thread = threading.Thread(
            target=mockup_marker_stream, kwargs=dict(stop_event=mockup_marker_event))
        mockup_marker_thread.start()

    embedder.init_all()
    embeder_thread, embedder_event = embedder.run()

    time.sleep(5)

    mockup_event.set()
    embedder_event.set()
