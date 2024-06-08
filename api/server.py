from pathlib import Path

from jsonargparse import CLI

from dareplane_utils.default_server.server import DefaultServer

from signal_embedding.utils.logging import logger
from signal_embedding.controller import SignalEmbedder


def main(signal_embedder: SignalEmbedder, port: int = 8080, ip: str = "127.0.0.1",
         loglevel: int = 10):
    logger.setLevel(loglevel)

    pcommand_map = {
        "RUN": signal_embedder.run,  # update at regular intervals
        # "STOP": controler.stop,  # stop updates -> command automatically created by dareplane-utils
        "UPDATE": signal_embedder.update,  # update once only
        "INIT_ALL": signal_embedder.init_all,  # initialize the streams and the model
        "CONNECT_INPUT_STREAM": signal_embedder.connect_input_stream,
        "CREATE_MODEL": signal_embedder.create_model,
        "CREATE_OUTPUT_STREAM": signal_embedder.create_output_stream,
    }

    server = DefaultServer(
        port, ip=ip, pcommand_map=pcommand_map, name="signal_embedding_server"
    )

    # initialize to start the socket
    server.init_server()
    # start processing of the server
    server.start_listening()

    return 0


if __name__ == "__main__":
    config_dir = (Path(__file__).parent / "../configs").resolve()
    # config_file = "skorch_nn_Lee2019_MI.yaml"

    config_file = "jumping_means.yaml"

    CLI(main, default_config_files=[config_dir / config_file])
