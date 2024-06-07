from jsonargparse import CLI

from dareplane_utils.default_server.server import DefaultServer

from signal_embedding.main import init_signal_embedding
from signal_embedding.utils.logging import logger


def main(port: int = 8080, ip: str = "127.0.0.1", loglevel: int = 10):
    logger.setLevel(loglevel)

    # Get the communication manager
    controler = init_signal_embedding()

    pcommand_map = {
        "RUN": controler.run,  # update at regular intervals
        # "STOP": controler.stop,  # stop updates -> command automatically created by dareplane-utils
        "UPDATE": controler.update,  # update once only
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
    CLI(main)
