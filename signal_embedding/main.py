from pathlib import Path

from jsonargparse import CLI

from signal_embedding.controller import SignalEmbedder

config_dir = (Path(__file__).parent / "../configs").resolve()
config_file = "jumping_means.yaml"


def init_signal_embedding(
        config_path: Path | str = config_dir / config_file,
) -> SignalEmbedder:
    return CLI(SignalEmbedder, args=[f'--config={config_path}', 'get_self'], as_positional=False)
