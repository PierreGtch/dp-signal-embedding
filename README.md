# Dareplane Signal Embedding

Dareplane module for transforming an LSL stream into a stream of embedding vectors.

## Getting Started

### Installation

1. Install poetry system-wide in an isolated environment (https://python-poetry.org/docs/)
2. Create your environment (conda, venv...)
3. Install dependencies with poetry: `poetry install`

### Testing

#### Controller

Testing the controller only :

```bash
python signal_embedding/main.py
```

By default, it uses the `jumping_means.yaml` config. You can use another one using the `--config` argument (more details
in [jsonargparse](https://jsonargparse.readthedocs.io)):

```bash
python signal_embedding/main.py --config=configs/skorch_nn_Lee2019_MI.yaml 
```

Alternatively, you can overwrite one specific parameter of the config (more details
in [jsonargparse](https://jsonargparse.readthedocs.io)):

```bash
python signal_embedding/main.py --signal_embedder.input_stream_name=MockupStream
``` 

#### Server

To start the server, you can use the following command:

```bash
python api/server.py
```

Again, the config can be changed via the `--config` argument:

```bash
python api/server.py --config=configs/skorch_nn_Lee2019_MI.yaml 
```

Then, you can test the server using `tellnet`:

```bash
telnet 127.0.0.1 8080
RUN
```