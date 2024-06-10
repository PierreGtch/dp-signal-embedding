import pylsl
import numpy as np
from dareplane_utils.general.time import sleep_s


def mockup_stream(stop_event):
    # create mockup input stream:
    info = pylsl.StreamInfo(
        name="AODataStream",
        type="EEG",
        channel_count=3,
        nominal_srate=128,
    )
    outlet = pylsl.StreamOutlet(info)

    while not stop_event.is_set():
        data = np.random.randn(
            10,
            3,
        )
        outlet.push_chunk(data)
        sleep_s(.1)
    return 0


def mockup_marker_stream(stop_event):
    # create mockup input stream:
    info = pylsl.StreamInfo(
        name="MarkersStream",
        type="Markers",
        channel_count=1,
        nominal_srate=pylsl.IRREGULAR_RATE,
        channel_format="string",
    )
    outlet = pylsl.StreamOutlet(info)

    while not stop_event.is_set():
        mrk = "right_hand"
        outlet.push_sample([mrk])
        sleep_s(.1)
    return 0
