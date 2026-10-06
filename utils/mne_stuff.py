def find_channel(raw, name):
    for ch_name in raw.ch_names:
        if (ch_name.rstrip(".") == name or ch_name == name):
            return ch_name

    raise ValueError(f"channel {name!r} not found in {raw.ch_names}")

