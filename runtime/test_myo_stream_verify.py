from myo_stream_verify import arrival_quality


def test_quality_uses_actual_delivery_rate():
    arrivals = [round(i * 1e9 / 134.5) for i in range(1346)]
    rate, gap = arrival_quality(arrivals, 10.0)
    assert rate == 134.5
    assert gap < 10


def test_quality_includes_silent_tail_in_rate():
    arrivals = [i * 5_000_000 for i in range(1001)]
    rate, _ = arrival_quality(arrivals, 10.0)
    assert rate == 100.0


def test_quality_exposes_gap_hidden_by_sample_pll():
    arrivals = [0, 5_000_000, 10_000_000, 2_010_000_000]
    _, gap = arrival_quality(arrivals, 2.01)
    assert gap == 2000
