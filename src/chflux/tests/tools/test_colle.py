from chflux.tools import filter_str, flatten_dict, update_dict


def test_filter_str():
    xs = ['timestamp_utc', 'timestamp_local', 'datetime_utc',
          'datetime_local', 'time_local', 'time_utc']
    assert filter_str(xs, '_utc') == ['timestamp_utc', 'datetime_utc',
                                      'time_utc']
    assert filter_str(xs, 'timestamp') == ['timestamp_utc', 'timestamp_local']
    assert filter_str(xs, 'datetime') == ['datetime_utc', 'datetime_local']
    assert filter_str(xs, 'time') == xs


def test_flatten_dict():
    d = {
        'A': {
            '1': 'entry A.1',
            '2': 'entry A.2',
        },
        'B': {
            '1': 'entry B.1',
        },
        'C': {
            '1': {
                '1': 'entry C.1.1',
            }
        }
    }
    d_flatten = flatten_dict(d)
    assert set(d_flatten.keys()) == {'A.1', 'B.1', 'A.2', 'C.1.1'}


def test_update_dict():
    d = {
        'A': {
            '1': 'entry A.1',
            '2': 'entry A.2',
        },
        'B': {
            '1': 'entry B.1',
        },
        'C': {
            '1': {
                '1': 'entry C.1.1',
            }
        }
    }
    updater = {
        'A': {
            '3': 'entry A.3',
            '2': 'entry A.2222',
        },
        'C': {
            '1': 'entry C.1'
        }
    }
    d_updated = update_dict(d, updater)
    assert '3' in d_updated['A']
    assert d_updated['A']['2'] == updater['A']['2']
    assert isinstance(d_updated['C']['1'], str)
