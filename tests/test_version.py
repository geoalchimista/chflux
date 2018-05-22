import chflux


def test_version():
    assert hasattr(chflux, '__version__')
    assert isinstance(chflux.__version__, str)
