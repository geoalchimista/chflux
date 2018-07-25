import chflux


def test_version():
    assert hasattr(chflux, '__version__')
    assert isinstance(chflux.__version__, str)
    # make sure that the version adheres to SemVer for the first two numbers
    subver = chflux.__version__.split('.')
    assert all(map(str.isdigit, subver[0:3]))
