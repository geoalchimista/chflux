import sys
from io import StringIO
import contextlib


@contextlib.contextmanager
def stdoutIO(stdout=None):
    old = sys.stdout
    if stdout is None:
        stdout = StringIO()
    sys.stdout = stdout
    yield stdout
    sys.stdout = old


def test_check_pkgreqs():
    test1 = "from chflux.tools import check_pkgreqs; check_pkgreqs(True)"
    test2 = "from chflux.tools import check_pkgreqs; check_pkgreqs(False)"

    with stdoutIO() as s1:
        try:
            exec(test1)
        except:
            print("Test failed: check_pkgreqs")

    assert len(s1.getvalue()) > 0

    with stdoutIO() as s2:
        try:
            exec(test2)
        except:
            print("Test failed: check_pkgreqs")

    assert len(s2.getvalue()) == 0
