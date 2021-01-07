import pytest


@pytest.fixture
def skip_integration(request):
    return request.config.getoption('--skipintegration')
