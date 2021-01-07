import pytest


def pytest_addoption(parser):
    parser.addoption('--skipintegration', action='store_true',
                     default=False, help='Skip integration tests.')


@pytest.fixture
def skip_integration(request):
    return request.config.getoption('--skipintegration')
