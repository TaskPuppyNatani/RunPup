import os

import pytest

from runpup.domain.models import StorageRoot
from runpup.persistence.database import Database
from runpup.persistence.repository import CatalogRepository


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture
def database(tmp_path):
    database = Database(tmp_path / "catalog" / "runpup.sqlite3")
    database.initialize()
    return database


@pytest.fixture
def repository(database):
    return CatalogRepository(database)


@pytest.fixture
def storage_root(tmp_path, repository):
    location = tmp_path / "external"
    location.mkdir()
    root = StorageRoot(name="Training outputs", location=str(location))
    repository.save_storage_root(root)
    return root
