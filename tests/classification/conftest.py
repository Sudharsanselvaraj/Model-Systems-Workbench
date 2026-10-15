"""Test guards for the offline classification suite: any network attempt fails the test."""

import socket

import pytest


@pytest.fixture(autouse=True)
def _block_network(monkeypatch):
    def _deny(*args, **kwargs):
        raise AssertionError("Network access attempted in an offline classification test")

    monkeypatch.setattr(socket.socket, "connect", _deny)
    monkeypatch.setattr(socket, "create_connection", _deny)
    monkeypatch.setattr(socket, "getaddrinfo", _deny)
