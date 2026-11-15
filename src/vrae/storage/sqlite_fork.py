"""SQLite connections that stay correct across fork().

DataLoader workers are forked from rank processes whose prefetch threads use SQLite. A child that inherits
SQLite's per-process file-lock bookkeeping, or a mutex a parent thread held at that instant, can report
"database is locked" or deadlock. So every use of a connection opened here holds GUARD, and just before any
fork (os.register_at_fork) GUARD is taken and every such connection is closed: the child inherits no SQLite
state at all. Threads reopen their connections lazily (thread_connection), in the parent and in the child.
"""
import os
import sqlite3
import threading

GUARD = threading.RLock()
_generation = 0
_open = []


def connect(*args, **kwargs) -> sqlite3.Connection:
    connection = sqlite3.connect(*args, check_same_thread=False, **kwargs)
    with GUARD:
        _open.append(connection)
    return connection


def thread_connection(local: threading.local, factory) -> sqlite3.Connection:
    """This thread's connection from factory(), reopened after a fork or a pre-fork close. Call under GUARD."""
    if getattr(local, "generation", None) != (os.getpid(), _generation):
        local.connection = factory()
        local.generation = (os.getpid(), _generation)
    return local.connection


def _before_fork():
    global _generation
    GUARD.acquire()
    for connection in _open:
        try:
            connection.close()
        except sqlite3.Error:
            pass
    _open.clear()
    _generation += 1


def _after_fork_in_parent():
    GUARD.release()


def _after_fork_in_child():
    global GUARD
    GUARD = threading.RLock()  # the inherited one belongs to the parent's forking thread


os.register_at_fork(before=_before_fork, after_in_parent=_after_fork_in_parent,
                    after_in_child=_after_fork_in_child)
