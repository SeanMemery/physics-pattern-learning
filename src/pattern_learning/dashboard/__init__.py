from __future__ import annotations


def create_dashboard_app(*args, **kwargs):
    from dashboard.app import create_dashboard_app as _create_dashboard_app

    return _create_dashboard_app(*args, **kwargs)


__all__ = ["create_dashboard_app"]
