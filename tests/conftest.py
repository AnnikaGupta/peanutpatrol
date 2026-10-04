"""Shared pytest config. No autouse fixtures here -- they'd apply to
test_integration.py too and override the real GOOGLE_CLOUD_PROJECT needed
for live API calls. See test_tools.py's local gcp_project_env fixture for
the mocked tests' version of this.
"""
