source venv/bin/activate
coverage run -m pytest --no-header | tee pytest_report
coverage report | tee coverage_report
