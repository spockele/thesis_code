source venv/bin/activate
coverage run -m unittest discover
coverage report > coverage_report
