source venv/bin/activate
coverage run -m pytest --no-header | tee ./unittest/pytest_report
coverage report | tee ./unittest/coverage_report
cp ./unittest/*_report ../thesis_report/source/.
deactivate