source venv_3_11_3/bin/activate
coverage run -m pytest ./test ./helper_functions/test --no-header | tee ./unittest/pytest_report
coverage report | tee ./unittest/coverage_report
cp ./unittest/*_report ../thesis_report/source/.
deactivate