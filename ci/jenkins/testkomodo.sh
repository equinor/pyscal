install_test_dependencies () {
  pip install -r test_requirements.txt
}

start_tests () {
  pytest --hypothesis-profile ci tests/
}