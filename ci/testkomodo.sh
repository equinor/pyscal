install_test_dependencies () {
  pip install ".[tests]"
}

start_tests () {
  pytest --hypothesis-profile ci tests/
}