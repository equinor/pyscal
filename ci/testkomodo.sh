install_test_dependencies () {
  pip install ".[tests,docs]"
}

start_tests () {
  pytest --hypothesis-profile ci tests/
}