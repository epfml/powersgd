name: Wheels

on:
  workflow_dispatch:
  pull_request:
  push:
    branches:
      - master
  release:
    types:
      - published

concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true

jobs:
  build:
    name: Build
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
        with:
          submodules: true

      - name: Build
        run: pipx run build

      - name: Check metadata
        run: pipx run twine check dist/*

      - uses: actions/upload-artifact@v3
        with:
          path: dist/*

  upload_all:
    name: Upload if release
    needs: [build]
    runs-on: ubuntu-latest
    if: github.event_name == 'release' && github.event.action == 'published'

    steps:
      - uses: actions/setup-python@v4
        with:
          python-version: "3.x"

      - uses: actions/download-artifact@v3
        with:
          name: artifact
          path: dist

      - uses: pypa/gh-action-pypi-publish@v1.5.1
        with:
          password: ${{ secrets.pypi_password }}
