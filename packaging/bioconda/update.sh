#!/usr/bin/env bash
# Point the recipe at a release tag: packaging/bioconda/update.sh 0.9.0
# Fills in the version and the sha256 of GitHub's source tarball for v$1. Run after the tag is
# pushed; the result is what gets copied into a bioconda-recipes pull request.
set -euo pipefail
v="${1:?usage: update.sh VERSION}"
here="$(cd "$(dirname "$0")" && pwd)"
sum="$(curl -fsSL "https://github.com/OtoYuki/proteus/archive/refs/tags/v$v.tar.gz" | sha256sum | cut -d' ' -f1)"
sed -i -e "s/{% set version = \".*\" %}/{% set version = \"$v\" %}/" \
       -e "s/{% set sha256 = \".*\" %}/{% set sha256 = \"$sum\" %}/" "$here/proteus/meta.yaml"
echo "proteus $v sha256 $sum"
