#! /bin/bash

isort  $(find src -name '*.py' )  ; black  $(find src  -name '*.py' )  ; isort  $(find tests -name '*.py') ; black  $(find tests -name '*.py')

npx prettier --write  .github/workflows/*.yml

