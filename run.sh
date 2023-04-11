#!/bin/bash
set -e

if command -v npm >/dev/null 2>&1; then
  if npm list -g docsify-cli >/dev/null 2>&1; then
    echo "docsify-cli is installed"
  else
    read -p "docsify-cli is not installed. Do you want to install it? (y/n) " choice
    case "$choice" in
      y|Y )
        npm i docsify-cli -g
        ;;
      n|N ) 
        echo "Okay, docsify-cli will not be installed."
        exit 1
        ;;
      * )
        echo "Invalid choice, docsify-cli will not be installed."
        exit 1
        ;;
    esac
  fi

  docsify serve --port 3000

else
  echo "npm is not installed. Please install npm before running this script."
fi
