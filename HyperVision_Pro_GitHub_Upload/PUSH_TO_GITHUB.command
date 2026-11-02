#!/bin/bash
cd "$(dirname "$0")"
echo "=========================================================="
echo "   PUSHING HYPERVISION PRO REPOSITORY TO GITHUB"
echo "=========================================================="
echo ""
echo "Remote URL: https://github.com/lalitpatil36/HyperVision-Pro.git"
echo ""
echo "Attempting git push to main..."
git branch -M main
git push -u origin main

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================================="
    echo " SUCCESS! Your code is now live on GitHub:"
    echo " https://github.com/lalitpatil36/HyperVision-Pro"
    echo "=========================================================="
else
    echo ""
    echo "=========================================================="
    echo " NOTE: If this is a new repository, make sure you created"
    echo " the empty repository 'HyperVision-Pro' on your GitHub account:"
    echo " https://github.com/new"
    echo " Then double-click this script again!"
    echo "=========================================================="
fi
echo ""
read -p "Press [Enter] to exit..."
