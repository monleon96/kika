#!/bin/bash
# Script to start Docker with proxy settings in WSL2

echo "Starting Docker with proxy configuration..."
sudo HTTP_PROXY="http://MONLEON-DE-LA-JAN:Navier-Stokes2024@wpadirsn.proton.intra.irsn.fr:8088" \
     HTTPS_PROXY="http://MONLEON-DE-LA-JAN:Navier-Stokes2024@wpadirsn.proton.intra.irsn.fr:8088" \
     service docker start

echo "Docker status:"
sudo service docker status
