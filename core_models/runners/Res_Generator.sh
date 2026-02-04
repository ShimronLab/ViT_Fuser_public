#!/bin/bash

for slice in {8..20}; do
    python3 Test_and_Compare.py --subject 4 --slice $slice --mask 1D --factor 3 --snr 10
done