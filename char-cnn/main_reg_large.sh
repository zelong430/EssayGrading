#!/bin/bash

for maxlen in 2048 4096
do
	for numfilter in 64 128 256
	do
		for dense in 64 128
		do
			for kernel in 64 128 256 512
			do
				for pool in 24 64 256
				do
					python main_reg.py $maxlen $numfilter $dense $kernel $pool
				done
			done
		done
	done
done

#1000 256 256 10 4 0.51 0.565
# 2048 256 256 64 4