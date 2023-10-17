#!/bin/bash
nohup sh -c 'python main.py > out.log 2>&1' 'python eval_qa.py > out.log 2>&1' &