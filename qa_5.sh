for r in 2 #-3 3 4 5 6 7 #r loss =  ((loss_0.pow(r) + loss_1.pow(r))/2).pow(1/r) -2 -1 0 1 2
do
    for s in 0  # 1 2 3 4 # seed
    do
        python main01.py --manual_seed=${s} --loss_r=${r}
        python eval_qa01.py --manual_seed=${s} --dataset='dailydialog' --loss_r=${r}
        python eval_qa01.py --manual_seed=${s} --dataset='iemocap' --loss_r=${r}
    done
done