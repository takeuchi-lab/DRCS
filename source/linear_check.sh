OUTDIR=linear_check_sh_output
mkdir -p $OUTDIR

for s in 7; do
    for t in 0 2 5; do # 0: "LINEAR", 5: "LINEAR_KERNEL", 2: "RBF"
        for B in -1 1; do
            for W in "" heart_scale.wgt_by_label; do
                if [ "$B" != "-1" ]; then
                    B_param="-B $B"
                    B_param_file="-B$B"
                else
                    B_param=
                    B_param_file=
                fi
                
                if [ "$W" = "" ]; then
                    W_param=
                    W_param_file=
                else
                    W_param="-W $W"
                    W_param_file="-W$W"
                fi

                ./train -e 0.000001 -s $s -t $t $B_param $W_param heart_scale $OUTDIR/model-heart_scale-s$s-t$t$B_param_file$W_param_file.log
            done
        done
    done
done
