import numpy as np

FILES = [
["linear_check_py_output/model-heart_scale-s7-t0-B1-W-1,1.1,2.log.primal", "linear_check_sh_output/model-heart_scale-s7-t0-B1-Wheart_scale.wgt_by_label.log"],
["linear_check_py_output/model-heart_scale-s7-t0-B1.log.primal", "linear_check_sh_output/model-heart_scale-s7-t0-B1.log"],
["linear_check_py_output/model-heart_scale-s7-t0-W-1,1.1,2.log.primal", "linear_check_sh_output/model-heart_scale-s7-t0-Wheart_scale.wgt_by_label.log"],
["linear_check_py_output/model-heart_scale-s7-t0.log.primal", "linear_check_sh_output/model-heart_scale-s7-t0.log"],
["linear_check_py_output/model-heart_scale-s7-t2-B1-W-1,1.1,2.log.dual", "linear_check_sh_output/model-heart_scale-s7-t2-B1-Wheart_scale.wgt_by_label.log"],
["linear_check_py_output/model-heart_scale-s7-t2-B1.log.dual", "linear_check_sh_output/model-heart_scale-s7-t2-B1.log"],
["linear_check_py_output/model-heart_scale-s7-t2-W-1,1.1,2.log.dual", "linear_check_sh_output/model-heart_scale-s7-t2-Wheart_scale.wgt_by_label.log"],
["linear_check_py_output/model-heart_scale-s7-t2.log.dual", "linear_check_sh_output/model-heart_scale-s7-t2.log"],
["linear_check_py_output/model-heart_scale-s7-t5-B1-W-1,1.1,2.log.dual", "linear_check_sh_output/model-heart_scale-s7-t5-B1-Wheart_scale.wgt_by_label.log"],
["linear_check_py_output/model-heart_scale-s7-t5-B1.log.dual", "linear_check_sh_output/model-heart_scale-s7-t5-B1.log"],
["linear_check_py_output/model-heart_scale-s7-t5-W-1,1.1,2.log.dual", "linear_check_sh_output/model-heart_scale-s7-t5-Wheart_scale.wgt_by_label.log"],
["linear_check_py_output/model-heart_scale-s7-t5.log.dual", "linear_check_sh_output/model-heart_scale-s7-t5.log"],
]

for files in FILES:
    # Result by CVXPY: Just read
    cvxpy_result = np.loadtxt(files[0])
    
    # Result by LIBLINEAR: Extract and read
    liblinear_result = None
    with open(files[1]) as f:
        for line in f:
            line = line.strip()
            if line == 'w' or line == 'SV':
                liblinear_result = []
            elif liblinear_result is None:
                continue
            else:
                liblinear_result.append(float((line.split())[0]))
    liblinear_result = np.array(liblinear_result)
    
    print(files[0])
    print(f'Size: {cvxpy_result.shape} {liblinear_result.shape}')
    if cvxpy_result.shape == liblinear_result.shape:
        print(f'Difference: {np.linalg.norm(cvxpy_result - liblinear_result)}')
    else:
        print(f'Size mismatch')
    
    
