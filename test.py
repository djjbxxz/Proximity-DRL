import sys
sys.path.append('.')


if __name__=='__main__':
    from Test_with_baseline.gpuselector import *
    from Test_with_baseline.jobselector import *

    from Test_with_baseline.base import TestCase
    from Test_with_baseline.GUI import dump
    
    num_episode = 1
    seed = 2018

    testcase = TestCase(firstcomesfirstserve(),PPOGpuSelector('runs/ppo/2023-10-31 19_56_05_continueWitLlr_5e-6_noEntropyReg'),seed)
    # dump(testcase)
    testcase.run(num_episode)

    # testcase = TestCase(firstcomesfirstserve(),FirstAvlGpuSelector(),seed)
    # testcase.run(num_episode)