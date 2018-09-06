from subprocess import call

import os


models_group1 = ['deepmedic_ce_c25_90_120_150_b50_mb50_all', 'deepmedic_ce_50_50_c25_all']

models_group2 = ['deepmedic_ce_c25_75_100_125_b50_mb50_all', 'deepmedic_ce_45_60_75_b50_mb50_all', 'deepmedic_ce_50_50_c25_all_noaug', 'deepmedic_ce_c25_60_80_100_b50_mb50_all']

models_group3 = ['deepmedic_ce_75_100_125_b50_mb50_all', 'deepmedic_ce_c25_45_60_75_b50_mb50_all']

models_group4 = ['deepmedic_ce_50_50_c25_all', 'deepmedic_ce_50_50_c25_all_noaug', 'deepmedic_ce_c25_60_80_100_b50_mb50_all']


for model in models_group1:
	print(model)
	call(['python', 'predict.py', '--gpu', '1', '--cfg', model])
	call(['python', 'compress_data.py', '--cfg', model])
