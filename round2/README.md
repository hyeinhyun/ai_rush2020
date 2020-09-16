AI RUSH 2020_round 2
===================
1. Tagging
	1. Problem 
	Image Classification (5class)
	2. Data
		- size : 32*32~
		- labeled data : 176752
		- meta data : category_1(8), category_2(118), category_3(851), category_4(1168)
	3. Evaluation Metric
		F1 score geometric mean
	4. Model
		- Efficientnet 3 + Efficientnet 3 (cutmix) + Efficientnet 5 (soft voting ensemble) 
			- label smoothing (0.2)
			- auto augmentation(cifar10)
			- cosine annealing scheduler

2. Music
		1. Problem 
		Music station Classification (5class)
		2. Model
		- Efficientnet 3 (cross validation ensemble)
				- training data : random crop & resize / stacking for making 3 channels / test data : center crop
