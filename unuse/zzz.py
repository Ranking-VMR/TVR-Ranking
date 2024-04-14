import numpy as np

a = [76292, 66863, 56876, 21582, 66233, 37222, 57872, 65690, 17797, 77353, 30616, 68082, 27960, 31550, 93483, 48526, 21537, 27565, 38577, 32618, 97047, 93900, 7322, 63811, 87922, 58197, 27098, 43326, 33695, 18328, 58609, 17460, 92706, 62721, 35877, 79334, 30985, 57640, 16211, 1694, 64252, 77726, 80890, 85745, 7684, 61431, 74269, 63662, 49645, 1441, 60957, 65241, 89491, 49477, 78574, 80390, 3190, 26746, 65032, 46685, 41480, 39490, 37638, 9360, 88450, 6928, 80201, 63951, 91263, 40077, 41726, 59822, 25379, 9092, 21284, 5631, 44391, 71773, 84285, 11296, 13542, 23097, 13444, 85646, 90082, 35002, 30201, 91705, 72531, 17260, 67741, 56935, 97001, 45200, 67666, 93052, 91337, 10195, 32651, 88421, 11283, 72055, 82280, 55142, 87656, 33335, 25972, 21762, 75552, 2544, 24155, 69236, 33817, 62504, 93012, 92992, 20171, 70241, 2140, 84204, 35786, 79792, 86125, 93200, 32897, 50491, 59549, 25288]
a_np = np.array(a)

# Vectorized comparison
matrix = (a_np[:, None] != a_np).astype(int)

print(matrix)