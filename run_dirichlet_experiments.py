from car import run_c_experiments
from nursery import run_n_experiments

def main():
	print("RUNNING EXPERIMENTS")

	print("Dirichlet ENCODING")
	print("DATASET: CAR")
	run_c_experiments()
	print("DATASET: Nursery")
	run_n_experiments()


if __name__ == '__main__':
	main()