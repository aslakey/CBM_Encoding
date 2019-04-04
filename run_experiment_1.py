from adult import run_a_experiments
from road_safety import run_rs_experiments
from car import run_c_experiments
from nursery import run_n_experiments
from insurance import run_i_experiments
from bike_sharing import run_bs_experiments

def main():
	print("RUNNING EXPERIMENTS")

	print("BETA ENCODING")
	print("DATASET: ADULT")
	run_a_experiments()
	print("DATASET: Road Safety")
	run_rs_experiments()

	print("Dirichlet ENCODING")
	print("DATASET: CAR")
	run_c_experiments()
	print("DATASET: Nursery")
	run_n_experiments()

	print("GIG ENCODING")
	print("DATASET: Insurance")
	run_i_experiments()
	print("DATASET: Bike Sharing")
	run_bs_experiments()

if __name__ == '__main__':
	main()