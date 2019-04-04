from insurance import run_i_experiments
from bike_sharing import run_bs_experiments

def main():
	print("RUNNING EXPERIMENTS")

	print("GIG ENCODING")
	print("DATASET: Insurance")
	run_i_experiments()
	print("DATASET: Bike Sharing")
	run_bs_experiments()

if __name__ == '__main__':
	main()