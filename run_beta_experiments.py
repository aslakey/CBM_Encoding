from lead_scoring import run_ls_experiments
from adult import run_a_experiments
from road_safety import run_rs_experiments

def main():
	print("RUNNING EXPERIMENTS")

	print("BETA ENCODING")
	print("DATASET: ADULT")
	run_a_experiments()
	print("DATASET: Road Safety")
	run_rs_experiments()


if __name__ == '__main__':
	main()