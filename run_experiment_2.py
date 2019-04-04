from lead_scoring import run_ls_experiments
from petfinder import run_pf_experiments

def main():
	print("RUNNING EXPERIMENTS")

	print("Lead Scoring")
	run_ls_experiments()

	# print("DATASET: PetFinder")
	# run_pf_experiments()

if __name__ == '__main__':
	main()