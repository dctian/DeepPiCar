import DeepPiCar

import logging
import sys

def main():
	# print system info
	logging.info('Starting DeepPiCar, system info: ' + sys.version)
	
	with DeepPiCar() as car:
		car.drive()
	
if __name__ == '__main__':
	main()