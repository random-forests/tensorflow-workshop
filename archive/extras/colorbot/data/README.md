## About the dataset

The data available on this repo was taken from Wikipedia color dataset:

https://en.wikipedia.org/wiki/List_of_colors:_A-F  
https://en.wikipedia.org/wiki/List_of_colors:_G-M  
https://en.wikipedia.org/wiki/List_of_colors:_N-Z

The data was preprocessed and the final result is train.csv and test.csv

## How this data was generated?

* wikipedia/A_F.txt, wikipedia/G_M.txt, wikipedia/N_Z.txt were generated
manually using regex from the Wikipedia datasets:
	1. Copy manually the table available at the wiki page
	2. Remove header
	3. Change Olive Drab (#3) to Olive Drab 3 and Olive Drab #7 to Olive
	   Drab 7 in wikipedia/N_Z.txt
	3. Remove everything after the HEX number with regex:
		* replace: (#[0-9a-fA-F]*).*$ with \1  

**make sure to delete the files if you want to generate them again**
* Run *python aggregate_wiki.py* to generate the wiki.csv file
* Run *python partitioning_data.py* to generate the train.csv and
test.csv files
