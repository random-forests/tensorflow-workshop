import argparse
import csv
import re

# parser definition
parser = argparse.ArgumentParser(prog='Make csv file from wikipedia data')

parser.add_argument('--data_dir', type=str, default='wikipedia/',
                    help='Local dir to the txt files.')

parser.add_argument('--output_path', type=str, default='wiki.csv',
                    help='Local path to the csv output.')

# constants
FILES = ['A_F.txt', 'G_M.txt', 'N_Z.txt']


def hex_to_rgb(hex_color):
  hex_color = int(hex_color, 16)
  r = (hex_color >> 16) & 0xFF
  g = (hex_color >> 8) & 0xFF
  b = hex_color & 0xFF
  return (r, g, b)


def preprocess_name(name):
  # keeping -, spaces, letters and numbers only
  name = re.sub(r'[^a-zA-Z0-9 -]', r'', name)
  # make all letters lower case
  name = name.lower()

  return name


def read_and_save_text(file_path, csv_writer):
  with open(file_path, 'r') as f:
    for line in f.readlines():
      # between each name and color there's a \t#
      name, hex_color = line.split('\t#')
      hex_color = re.sub('\n', '', hex_color)  # remove \n

      name = preprocess_name(name)
      r, g, b = hex_to_rgb(hex_color)

      csv_writer.writerow([name, r, g, b])


def main():
  try:
    args = parser.parse_args()
  except:
    print(parser.print_help())
    exit()

  # get args
  data_dir = args.data_dir
  output_path = args.output_path

  # create csv writer for the output
  output_file = open(output_path, 'a+')
  csv_writer = csv.writer(output_file)

  # read each wikipedia file and save in the csv
  for file_name in FILES:
    file_path = data_dir + file_name
    read_and_save_text(file_path, csv_writer)

if __name__ == '__main__':
  main()
