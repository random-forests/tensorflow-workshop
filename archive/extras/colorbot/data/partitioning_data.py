import argparse
import csv
import random


# parser definition
parser = argparse.ArgumentParser(prog='Generate data partition')

parser.add_argument('--data_path', type=str, default='wiki.csv',
                    help='Local path to the csv file in your machine.'
                         'The file is not expected to have a header, '
                         'but must have the following structure:'
                         'name, red, blue, green')

parser.add_argument('--test_size', type=int, default=100,
                    help='Expected size of the training dataset.'
                         'Default=100')

parser.add_argument('--test_path', type=str, default='test.csv',
                    help='Path to the test csv.')

parser.add_argument('--train_path', type=str, default='train.csv',
                    help='Path to the train csv.')


# constants
HEADER = ['name', 'red', 'green', 'blue']


# helper functions
def write_csv(csv_file, content):
  def _write_csv_header(csv_writer):
    csv_writer.writerow(HEADER)

  def _write_csv_content(csv_writer, content):
    for row in content:
      csv_writer.writerow(row)

  csv_file = open(csv_file, 'a+')
  csv_writer = csv.writer(csv_file)

  _write_csv_header(csv_writer)
  _write_csv_content(csv_writer, content)

  csv_file.close()


def read_csv(csv_reader):
  data = []

  # dictionary to keep track of color_names
  color_names = {}

  for line in csv_reader:
    name, red, green, blue = line
    # ignore duplicates
    if name not in color_names:
      color_names[name] = 1
      data.append([name, red, green, blue])

  return data


def main():
  try:
    args = parser.parse_args()
  except:
    print(parser.print_help())
    exit()

  # get args
  data_path = args.data_path
  test_size = args.test_size
  train_file = args.train_path
  test_file = args.test_path

  # csv file: name, red, green, blue
  csv_file = open(data_path, 'rb')
  csv_reader = csv.reader(csv_file)

  # save all csv data in memory
  data = read_csv(csv_reader)
  csv_file.close()

  # shuflle data
  random.shuffle(data)

  # partionate data and save into train and test file
  write_csv(test_file, data[:test_size])
  write_csv(train_file, data[test_size:])

if __name__ == '__main__':
  main()
